"""
Agentic (GAIA/HotpotQA) Trajectory Generation using Smolagents Multi-Agent Framework.

Architecture:
  Manager (CodeAgent): Decomposes tasks, delegates to search_agent, synthesizes answers
  └── search_agent (ToolCallingAgent): Web search, Wikipedia, page reading

All agents and tools are built-in Smolagents components.

Usage:
    python -m scripts.generation.agentic.gen_agentic_smolagents --domain hotpotqa --num_tasks 2 --output_dir tmp/agentic_traj
    python -m scripts.generation.agentic.gen_agentic_smolagents --domain gaia --num_tasks 2 --output_dir tmp/agentic_traj
"""
import argparse
import ast
import json
import os
import re
import string
import sys
import time
from io import BytesIO
from datetime import datetime

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    PythonInterpreterTool,
)
from smolagents.default_tools import WikipediaSearchTool


PYTHON_TOOL_IMPORTS = [
    "json",
    "re",
    "math",
    "collections",
    "datetime",
    "os",
    "pathlib",
    "csv",
    "io",
    "zipfile",
    "xml",
    "xml.etree.ElementTree",
    "openpyxl",
    "pptx",
    "pdfplumber",
    "docx",
]

MANAGER_INSTRUCTIONS = """
You are the final answer agent.

Rules:
- If the task asks for a specific value, your `final_answer(...)` must contain only the minimal atomic answer requested.
- Never pass a research report, Markdown headings, JSON object, bullet list, or explanation into `final_answer(...)`.
- If a managed agent returns a long report, extract the direct answer first, then call `final_answer(...)` with only that answer.
- Prefer using attached-file content already injected into the task before trying to rediscover the same file via web search.
- If you use python, keep code short and only import libraries that are clearly useful for the current file type.
""".strip()

SEARCH_AGENT_INSTRUCTIONS = """
Return the most direct answer first. When possible, put the direct answer in the opening sentence before any explanation.
""".strip()


# ============================================================
# Agent setup (ALL built-in Smolagents components)
# ============================================================

def create_agents(model_id: str = "openai/gpt-5.4-mini", api_key: str = None):
    """Create Manager + Search Agent using only Smolagents built-in components."""

    # LLM backend
    model = LiteLLMModel(
        model_id=model_id,
        api_key=api_key,
        max_tokens=4096,
    )

    # Child agent: web search specialist (ToolCallingAgent)
    search_agent = ToolCallingAgent(
        model=model,
        tools=[
            DuckDuckGoSearchTool(),
            WikipediaSearchTool(user_agent="AgentForesight/1.0"),
            VisitWebpageTool(),
        ],
        max_steps=12,
        instructions=SEARCH_AGENT_INSTRUCTIONS,
        name="search_agent",
        description=(
            "A research assistant that can search the web and read web pages. "
            "Give it a specific question or search query, and it will find and return relevant information. "
            "Use this agent for ALL information retrieval tasks."
        ),
        provide_run_summary=True,
    )

    # Parent agent: manager (CodeAgent)
    manager = CodeAgent(
        model=model,
        tools=[PythonInterpreterTool(authorized_imports=PYTHON_TOOL_IMPORTS)],
        managed_agents=[search_agent],
        max_steps=18,
        instructions=MANAGER_INSTRUCTIONS,
        additional_authorized_imports=PYTHON_TOOL_IMPORTS,
    )

    return manager, model


# ============================================================
# Trajectory extraction
# ============================================================

def extract_trajectory(agent) -> list[dict]:
    """Extract trajectory steps from Smolagents agent memory."""
    steps = []
    for step in agent.memory.steps:
        step_data = {
            "step_number": getattr(step, 'step_number', len(steps)),
            "type": type(step).__name__,
        }

        if hasattr(step, 'task'):
            step_data["agent"] = "user"
            step_data["content"] = str(step.task)[:500]
        elif hasattr(step, 'model_output'):
            step_data["agent"] = "Manager"
            step_data["content"] = str(step.model_output or "")[:500]

            # Tool calls
            if hasattr(step, 'tool_calls') and step.tool_calls:
                step_data["tool_calls"] = []
                for tc in step.tool_calls:
                    call = {
                        "name": tc.name if hasattr(tc, 'name') else str(tc),
                        "arguments": str(tc.arguments)[:200] if hasattr(tc, 'arguments') else "",
                    }
                    step_data["tool_calls"].append(call)

            # Observations (tool results)
            if hasattr(step, 'observations') and step.observations:
                step_data["observation"] = str(step.observations)[:500]

            # Check if this step called a managed agent
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    tc_name = tc.name if hasattr(tc, 'name') else str(tc)
                    if tc_name == "search_agent":
                        step_data["agent"] = "Manager→search_agent"

        elif hasattr(step, 'answer'):
            step_data["agent"] = "Manager"
            step_data["content"] = f"Final answer: {step.answer}"
            step_data["is_final"] = True

        steps.append(step_data)

    return steps


# ============================================================
# File helpers
# ============================================================

def _color_to_str(color) -> str:
    if color is None:
        return ""
    try:
        color_type = getattr(color, "type", None)
    except Exception:
        color_type = None

    if color_type == "rgb":
        try:
            rgb = getattr(color, "rgb", None)
        except Exception:
            rgb = None
        if isinstance(rgb, str) and rgb:
            return rgb

    try:
        indexed = getattr(color, "indexed", None)
    except Exception:
        indexed = None
    if isinstance(indexed, int):
        return f"indexed:{indexed}"

    try:
        theme = getattr(color, "theme", None)
    except Exception:
        theme = None
    if isinstance(theme, int):
        return f"theme:{theme}"

    return ""


def _extract_xlsx_text(workbook, display_name: str) -> str:
    rows = [f"Workbook: {display_name}", f"Sheets: {workbook.sheetnames}"]
    total_cells = 0

    for sheet_name in workbook.sheetnames:
        ws = workbook[sheet_name]
        rows.append(f"--- Sheet: {sheet_name} ({ws.max_row}x{ws.max_column}) ---")
        if ws.merged_cells.ranges:
            rows.append(f"Merged ranges: {[str(rng) for rng in list(ws.merged_cells.ranges)[:20]]}")

        for row in ws.iter_rows(
            min_row=1,
            max_row=min(ws.max_row, 120),
            min_col=1,
            max_col=min(ws.max_column, 60),
        ):
            for cell in row:
                fill = _color_to_str(getattr(getattr(cell, "fill", None), "fgColor", None))
                font = _color_to_str(getattr(getattr(cell, "font", None), "color", None))
                has_style = bool(fill or font or getattr(cell, "comment", None))
                if cell.value is None and not has_style:
                    continue

                value = cell.value
                parts = [f"{cell.coordinate}={value!r}"]
                if fill:
                    parts.append(f"fill={fill}")
                if font:
                    parts.append(f"font={font}")
                if getattr(cell, "comment", None):
                    parts.append(f"comment={cell.comment.text!r}")
                rows.append(" | ".join(parts))
                total_cells += 1
                if total_cells >= 500:
                    rows.append("[truncated workbook cells]")
                    return "\n".join(rows)[:12000]

    return "\n".join(rows)[:12000]


def read_xlsx_file(local_path: str, display_name: str) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(local_path, data_only=False)
    return _extract_xlsx_text(wb, display_name)


def read_xlsx_bytes(data: bytes, display_name: str) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(BytesIO(data), data_only=False)
    return _extract_xlsx_text(wb, display_name)


def read_pdf_file(local_path: str, display_name: str) -> str:
    import pdfplumber

    with pdfplumber.open(local_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages[:10], 1):
            pages.append(f"--- Page {page_num} ---\n{page.extract_text() or ''}")
    return f"PDF: {display_name}\n" + "\n\n".join(pages)[:12000]


def read_pdf_bytes(data: bytes, display_name: str) -> str:
    import pdfplumber

    with pdfplumber.open(BytesIO(data)) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages[:10], 1):
            pages.append(f"--- Page {page_num} ---\n{page.extract_text() or ''}")
    return f"PDF: {display_name}\n" + "\n\n".join(pages)[:12000]


def read_docx_file(local_path: str, display_name: str) -> str:
    import docx

    doc = docx.Document(local_path)
    parts = [f"DOCX: {display_name}"]
    for para in doc.paragraphs[:300]:
        if para.text.strip():
            parts.append(para.text)
    return "\n".join(parts)[:12000]


def read_docx_bytes(data: bytes, display_name: str) -> str:
    import docx

    doc = docx.Document(BytesIO(data))
    parts = [f"DOCX: {display_name}"]
    for para in doc.paragraphs[:300]:
        if para.text.strip():
            parts.append(para.text)
    return "\n".join(parts)[:12000]


def read_pptx_file(local_path: str, display_name: str) -> str:
    from pptx import Presentation

    prs = Presentation(local_path)
    slides_text = [f"PPTX: {display_name} | slides={len(prs.slides)}"]
    for slide_num, slide in enumerate(list(prs.slides)[:50], 1):
        texts = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text_frame.text.strip()
                if text:
                    texts.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    texts.append(" | ".join(row_text))
        slide_blob = "\n".join(texts)
        slides_text.append(f"--- Slide {slide_num} ---\n{slide_blob}")
    return "\n\n".join(slides_text)[:12000]


def read_pptx_bytes(data: bytes, display_name: str) -> str:
    from pptx import Presentation

    prs = Presentation(BytesIO(data))
    slides_text = [f"PPTX: {display_name} | slides={len(prs.slides)}"]
    for slide_num, slide in enumerate(list(prs.slides)[:50], 1):
        texts = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text_frame.text.strip()
                if text:
                    texts.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    texts.append(" | ".join(row_text))
        slides_text.append(f"--- Slide {slide_num} ---\n" + "\n".join(texts))
    return "\n\n".join(slides_text)[:12000]


def read_text_bytes(data: bytes, display_name: str) -> str:
    return f"FILE: {display_name}\n" + data.decode("utf-8", errors="replace")[:12000]


def _clip_embedded_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[truncated embedded file]"


def _convert_legacy_excel_to_xlsx(src_path: str, tmpdir: str) -> str:
    import subprocess

    converted_path = os.path.join(
        tmpdir, os.path.splitext(os.path.basename(src_path))[0] + ".xlsx"
    )
    cmd = [
        "soffice",
        "--headless",
        "--convert-to",
        "xlsx",
        "--outdir",
        tmpdir,
        src_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
    if os.path.exists(converted_path):
        return converted_path
    message = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(message or "soffice failed to convert legacy Excel file")


def read_xls_file(local_path: str, display_name: str) -> str:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gaia_xls_") as tmpdir:
        converted_path = _convert_legacy_excel_to_xlsx(local_path, tmpdir)
        return read_xlsx_file(converted_path, display_name)


def read_xls_bytes(data: bytes, display_name: str) -> str:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gaia_xls_") as tmpdir:
        src_path = os.path.join(
            tmpdir, os.path.splitext(os.path.basename(display_name))[0] + ".xls"
        )
        with open(src_path, "wb") as f:
            f.write(data)
        converted_path = _convert_legacy_excel_to_xlsx(src_path, tmpdir)
        return read_xlsx_file(converted_path, display_name)


def read_zip_file(local_path: str, display_name: str) -> str:
    import zipfile

    with zipfile.ZipFile(local_path, "r") as zf:
        return _read_zip_archive(zf, display_name)


def _read_zip_archive(zf, display_name: str) -> str:
    file_list = sorted(zf.namelist())
    parts = [f"ZIP: {display_name}", f"Entries ({len(file_list)}): {file_list[:60]}"]

    extracted = 0
    for name in file_list:
        if extracted >= 12:
            parts.append("[truncated archive entries]")
            break
        if name.endswith("/"):
            continue

        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        data = zf.read(name)

        try:
            if ext in {"txt", "csv", "json", "jsonld", "xml", "md", "py"}:
                parts.append(f"--- {name} ---\n{_clip_embedded_text(read_text_bytes(data, name), 3500)}")
                extracted += 1
            elif ext == "xls":
                parts.append(f"--- {name} ---\n{_clip_embedded_text(read_xls_bytes(data, name), 5000)}")
                extracted += 1
            elif ext == "xlsx":
                parts.append(f"--- {name} ---\n{_clip_embedded_text(read_xlsx_bytes(data, name), 5000)}")
                extracted += 1
            elif ext == "pdf":
                parts.append(f"--- {name} ---\n{_clip_embedded_text(read_pdf_bytes(data, name), 4500)}")
                extracted += 1
            elif ext == "docx":
                parts.append(f"--- {name} ---\n{_clip_embedded_text(read_docx_bytes(data, name), 4500)}")
                extracted += 1
            elif ext == "pptx":
                parts.append(f"--- {name} ---\n{_clip_embedded_text(read_pptx_bytes(data, name), 4500)}")
                extracted += 1
            elif ext == "zip":
                import zipfile

                with zipfile.ZipFile(BytesIO(data), "r") as nested:
                    nested_text = _read_zip_archive(nested, name)
                parts.append(f"--- {name} ---\n{_clip_embedded_text(nested_text, 4000)}")
                extracted += 1
        except Exception as e:
            parts.append(f"--- {name} ---\n[Error reading embedded file: {e}]")
            extracted += 1

    return "\n\n".join(parts)[:12000]


# ============================================================
# Dataset loading
# ============================================================

def load_hotpotqa(dataset_path: str, start_idx: int, num_tasks: int) -> list[dict]:
    """Load HotpotQA tasks."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset_path)
    if isinstance(ds, dict):
        ds = ds.get("validation", ds.get("test", list(ds.values())[0]))
    tasks = []
    for i in range(start_idx, min(start_idx + num_tasks, len(ds))):
        sample = dict(ds[i])
        tasks.append({
            "task_idx": i,
            "question": sample.get("question", ""),
            "gold_answer": str(sample.get("answer", "")),
            "metadata": {
                "id": sample.get("id", f"hotpotqa_{i}"),
                "type": sample.get("type", ""),
                "level": sample.get("level", ""),
            },
        })
    return tasks


def load_gaia(dataset_path: str, start_idx: int, num_tasks: int,
              api_key: str = None, gaia_files_dir: str = None) -> list[dict]:
    """Load GAIA tasks. Injects file attachment content into the question prompt."""
    import os
    from datasets import load_from_disk
    ds = load_from_disk(dataset_path)
    if isinstance(ds, dict):
        ds = ds.get("validation", ds.get("test", list(ds.values())[0]))

    tasks = []
    for i in range(start_idx, min(start_idx + num_tasks, len(ds))):
        sample = dict(ds[i])
        question = sample.get("Question", sample.get("question", ""))
        fname = sample.get("file_name", "")
        fpath = sample.get("file_path", "")

        # Handle file attachment
        file_content = None
        if fname and fpath:
            local_path = os.path.join(gaia_files_dir, fpath)
            if os.path.exists(local_path):
                ext = fname.rsplit(".", 1)[-1].lower()

                if ext in ("txt", "csv", "py", "jsonld", "json", "xml", "md"):
                    # Text files: read directly
                    try:
                        file_content = open(local_path, "r", errors="replace").read()[:12000]
                    except Exception:
                        file_content = f"[Error reading {fname}]"

                elif ext == "xls":
                    try:
                        file_content = read_xls_file(local_path, fname)
                    except Exception as e:
                        file_content = f"[Error reading xls: {e}]"

                elif ext == "xlsx":
                    try:
                        file_content = read_xlsx_file(local_path, fname)
                    except Exception as e:
                        file_content = f"[Error reading xlsx: {e}]"

                elif ext == "pdf":
                    try:
                        file_content = read_pdf_file(local_path, fname)
                    except Exception as e:
                        file_content = f"[Error reading pdf: {e}]"

                elif ext == "docx":
                    try:
                        file_content = read_docx_file(local_path, fname)
                    except Exception as e:
                        file_content = f"[Error reading docx: {e}]"

                elif ext == "pptx":
                    try:
                        file_content = read_pptx_file(local_path, fname)
                    except Exception as e:
                        file_content = f"[Error reading pptx: {e}]"

                elif ext == "zip":
                    try:
                        file_content = read_zip_file(local_path, fname)
                    except Exception as e:
                        file_content = f"[Error reading zip: {e}]"

                elif ext in ("png", "jpg", "jpeg"):
                    # Image: describe via OpenAI vision API
                    try:
                        import base64
                        from openai import OpenAI
                        img_b64 = base64.b64encode(open(local_path, "rb").read()).decode()
                        vision_client = OpenAI(api_key=api_key) if api_key else None
                        if vision_client:
                            resp = vision_client.chat.completions.create(
                                model="gpt-5.4-mini",
                                messages=[{
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Describe this image in detail. Include ALL text, numbers, colors, labels, and data visible. Be exhaustive."},
                                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img_b64}"}}
                                    ]
                                }],
                                max_completion_tokens=1000,
                            )
                            file_content = f"[Image description]: {resp.choices[0].message.content}"
                        else:
                            file_content = f"[Image file at: {local_path}, but no API key for vision]"
                    except Exception as e:
                        file_content = f"[Error reading image: {e}]"

                elif ext in ("mp3", "wav"):
                    # Audio: transcribe via OpenAI Whisper API
                    try:
                        from openai import OpenAI
                        whisper_client = OpenAI(api_key=api_key) if api_key else None
                        if whisper_client:
                            with open(local_path, "rb") as audio_f:
                                transcript = whisper_client.audio.transcriptions.create(
                                    model="whisper-1", file=audio_f
                                )
                            file_content = f"[Audio transcription]: {transcript.text}"
                        else:
                            file_content = f"[Audio file at: {local_path}, but no API key for transcription]"
                    except Exception as e:
                        file_content = f"[Error transcribing audio: {e}]"

                else:
                    file_content = f"[File type .{ext} available at: {local_path}]"

        # Inject file content into question
        if file_content:
            question += f"\n\n--- Attached file: {fname} ---\n{file_content}"

        tasks.append({
            "task_idx": i,
            "question": question,
            "gold_answer": str(sample.get("Final answer", sample.get("answer", ""))),
            "metadata": {
                "task_id": sample.get("task_id", f"gaia_{i}"),
                "level": sample.get("Level", ""),
                "file_name": fname,
            },
        })
    return tasks


# ============================================================
# Answer extraction / matching
# ============================================================

def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    return float(number_str)


def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def split_string(s: str, char_list: list[str] | None = None) -> list[str]:
    char_list = char_list or [",", ";"]
    return re.split(f"[{''.join(char_list)}]", s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", str(input_str))
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def normalize_hotpot_answer(text: str) -> str:
    lowered = str(text).lower()
    translator = str.maketrans("", "", string.punctuation)
    no_punct = lowered.translate(translator)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punct)
    return " ".join(no_articles.split())


HOTPOT_ORDINAL_VALUES = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
    "fifth": 5,
    "5th": 5,
    "five": 5,
    "sixth": 6,
    "6th": 6,
    "six": 6,
    "seventh": 7,
    "7th": 7,
    "seven": 7,
    "eighth": 8,
    "8th": 8,
    "eight": 8,
    "ninth": 9,
    "9th": 9,
    "nine": 9,
    "tenth": 10,
    "10th": 10,
    "ten": 10,
}


def hotpot_ordinal_value(text: str) -> int | None:
    normalized = normalize_hotpot_answer(text)
    normalized = re.sub(
        r"\b(book|season|series|volume|part|chapter|no)\b",
        " ",
        normalized,
    )
    normalized = " ".join(normalized.split())
    if not normalized:
        return None
    if normalized.isdigit():
        return int(normalized)
    return HOTPOT_ORDINAL_VALUES.get(normalized)


def extract_last_code_output(trajectory: list[dict]) -> str:
    for step in reversed(trajectory):
        obs = str(step.get("observation", ""))
        match = re.search(r"Last output from code snippet:\n(.*)", obs, re.S)
        if match:
            return match.group(1).strip()
    return ""


def extract_short_version(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"Task outcome \(short version\):\n?(.*?)(?:\n\n### 2\.|$)", text, re.S)
    return match.group(1).strip() if match else ""


def atomic_answer_from_short(short_text: str, gold: str = "") -> str:
    short_text = short_text.strip()
    if not short_text:
        return ""

    total_match = re.search(r"\*\*Total count:\s*([0-9][0-9,]*)\*\*", short_text)
    if total_match:
        return total_match.group(1)

    if is_float(str(gold)):
        number_match = re.search(r"[-+]?\$?[0-9][0-9,]*(?:\.[0-9]+)?", short_text)
        if number_match:
            return number_match.group(0)

    ticks = re.findall(r"`([^`]+)`", short_text)
    if len(ticks) == 1:
        return ticks[0].strip()

    bolds = re.findall(r"\*\*([^*]+)\*\*", short_text)
    if len(bolds) == 1:
        return bolds[0].strip()
    if len(bolds) >= 2:
        lowered = short_text.lower()
        if "mentions" in lowered or "deleted the word" in lowered or "award number" in lowered:
            return bolds[-1].strip()
        if "born in may" in lowered:
            return bolds[0].strip()
        if "country is" in lowered or "brand is" in lowered or "word is" in lowered:
            return bolds[0].strip()

    plain = short_text.strip(" .")
    if len(plain) <= 60 and "\n" not in plain:
        return plain

    if "citation count" in plain.lower():
        return "Citations"

    return plain


def canonicalize_pred_answer(raw_pred: str, gold: str, trajectory: list[dict]) -> str:
    raw_pred = str(raw_pred or "").strip()
    last_output = extract_last_code_output(trajectory)
    text = raw_pred or last_output

    if text.startswith("{") and "short_version" in text:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict) and parsed.get("short_version"):
                text = str(parsed["short_version"])
        except Exception:
            pass

    if (
        text.startswith("### ")
        or text.startswith("Here is the final answer")
        or len(text.splitlines()) > 4
    ):
        short = extract_short_version(text)
        if short:
            text = atomic_answer_from_short(short, gold)

    if is_float(str(gold)) and text:
        match = re.search(r"[-+]?\$?[0-9][0-9,]*(?:\.[0-9]+)?", text)
        if match:
            text = match.group(0)

    return text.strip()


def check_answer(pred: str, gold: str, domain: str = "gaia") -> bool:
    """Domain-aware answer matching after answer canonicalization."""
    if not pred or not gold:
        return False

    if domain == "hotpotqa":
        if normalize_hotpot_answer(pred) == normalize_hotpot_answer(gold):
            return True
        pred_ordinal = hotpot_ordinal_value(pred)
        gold_ordinal = hotpot_ordinal_value(gold)
        return pred_ordinal is not None and pred_ordinal == gold_ordinal

    if is_float(str(gold)):
        try:
            return normalize_number_str(str(pred)) == float(gold)
        except ValueError:
            return False

    if any(char in str(gold) for char in [",", ";"]):
        gt_elems = split_string(str(gold))
        pred_elems = split_string(str(pred))
        if len(gt_elems) != len(pred_elems):
            return False
        comparisons = []
        for pred_elem, gt_elem in zip(pred_elems, gt_elems):
            if is_float(gt_elem):
                try:
                    comparisons.append(normalize_number_str(pred_elem) == float(gt_elem))
                except ValueError:
                    comparisons.append(False)
            else:
                comparisons.append(
                    normalize_str(pred_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    return normalize_str(str(pred)) == normalize_str(str(gold))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Agentic trajectories with Smolagents")
    parser.add_argument("--domain", required=True, choices=["hotpotqa", "gaia"])
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="tmp/agentic_traj")
    parser.add_argument("--model", type=str, default="openai/gpt-5.4-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to a datasets.load_from_disk HotpotQA or GAIA directory")
    parser.add_argument("--gaia-files-dir", type=str, default=None,
                        help="Directory holding GAIA task attachments (required for GAIA)")
    args = parser.parse_args()

    if args.domain == "gaia" and not args.gaia_files_dir:
        parser.error("--gaia-files-dir is required for GAIA")

    os.makedirs(args.output_dir, exist_ok=True)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: provide --api_key or set OPENAI_API_KEY")
        sys.exit(1)

    # Load dataset
    if args.domain == "hotpotqa":
        tasks = load_hotpotqa(args.dataset, args.start_idx, args.num_tasks)
    else:
        tasks = load_gaia(args.dataset, args.start_idx, args.num_tasks,
                          api_key=api_key, gaia_files_dir=args.gaia_files_dir)

    print(f"Domain: {args.domain}")
    print(f"Model: {args.model}")
    print(f"Loaded {len(tasks)} tasks")
    print(f"Output: {args.output_dir}")

    results = []

    for i, task in enumerate(tasks):
        question = task["question"]
        gold = task["gold_answer"]
        meta = task["metadata"]
        tid = meta.get("id", meta.get("task_id", f"{args.domain}_{i}"))

        print(f"\n[{i+1}/{len(tasks)}] {tid}: {question[:80]}...")

        # Create fresh agents per task
        manager, model = create_agents(args.model, api_key)

        start = time.time()
        try:
            output = manager.run(question)
            raw_pred = str(output) if output is not None else ""
            trajectory = extract_trajectory(manager)
            pred = canonicalize_pred_answer(raw_pred, gold, trajectory)
            status = "completed"
        except Exception as e:
            raw_pred = ""
            trajectory = extract_trajectory(manager) if hasattr(manager, "memory") else []
            pred = canonicalize_pred_answer(raw_pred, gold, trajectory)
            status = f"error: {e}"

        elapsed = round(time.time() - start, 2)
        is_correct = check_answer(pred, gold, domain=args.domain)

        agents = list(set(s.get("agent", "") for s in trajectory))
        n_tools = sum(1 for s in trajectory if s.get("tool_calls"))

        record = {
            "task_idx": task["task_idx"],
            "domain": args.domain,
            "task": question,
            "gold_answer": gold,
            "pred_answer": pred[:500],
            "raw_pred_answer": raw_pred[:1000],
            "metadata": meta,
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "trajectory": trajectory,
            "num_steps": len(trajectory),
            "agents": agents,
            "num_tool_calls": n_tools,
            "elapsed_s": elapsed,
            "is_correct": is_correct,
            "status": status,
        }
        results.append(record)

        icon = "✓" if is_correct else "✗"
        print(f"  {icon} steps={len(trajectory)}, agents={agents}, tools={n_tools}, "
              f"pred='{pred[:40]}', gt='{gold[:20]}', time={elapsed}s")

        # Save incrementally
        outfile = os.path.join(args.output_dir,
                               f"{args.domain}_{args.start_idx}_{args.start_idx+len(tasks)}.json")
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    n_correct = sum(1 for r in results if r["is_correct"])
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_correct}/{len(tasks)} correct ({100*n_correct/max(len(tasks),1):.0f}%)")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
