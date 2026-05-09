"""Loader for AFTraj-2K parquet artifacts.

Reads ``aftraj_safe.parquet`` and ``aftraj_unsafe.parquet`` and yields a
unified record schema used by the inference scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

@dataclass
class TrajectoryRecord:
    conv_id: str
    domain: str
    label: str                               
    task: str
    gold_answer: str
    num_turns: int
    turns: list[dict]
    tools: list[dict] = field(default_factory=list)
    mistake_step: int = -1                         
    mistake_agent: str = ""
    mistake_reason: str = ""
    unsafe_source: str = ""                                                 

def _as_list(value) -> list:
    if value is None:
        return []
    return [v for v in value]

def _row_to_record(row: dict, label: str) -> TrajectoryRecord:
                                                                           
    turns = [dict(t) for t in _as_list(row.get("turns"))]
    tools = [dict(t) for t in _as_list(row.get("tools"))]
    return TrajectoryRecord(
        conv_id=str(row["conv_id"]),
        domain=str(row["domain"]),
        label=label,
        task=str(row.get("task", "")),
        gold_answer=str(row.get("gold_answer", "")),
        num_turns=int(row.get("num_turns", len(turns))),
        turns=turns,
        tools=tools,
        mistake_step=int(row.get("mistake_step", -1)),
        mistake_agent=str(row.get("mistake_agent", "")),
        mistake_reason=str(row.get("mistake_reason", "")),
        unsafe_source=str(row.get("unsafe_source", "")),
    )

def load_aftraj(data_dir: str | Path,
                domains: list[str] | None = None,
                splits: tuple[str, ...] = ("safe", "unsafe"),
                limit: int | None = None,
                paper_test_split: bool = False) -> list[TrajectoryRecord]:
    data_dir = Path(data_dir)
    files = {
        "safe":   data_dir / "aftraj_safe.parquet",
        "unsafe": data_dir / "aftraj_unsafe.parquet",
    }
    test_ids: dict[str, set[str]] | None = None
    if paper_test_split:
        sp = data_dir / "splits_test.json"
        if not sp.exists():
            raise FileNotFoundError(f"paper_test_split=True but missing: {sp}")
        sj = json.load(open(sp))
        test_ids = {"safe": set(sj["test_safe"]), "unsafe": set(sj["test_unsafe"])}

    records: list[TrajectoryRecord] = []
    for split in splits:
        if split not in files:
            raise ValueError(f"unknown split: {split!r}")
        path = files[split]
        if not path.exists():
            raise FileNotFoundError(f"parquet not found: {path}")
        df = pd.read_parquet(path)
        if domains is not None:
            df = df[df["domain"].isin(domains)]
        if test_ids is not None:
            df = df[df["conv_id"].isin(test_ids[split])]
        for _, row in df.iterrows():
            records.append(_row_to_record(row.to_dict(), split))
    if limit is not None:
        records = records[:limit]
    return records
