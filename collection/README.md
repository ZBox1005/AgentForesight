# AFTraj trajectory collection scripts

The scripts in this directory are the Math AutoGen Swarm and Agentic smolagents rollout implementations used to build the corresponding AFTraj trajectory pools. They include the original role instructions, tools, trajectory extraction, and answer checks. Local absolute data paths were replaced with command-line arguments for release; the rollout logic was left in place.

Install the recorded collection-environment packages with `pip install -r collection/requirements.txt`. Set `OPENAI_API_KEY` in your environment. The scripts expect task corpora already stored using Hugging Face `datasets.save_to_disk`; these upstream corpora and GAIA attachments are not bundled with AFTraj.

## Math: MATH-500 with AutoGen Swarm

```bash
python -m collection.gen_math_autogen \
  --dataset /path/to/saved/MATH500 \
  --num_tasks 2 --start_idx 0 --output_dir ./outputs/math
```

`gen_math_autogen.py` creates `MathSolver` and `Verifier` agents with `compute` and `verify_answer` tools. It uses `gpt-5.4-mini` by default and writes JSON trajectory records incrementally.

## Agentic: HotpotQA or GAIA with smolagents

```bash
python -m collection.gen_agentic_smolagents \
  --domain hotpotqa --dataset /path/to/saved/hotpotqa_val \
  --num_tasks 2 --start_idx 0 --output_dir ./outputs/hotpotqa

python -m collection.gen_agentic_smolagents \
  --domain gaia --dataset /path/to/saved/gaia_full \
  --gaia-files-dir /path/to/gaia_files \
  --num_tasks 2 --start_idx 0 --output_dir ./outputs/gaia
```

`gen_agentic_smolagents.py` creates a `CodeAgent` manager and `search_agent` (`ToolCallingAgent`). The search tools are `DuckDuckGoSearchTool`, `WikipediaSearchTool`, and `VisitWebpageTool`. It uses `openai/gpt-5.4-mini` by default. GAIA attachment handling includes text, Office files, PDFs, images, and audio; the legacy `.xls` path uses the `soffice` executable. The source constructors set `max_tokens=4096` for the smolagents model and do not explicitly set temperature, `top_p`, or seed. Dynamic search results depend on the live backends.

The pinned package revisions above come from the recorded environment snapshot. Curated trajectories are available in [AFTraj](https://huggingface.co/datasets/ZBox008003/AFTraj).
