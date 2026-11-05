# Paper decision ledger

`paper_test_predictions.jsonl` contains the 332 saved per-trajectory decisions of the model used in the paper. It omits raw model responses and trajectory content. Run `python -m scripts.verify_paper_metrics` to recompute the Exact-F1, false-alarm rate, and absolute step shift. This ledger is derived from the AFTraj test split and is available under CC BY 4.0.
