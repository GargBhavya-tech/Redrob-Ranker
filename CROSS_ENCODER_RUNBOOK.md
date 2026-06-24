# Cross-Encoder Re-rank — Optional Upside (you benchmark it)

The cross-encoder stage (`src/stage3_cross_encoder.py`) is OFF unless the model
is available. The pipeline is strong without it (recent-role doc weighting
already captures most of its benefit — see DECISION_MAP ticket #3).

## To benchmark on your machine (where HuggingFace works)

```bash
pip install sentence-transformers
# 1. produce the baseline (no cross-encoder)
REDROB_DISABLE_CROSS_ENCODER=1 python -m src.run --candidates data/candidates.jsonl --out sub_base.csv
# 2. produce the cross-encoder version (wire rerank() into run.py after fusion)
python -m src.run --candidates data/candidates.jsonl --out sub_ce.csv
# 3. compare both against the human anchor
python eval/eval_anchor.py eval/anchor_labels.json sub_base.csv sub_ce.csv
```

Keep the cross-encoder ONLY if it raises anchor pairwise-agreement. Blend default
0.5; try 0.3–0.7. If it doesn't help (likely, given the doc weighting), leave it
disabled — "no external model download at inference" is the more reproducible,
more defensible story.
