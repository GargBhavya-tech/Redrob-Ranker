"""
Stage 1: stream all 100K candidates, extract structured features, and write
the result to a Polars DataFrame (parquet) for fast downstream use.

This is the "entire pool" pass — must be fast and memory-light. Embeddings
are NOT computed here; that happens only for the Stage 1 shortlist.
"""

import time
import polars as pl

from .schema import iter_candidates
from .features import extract_features
from .date_utils import parse_date


def build_feature_table(jsonl_path: str, dataset_now=None) -> pl.DataFrame:
    if dataset_now is None:
        # Two passes would be wasteful; instead use a fixed known dataset
        # "now" date. The dataset's last_active_date values cluster near
        # 2026-05-27 (observed). Hardcode for determinism / speed.
        from datetime import date
        dataset_now = date(2026, 5, 27)

    rows = []
    for cand in iter_candidates(jsonl_path):
        rows.append(extract_features(cand, dataset_now))

    return pl.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    df = build_feature_table("data/candidates.jsonl")
    print(f"Extracted features for {len(df)} candidates in {time.time()-t0:.2f}s")
    print(df.describe())

    out_path = "data/features.parquet"
    df.write_parquet(out_path)
    print(f"Wrote {out_path}")
