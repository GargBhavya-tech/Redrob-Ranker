"""
End-to-end pipeline: candidates.jsonl -> submission.csv

Run with:
    python -m src.run --candidates data/candidates.jsonl --out data/submission.csv

Stages:
  1. Stream-decode all 100K candidates (msgspec), extract structured features
  2. Stage 1 score + shortlist top-K by structural signals
  3. Stage 2 TF-IDF cosine similarity (semantic) on the shortlist
  4. RRF fusion -> top 100
  5. Enrich top 100 with grounding facts
  6. Generate reasoning + write submission CSV
"""

import argparse
import time

from .build_features import build_feature_table
from .stage1_shortlist import compute_stage1_score, shortlist
from .stage2_semantic import compute_semantic_scores
from .fusion import top100
from .enrich import enrich
from .reasoning import build_submission

import polars as pl


def run(candidates_path: str, out_path: str, shortlist_k: int = 1000):
    t0 = time.time()

    print("[1/6] Extracting features for all candidates...")
    df = build_feature_table(candidates_path)
    print(f"      {len(df)} candidates, {time.time()-t0:.1f}s elapsed")

    print("[2/6] Computing Stage 1 score + shortlist...")
    df = compute_stage1_score(df)
    sl = shortlist(df, k=shortlist_k)
    print(f"      shortlist size {len(sl)}, {time.time()-t0:.1f}s elapsed")

    print("[3/6] Computing Stage 2 semantic similarity (TF-IDF)...")
    shortlist_ids = set(sl["candidate_id"].to_list())
    sem_scores = compute_semantic_scores(shortlist_ids, candidates_path)
    sl = sl.with_columns(
        pl.col("candidate_id").map_elements(
            lambda cid: sem_scores.get(cid, 0.0), return_dtype=pl.Float64
        ).alias("s_sem")
    )
    print(f"      {time.time()-t0:.1f}s elapsed")

    print("[4/6] RRF fusion -> top 100...")
    top = top100(sl)
    print(f"      {time.time()-t0:.1f}s elapsed")

    print("[5/6] Enriching top 100 with grounding facts...")
    top = enrich(top, candidates_path)
    print(f"      {time.time()-t0:.1f}s elapsed")

    print("[6/6] Generating reasoning + writing submission CSV...")
    sub = build_submission(top)
    sub.write_csv(out_path)
    print(f"      wrote {out_path}, total time {time.time()-t0:.1f}s elapsed")

    n_stuffers = top.filter(pl.col("is_keyword_stuffer")).height
    n_honeypots = top.filter(pl.col("is_honeypot")).height
    print(f"\nTrap check: {n_stuffers} keyword-stuffers, {n_honeypots} honeypots in top 100")

    return sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--out", default="data/submission.csv")
    parser.add_argument("--shortlist-k", type=int, default=1000)
    args = parser.parse_args()

    run(args.candidates, args.out, args.shortlist_k)
