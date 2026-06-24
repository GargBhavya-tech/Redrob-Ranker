"""
Stage 1 shortlisting.

Combines the structural signals from features.py into a single fast score
used purely to truncate 100,000 -> top-K candidates before the expensive
Stage 2 semantic embedding pass.

This is intentionally generous (high recall) — it just needs to ensure that
genuinely strong candidates aren't dropped before semantic re-ranking can see
them. The final ranking is decided later via RRF across signals including
the Stage 2 embedding similarity.
"""

import polars as pl


# Weights for the Stage 1 combined score. Title and skills dominate because
# they're the strongest discriminators against the JD's "trap" candidates
# (keyword stuffers, off-track titles). YoE and location are secondary.
STAGE1_WEIGHTS = {
    # Tuned against the HUMAN-ANCHORED gold (non-circular), applying the grid's
    # DIRECTION not its argmax: titles oversell in this dataset, so s_title is
    # down-weighted and s_skills/s_evidence (what they actually have/built) carry
    # more. Anchor pairwise agreement 0.853 -> 0.866 with this direction; k=60
    # re-confirmed. Not snapped to the exact max (0.868) to avoid overfitting 49
    # points.
    "s_title": 0.20,
    "s_skills": 0.32,
    "s_evidence": 0.26,
    "s_yoe": 0.12,
    "s_seniority": 0.06,
    "s_loc": 0.04,
}


def compute_stage1_score(df: pl.DataFrame) -> pl.DataFrame:
    expr = sum(pl.col(k) * w for k, w in STAGE1_WEIGHTS.items())
    return df.with_columns(expr.alias("stage1_score"))


def shortlist(df: pl.DataFrame, k: int = 1000) -> pl.DataFrame:
    """
    Returns the top-k candidates by stage1_score, with honeypots dropped
    (we don't want to waste embedding budget on them) but keyword-stuffers
    KEPT (their low s_title/s_skills will naturally rank them low; we still
    want them visible to the trap-rate check / debugging).
    """
    df = compute_stage1_score(df)
    df = df.filter(~pl.col("is_honeypot"))
    return df.sort("stage1_score", descending=True).head(k)


if __name__ == "__main__":
    df = pl.read_parquet("data/features.parquet")
    df = compute_stage1_score(df)
    df.write_parquet("data/features.parquet")  # persist stage1_score too

    sl = shortlist(df, k=1000)
    print(f"Shortlist size: {len(sl)}")
    print(sl.select(
        "candidate_id", "current_title", "stage1_score",
        "s_title", "s_skills", "s_yoe", "core_skill_hits",
        "is_keyword_stuffer",
    ).head(20))

    print()
    print("Keyword stuffers in shortlist:", sl.filter(pl.col("is_keyword_stuffer")).height)

    sl.write_parquet("data/shortlist.parquet")
    print("Wrote data/shortlist.parquet")
