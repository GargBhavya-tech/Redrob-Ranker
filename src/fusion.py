"""
Final fusion via Reciprocal Rank Fusion (RRF).

Each candidate in the shortlist is ranked independently on several signals;
RRF combines these ordinal ranks (k=60 smoothing constant, per research
synthesis) into a single fused score, avoiding the "score domination"
problem of naive linear weighting across signals with very different
distributions (TF-IDF cosine ~0.05-0.2, behavioral ~0-1, etc.).

Signals fused:
  - s_sem        (Stage 2 TF-IDF cosine similarity to JD)
  - s_title      (title lexical match)
  - s_skills     (core/secondary skill overlap)
  - s_yoe        (years-of-experience fit)
  - s_seniority  (career-trajectory alignment)
  - behavioral_score (recency + engagement, notice-period decayed)

The is_keyword_stuffer flag is applied as a hard penalty (large rank offset)
rather than fused, since it's a binary trap signal, not a graded one.
"""

import polars as pl

RRF_K = 60

FUSION_SIGNALS = [
    "s_sem",
    "s_evidence",
    "s_title",
    "s_skills",
    "s_yoe",
    "s_seniority",
    "behavioral_score",
]

# Keyword-stuffer trap penalty: effectively pushes flagged candidates'
# rank-in-each-list down by this many positions before RRF.
STUFFER_RANK_PENALTY = 2000
# Soft-honeypot penalty: DISABLED. Ablation (with/without) showed exactly zero
# effect on the top 100 -- soft-honeypots (one skill claimed far longer than the
# career) are overwhelmingly junior self-learners that already rank low on
# title/skills/seniority, so they never approached the cut. The is_honeypot_soft
# flag is still computed and surfaced for transparency, but applying a rank
# penalty here is dead code, so we don't.
SOFT_HONEYPOT_RANK_PENALTY = 0


def add_rank_columns(df: pl.DataFrame) -> pl.DataFrame:
    """For each fusion signal, add a `<signal>_rank` column (1 = best)."""
    has_soft = "is_honeypot_soft" in df.columns
    out = df
    for sig in FUSION_SIGNALS:
        # Higher signal value = better = rank 1. Apply keyword-stuffer
        # penalty by adding a large constant to the *rank* (not the score).
        rank_expr = (
            pl.col(sig).rank(method="ordinal", descending=True)
        )
        out = out.with_columns(rank_expr.alias(f"{sig}_rank"))

        out = out.with_columns(
            pl.when(pl.col("is_keyword_stuffer"))
            .then(pl.col(f"{sig}_rank") + STUFFER_RANK_PENALTY)
            .otherwise(pl.col(f"{sig}_rank"))
            .alias(f"{sig}_rank")
        )
        if has_soft and SOFT_HONEYPOT_RANK_PENALTY:
            out = out.with_columns(
                pl.when(pl.col("is_honeypot_soft"))
                .then(pl.col(f"{sig}_rank") + SOFT_HONEYPOT_RANK_PENALTY)
                .otherwise(pl.col(f"{sig}_rank"))
                .alias(f"{sig}_rank")
            )
        # JD disqualifiers (off-domain-only / research-only): large rank push,
        # smaller than keyword-stuffer since they may still have adjacent value.
        for flag_col in ("off_domain_only", "research_only"):
            if flag_col in out.columns:
                out = out.with_columns(
                    pl.when(pl.col(flag_col))
                    .then(pl.col(f"{sig}_rank") + 1000)
                    .otherwise(pl.col(f"{sig}_rank"))
                    .alias(f"{sig}_rank")
                )
    return out


# Per-signal weights for RRF. Standard RRF is uniform (all 1.0). The recent-role-
# weighted semantic signal is now the single best predictor of the human-anchor
# tiers, so it earns a heavier weight; s_evidence (built-vs-listed from prose) is
# the next strongest. Title/skills/yoe/behavioral stay light because the dataset
# deliberately scrambles title/skill keywords. Tuned against the human anchor.
SIGNAL_WEIGHTS = {
    "s_sem": 2.4,
    "s_evidence": 1.6,
    "s_title": 0.8,
    "s_skills": 0.8,
    "s_yoe": 0.7,
    "s_seniority": 0.7,
    "behavioral_score": 0.6,
}


def compute_rrf(df: pl.DataFrame, k: int = RRF_K) -> pl.DataFrame:
    df = add_rank_columns(df)

    rrf_expr = sum(
        (SIGNAL_WEIGHTS.get(sig, 1.0) / (k + pl.col(f"{sig}_rank")))
        for sig in FUSION_SIGNALS
    )
    df = df.with_columns(rrf_expr.alias("rrf_score"))

    return df.sort("rrf_score", descending=True)


def top100(df: pl.DataFrame) -> pl.DataFrame:
    fused = compute_rrf(df)
    top = fused.head(100)

    # Rescale rrf_score to a [0,1]-ish "score" column for the submission CSV,
    # preserving monotonic order. Min-max normalize within the top 100.
    min_rrf = top["rrf_score"].min()
    max_rrf = top["rrf_score"].max()
    span = max(max_rrf - min_rrf, 1e-9)

    top = top.with_columns(
        (0.40 + 0.59 * (pl.col("rrf_score") - min_rrf) / span).alias("score")
    )

    # Sort by (rrf_score desc, candidate_id asc) so that equal post-rounding
    # scores satisfy the tiebreak rule (ties broken by candidate_id ascending).
    top = top.sort(["rrf_score", "candidate_id"], descending=[True, False])

    # Add explicit rank 1..100
    top = top.with_columns(pl.int_range(1, len(top) + 1).alias("rank"))

    return top


if __name__ == "__main__":
    sl = pl.read_parquet("data/shortlist_sem.parquet")
    top = top100(sl)

    print(top.select(
        "rank", "candidate_id", "current_title", "score",
        "rrf_score", "is_keyword_stuffer", "is_honeypot",
    ).head(20))

    print()
    print("Keyword stuffers in top 100:", top.filter(pl.col("is_keyword_stuffer")).height)
    print("Honeypots in top 100:", top.filter(pl.col("is_honeypot")).height)

    top.write_parquet("data/top100.parquet")
    print("Wrote data/top100.parquet")
