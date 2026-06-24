"""
stage3_cross_encoder.py
=======================
Cross-encoder re-rank of the TOP-N fused candidates.

WHY (proven in decision-map ticket #2): the bi-encoder (cosine of JD-query vs
candidate doc) cannot tell "I BUILT ranking systems" from "I MENTION ranking" --
it ranks true tier-5 retrieval engineers BELOW tier-2 churn/fraud candidates
whose summary boilerplate is stuffed with retrieval keywords. A cross-encoder
reads the JD and the candidate's RECENT-ROLE text JOINTLY through attention, so
it can distinguish built-vs-mentioned. We bound it to the top-N (default 50) so
the O(N) transformer cost stays trivially within the 5-min CPU budget.

DESIGN for reproducibility (decision-map: offline is load-bearing):
  - This stage is OPTIONAL. If the cross-encoder model is unavailable (no network
    / not cached), `rerank()` returns the input order UNCHANGED and the pipeline
    proceeds on the bi-encoder fusion. The baseline never breaks.
  - When present, it blends the cross-encoder score with the existing fused score
    (convex blend, default 0.5) rather than overriding it, so a single noisy
    cross-encoder judgment can't catastrophically reorder the list.

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU-friendly, ~80MB). At top-50 this
is ~50 forward passes, well under budget.

The candidate text fed to the cross-encoder is RECENT-ROLE-WEIGHTED (same rule as
the anchor): recent role description first, because that is the ground truth of
what they actually did.
"""
from __future__ import annotations
import os


def _candidate_query_text(cand) -> str:
    """Recent-role-first text; cross-encoder reads this against the JD."""
    ch = cand.career_history or []
    recent = (ch[0].description if ch else "") or ""
    recent_title = (ch[0].title if ch else "") or ""
    summary = cand.profile.summary or ""
    # recent role leads; summary trails (it oversells). Keep it short -- cross
    # encoders truncate, and the recent role is what matters.
    return f"{recent_title}. {recent} {summary}"[:1200]


def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Return a CrossEncoder or None if unavailable (offline-safe)."""
    if os.environ.get("REDROB_DISABLE_CROSS_ENCODER"):
        return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name, max_length=512)
    except Exception as e:
        print(f"[stage3] cross-encoder unavailable ({type(e).__name__}); "
              f"keeping bi-encoder order.")
        return None


def rerank(top_df, candidates_by_id, jd_query: str,
           top_n: int = 50, blend: float = 0.5, model=None):
    """
    Re-rank the top `top_n` rows of `top_df` (a polars DataFrame already sorted
    by fused score, with a 'candidate_id' and 'rrf_score' column).

    Returns a new polars DataFrame, re-sorted. If no model, returns top_df
    unchanged.
    """
    import polars as pl
    if model is None:
        model = load_cross_encoder()
    if model is None:
        return top_df  # graceful no-op

    head = top_df.head(top_n)
    ids = head["candidate_id"].to_list()
    pairs = [(jd_query, _candidate_query_text(candidates_by_id[i])) for i in ids]

    ce_scores = model.predict(pairs, convert_to_numpy=True)
    # min-max normalize ce scores to [0,1] so the blend is scale-stable
    lo, hi = float(ce_scores.min()), float(ce_scores.max())
    ce_norm = [(s - lo) / (hi - lo) if hi > lo else 0.5 for s in ce_scores]

    # normalize the existing fused score over the same head for a fair blend
    rrf = head["rrf_score"].to_list()
    rlo, rhi = min(rrf), max(rrf)
    rrf_norm = [(r - rlo) / (rhi - rlo) if rhi > rlo else 0.5 for r in rrf]

    blended = {ids[i]: blend * ce_norm[i] + (1 - blend) * rrf_norm[i]
               for i in range(len(ids))}

    # rows beyond top_n keep their fused order, appended after the re-ranked head
    tail = top_df.slice(top_n, top_df.height - top_n) if top_df.height > top_n else None

    head = head.with_columns(
        pl.col("candidate_id").replace_strict(blended, default=0.0).alias("ce_blended")
    ).sort("ce_blended", descending=True)

    if tail is not None:
        tail = tail.with_columns(pl.lit(-1.0).alias("ce_blended"))
        out = pl.concat([head, tail], how="vertical_relaxed")
    else:
        out = head
    return out
