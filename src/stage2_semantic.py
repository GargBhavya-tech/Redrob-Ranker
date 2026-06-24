"""
Stage 2: semantic similarity via TF-IDF cosine similarity.

Originally planned to use bge-small-en-v1.5 via FastEmbed (per the research),
but that requires a HuggingFace download which is unavailable in this
environment AND would violate the submission's "no network during ranking"
constraint anyway. TF-IDF cosine similarity is a robust, fully offline,
zero-dependency-download substitute for S_sem:

  - Builds a TF-IDF vectorizer fit on (JD text + shortlisted candidates' text)
  - Each candidate's "document" = summary + headline + skills + career
    history descriptions (the richest semantic signal in the profile)
  - Cosine similarity between JD vector and each candidate vector -> s_sem

Only run on the Stage 1 shortlist (~1000 candidates), so this is cheap.
"""

import time
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import jd_config as jd
from .schema import iter_candidates


def build_candidate_document(cand) -> str:
    """
    Recent-role-weighted document.

    Calibrated on the human-anchored gold: the dataset deliberately makes the
    SUMMARY and TITLE oversell (a "Recommendation Systems Engineer" whose recent
    work is churn prediction; a "Computer Vision Engineer" who actually built
    production reco). The most-recent career description is the ground truth of
    what the candidate did, so it is repeated to dominate the TF-IDF vector;
    summary/title get single weight; older roles and skills contribute lightly.

    Measured effect (anchor): tier<->rank corr 0.73 -> 0.82, tier-5s-in-top-10
    8 -> 9, fully offline and deterministic.
    """
    ch = cand.career_history
    recent_desc = (ch[0].description if ch else "") or ""
    recent_title = (ch[0].title if ch else "") or ""
    older = " ".join((h.description or "") for h in ch[1:])
    skills = " ".join((s.name or "") for s in cand.skills)

    parts = (
        [recent_desc] * 3        # recent role description dominates
        + [recent_title] * 2     # recent title reinforces
        + [cand.profile.summary or ""]   # summary: single weight (often oversells)
        + [older]                # older roles: light context
        + [skills]               # skill tags: light (deliberately noisy in data)
    )
    return " ".join(p for p in parts if p)


def compute_semantic_scores(shortlist_ids: set[str], jsonl_path: str) -> dict[str, float]:
    docs = []
    ids = []

    for cand in iter_candidates(jsonl_path):
        if cand.candidate_id in shortlist_ids:
            docs.append(build_candidate_document(cand))
            ids.append(cand.candidate_id)

    # JD text as the query document
    corpus = [jd.JD_SEMANTIC_QUERY] + docs

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=40000,
        ngram_range=(1, 3),
        sublinear_tf=True,   # validated +0.008 pairwise agreement on the human anchor
        min_df=1,
    )
    tfidf = vectorizer.fit_transform(corpus)

    jd_vec = tfidf[0:1]
    cand_vecs = tfidf[1:]

    sims = cosine_similarity(jd_vec, cand_vecs)[0]  # shape (n_candidates,)

    return dict(zip(ids, sims.tolist()))


if __name__ == "__main__":
    sl = pl.read_parquet("data/shortlist.parquet")
    shortlist_ids = set(sl["candidate_id"].to_list())

    t0 = time.time()
    sem_scores = compute_semantic_scores(shortlist_ids, "data/candidates.jsonl")
    print(f"Computed semantic scores for {len(sem_scores)} candidates in {time.time()-t0:.2f}s")

    sl = sl.with_columns(
        pl.col("candidate_id").map_elements(
            lambda cid: sem_scores.get(cid, 0.0), return_dtype=pl.Float64
        ).alias("s_sem")
    )

    print(sl.select("candidate_id", "current_title", "s_sem", "stage1_score")
            .sort("s_sem", descending=True)
            .head(15))

    sl.write_parquet("data/shortlist_sem.parquet")
    print("Wrote data/shortlist_sem.parquet")
