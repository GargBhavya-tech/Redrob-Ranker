"""
Enrich the top-100 DataFrame with additional fact fields pulled directly
from candidates.jsonl, needed to ground the reasoning text:
  - recruiter_response_rate
  - matched core/secondary skill names (actual strings, for citation)
  - open_to_work_flag
"""

import polars as pl

from . import jd_config as jd
from .schema import iter_candidates


def _norm(s: str) -> str:
    return (s or "").lower().strip()


def matched_skill_names(skills: list[str], skill_set: set[str], limit: int) -> list[str]:
    out = []
    for s in skills:
        sn = _norm(s)
        if any(c in sn or sn in c for c in skill_set):
            out.append(s)
        if len(out) >= limit:
            break
    return out


def enrich(top: pl.DataFrame, jsonl_path: str) -> pl.DataFrame:
    ids = set(top["candidate_id"].to_list())
    extra = {}

    for cand in iter_candidates(jsonl_path):
        if cand.candidate_id not in ids:
            continue
        skill_names = [s.name for s in cand.skills]
        core_matches = matched_skill_names(skill_names, jd.CORE_SKILLS, 3)
        sec_matches = matched_skill_names(skill_names, jd.SECONDARY_SKILLS, 2)

        extra[cand.candidate_id] = {
            "recruiter_response_rate": cand.redrob_signals.recruiter_response_rate,
            "open_to_work_flag": cand.redrob_signals.open_to_work_flag,
            "matched_core_skills": ", ".join(core_matches) if core_matches else "",
            "matched_secondary_skills": ", ".join(sec_matches) if sec_matches else "",
            "current_company_size": cand.profile.current_company_size,
            "country": cand.profile.country,
        }

    rrr = []
    otw = []
    mcs = []
    mss = []
    ccs = []
    country = []
    for cid in top["candidate_id"].to_list():
        e = extra.get(cid, {})
        rrr.append(e.get("recruiter_response_rate", 0.0))
        otw.append(e.get("open_to_work_flag", False))
        mcs.append(e.get("matched_core_skills", ""))
        mss.append(e.get("matched_secondary_skills", ""))
        ccs.append(e.get("current_company_size", ""))
        country.append(e.get("country", ""))

    return top.with_columns(
        pl.Series("recruiter_response_rate", rrr),
        pl.Series("open_to_work_flag", otw),
        pl.Series("matched_core_skills", mcs),
        pl.Series("matched_secondary_skills", mss),
        pl.Series("current_company_size", ccs),
        pl.Series("country", country),
    )


if __name__ == "__main__":
    top = pl.read_parquet("data/top100.parquet")
    enriched = enrich(top, "data/candidates.jsonl")
    enriched.write_parquet("data/top100_enriched.parquet")
    print(enriched.select(
        "candidate_id", "current_title", "matched_core_skills",
        "recruiter_response_rate", "open_to_work_flag",
    ).head(10))
    print("Wrote data/top100_enriched.parquet")
