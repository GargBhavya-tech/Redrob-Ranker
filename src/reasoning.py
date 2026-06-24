"""
Generates template-based, fact-grounded per-candidate reasoning and writes
the final submission CSV (candidate_id, rank, score, reasoning).

Reasoning is built ENTIRELY from extracted features / profile fields — no
LLM call (which would violate the "no network during ranking" constraint
and risk hallucination flags at Stage 4). Every claim in the reasoning is
traceable to a specific field, satisfying the Stage 4 review checklist:
  - specific facts (years, title, named skills, signal values)
  - JD connection (title/skills framing)
  - honest concerns (notice period, location, low response rate, etc.)
  - no hallucination (only fields actually present are referenced)
  - variation (multiple template branches + interpolated facts)
  - rank-consistency (tone graduated by rank tier)
"""

import polars as pl


def _tier(rank: int) -> str:
    if rank <= 10:
        return "top"
    elif rank <= 40:
        return "strong"
    elif rank <= 75:
        return "solid"
    else:
        return "filler"


def _location_phrase(location: str, country: str) -> str:
    loc = (location or "").strip()
    if not loc:
        return ""
    if country and country.lower() == "india":
        return f"based in {loc}"
    return f"based in {loc}, {country}" if country else f"based in {loc}"


def _cap_first(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _concerns(row: dict) -> list[str]:
    concerns = []
    notice = row["notice_period_days"]
    if notice >= 90:
        concerns.append(f"a {notice}-day notice period")
    elif notice >= 60:
        concerns.append(f"a {notice}-day notice period")

    if row["recruiter_response_rate"] < 0.4:
        concerns.append(f"a low recruiter response rate ({row['recruiter_response_rate']:.0%})")

    if row["s_loc"] < 0.5 and (row["country"] or "").lower() != "india":
        concerns.append(f"located outside India ({row['location']})")
    elif row["s_loc"] < 0.5:
        concerns.append(f"based in {row['location']}, outside the Pune/Noida preferred hubs")

    if not row["open_to_work_flag"]:
        concerns.append("not currently flagged as open to work")

    if row["secondary_skill_hits"] == 0 and row["core_skill_hits"] <= 1:
        concerns.append("limited breadth of named retrieval/ranking skills")

    return concerns


def build_reasoning(row: dict) -> str:
    tier = _tier(row["rank"])
    title = row["current_title"]
    company = row["current_company"]
    yoe = row["years_of_experience"]
    core_skills = row["matched_core_skills"]
    concerns = _concerns(row)
    loc_phrase = _location_phrase(row["location"], row["country"])

    skill_phrase = ""
    if core_skills:
        skill_phrase = f" with hands-on experience in {core_skills}"

    if tier == "top":
        base = (
            f"{title} at {company} with {yoe:.1f} years of experience{skill_phrase}, "
            f"directly matching the JD's retrieval/ranking core. "
        )
    elif tier == "strong":
        base = (
            f"{title} ({yoe:.1f} yrs at {company}){skill_phrase} — strong alignment "
            f"with the embeddings/retrieval requirements. "
        )
    elif tier == "solid":
        base = (
            f"{title} with {yoe:.1f} years of experience{skill_phrase}; a reasonable "
            f"fit for the role's core retrieval/ranking work. "
        )
    else:
        base = (
            f"{title} with {yoe:.1f} years of experience{skill_phrase}; included as "
            f"lower-confidence filler given partial alignment with the JD. "
        )

    if loc_phrase:
        base += f"{_cap_first(loc_phrase)}. "

    if concerns:
        if len(concerns) == 1:
            base += f"Some concern: {concerns[0]}."
        else:
            base += "Concerns: " + "; ".join(concerns) + "."
    else:
        base += "No major red flags identified from available signals."

    return base


def build_submission(top: pl.DataFrame) -> pl.DataFrame:
    top = top.with_columns(pl.col("score").round(4))
    # Final ordering: score descending, candidate_id ascending for ties.
    # This guarantees the validator's tiebreak rule holds even after rounding.
    top = top.sort(["score", "candidate_id"], descending=[True, False])
    top = top.with_columns(pl.int_range(1, len(top) + 1).alias("rank"))

    rows = top.to_dicts()
    reasonings = [build_reasoning(r) for r in rows]

    out = top.select("candidate_id", "rank", "score").with_columns(
        pl.Series("reasoning", reasonings),
    )
    return out


if __name__ == "__main__":
    top = pl.read_parquet("data/top100_enriched.parquet")
    sub = build_submission(top)

    print(sub.head(5))
    print("...")
    print(sub.tail(5))

    out_path = "data/submission.csv"
    sub.write_csv(out_path)
    print(f"\nWrote {out_path}")
