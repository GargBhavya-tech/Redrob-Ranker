"""
gold_scorer_v2.py
=================
HARDENED, INDEPENDENT proxy ground-truth grader.

Lessons from hand-reviewing ~40 candidates:
  - The TITLE oversells and the SKILLS list is deliberately noisy (a
    "Recommendation Systems Engineer" whose actual recent work is churn
    prediction; FAISS/Weaviate sprinkled onto fraud-detection engineers).
  - The RECENT-ROLE DESCRIPTION is the ground truth of what they actually did.
  - The SUMMARY contains honest self-disclosure ("lighter on the deep-learning
    side", "haven't done it professionally yet", "split between dashboarding and
    ML") that reveals true depth.
  - Some profiles embed a contradiction (summary says "6.3 years" while the
    profile says 2.7) -> honeypot.

So this grader:
  * grades PRIMARILY off the most-recent career description + summary prose,
  * treats title and skill tags as weak priors only,
  * forces honeypots to tier 0 using TWO INDEPENDENT signals (summary/profile
    YoE contradiction; impossible company tenure) -- deliberately NOT importing
    the ranker's honeypot function, to keep the gold independent.

This is intentionally STRICTER than the rankers, so agreement is meaningful.
"""
from __future__ import annotations
import re

# --- what the candidate ACTUALLY BUILT (in recent role / summary prose) ------
BUILT_RETRIEVAL = [
    r"\bbuilt\b.{0,50}(search|ranking|recommendation|retrieval|matching|relevance)",
    r"(shipped|rebuilt|migrated).{0,50}(search|ranking|retrieval|embedding|recommendation)",
    r"(owned|designed|architected).{0,50}(ranking|retrieval|relevance|search|recommendation)",
    r"(rag|dense retrieval|hybrid retrieval|hybrid search).{0,60}(production|serving|queries|users|scale)",
    r"learning[- ]to[- ]rank|lambdamart|\bndcg\b",
    r"(bm25|bge|faiss|hnsw|pinecone|weaviate|qdrant|milvus|opensearch|elasticsearch).{0,60}(retriev|rank|embed|search|serving|production)",
    r"recommendation system",
    r"led the team .{0,40}(retrieval|ranking|search|embedding)",
]
BUILT_ML = [
    r"(built|shipped|deployed|trained).{0,50}(model|pipeline|classifier|nlp|ml)",
    r"transformer[- ]based (classifier|model)",
    r"fine[- ]?tun.{0,40}(model|llm|production)",
    r"production ml (pipeline|model|system)",
    r"information extraction|document classification|sentiment analysis",
]
# honest hedging that reveals the depth is shallow / not production retrieval
HEDGE = [
    r"lighter (on|weight)", r"lighter than ranking systems",
    r"haven'?t done .{0,30}(professional|production)", r"not in a professional capacity",
    r"self[- ]?(taught|learner|directed|study)", r"side project", r"online courses?",
    r"experimented with", r"curious about", r"building competence",
    r"strongest at the modeling and analysis side", r"classical methods",
    r"split between (dashboard|analytics)", r"at a self-learner level",
    r"mostly .{0,15}(classical|collaborative filtering|gradient-boosted)",
    r"transition(ing)? (toward|into)", r"kaggle", r"playing with",
]

NON_TECH_TITLES = {
    "hr manager", "marketing manager", "sales executive", "accountant",
    "business analyst", "operations manager", "customer support",
    "content writer", "graphic designer", "project manager", "civil engineer",
    "mechanical engineer", "recruiter", "office manager",
}
CONSULTING = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
              "mindtree", "mphasis", "hcl", "tech mahindra", "ltimindtree"}
OFFDOMAIN = ["computer vision", "image classification", "object detection",
             "opencv", "yolo", "speech recognition", "asr", "robotics", "lidar"]
TARGET_LOC = {"pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
              "bengaluru", "gurgaon", "gurugram", "ncr"}


def _low(s):
    return s.lower() if isinstance(s, str) else ""


def _hits(pats, text):
    return sum(1 for p in pats if re.search(p, text))


def _independent_honeypot(c: dict) -> bool:
    """Two signals that DON'T reuse the ranker's structural rule."""
    p = c.get("profile", {})
    yoe = float(p.get("years_of_experience", 0) or 0)
    summary = p.get("summary", "")
    # signal 1: summary states a YoE materially different from the profile field
    m = re.search(r"(\d+\.?\d*)\s*\+?\s*years? of experience", summary, re.I)
    if m:
        try:
            if abs(float(m.group(1)) - yoe) > 2.0:
                return True
        except ValueError:
            pass
    # signal 2: impossible company tenure (role longer than entire career)
    for r in c.get("career_history", []):
        if int(r.get("duration_months", 0) or 0) > yoe * 12 + 18:
            return True
    return False


def grade(c: dict) -> int:
    p = c.get("profile", {})
    title = _low(p.get("current_title", ""))
    yoe = float(p.get("years_of_experience", 0) or 0)
    summary = _low(p.get("summary", ""))
    ch = c.get("career_history", [])
    # most-recent role description carries the most weight
    recent = _low(ch[0].get("description", "")) if ch else ""
    all_career = " ".join(_low(r.get("description", "")) for r in ch)
    prose = summary + " \n " + recent + " \n " + all_career

    if _independent_honeypot(c):
        return 0

    built_r = _hits(BUILT_RETRIEVAL, prose)
    built_ml = _hits(BUILT_ML, prose)
    hedge = _hits(HEDGE, prose)

    # --- disqualifier caps ---------------------------------------------------
    companies = [_low(r.get("company", "")) for r in ch]
    all_consulting = bool(companies) and all(
        any(cc in comp for cc in CONSULTING) for comp in companies)
    is_nontech = title in NON_TECH_TITLES
    offdomain = sum(1 for o in OFFDOMAIN if o in prose)
    has_ir = built_r > 0 or "retrieval" in prose or "ranking" in prose or "nlp" in prose

    cap = 5
    if is_nontech:
        cap = min(cap, 1)
    if all_consulting:
        cap = min(cap, 2)
    if offdomain >= 2 and not has_ir:
        cap = min(cap, 2)
    if yoe < 3:
        cap = min(cap, 2)

    # --- base tier from what they ACTUALLY BUILT ----------------------------
    if built_r >= 2:
        base = 5
    elif built_r == 1:
        base = 4
    elif built_ml >= 2:
        base = 3
    elif built_ml == 1:
        base = 2
    else:
        base = 1

    # honest hedging caps aspirational candidates (key independence lever)
    if built_r == 0:
        if hedge >= 2:
            base = min(base, 2)
        elif hedge >= 1:
            base = min(base, 3)

    tier = min(base, cap)

    # --- seniority / band / location nudges ---------------------------------
    in_band = 5 <= yoe <= 9
    senior = any(k in title for k in ("senior", "lead", "principal", "staff"))
    loc_ok = any(t in _low(p.get("location", "")) for t in TARGET_LOC) or \
        bool(c.get("redrob_signals", {}).get("willing_to_relocate", False))

    if tier >= 4 and not (in_band or senior):
        tier -= 1
    if tier >= 3 and not loc_ok and _low(p.get("country", "")) != "india":
        tier -= 1

    return max(0, min(5, tier))
