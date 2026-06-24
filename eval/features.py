"""
features.py
-----------
Turns a raw candidate dict into:
  - a text blob (for semantic + lexical matching)
  - structured signals (retrieval evidence, seniority, product-company, location, yoe fit)
  - disqualifier penalties (consulting-only, off-domain-only, research-only, keyword-stuffer)
  - a honeypot verdict (hard / soft) from structural impossibilities
  - a behavioral multiplier from the 23 redrob_signals

Everything here is pure-Python and O(candidate); it runs comfortably inside the
5-minute CPU budget for 100k candidates.
"""
from __future__ import annotations
import math
from datetime import date

from jd_config import (
    CORE_RETRIEVAL, ML_GENERAL, BUILD_EVIDENCE, CONSULTING_FIRMS, OFF_DOMAIN,
    RESEARCH_ONLY, NON_TECH_TITLE_HINTS, TECH_TITLE_HINTS, TARGET_LOCATIONS,
    YOE_IDEAL_LOW, YOE_IDEAL_HIGH, YOE_HARD_LOW,
    W_RETRIEVAL_EVIDENCE, W_CORE_SKILLS, W_SENIORITY, W_PRODUCT_COMPANY,
    W_YOE_FIT, W_ML_GENERAL, W_LOCATION,
    P_CONSULTING_ONLY, P_OFF_DOMAIN_ONLY, P_RESEARCH_ONLY, P_KEYWORD_STUFFER,
    P_TITLE_CHASER, BEHAVIOR_MIN, BEHAVIOR_MAX,
)

REFERENCE_TODAY = date(2026, 6, 17)   # challenge "today"


def _low(s) -> str:
    return s.lower() if isinstance(s, str) else ""


def _count_terms(text: str, vocab) -> int:
    return sum(1 for t in vocab if t in text)


def build_text(c: dict) -> str:
    """One text blob per candidate for embedding + TF-IDF."""
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""), p.get("summary", ""), p.get("current_title", ""),
        p.get("current_company", ""), p.get("current_industry", ""),
    ]
    for r in c.get("career_history", []):
        parts += [r.get("title", ""), r.get("company", ""), r.get("description", "")]
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    for e in c.get("education", []):
        parts += [e.get("degree", ""), e.get("field_of_study", "")]
    return " ".join(x for x in parts if x)


# ---------------------------------------------------------------------------
# Honeypot detection -- hard structural impossibilities only.
# Calibrated on the real 100k pool: the "rare" structural flags land near the
# spec's ~80 honeypots; the noisier skill-duration flag is treated as SOFT.
# ---------------------------------------------------------------------------
def honeypot_flags(c: dict):
    p = c.get("profile", {}); skills = c.get("skills", []); ch = c.get("career_history", [])
    yoe = float(p.get("years_of_experience", 0) or 0)
    rare, soft = [], []

    total_months = sum(int(r.get("duration_months", 0) or 0) for r in ch)
    if total_months > yoe * 12 + 36:
        rare.append("tenure_sum_exceeds_career")

    if ch:
        max_role = max(int(r.get("duration_months", 0) or 0) for r in ch)
        if max_role > yoe * 12 + 12:
            rare.append("single_role_exceeds_career")

    expert_zero = [s for s in skills
                   if s.get("proficiency") in ("expert", "advanced")
                   and int(s.get("duration_months", 1) or 0) == 0]
    if len(expert_zero) >= 2:
        rare.append("expert_skill_zero_months")

    if skills and yoe > 0:
        max_sd = max(int(s.get("duration_months", 0) or 0) for s in skills)
        if max_sd > yoe * 12 * 2.5 + 12:
            soft.append("skill_duration_implausible")

    return rare, soft


def honeypot_verdict(c: dict):
    """Return ('hard'|'soft'|None, flags). Hard => force to the bottom."""
    rare, soft = honeypot_flags(c)
    if rare:                       # near-certain impossible profile
        return "hard", rare + soft
    if soft:                       # suspicious but not by itself impossible
        return "soft", soft
    return None, []


# ---------------------------------------------------------------------------
# Structured relevance signals
# ---------------------------------------------------------------------------
def structured_signals(c: dict) -> dict:
    p = c.get("profile", {}); ch = c.get("career_history", []); skills = c.get("skills", [])
    title = _low(p.get("current_title", ""))
    summary = _low(p.get("summary", ""))
    career_text = " ".join(_low(r.get("description", "")) for r in ch)
    skill_text = " ".join(_low(s.get("name", "")) for s in skills)
    full_text = " ".join([title, summary, career_text, skill_text,
                          _low(p.get("headline", ""))])
    yoe = float(p.get("years_of_experience", 0) or 0)

    is_tech_title = any(k in title for k in TECH_TITLE_HINTS) and \
        not any(k == title for k in NON_TECH_TITLE_HINTS)
    is_nontech_title = any(k in title for k in NON_TECH_TITLE_HINTS) and not is_tech_title

    retrieval_in_career = _count_terms(career_text + " " + summary, CORE_RETRIEVAL)
    retrieval_in_skills = _count_terms(skill_text, CORE_RETRIEVAL)
    ml_general_hits = _count_terms(full_text, ML_GENERAL)
    build_evidence = _count_terms(career_text + " " + summary, BUILD_EVIDENCE)

    companies = [_low(r.get("company", "")) for r in ch]
    all_consulting = bool(companies) and all(
        any(cf in comp for cf in CONSULTING_FIRMS) for comp in companies)
    # product company = at least one non-consulting employer + build evidence
    product_company = (build_evidence >= 1) and not all_consulting

    loc = _low(p.get("location", ""))
    relocate = bool(c.get("redrob_signals", {}).get("willing_to_relocate", False))
    location_ok = any(t in loc for t in TARGET_LOCATIONS) or relocate

    # off-domain (CV/speech/robotics) without NLP/IR
    offdomain_hits = _count_terms(full_text, OFF_DOMAIN)
    has_nlp_ir = ("nlp" in full_text) or (retrieval_in_career + retrieval_in_skills > 0)
    off_domain_only = offdomain_hits >= 3 and not has_nlp_ir

    research_only = (_count_terms(full_text, RESEARCH_ONLY) >= 2) and build_evidence == 0

    # keyword stuffer: non-tech title, AI skills listed, but NO build evidence in career
    keyword_stuffer = is_nontech_title and retrieval_in_skills + \
        _count_terms(skill_text, ML_GENERAL) >= 4 and build_evidence == 0

    # title chaser: many short stints
    durations = [int(r.get("duration_months", 0) or 0) for r in ch if r.get("duration_months")]
    avg_tenure = (sum(durations) / len(durations)) if durations else 99
    title_chaser = len(durations) >= 3 and avg_tenure < 18

    # seniority hint
    seniority = 0.0
    if any(k in title for k in ("senior", "lead", "principal", "staff")):
        seniority = 1.0
    elif any(k in title for k in ("junior", "intern", "associate", "trainee")):
        seniority = -0.5

    return dict(
        yoe=yoe, is_tech_title=is_tech_title, is_nontech_title=is_nontech_title,
        retrieval_in_career=retrieval_in_career, retrieval_in_skills=retrieval_in_skills,
        ml_general_hits=ml_general_hits, build_evidence=build_evidence,
        product_company=product_company, all_consulting=all_consulting,
        location_ok=location_ok, off_domain_only=off_domain_only,
        research_only=research_only, keyword_stuffer=keyword_stuffer,
        title_chaser=title_chaser, seniority=seniority, avg_tenure=avg_tenure,
    )


def structured_score(sig: dict) -> float:
    """Deterministic structured relevance score (higher = better)."""
    yoe = sig["yoe"]
    # retrieval evidence in career counts double vs skills (built > listed)
    retr = min(sig["retrieval_in_career"], 6) * 1.0 + min(sig["retrieval_in_skills"], 6) * 0.4
    s = 0.0
    s += W_RETRIEVAL_EVIDENCE * (retr / 6.0)
    s += W_CORE_SKILLS * min(sig["retrieval_in_skills"], 5) / 5.0
    s += W_ML_GENERAL * min(sig["ml_general_hits"], 6) / 6.0
    s += W_SENIORITY * (sig["seniority"] + 0.5) / 1.5
    s += W_PRODUCT_COMPANY * (1.0 if sig["product_company"] else 0.0)
    s += W_LOCATION * (1.0 if sig["location_ok"] else 0.0)

    # YoE fit: 1.0 inside ideal band, asymmetric Gaussian decay below, mild decay above
    if YOE_IDEAL_LOW <= yoe <= YOE_IDEAL_HIGH:
        yoe_fit = 1.0
    elif yoe < YOE_IDEAL_LOW:
        yoe_fit = math.exp(-0.25 * (YOE_IDEAL_LOW - yoe) ** 2)
    else:
        yoe_fit = math.exp(-0.05 * (yoe - YOE_IDEAL_HIGH) ** 2)
    s += W_YOE_FIT * yoe_fit
    if yoe < YOE_HARD_LOW:
        s -= 1.0   # senior role; very junior is a poor fit

    if not sig["is_tech_title"]:
        s -= 1.0   # non-engineering current role is a strong negative for THIS JD

    # Penalties
    if sig["all_consulting"]:
        s -= P_CONSULTING_ONLY
    if sig["off_domain_only"]:
        s -= P_OFF_DOMAIN_ONLY
    if sig["research_only"]:
        s -= P_RESEARCH_ONLY
    if sig["keyword_stuffer"]:
        s -= P_KEYWORD_STUFFER
    if sig["title_chaser"]:
        s -= P_TITLE_CHASER
    return s


# ---------------------------------------------------------------------------
# Behavioral multiplier from the 23 redrob_signals
# ---------------------------------------------------------------------------
def _days_since(date_str: str) -> float:
    try:
        d = date.fromisoformat(date_str[:10])
        return max(0.0, (REFERENCE_TODAY - d).days)
    except Exception:
        return 365.0


def behavioral_multiplier(c: dict) -> float:
    """
    A perfect-on-paper candidate who is inactive / unresponsive is, for hiring,
    not available -> down-weight. Bounded multiplier so it modulates but never
    dominates the relevance signal.
    """
    sig = c.get("redrob_signals", {})

    # recency: exponential decay, 90-day half-life (W = 0.5 ** days/90)
    days = _days_since(sig.get("last_active_date", ""))
    recency = 0.5 ** (days / 90.0)                      # in (0,1]

    rrr = float(sig.get("recruiter_response_rate", 0.0) or 0.0)   # 0..1
    icr = float(sig.get("interview_completion_rate", 0.0) or 0.0)  # 0..1
    otw = 1.0 if sig.get("open_to_work_flag") else 0.0
    completeness = float(sig.get("profile_completeness_score", 0) or 0) / 100.0

    # verification adds a touch of trust
    verified = sum(bool(sig.get(k)) for k in
                   ("verified_email", "verified_phone", "linkedin_connected")) / 3.0

    # Weighted blend in [0,1]
    raw = (0.34 * recency + 0.24 * rrr + 0.14 * icr +
           0.12 * otw + 0.10 * completeness + 0.06 * verified)

    return BEHAVIOR_MIN + (BEHAVIOR_MAX - BEHAVIOR_MIN) * max(0.0, min(1.0, raw))
