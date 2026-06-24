"""
Structured feature extraction for the Redrob ranker.

For each candidate, computes a dict of normalized [0,1]-ish signals derived
from profile/career/skills/redrob_signals, used in:
  - Stage 1 fast filtering (combined score -> top-K shortlist)
  - Final RRF fusion (each signal contributes its own ranked list)
  - Trap detection (keyword-stuffer, honeypot/consistency checks)

Design choices follow the research synthesis:
  - S_yoe: asymmetric Gaussian decay (no penalty if over-qualified)
  - S_title: lexical/Jaccard match against JD title-token target set
  - S_loc: location match via TARGET_LOCATIONS lookup table
  - S_skills: weighted overlap against CORE_SKILLS / SECONDARY_SKILLS
  - Behavioral composite: recency decay + engagement + notice-period decay
  - Trap flags: keyword-stuffer, honeypot/consistency
"""

import math
from datetime import date

from . import jd_config as jd
from .date_utils import parse_date, days_between, half_life_decay
from .schema import Candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_text(s: str) -> str:
    return (s or "").lower().strip()


def _skill_names(cand: Candidate) -> set[str]:
    return {_norm_text(s.name) for s in cand.skills}


# ---------------------------------------------------------------------------
# Individual signal scores
# ---------------------------------------------------------------------------

def score_years_of_experience(yoe: float, required: float = jd.REQUIRED_YOE,
                               lam: float = 0.08) -> float:
    """
    Asymmetric: 1.0 if candidate meets/exceeds required YoE, otherwise a
    Gaussian decay penalizing the shortfall.
    S_yoe = 1.0                          if yoe >= required
          = exp(-lam * (required-yoe)^2) otherwise
    """
    if yoe >= required:
        return 1.0
    deficit = required - yoe
    return math.exp(-lam * deficit * deficit)


def score_title_match(current_title: str) -> float:
    """
    Jaccard-style overlap between tokens of the candidate's current title
    and the JD's target title-token set. Catches "Senior AI Engineer",
    "ML Engineer - Search", "Applied Scientist, Ranking" etc., while scoring
    "Marketing Manager" or "Graphic Designer" near zero.
    """
    title = _norm_text(current_title)
    tokens = set(title.replace("-", " ").replace("/", " ").split())
    if not tokens:
        return 0.0
    overlap = tokens & jd.TARGET_TITLE_TOKENS
    # Jaccard against the *candidate's* tokens (precision-oriented): a short,
    # highly-aligned title (e.g. "ml engineer") scores high even though the
    # JD token set is much larger than the title itself.
    return len(overlap) / len(tokens)


def score_location(location: str, country: str) -> float:
    """
    Looks up the candidate's location string against TARGET_LOCATIONS.
    Falls back to a low baseline for India-but-unlisted-city, and near-zero
    for outside India (JD doesn't sponsor visas, though "case-by-case" is
    allowed — we don't hard-zero it).
    """
    loc = _norm_text(location)
    country_n = _norm_text(country)

    for key, val in jd.TARGET_LOCATIONS.items():
        if key in loc:
            return val

    if country_n == "india":
        return 0.4  # other Indian city, plausibly relocatable
    return 0.1  # outside India


def score_evidence(cand: Candidate) -> tuple[float, bool, bool, bool]:
    """
    Ported from the independent eval-harness scorer. Reads CAREER-DESCRIPTION
    PROSE (not just skill tags) to separate candidates who BUILT retrieval/
    ranking systems from those who merely LIST the keywords, and applies the
    JD's explicit disqualifiers as down-weights.

    Returns (evidence_score in [0,1], off_domain_only, research_only, title_chaser).

    Rationale: the dataset's titles oversell and skill tags are deliberately
    noisy; the recent-role description is the ground truth of what was built.
    Weighting retrieval-in-career above retrieval-in-skills is the single
    highest-signal lever for this JD.
    """
    import re
    p = cand.profile
    ch = cand.career_history or []
    summary = _norm_text(p.summary)
    career = " ".join(_norm_text(h.description) for h in ch)
    prose = summary + " \n " + career

    built = sum(1 for pat in jd.BUILD_EVIDENCE_PATTERNS if re.search(pat, prose))
    # retrieval terms appearing in career prose are worth ~2.5x the same term
    # appearing only as a skill tag.
    retr_career = sum(1 for t in jd.CORE_SKILLS if t in career)
    skill_text = " ".join(_norm_text(s.name) for s in (cand.skills or []))
    retr_skills = sum(1 for t in jd.CORE_SKILLS if t in skill_text)

    # off-domain-only (CV/speech/robotics, no NLP/IR)
    offdomain = sum(1 for o in jd.OFF_DOMAIN_TERMS if o in prose)
    has_ir = built > 0 or retr_career > 0 or "nlp" in prose
    off_domain_only = offdomain >= 2 and not has_ir

    # research-only (academic language, no build evidence)
    research_hits = sum(1 for r in jd.RESEARCH_ONLY_TERMS if r in prose)
    research_only = research_hits >= 2 and built == 0

    # title-chaser: 3+ roles averaging < 18 months
    durs = [int(h.duration_months or 0) for h in ch if h.duration_months]
    title_chaser = len(durs) >= 3 and (sum(durs) / len(durs)) < 18

    # combine: built-evidence dominates, retrieval-in-career strong, skills weak
    raw = (0.55 * min(built, 3) / 3.0
           + 0.30 * min(retr_career, 4) / 4.0
           + 0.15 * min(retr_skills, 5) / 5.0)
    if off_domain_only:
        raw -= 0.40
    if research_only:
        raw -= 0.30
    if title_chaser:
        raw -= 0.15
    return max(0.0, min(1.0, raw)), off_domain_only, research_only, title_chaser


def score_skills(skills: set[str]) -> tuple[float, int, int]:
    """
    Returns (score, n_core_hits, n_secondary_hits).
    Score weights core JD skills heavily, secondary skills lightly, capped at 1.0.
    """
    core_hits = sum(1 for s in skills if any(c in s or s in c for c in jd.CORE_SKILLS))
    secondary_hits = sum(1 for s in skills if any(c in s or s in c for c in jd.SECONDARY_SKILLS))

    score = min(1.0, 0.12 * core_hits + 0.04 * secondary_hits)
    return score, core_hits, secondary_hits


def score_seniority_context(cand: Candidate) -> float:
    """
    Rewards career trajectories that match the JD's "product company,
    applied ML, shipped end-to-end systems" framing:
      - penalize candidates whose entire visible career is at pure-services
        companies (TCS/Infosys/etc.)
      - reward presence of ML/AI-flavored titles across career history
        (not just current title) as a signal of sustained trajectory
    """
    history = cand.career_history
    if not history:
        return 0.5  # neutral, insufficient data

    companies = {_norm_text(h.company) for h in history}
    all_services = all(
        any(sc in c for sc in jd.SERVICES_COMPANIES) for c in companies
    )
    services_penalty = -0.2 if all_services and len(companies) >= 1 else 0.0

    ml_title_hits = 0
    for h in history:
        toks = set(_norm_text(h.title).replace("-", " ").split())
        if toks & jd.TARGET_TITLE_TOKENS:
            ml_title_hits += 1
    trajectory_bonus = min(0.3, 0.1 * ml_title_hits)

    return max(0.0, min(1.0, 0.5 + trajectory_bonus + services_penalty))


# ---------------------------------------------------------------------------
# Behavioral / availability composite
# ---------------------------------------------------------------------------

def score_behavioral(cand: Candidate, dataset_now: date,
                      notice_market_standard: int = 60,
                      notice_lambda: float = 0.02,
                      recency_half_life: int = 60) -> dict:
    """
    Returns a dict of behavioral sub-scores plus a combined 'behavioral_score'
    in [0,1], following the research synthesis:
      - recency: half-life decay on days-since-last-active
      - engagement: blend of response rate, interview completion, profile
        completeness, open_to_work
      - notice_penalty: exponential decay only beyond market-standard notice
    """
    sig = cand.redrob_signals

    last_active = parse_date(sig.last_active_date)
    if last_active is not None:
        days_inactive = max(0, days_between(last_active, dataset_now))
        recency = half_life_decay(days_inactive, recency_half_life)
    else:
        recency = 0.3  # unknown -> mildly penalized

    engagement = (
        0.35 * sig.recruiter_response_rate
        + 0.25 * min(1.0, sig.profile_completeness_score / 100.0)
        + 0.20 * max(0.0, sig.interview_completion_rate)
        + 0.20 * (1.0 if sig.open_to_work_flag else 0.0)
    )
    engagement = max(0.0, min(1.0, engagement))

    notice = sig.notice_period_days
    if notice <= notice_market_standard:
        notice_penalty_mult = 1.0
    else:
        notice_penalty_mult = math.exp(-notice_lambda * (notice - notice_market_standard))

    behavioral_score = (0.5 * recency + 0.5 * engagement) * notice_penalty_mult

    return {
        "recency": recency,
        "engagement": engagement,
        "notice_penalty_mult": notice_penalty_mult,
        "behavioral_score": max(0.0, min(1.0, behavioral_score)),
    }


# ---------------------------------------------------------------------------
# Trap / fraud heuristics
# ---------------------------------------------------------------------------

def keyword_stuffer_flag(cand: Candidate, skills: set[str], core_hits: int) -> bool:
    """
    Flags the "AI keyword stuffer" trap: many AI/ML keywords in the skills
    list, but a current title and career history with zero alignment to the
    AI/ML/Engineering field (e.g. Marketing Manager, Graphic Designer with
    9 AI skills — seen literally in sample_submission.csv).
    """
    title = _norm_text(cand.profile.current_title)
    title_tokens = set(title.replace("-", " ").split())

    is_non_fit_title = any(kw in title for kw in jd.NON_FIT_TITLE_KEYWORDS)
    has_ml_title_alignment = bool(title_tokens & jd.TARGET_TITLE_TOKENS)

    # Also check career history for any engineering/ML-flavored title ever
    history_alignment = any(
        set(_norm_text(h.title).replace("-", " ").split()) & jd.TARGET_TITLE_TOKENS
        for h in cand.career_history
    )

    high_ai_keyword_count = sum(1 for s in skills if s in jd.AI_KEYWORD_TERMS) >= 5

    return bool(
        high_ai_keyword_count
        and is_non_fit_title
        and not has_ml_title_alignment
        and not history_alignment
    )


def honeypot_flag(cand: Candidate) -> bool:
    """
    Honeypot = a STRUCTURALLY IMPOSSIBLE profile (per the spec's own examples:
    "8 years experience at a company founded 3 years ago", "expert proficiency
    in 10 skills with 0 years used"). Calibrated on the real 100k pool to land
    near the spec's ~80 honeypots.

    Replaces the original chronological-overlap/tenure-mirroring heuristic, which
    (verified on the dataset) flagged normal repeated-duration careers and missed
    every actual planted contradiction.

    Hard signals (any one => honeypot):
      H1: total tenure across roles exceeds the stated career by >3 years
      H2: a single role is longer than the entire stated career (+1yr slack)
      H3: >=2 expert/advanced skills claimed with 0 months of usage
      H4: summary states a years-of-experience materially different from the
          profile field (text/data contradiction)
    """
    p = cand.profile
    yoe = float(p.years_of_experience or 0)
    ch = cand.career_history or []
    skills = cand.skills or []

    # H1: total tenure >> career
    total_months = sum(int(h.duration_months or 0) for h in ch)
    if total_months > yoe * 12 + 36:
        return True

    # H2: single role longer than whole career
    if ch:
        max_role = max(int(h.duration_months or 0) for h in ch)
        if max_role > yoe * 12 + 12:
            return True

    # H3: >=2 expert/advanced skills with 0 months used
    expert_zero = sum(
        1 for s in skills
        if getattr(s, "proficiency", None) in ("expert", "advanced")
        and int(getattr(s, "duration_months", 1) or 0) == 0
    )
    if expert_zero >= 2:
        return True

    # H4: summary vs profile YoE contradiction
    import re
    m = re.search(r"(\d+\.?\d*)\s*\+?\s*years? of experience",
                  p.summary or "", re.I)
    if m:
        try:
            if abs(float(m.group(1)) - yoe) > 2.0:
                return True
        except ValueError:
            pass

    return False


def honeypot_soft_flag(cand: Candidate) -> bool:
    """Softer signal: a single skill claimed for >2.5x the entire career.
    Used for score damping, not a hard drop (it also catches genuine junior
    self-learners, so it must not be a hard filter)."""
    p = cand.profile
    yoe = float(p.years_of_experience or 0)
    if yoe <= 0 or not cand.skills:
        return False
    max_sd = max(int(getattr(s, "duration_months", 0) or 0) for s in cand.skills)
    return max_sd > yoe * 12 * 2.5 + 12


# ---------------------------------------------------------------------------
# Top-level feature record
# ---------------------------------------------------------------------------

def extract_features(cand: Candidate, dataset_now: date) -> dict:
    skills = _skill_names(cand)
    skills_score, core_hits, secondary_hits = score_skills(skills)
    behavioral = score_behavioral(cand, dataset_now)
    evidence, off_domain_only, research_only, title_chaser = score_evidence(cand)

    return {
        "candidate_id": cand.candidate_id,
        "s_yoe": score_years_of_experience(cand.profile.years_of_experience),
        "s_title": score_title_match(cand.profile.current_title),
        "s_loc": score_location(cand.profile.location, cand.profile.country),
        "s_skills": skills_score,
        "s_evidence": evidence,
        "off_domain_only": off_domain_only,
        "research_only": research_only,
        "title_chaser": title_chaser,
        "core_skill_hits": core_hits,
        "secondary_skill_hits": secondary_hits,
        "s_seniority": score_seniority_context(cand),
        "behavioral_score": behavioral["behavioral_score"],
        "recency": behavioral["recency"],
        "engagement": behavioral["engagement"],
        "notice_penalty_mult": behavioral["notice_penalty_mult"],
        "is_keyword_stuffer": keyword_stuffer_flag(cand, skills, core_hits),
        "is_honeypot": honeypot_flag(cand),
        "is_honeypot_soft": honeypot_soft_flag(cand),
        "years_of_experience": cand.profile.years_of_experience,
        "notice_period_days": cand.redrob_signals.notice_period_days,
        "current_title": cand.profile.current_title,
        "current_company": cand.profile.current_company,
        "location": cand.profile.location,
    }
