"""
msgspec.Struct definitions mirroring candidate_schema.json / candidates.jsonl.

Using typed Structs lets msgspec decode each ~4.65KB JSON line directly into
packed C-struct-like objects, skipping Python dict/hash overhead — this is
what makes it feasible to stream all 100K records well inside the 5-minute
CPU budget.
"""

from typing import Optional
import msgspec


class CareerEntry(msgspec.Struct):
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    is_current: bool = False
    industry: Optional[str] = None
    company_size: Optional[str] = None
    description: str = ""


class EducationEntry(msgspec.Struct):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    grade: Optional[str] = None
    tier: Optional[str] = None


class SkillEntry(msgspec.Struct):
    name: str
    proficiency: str = ""
    endorsements: int = 0
    duration_months: int = 0


class CertificationEntry(msgspec.Struct):
    name: str = ""
    issuer: Optional[str] = None
    year: Optional[int] = None


class LanguageEntry(msgspec.Struct):
    language: str
    proficiency: str = ""


class Profile(msgspec.Struct):
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""


class SalaryRange(msgspec.Struct):
    min: float = 0.0
    max: float = 0.0


class RedrobSignals(msgspec.Struct):
    profile_completeness_score: float = 0.0
    signup_date: Optional[str] = None
    last_active_date: Optional[str] = None
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: dict = msgspec.field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    expected_salary_range_inr_lpa: Optional[SalaryRange] = None
    preferred_work_mode: str = ""
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


class Candidate(msgspec.Struct):
    candidate_id: str
    profile: Profile
    career_history: list[CareerEntry] = msgspec.field(default_factory=list)
    education: list[EducationEntry] = msgspec.field(default_factory=list)
    skills: list[SkillEntry] = msgspec.field(default_factory=list)
    certifications: list[CertificationEntry] = msgspec.field(default_factory=list)
    languages: list[LanguageEntry] = msgspec.field(default_factory=list)
    redrob_signals: RedrobSignals = msgspec.field(default_factory=RedrobSignals)


# Pre-built decoder, reused across the whole stream for max throughput.
decoder = msgspec.json.Decoder(Candidate)


def iter_candidates(path: str):
    """Stream-decode candidates.jsonl, one Candidate Struct per line."""
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield decoder.decode(line)
