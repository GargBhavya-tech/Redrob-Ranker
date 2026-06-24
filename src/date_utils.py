"""
Small date/time utilities used across the scoring pipeline.

The dataset's "current date" context is the hackathon's data-generation date.
We use the latest signup_date/last_active_date observed in the dataset as a
practical proxy for "now" so recency decay is relative to the dataset itself,
not the wall-clock date the ranker happens to run on.
"""

from datetime import date, datetime
from typing import Optional


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def days_between(a: date, b: date) -> int:
    """Days from a to b (positive if b is after a)."""
    return (b - a).days


def half_life_decay(days_ago: float, half_life_days: float) -> float:
    """
    W(t) = e^{-lambda * t}, lambda = ln(2) / half_life.
    Returns a value in (0, 1]; 1.0 at t=0, 0.5 at t=half_life.
    """
    if days_ago <= 0:
        return 1.0
    lam = 0.6931471805599453 / half_life_days  # ln(2) / half_life
    return pow(2.718281828459045, -lam * days_ago)
