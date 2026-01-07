"""Shared helpers for report calculations."""
from typing import Union

PRIORITY_FREQUENCY_WEIGHT = 2
TIER_TOP_THRESHOLD = 4
TIER_HIGH_THRESHOLD = 3
TIER_MEDIUM_THRESHOLD = 2


def calculate_priority_score(frequency_years: int, average_marks: float) -> float:
    """
    Compute priority score using distinct-year frequency and average marks.
    Formula: (2 × Frequency) + Average Marks.
    """
    return round((PRIORITY_FREQUENCY_WEIGHT * frequency_years) + (average_marks or 0), 1)


def calculate_average_marks(total_marks: Union[int, float], question_count: int) -> float:
    """Safely compute average marks for a cluster."""
    if question_count > 0:
        return round(total_marks / question_count, 1)
    return 0.0


def calculate_confidence(frequency_years: int, total_years: int) -> float:
    """Calculate confidence percentage from distinct year coverage."""
    if total_years:
        return round((frequency_years / total_years) * 100, 1)
    return 0.0


def format_priority_details(item: dict) -> str:
    """Create a consistent priority detail string used across outputs."""
    return (
        f"Priority Score: {item.get('priority_score', 0)} | "
        f"Confidence: {item.get('confidence', 0)}% | "
        f"Part A: {item.get('part_a', item.get('part_a_count', 0))} | "
        f"Part B: {item.get('part_b', item.get('part_b_count', 0))}"
    )


def priority_sort_key(item: dict) -> tuple:
    """
    Shared sort key for priority items.
    Orders by priority score, then frequency, then topic/topic_name.
    """
    return (
        -item.get('priority_score', 0),
        -item.get('frequency', 0),
        item.get('topic_name') or item.get('topic', '')
    )
