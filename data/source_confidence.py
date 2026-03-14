"""Source confidence scoring for data provenance tracking."""
from __future__ import annotations

from data.entity_resolution import SOURCE_CONFIDENCE


def get_confidence(source: str) -> float:
    """
    Return the confidence score for a named data source.

    Parameters
    ----------
    source:
        Source name (e.g. ``"pdc"``, ``"dartsorakel"``).

    Returns
    -------
    float
        Confidence in [0, 1].
    """
    return SOURCE_CONFIDENCE.get(source.lower(), 0.5)
