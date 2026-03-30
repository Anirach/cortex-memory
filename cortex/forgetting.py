"""
Ebbinghaus forgetting curve implementation.

Retention R = e^(-t / S) where:
- t = time since last access (in hours)
- S = stability (increases with each successful retrieval)

Memories with R below a threshold are candidates for pruning.
"""

from __future__ import annotations

import math
import time


def retention(
    last_accessed: float,
    access_count: int = 1,
    importance: float = 0.5,
    base_stability: float = 24.0,
    now: float | None = None,
) -> float:
    """
    Calculate memory retention score [0, 1].

    Parameters
    ----------
    last_accessed : float
        Unix timestamp of last access.
    access_count : int
        Number of times this memory has been retrieved.
    importance : float
        Subjective importance [0, 1].
    base_stability : float
        Base half-life in hours.
    now : float | None
        Current time (unix). Defaults to time.time().
    """
    if now is None:
        now = time.time()

    elapsed_hours = max(0, (now - last_accessed) / 3600.0)

    # Stability grows with access count and importance
    stability = base_stability * (1 + math.log1p(access_count)) * (0.5 + importance)

    return math.exp(-elapsed_hours / stability) if stability > 0 else 0.0


def should_forget(
    last_accessed: float,
    access_count: int = 1,
    importance: float = 0.5,
    threshold: float = 0.1,
    now: float | None = None,
) -> bool:
    """Return True if this memory should be pruned."""
    return retention(last_accessed, access_count, importance, now=now) < threshold


def decay_importance(
    current_importance: float,
    last_accessed: float,
    decay_rate: float = 0.01,
    now: float | None = None,
) -> float:
    """Gradually reduce importance over time."""
    if now is None:
        now = time.time()
    elapsed_days = max(0, (now - last_accessed) / 86400.0)
    return max(0.0, current_importance * math.exp(-decay_rate * elapsed_days))
