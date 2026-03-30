"""
Self-Improvement Engine — tracks errors, learns from corrections, detects patterns.

The engine maintains an error log, correction registry, and improvement metrics.
It feeds back into procedural memory to upgrade skills over time.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cortex.storage import Storage

if TYPE_CHECKING:
    from cortex.memories.procedural import ProceduralMemory


@dataclass
class Error:
    id: str
    description: str
    context: str = ""
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    correction: str = ""
    category: str = "unknown"
    severity: float = 0.5


@dataclass
class Correction:
    id: str
    error_id: str
    wrong: str
    correct: str
    pattern: str = ""
    created_at: float = field(default_factory=time.time)
    times_applied: int = 0


@dataclass
class ImprovementReport:
    total_errors: int = 0
    resolved_errors: int = 0
    resolution_rate: float = 0.0
    total_corrections: int = 0
    top_error_categories: list[tuple[str, int]] = field(default_factory=list)
    learning_velocity: float = 0.0  # errors resolved per day
    correction_patterns: list[dict[str, Any]] = field(default_factory=list)
    improvement_trend: str = "stable"  # improving, stable, declining


class SelfImprovementEngine:
    """Tracks mistakes, learns correction patterns, and measures learning velocity."""

    def __init__(self, storage: Storage, procedural: "ProceduralMemory | None" = None) -> None:
        self.storage = storage
        self.procedural = procedural

    # ── Error Logging ───────────────────────────────────────

    def log_error(
        self,
        description: str,
        context: str = "",
        category: str = "unknown",
        severity: float = 0.5,
    ) -> Error:
        """Log a new error."""
        error = Error(
            id=f"err_{uuid.uuid4().hex[:8]}",
            description=description,
            context=context,
            category=category,
            severity=max(0.0, min(1.0, severity)),
        )
        self.storage.insert("error_log", {
            "id": error.id,
            "description": error.description,
            "context": error.context,
            "created_at": error.created_at,
            "resolved": 0,
            "correction": "",
            "category": error.category,
            "severity": error.severity,
        })
        return error

    def resolve_error(self, error_id: str, correction_text: str = "") -> None:
        """Mark an error as resolved."""
        self.storage.execute(
            "UPDATE error_log SET resolved = 1, correction = ? WHERE id = ?",
            (correction_text, error_id),
        )

    # ── Correction Learning ─────────────────────────────────

    def log_correction(
        self,
        correct: str,
        error_id: str = "",
        wrong: str = "",
        pattern: str = "",
    ) -> Correction:
        """Log a correction pattern (how to fix a class of errors)."""
        correction = Correction(
            id=f"cor_{uuid.uuid4().hex[:8]}",
            error_id=error_id,
            wrong=wrong,
            correct=correct,
            pattern=pattern,
        )
        self.storage.insert("corrections", {
            "id": correction.id,
            "error_id": correction.error_id,
            "wrong": correction.wrong,
            "correct": correction.correct,
            "pattern": correction.pattern,
            "created_at": correction.created_at,
            "times_applied": 0,
        })

        # Resolve the linked error if provided
        if error_id:
            self.resolve_error(error_id, correct)

        # Optionally create a procedural skill from the correction
        if self.procedural and pattern:
            self.procedural.store(
                content=f"Correction: {correct}",
                pattern=pattern,
                trigger=wrong or pattern,
                importance=0.7,
                tags=["auto-correction"],
                metadata={"error_id": error_id, "correction_id": correction.id},
            )

        return correction

    def apply_correction(self, correction_id: str) -> None:
        """Increment the application count for a correction."""
        self.storage.execute(
            "UPDATE corrections SET times_applied = times_applied + 1 WHERE id = ?",
            (correction_id,),
        )

    def find_correction(self, error_description: str) -> Correction | None:
        """Find a matching correction for an error (simple keyword match)."""
        corrections = self.storage.fetch_all("corrections")
        desc_lower = error_description.lower()
        best: Correction | None = None
        best_overlap = 0

        for row in corrections:
            pattern = row.get("pattern", "").lower()
            wrong = row.get("wrong", "").lower()
            # Score by keyword overlap
            overlap = sum(1 for w in desc_lower.split() if w in pattern or w in wrong)
            if overlap > best_overlap:
                best_overlap = overlap
                best = self._correction_from_row(row)

        return best if best_overlap > 0 else None

    # ── Pattern Extraction ──────────────────────────────────

    def get_error_patterns(self) -> list[tuple[str, int]]:
        """Find the most common error categories."""
        rows = self.storage.fetch_all("error_log")
        counter = Counter(r["category"] for r in rows)
        return counter.most_common(10)

    # ── Improvement Report ──────────────────────────────────

    def get_improvement_report(self) -> ImprovementReport:
        """Generate a comprehensive improvement report."""
        errors = self.storage.fetch_all("error_log")
        corrections = self.storage.fetch_all("corrections")

        total = len(errors)
        resolved = sum(1 for e in errors if e["resolved"])
        resolution_rate = resolved / total if total > 0 else 0.0

        # Learning velocity: errors resolved per day (last 7 days)
        week_ago = time.time() - 7 * 86400
        recent_resolved = sum(
            1 for e in errors
            if e["resolved"] and e["created_at"] >= week_ago
        )
        learning_velocity = recent_resolved / 7.0

        # Trend detection
        two_weeks_ago = time.time() - 14 * 86400
        prev_resolved = sum(
            1 for e in errors
            if e["resolved"] and two_weeks_ago <= e["created_at"] < week_ago
        )
        if recent_resolved > prev_resolved + 1:
            trend = "improving"
        elif recent_resolved < prev_resolved - 1:
            trend = "declining"
        else:
            trend = "stable"

        # Correction patterns
        correction_patterns = []
        for c in corrections:
            correction_patterns.append({
                "id": c["id"],
                "pattern": c.get("pattern", ""),
                "correct": c["correct"],
                "times_applied": c.get("times_applied", 0),
            })
        correction_patterns.sort(key=lambda x: x["times_applied"], reverse=True)

        return ImprovementReport(
            total_errors=total,
            resolved_errors=resolved,
            resolution_rate=resolution_rate,
            total_corrections=len(corrections),
            top_error_categories=self.get_error_patterns(),
            learning_velocity=learning_velocity,
            correction_patterns=correction_patterns[:10],
            improvement_trend=trend,
        )

    # ── Fill Accuracy Tracking ──────────────────────────────

    def log_fill_outcome(self, gap_id: str, success: bool, confidence: float = 0.5) -> None:
        """Track whether a gap fill was accurate (for gap_filling integration)."""
        self.log_error(
            description=f"Gap fill {'succeeded' if success else 'failed'}: {gap_id}",
            context=f"confidence={confidence:.2f}",
            category="gap_fill_success" if success else "gap_fill_failure",
            severity=0.0 if success else 0.6,
        )
        if success:
            self.resolve_error(
                self.storage.fetch_all("error_log", "description LIKE ?", (f"%{gap_id}%",))[-1]["id"]
                if self.storage.fetch_all("error_log", "description LIKE ?", (f"%{gap_id}%",))
                else "",
                "Confirmed correct",
            )

    def get_fill_accuracy(self) -> float:
        """Return accuracy rate of gap fills."""
        successes = self.storage.count("error_log", "category = 'gap_fill_success'")
        failures = self.storage.count("error_log", "category = 'gap_fill_failure'")
        total = successes + failures
        return successes / total if total > 0 else 0.0

    # ── Internal ────────────────────────────────────────────

    def _correction_from_row(self, row: dict) -> Correction:
        return Correction(
            id=row["id"],
            error_id=row.get("error_id", ""),
            wrong=row.get("wrong", ""),
            correct=row["correct"],
            pattern=row.get("pattern", ""),
            created_at=row.get("created_at", 0),
            times_applied=row.get("times_applied", 0),
        )
