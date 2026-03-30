"""
Procedural Memory — learned skills, patterns, and how-to knowledge.

Stores actionable knowledge: "when X happens, do Y". Tracks success/failure
rates to rank skill reliability. Supports pattern matching for automatic
skill selection.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cortex.embeddings import embed_text, cosine_similarity
from cortex.storage import Storage


@dataclass
class Skill:
    id: str
    content: str
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    importance: float = 0.5
    success_count: int = 0
    failure_count: int = 0
    pattern: str = ""       # Trigger pattern description
    trigger: str = ""       # When to apply this skill
    tags: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def reliability(self) -> float:
        """Bayesian reliability: weighted success rate with prior."""
        total = self.success_count + self.failure_count
        # Laplace smoothing with prior=0.5
        return (self.success_count + 1) / (total + 2)


class ProceduralMemory:
    """Persistent procedural memory (skills and patterns) backed by SQLite."""

    TABLE = "procedural_memory"

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def store(
        self,
        content: str,
        pattern: str = "",
        trigger: str = "",
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Skill:
        """Store a new skill/procedure."""
        now = time.time()
        skill = Skill(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            content=content,
            embedding=embed_text(content),
            metadata=metadata or {},
            created_at=now,
            last_accessed=now,
            importance=max(0.0, min(1.0, importance)),
            pattern=pattern,
            trigger=trigger,
            tags=tags or [],
        )
        self.storage.insert(self.TABLE, self._to_row(skill))
        return skill

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_reliability: float = 0.0,
    ) -> list[Skill]:
        """Retrieve skills by semantic similarity, weighted by reliability."""
        query_vec = embed_text(query)
        rows = self.storage.fetch_all(self.TABLE)
        if not rows:
            return []

        scored: list[tuple[float, Skill]] = []
        for row in rows:
            skill = self._from_row(row)
            sim = 0.0
            if skill.embedding is not None:
                sim = cosine_similarity(query_vec, skill.embedding)

            if skill.reliability < min_reliability:
                continue

            score = sim * skill.reliability * (0.5 + skill.importance)
            scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [s for _, s in scored[:top_k]]

        for skill in results:
            self._touch(skill.id)

        return results

    def record_outcome(self, skill_id: str, success: bool) -> None:
        """Record success or failure for a skill."""
        col = "success_count" if success else "failure_count"
        self.storage.execute(
            f"UPDATE {self.TABLE} SET {col} = {col} + 1 WHERE id = ?",
            (skill_id,),
        )

    def get(self, skill_id: str) -> Skill | None:
        row = self.storage.fetch_one(self.TABLE, skill_id)
        if row:
            self._touch(skill_id)
            return self._from_row(row)
        return None

    def get_all(self) -> list[Skill]:
        return [self._from_row(r) for r in self.storage.fetch_all(self.TABLE)]

    def get_by_trigger(self, trigger_query: str) -> list[Skill]:
        """Find skills whose trigger matches the query."""
        rows = self.storage.fetch_all(self.TABLE, "trigger != ''")
        q = trigger_query.lower()
        return [
            self._from_row(r) for r in rows
            if q in r["trigger"].lower() or r["trigger"].lower() in q
        ]

    def delete(self, skill_id: str) -> None:
        self.storage.delete(self.TABLE, skill_id)

    def count(self) -> int:
        return self.storage.count(self.TABLE)

    # ── Internal ────────────────────────────────────────────

    def _touch(self, skill_id: str) -> None:
        self.storage.execute(
            f"UPDATE {self.TABLE} SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (time.time(), skill_id),
        )

    def _to_row(self, skill: Skill) -> dict:
        return {
            "id": skill.id,
            "content": skill.content,
            "embedding": skill.embedding.tobytes() if skill.embedding is not None else None,
            "metadata": json.dumps(skill.metadata),
            "created_at": skill.created_at,
            "last_accessed": skill.last_accessed,
            "access_count": skill.access_count,
            "importance": skill.importance,
            "success_count": skill.success_count,
            "failure_count": skill.failure_count,
            "pattern": skill.pattern,
            "trigger": skill.trigger,
            "tags": json.dumps(skill.tags),
        }

    def _from_row(self, row: dict) -> Skill:
        emb = None
        if row.get("embedding"):
            emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        return Skill(
            id=row["id"],
            content=row["content"],
            embedding=emb,
            metadata=json.loads(row.get("metadata") or "{}"),
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row.get("access_count", 1),
            importance=row.get("importance", 0.5),
            success_count=row.get("success_count", 0),
            failure_count=row.get("failure_count", 0),
            pattern=row.get("pattern", ""),
            trigger=row.get("trigger", ""),
            tags=json.loads(row.get("tags") or "[]"),
        )
