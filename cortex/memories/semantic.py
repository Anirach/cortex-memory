"""
Semantic Memory — facts, knowledge, and entities with importance scoring.

Long-term storage for declarative knowledge. Supports categories, confidence
scores, and automatic importance adjustment based on access patterns.
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
class Fact:
    id: str
    content: str
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    importance: float = 0.5
    category: str = "general"
    confidence: float = 1.0
    source: str = "user"
    tags: list[str] = field(default_factory=list)


class SemanticMemory:
    """Persistent semantic memory (facts and knowledge) backed by SQLite."""

    TABLE = "semantic_memory"

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def store(
        self,
        content: str,
        importance: float = 0.5,
        category: str = "general",
        confidence: float = 1.0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "user",
    ) -> Fact:
        """Store a new fact."""
        now = time.time()
        fact = Fact(
            id=f"sem_{uuid.uuid4().hex[:12]}",
            content=content,
            embedding=embed_text(content),
            metadata=metadata or {},
            created_at=now,
            last_accessed=now,
            importance=max(0.0, min(1.0, importance)),
            category=category,
            confidence=max(0.0, min(1.0, confidence)),
            tags=tags or [],
            source=source,
        )
        self.storage.insert(self.TABLE, self._to_row(fact))
        return fact

    def recall(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[Fact]:
        """Retrieve facts by semantic similarity."""
        query_vec = embed_text(query)

        where_parts = []
        params: list = []
        if category:
            where_parts.append("category = ?")
            params.append(category)
        if min_confidence > 0:
            where_parts.append("confidence >= ?")
            params.append(min_confidence)

        where = " AND ".join(where_parts) if where_parts else ""
        rows = self.storage.fetch_all(self.TABLE, where, tuple(params))
        if not rows:
            return []

        scored: list[tuple[float, Fact]] = []
        for row in rows:
            fact = self._from_row(row)
            sim = 0.0
            if fact.embedding is not None:
                sim = cosine_similarity(query_vec, fact.embedding)
            score = sim * fact.confidence * (0.5 + fact.importance)
            scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [f for _, f in scored[:top_k]]

        for fact in results:
            self._touch(fact.id)

        return results

    def get(self, fact_id: str) -> Fact | None:
        row = self.storage.fetch_one(self.TABLE, fact_id)
        if row:
            self._touch(fact_id)
            return self._from_row(row)
        return None

    def get_all(self) -> list[Fact]:
        return [self._from_row(r) for r in self.storage.fetch_all(self.TABLE)]

    def get_categories(self) -> list[str]:
        rows = self.storage.execute(f"SELECT DISTINCT category FROM {self.TABLE}")
        return [r["category"] for r in rows]

    def update_confidence(self, fact_id: str, new_confidence: float) -> None:
        self.storage.execute(
            f"UPDATE {self.TABLE} SET confidence = ? WHERE id = ?",
            (max(0.0, min(1.0, new_confidence)), fact_id),
        )

    def boost_importance(self, fact_id: str, delta: float = 0.1) -> None:
        self.storage.execute(
            f"UPDATE {self.TABLE} SET importance = MIN(1.0, importance + ?) WHERE id = ?",
            (delta, fact_id),
        )

    def delete(self, fact_id: str) -> None:
        self.storage.delete(self.TABLE, fact_id)

    def count(self) -> int:
        return self.storage.count(self.TABLE)

    # ── Internal ────────────────────────────────────────────

    def _touch(self, fact_id: str) -> None:
        self.storage.execute(
            f"UPDATE {self.TABLE} SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (time.time(), fact_id),
        )

    def _to_row(self, fact: Fact) -> dict:
        return {
            "id": fact.id,
            "content": fact.content,
            "embedding": fact.embedding.tobytes() if fact.embedding is not None else None,
            "metadata": json.dumps(fact.metadata),
            "created_at": fact.created_at,
            "last_accessed": fact.last_accessed,
            "access_count": fact.access_count,
            "importance": fact.importance,
            "category": fact.category,
            "confidence": fact.confidence,
            "source": fact.source,
            "tags": json.dumps(fact.tags),
        }

    def _from_row(self, row: dict) -> Fact:
        emb = None
        if row.get("embedding"):
            emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        return Fact(
            id=row["id"],
            content=row["content"],
            embedding=emb,
            metadata=json.loads(row.get("metadata") or "{}"),
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row.get("access_count", 1),
            importance=row.get("importance", 0.5),
            category=row.get("category", "general"),
            confidence=row.get("confidence", 1.0),
            source=row.get("source", "user"),
            tags=json.loads(row.get("tags") or "[]"),
        )
