"""
Episodic Memory — timestamped events and conversations with temporal decay.

Each memory is an "episode" with rich temporal metadata, importance scoring,
and automatic decay via the Ebbinghaus forgetting curve.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cortex.embeddings import embed_text, cosine_similarity
from cortex.forgetting import retention, should_forget
from cortex.storage import Storage


@dataclass
class Episode:
    id: str
    content: str
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    importance: float = 0.5
    decay_factor: float = 1.0
    tags: list[str] = field(default_factory=list)
    source: str = "user"


class EpisodicMemory:
    """Persistent episodic memory backed by SQLite."""

    TABLE = "episodic_memory"

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def store(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "user",
    ) -> Episode:
        """Store a new episode."""
        now = time.time()
        ep = Episode(
            id=f"ep_{uuid.uuid4().hex[:12]}",
            content=content,
            embedding=embed_text(content),
            metadata=metadata or {},
            created_at=now,
            last_accessed=now,
            importance=max(0.0, min(1.0, importance)),
            tags=tags or [],
            source=source,
        )
        self.storage.insert(self.TABLE, self._to_row(ep))
        return ep

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_retention: float = 0.05,
    ) -> list[Episode]:
        """Retrieve episodes by semantic + temporal relevance."""
        query_vec = embed_text(query)
        rows = self.storage.fetch_all(self.TABLE)
        if not rows:
            return []

        scored: list[tuple[float, Episode]] = []
        now = time.time()

        for row in rows:
            ep = self._from_row(row)
            ret = retention(ep.last_accessed, ep.access_count, ep.importance, now=now)
            if ret < min_retention:
                continue

            sim = 0.0
            if ep.embedding is not None:
                sim = cosine_similarity(query_vec, ep.embedding)

            # Combined score: similarity * retention * importance
            score = sim * ret * (0.5 + ep.importance)
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in scored[:top_k]]

        # Update access timestamps
        for ep in results:
            self._touch(ep.id)

        return results

    def get(self, episode_id: str) -> Episode | None:
        row = self.storage.fetch_one(self.TABLE, episode_id)
        if row:
            self._touch(episode_id)
            return self._from_row(row)
        return None

    def get_all(self) -> list[Episode]:
        return [self._from_row(r) for r in self.storage.fetch_all(self.TABLE)]

    def delete(self, episode_id: str) -> None:
        self.storage.delete(self.TABLE, episode_id)

    def prune_forgotten(self, threshold: float = 0.05) -> int:
        """Remove memories below retention threshold. Returns count removed."""
        rows = self.storage.fetch_all(self.TABLE)
        now = time.time()
        pruned = 0
        for row in rows:
            if should_forget(row["last_accessed"], row["access_count"], row["importance"], threshold, now):
                self.storage.delete(self.TABLE, row["id"])
                pruned += 1
        return pruned

    def count(self) -> int:
        return self.storage.count(self.TABLE)

    # ── Internal ────────────────────────────────────────────

    def _touch(self, episode_id: str) -> None:
        self.storage.execute(
            f"UPDATE {self.TABLE} SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (time.time(), episode_id),
        )

    def _to_row(self, ep: Episode) -> dict:
        return {
            "id": ep.id,
            "content": ep.content,
            "embedding": ep.embedding.tobytes() if ep.embedding is not None else None,
            "metadata": json.dumps(ep.metadata),
            "created_at": ep.created_at,
            "last_accessed": ep.last_accessed,
            "access_count": ep.access_count,
            "importance": ep.importance,
            "decay_factor": ep.decay_factor,
            "tags": json.dumps(ep.tags),
            "source": ep.source,
        }

    def _from_row(self, row: dict) -> Episode:
        emb = None
        if row.get("embedding"):
            emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        return Episode(
            id=row["id"],
            content=row["content"],
            embedding=emb,
            metadata=json.loads(row.get("metadata") or "{}"),
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row.get("access_count", 1),
            importance=row.get("importance", 0.5),
            decay_factor=row.get("decay_factor", 1.0),
            tags=json.loads(row.get("tags") or "[]"),
            source=row.get("source", "user"),
        )
