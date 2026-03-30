"""
Hippocampal Index — hybrid search combining multiple retrieval signals.

Combines:
1. Vector similarity (cosine)
2. BM25 text search
3. Temporal proximity scoring
4. Graph-based relationship traversal
5. Cross-memory inference
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cortex.embeddings import embed_text, cosine_similarity, tokenize
from cortex.storage import Storage


@dataclass
class SearchResult:
    """A unified search result from any memory type."""
    id: str
    content: str
    memory_type: str       # episodic, semantic, procedural
    score: float
    signals: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class HippocampalIndex:
    """
    Hybrid retrieval engine that merges multiple search signals
    into a single relevance score.
    """

    # Tables to search
    MEMORY_TABLES = {
        "episodic": "episodic_memory",
        "semantic": "semantic_memory",
        "procedural": "procedural_memory",
    }

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self._idf_cache: dict[str, float] | None = None

    def search(
        self,
        query: str,
        top_k: int = 10,
        memory_types: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search across all memory types.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Max results to return.
        memory_types : list[str] | None
            Restrict to specific types (default: all).
        weights : dict[str, float] | None
            Override signal weights. Keys: similarity, bm25, temporal, graph, importance.
        """
        if weights is None:
            weights = {
                "similarity": 1.5,
                "bm25": 1.0,
                "temporal": 0.3,
                "graph": 0.5,
                "importance": 0.8,
            }

        query_vec = embed_text(query)
        query_tokens = tokenize(query)
        now = time.time()

        # Build IDF index if needed
        self._build_idf()

        types_to_search = memory_types or list(self.MEMORY_TABLES.keys())
        all_results: list[SearchResult] = []

        for mem_type in types_to_search:
            table = self.MEMORY_TABLES.get(mem_type)
            if not table:
                continue

            rows = self.storage.fetch_all(table)
            for row in rows:
                signals = {}

                # 1. Vector similarity
                emb = row.get("embedding")
                if emb:
                    mem_vec = np.frombuffer(emb, dtype=np.float32)
                    signals["similarity"] = max(0.0, cosine_similarity(query_vec, mem_vec))
                else:
                    signals["similarity"] = 0.0

                # 2. BM25 score
                signals["bm25"] = self._bm25_score(query_tokens, row["content"])

                # 3. Temporal proximity (more recent = higher)
                age_hours = max(1, (now - row.get("last_accessed", row["created_at"])) / 3600)
                signals["temporal"] = 1.0 / math.log1p(age_hours)

                # 4. Graph connectivity (number of edges)
                edge_count = self.storage.count(
                    "memory_edges",
                    "source_id = ? OR target_id = ?",
                    (row["id"], row["id"]),
                )
                signals["graph"] = min(1.0, edge_count / 5.0)

                # 5. Importance
                signals["importance"] = row.get("importance", 0.5)

                # Weighted combination
                total_score = sum(
                    weights.get(signal, 0) * value
                    for signal, value in signals.items()
                )

                all_results.append(SearchResult(
                    id=row["id"],
                    content=row["content"],
                    memory_type=mem_type,
                    score=total_score,
                    signals=signals,
                    metadata=json.loads(row.get("metadata") or "{}"),
                ))

        # Sort and return top-k
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    # ── BM25 ────────────────────────────────────────────────

    def _build_idf(self) -> None:
        """Build IDF scores across all memories."""
        if self._idf_cache is not None:
            return

        doc_count = 0
        df: Counter = Counter()

        for table in self.MEMORY_TABLES.values():
            rows = self.storage.execute(f"SELECT content FROM {table}")
            for row in rows:
                doc_count += 1
                tokens = set(tokenize(row["content"]))
                df.update(tokens)

        doc_count = max(doc_count, 1)
        self._idf_cache = {
            token: math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in df.items()
        }

    def _bm25_score(
        self,
        query_tokens: list[str],
        document: str,
        k1: float = 1.5,
        b: float = 0.75,
        avg_dl: float = 50.0,
    ) -> float:
        """Okapi BM25 score for a single document."""
        if not self._idf_cache or not query_tokens:
            return 0.0

        doc_tokens = tokenize(document)
        dl = len(doc_tokens)
        tf = Counter(doc_tokens)

        score = 0.0
        for qt in query_tokens:
            if qt not in tf:
                continue
            idf = self._idf_cache.get(qt, 0.0)
            term_freq = tf[qt]
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * dl / avg_dl)
            score += idf * numerator / denominator

        return score

    def invalidate_cache(self) -> None:
        """Force IDF rebuild on next search."""
        self._idf_cache = None

    # ── Graph Operations ────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
    ) -> None:
        """Add a relationship between two memories."""
        self.storage.insert("memory_edges", {
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation,
            "weight": weight,
            "created_at": time.time(),
        })

    def get_neighbors(self, memory_id: str, max_depth: int = 2) -> list[dict[str, Any]]:
        """BFS traversal of memory graph from a starting node."""
        visited: set[str] = {memory_id}
        queue: list[tuple[str, int]] = [(memory_id, 0)]
        results: list[dict[str, Any]] = []

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            edges = self.storage.fetch_all(
                "memory_edges",
                "source_id = ? OR target_id = ?",
                (current, current),
            )
            for edge in edges:
                neighbor = (
                    edge["target_id"] if edge["source_id"] == current
                    else edge["source_id"]
                )
                if neighbor not in visited:
                    visited.add(neighbor)
                    results.append({
                        "id": neighbor,
                        "relation": edge["relation"],
                        "weight": edge["weight"],
                        "depth": depth + 1,
                    })
                    queue.append((neighbor, depth + 1))

        return results
