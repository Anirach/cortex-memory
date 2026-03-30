"""
Memory Consolidation — the "Sleep/Dream" cycle.

Moves memories between layers based on access patterns and importance:
- Working → Episodic  (after session flush)
- Episodic → Semantic  (repeated access or high importance)
- Episodic → Procedural (when action patterns are detected)
- Low-retention memories are pruned via forgetting curve.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cortex.forgetting import retention, should_forget

if TYPE_CHECKING:
    from cortex.storage import Storage
    from cortex.memories.working import WorkingMemory
    from cortex.memories.episodic import EpisodicMemory
    from cortex.memories.semantic import SemanticMemory
    from cortex.memories.procedural import ProceduralMemory


# ── Pattern detectors for procedural promotion ──────────────

_ACTION_PATTERNS = [
    re.compile(r"\b(when|if|whenever)\b.*\b(then|do|use|run|call|execute)\b", re.I),
    re.compile(r"\b(to|how to|steps? to)\b.*\b(first|then|next|finally)\b", re.I),
    re.compile(r"\b(always|never|should|must)\b.*\b(before|after|instead)\b", re.I),
]


def _looks_procedural(text: str) -> bool:
    return any(p.search(text) for p in _ACTION_PATTERNS)


@dataclass
class ConsolidationReport:
    """Summary of a consolidation cycle."""
    working_to_episodic: int = 0
    episodic_to_semantic: int = 0
    episodic_to_procedural: int = 0
    pruned_episodic: int = 0
    pruned_semantic: int = 0
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ConsolidationEngine:
    """Orchestrates the sleep/dream memory consolidation cycle."""

    # Thresholds
    SEMANTIC_PROMOTION_ACCESS = 3      # min access count to promote
    SEMANTIC_PROMOTION_IMPORTANCE = 0.7 # OR importance above this
    PRUNE_RETENTION_THRESHOLD = 0.05   # forget below this retention

    def __init__(
        self,
        storage: "Storage",
        working: "WorkingMemory",
        episodic: "EpisodicMemory",
        semantic: "SemanticMemory",
        procedural: "ProceduralMemory",
    ) -> None:
        self.storage = storage
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

    def consolidate(self) -> ConsolidationReport:
        """Run a full consolidation cycle. Returns a report."""
        t0 = time.time()
        report = ConsolidationReport()

        report.working_to_episodic = self._flush_working()
        report.episodic_to_semantic = self._promote_to_semantic()
        report.episodic_to_procedural = self._promote_to_procedural()
        report.pruned_episodic = self.episodic.prune_forgotten(self.PRUNE_RETENTION_THRESHOLD)
        report.pruned_semantic = self._prune_low_confidence_semantic()

        report.duration_ms = (time.time() - t0) * 1000
        report.timestamp = time.time()
        return report

    def _flush_working(self) -> int:
        """Move working memory items into episodic memory."""
        items = self.working.clear()
        for item in items:
            self.episodic.store(
                content=item.content,
                importance=0.4,  # working memory items start at moderate importance
                metadata={**item.metadata, "source": "working_memory"},
                source="consolidation",
            )
        return len(items)

    def _promote_to_semantic(self) -> int:
        """Promote frequently-accessed or high-importance episodes to semantic memory."""
        episodes = self.episodic.get_all()
        promoted = 0

        for ep in episodes:
            qualifies = (
                ep.access_count >= self.SEMANTIC_PROMOTION_ACCESS
                or ep.importance >= self.SEMANTIC_PROMOTION_IMPORTANCE
            )
            if not qualifies:
                continue

            # Check if already stored as semantic (by content similarity)
            existing = self.semantic.recall(ep.content, top_k=1)
            if existing and existing[0].content == ep.content:
                # Boost existing instead of duplicating
                self.semantic.boost_importance(existing[0].id, 0.05)
                continue

            self.semantic.store(
                content=ep.content,
                importance=min(1.0, ep.importance + 0.1),
                category="consolidated",
                confidence=0.9,
                tags=ep.tags,
                metadata={**ep.metadata, "promoted_from": ep.id},
                source="consolidation",
            )
            promoted += 1

        return promoted

    def _promote_to_procedural(self) -> int:
        """Detect action patterns in episodes and promote to procedural memory."""
        episodes = self.episodic.get_all()
        promoted = 0

        for ep in episodes:
            if _looks_procedural(ep.content):
                # Check for duplicates
                existing = self.procedural.recall(ep.content, top_k=1)
                if existing and existing[0].content == ep.content:
                    continue

                self.procedural.store(
                    content=ep.content,
                    pattern="auto-detected",
                    trigger="",
                    importance=min(1.0, ep.importance + 0.1),
                    tags=ep.tags,
                    metadata={**ep.metadata, "promoted_from": ep.id},
                )
                promoted += 1

        return promoted

    def _prune_low_confidence_semantic(self) -> int:
        """Remove semantic memories with very low confidence."""
        rows = self.storage.fetch_all(
            "semantic_memory", "confidence < ? AND access_count < ?", (0.2, 2)
        )
        for row in rows:
            self.storage.delete("semantic_memory", row["id"])
        return len(rows)

    def consolidate_fragments(self, topic: str, max_fragments: int = 10) -> str | None:
        """
        Merge multiple episodic memories about the same topic into a single
        semantic summary. Returns the consolidated content or None.
        """
        episodes = self.episodic.recall(topic, top_k=max_fragments)
        if len(episodes) < 2:
            return None

        # Build consolidated summary from fragments
        contents = [ep.content for ep in episodes]
        # Simple merge: deduplicate sentences and join
        seen_sentences: set[str] = set()
        merged_parts: list[str] = []
        for content in contents:
            for sentence in re.split(r'[.!?]+', content):
                sentence = sentence.strip()
                if sentence and sentence.lower() not in seen_sentences:
                    seen_sentences.add(sentence.lower())
                    merged_parts.append(sentence)

        if not merged_parts:
            return None

        consolidated = ". ".join(merged_parts) + "."
        avg_importance = sum(ep.importance for ep in episodes) / len(episodes)

        self.semantic.store(
            content=consolidated,
            importance=min(1.0, avg_importance + 0.15),
            category="consolidated",
            confidence=0.85,
            tags=["consolidated", topic.lower().replace(" ", "_")],
            metadata={
                "source_episodes": [ep.id for ep in episodes],
                "fragment_count": len(episodes),
            },
            source="consolidation",
        )
        return consolidated
