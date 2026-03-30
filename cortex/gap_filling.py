"""
Memory Gap Filling Engine — actively fills knowledge gaps rather than just reporting them.

Integrates with metacognition (gap detection), self-improvement (fill accuracy tracking),
evolution (evolving fill strategies), and consolidation (consolidation fills).

Gap Types:
- INFERRABLE: can be filled by reasoning from existing memories
- SEARCHABLE: needs external data (returns question/query for the caller)
- ASKABLE: needs to ask the user (returns question to present)
- CONSOLIDATABLE: can be filled by merging fragmented memories
"""

from __future__ import annotations

import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

from cortex.embeddings import embed_text, cosine_similarity, tokenize
from cortex.storage import Storage

if TYPE_CHECKING:
    from cortex.memories.episodic import EpisodicMemory
    from cortex.memories.semantic import SemanticMemory
    from cortex.memories.procedural import ProceduralMemory
    from cortex.metacognition import MetaCognitionEngine, KnowledgeGap
    from cortex.self_improvement import SelfImprovementEngine
    from cortex.consolidation import ConsolidationEngine


class GapType(str, Enum):
    INFERRABLE = "inferrable"
    SEARCHABLE = "searchable"
    ASKABLE = "askable"
    CONSOLIDATABLE = "consolidatable"


@dataclass
class Gap:
    """A detected knowledge gap with priority and fill strategy."""
    id: str
    topic: str
    description: str
    gap_type: GapType
    priority: float                           # [0, 1]
    related_memory_ids: list[str] = field(default_factory=list)
    detected_at: float = field(default_factory=time.time)
    filled: bool = False
    fill_content: str = ""
    fill_confidence: float = 0.0
    fill_strategy: str = ""
    fill_memory_id: str = ""                  # ID of the memory created to fill this gap
    confirmed: bool | None = None             # None=unverified, True=correct, False=wrong
    related_queries: int = 0                  # how many queries touched this gap area
    existing_coverage: float = 0.0            # [0, 1] how much we already know


@dataclass
class FillResult:
    """Result of attempting to fill a gap."""
    gap_id: str
    success: bool
    strategy: str
    content: str = ""
    confidence: float = 0.0
    memory_id: str = ""
    reason: str = ""


@dataclass
class CoverageCluster:
    """A topic cluster in the knowledge coverage map."""
    topic: str
    coverage: float         # [0, 1]
    memory_count: int
    gap_count: int
    avg_confidence: float
    avg_importance: float


@dataclass
class GapReport:
    """Full gap analysis with coverage map."""
    total_gaps: int
    filled_gaps: int
    fill_accuracy: float
    gaps_by_type: dict[str, int]
    top_priority_gaps: list[Gap]
    coverage_clusters: list[CoverageCluster]
    timeline_gaps: list[dict[str, Any]]
    recommendations: list[str]
    generated_at: float = field(default_factory=time.time)


class GapFillingEngine:
    """
    Actively detects and fills knowledge gaps across the memory system.

    Works with metacognition for detection, self-improvement for accuracy
    tracking, and consolidation for merge-based fills.
    """

    def __init__(
        self,
        storage: Storage,
        episodic: "EpisodicMemory",
        semantic: "SemanticMemory",
        procedural: "ProceduralMemory",
        metacognition: "MetaCognitionEngine | None" = None,
        self_improvement: "SelfImprovementEngine | None" = None,
        consolidation: "ConsolidationEngine | None" = None,
    ) -> None:
        self.storage = storage
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.metacognition = metacognition
        self.self_improvement = self_improvement
        self.consolidation = consolidation
        self._gap_cache: dict[str, Gap] = {}

    # ── Gap Detection ───────────────────────────────────────

    def find_gaps(self, top_k: int = 20) -> list[Gap]:
        """
        Detect and prioritise knowledge gaps across all memory types.
        Delegates to metacognition for raw detection, then enriches with
        priority scoring and classification.
        """
        raw_gaps: list[Gap] = []

        # 1. Use metacognition if available
        if self.metacognition:
            kg_gaps = self.metacognition.find_knowledge_gaps(top_k=top_k * 2)
            for kg in kg_gaps:
                gap = Gap(
                    id=kg.id,
                    topic=kg.topic,
                    description=kg.description,
                    gap_type=GapType(kg.gap_type) if kg.gap_type in GapType.__members__.values() else GapType.SEARCHABLE,
                    priority=kg.priority,
                    related_memory_ids=kg.related_memories,
                    detected_at=kg.detected_at,
                )
                raw_gaps.append(gap)

        # 2. Find inference chain gaps (A→B, B→?, ?→D)
        raw_gaps.extend(self._find_inference_gaps())

        # 3. Find fragmentation gaps (many small episodic memories, no semantic summary)
        raw_gaps.extend(self._find_fragmentation_gaps())

        # 4. Find pattern gaps (known patterns with missing instances)
        raw_gaps.extend(self._find_pattern_gaps())

        # 5. Score priorities
        for gap in raw_gaps:
            gap.priority = self._score_priority(gap)

        # Deduplicate by topic
        seen: set[str] = set()
        unique: list[Gap] = []
        for g in sorted(raw_gaps, key=lambda x: x.priority, reverse=True):
            key = g.topic.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(g)
                self._gap_cache[g.id] = g

        return unique[:top_k]

    def _find_inference_gaps(self) -> list[Gap]:
        """Find facts that could be inferred from existing knowledge."""
        gaps = []
        sem_facts = self.semantic.get_all()

        # Build a simple token-overlap graph to find "near" facts
        fact_tokens: dict[str, set[str]] = {}
        for f in sem_facts:
            fact_tokens[f.id] = set(tokenize(f.content))

        # Find pairs that share tokens but don't have a connecting fact
        fact_list = list(fact_tokens.items())
        for i in range(len(fact_list)):
            for j in range(i + 1, min(i + 20, len(fact_list))):
                id_a, tokens_a = fact_list[i]
                id_b, tokens_b = fact_list[j]
                shared = tokens_a & tokens_b
                only_a = tokens_a - tokens_b
                only_b = tokens_b - tokens_a

                if len(shared) >= 2 and only_a and only_b:
                    # These facts share context but have unique info → potential inference
                    topic = " ".join(list(shared)[:3])
                    # Check if inference already exists
                    existing = self.semantic.recall(topic, top_k=1)
                    if not existing or existing[0].id in (id_a, id_b):
                        gaps.append(Gap(
                            id=f"gap_infer_{id_a[:8]}_{id_b[:8]}",
                            topic=topic,
                            description=f"Can infer relationship between concepts: {topic}",
                            gap_type=GapType.INFERRABLE,
                            priority=0.5,
                            related_memory_ids=[id_a, id_b],
                        ))

        return gaps[:10]  # Limit to avoid explosion

    def _find_fragmentation_gaps(self) -> list[Gap]:
        """Find topics with many episodic fragments but no semantic summary."""
        gaps = []
        episodes = self.episodic.get_all()

        # Group episodes by rough topic (shared tokens)
        topic_episodes: dict[str, list] = defaultdict(list)
        for ep in episodes:
            top_tokens = tokenize(ep.content)[:5]
            for token in top_tokens:
                topic_episodes[token].append(ep)

        for token, eps in topic_episodes.items():
            if len(eps) >= 3:
                # Check if semantic memory has coverage
                sem_results = self.semantic.recall(token, top_k=2)
                if len(sem_results) < 1:
                    gaps.append(Gap(
                        id=f"gap_frag_{token}",
                        topic=token,
                        description=f"Topic '{token}' has {len(eps)} episodic fragments but no semantic summary",
                        gap_type=GapType.CONSOLIDATABLE,
                        priority=min(1.0, len(eps) / 8.0),
                        related_memory_ids=[ep.id for ep in eps[:5]],
                    ))

        return gaps[:10]

    def _find_pattern_gaps(self) -> list[Gap]:
        """Find recurring patterns with missing instances."""
        gaps = []
        skills = self.procedural.get_all()

        for skill in skills:
            if skill.trigger:
                # Check if there are episodic instances of this pattern
                related = self.episodic.recall(skill.trigger, top_k=3)
                if not related:
                    gaps.append(Gap(
                        id=f"gap_pattern_{skill.id}",
                        topic=skill.trigger,
                        description=f"Procedural pattern '{skill.trigger}' has no episodic instances",
                        gap_type=GapType.ASKABLE,
                        priority=0.4,
                        related_memory_ids=[skill.id],
                    ))

        return gaps

    def _score_priority(self, gap: Gap) -> float:
        """
        Priority = (importance × frequency_of_related_queries) / (age × existing_coverage)
        """
        # Importance: inferred from related memory importance
        importance = 0.5
        for mid in gap.related_memory_ids[:5]:
            row = (
                self.storage.fetch_one("semantic_memory", mid)
                or self.storage.fetch_one("episodic_memory", mid)
                or self.storage.fetch_one("procedural_memory", mid)
            )
            if row:
                importance = max(importance, row.get("importance", 0.5))

        # Frequency of related queries (approximate via access count)
        freq = max(1, gap.related_queries)

        # Age in days (capped)
        age_days = max(1.0, (time.time() - gap.detected_at) / 86400.0)

        # Existing coverage
        coverage = max(0.1, gap.existing_coverage)

        raw_priority = (importance * freq) / (age_days * coverage)
        return min(1.0, max(0.0, raw_priority))

    # ── Gap Filling ─────────────────────────────────────────

    def fill_gaps(self, max_fills: int = 10) -> list[FillResult]:
        """Auto-fill the top priority gaps that can be filled automatically."""
        gaps = self.find_gaps(top_k=max_fills * 2)
        results: list[FillResult] = []

        fillable_types = {GapType.INFERRABLE, GapType.CONSOLIDATABLE}
        fillable = [g for g in gaps if g.gap_type in fillable_types and not g.filled]

        for gap in fillable[:max_fills]:
            result = self.fill_gap(gap.id, strategy=gap.gap_type.value)
            results.append(result)

        return results

    def fill_gap(self, gap_id: str, strategy: str = "auto") -> FillResult:
        """Fill a specific gap using the given strategy."""
        gap = self._gap_cache.get(gap_id)
        if not gap:
            return FillResult(gap_id=gap_id, success=False, strategy=strategy, reason="Gap not found")

        if strategy == "auto":
            strategy = gap.gap_type.value

        fill_methods = {
            "inferrable": self._fill_by_inference,
            "inference": self._fill_by_inference,
            "consolidatable": self._fill_by_consolidation,
            "consolidation": self._fill_by_consolidation,
            "pattern": self._fill_by_pattern,
            "temporal": self._fill_by_temporal_interpolation,
            "cross_memory": self._fill_by_cross_memory,
        }

        method = fill_methods.get(strategy, self._fill_by_inference)
        result = method(gap)

        if result.success:
            gap.filled = True
            gap.fill_content = result.content
            gap.fill_confidence = result.confidence
            gap.fill_strategy = result.strategy
            gap.fill_memory_id = result.memory_id

        # Track fill outcome in self-improvement
        if self.self_improvement:
            self.self_improvement.log_fill_outcome(gap_id, result.success, result.confidence)

        return result

    def _fill_by_inference(self, gap: Gap) -> FillResult:
        """Infer missing knowledge from existing semantic + episodic memories."""
        # Gather related memories
        related_contents: list[str] = []
        for mid in gap.related_memory_ids:
            for table in ["semantic_memory", "episodic_memory"]:
                row = self.storage.fetch_one(table, mid)
                if row:
                    related_contents.append(row["content"])

        # Also search for topic
        sem_results = self.semantic.recall(gap.topic, top_k=5)
        ep_results = self.episodic.recall(gap.topic, top_k=5)
        related_contents.extend(f.content for f in sem_results)
        related_contents.extend(e.content for e in ep_results)

        if len(related_contents) < 2:
            return FillResult(
                gap_id=gap.id, success=False, strategy="inference",
                reason="Not enough related memories to infer",
            )

        # Simple inference: extract shared concepts and combine unique facts
        all_tokens: list[set[str]] = [set(tokenize(c)) for c in related_contents]
        shared = set.intersection(*all_tokens) if all_tokens else set()
        unique_sentences: list[str] = []
        seen: set[str] = set()

        for content in related_contents:
            for sentence in re.split(r'[.!?\n]+', content):
                sentence = sentence.strip()
                if sentence and sentence.lower() not in seen:
                    seen.add(sentence.lower())
                    unique_sentences.append(sentence)

        if not unique_sentences:
            return FillResult(
                gap_id=gap.id, success=False, strategy="inference",
                reason="Could not extract useful content",
            )

        inferred = ". ".join(unique_sentences[:5]) + "."
        confidence = min(0.8, 0.3 + 0.1 * len(related_contents))

        # Store as semantic memory marked as inferred
        fact = self.semantic.store(
            content=inferred,
            importance=gap.priority * 0.8,
            category="inferred",
            confidence=confidence,
            tags=["cortex_inferred", gap.topic.replace(" ", "_")],
            metadata={
                "gap_id": gap.id,
                "fill_strategy": "inference",
                "source_memories": gap.related_memory_ids[:5],
            },
            source="gap_fill",
        )

        return FillResult(
            gap_id=gap.id, success=True, strategy="inference",
            content=inferred, confidence=confidence, memory_id=fact.id,
        )

    def _fill_by_consolidation(self, gap: Gap) -> FillResult:
        """Merge fragmented episodic memories into a semantic summary."""
        if not self.consolidation:
            return FillResult(
                gap_id=gap.id, success=False, strategy="consolidation",
                reason="Consolidation engine not available",
            )

        consolidated = self.consolidation.consolidate_fragments(gap.topic)
        if not consolidated:
            return FillResult(
                gap_id=gap.id, success=False, strategy="consolidation",
                reason="Not enough fragments to consolidate",
            )

        # Find the newly created semantic memory
        results = self.semantic.recall(gap.topic, top_k=1)
        memory_id = results[0].id if results else ""

        return FillResult(
            gap_id=gap.id, success=True, strategy="consolidation",
            content=consolidated, confidence=0.85, memory_id=memory_id,
        )

    def _fill_by_pattern(self, gap: Gap) -> FillResult:
        """Use procedural patterns to fill gaps."""
        skills = self.procedural.recall(gap.topic, top_k=3)
        if not skills:
            return FillResult(
                gap_id=gap.id, success=False, strategy="pattern",
                reason="No matching procedural patterns",
            )

        # Apply the best matching pattern
        best = skills[0]
        content = f"Based on pattern '{best.pattern or best.trigger}': {best.content}"
        confidence = best.reliability * 0.9

        fact = self.semantic.store(
            content=content,
            importance=gap.priority * 0.7,
            category="inferred",
            confidence=confidence,
            tags=["cortex_inferred", "pattern_fill"],
            metadata={"gap_id": gap.id, "fill_strategy": "pattern", "pattern_id": best.id},
            source="gap_fill",
        )

        return FillResult(
            gap_id=gap.id, success=True, strategy="pattern",
            content=content, confidence=confidence, memory_id=fact.id,
        )

    def _fill_by_temporal_interpolation(self, gap: Gap) -> FillResult:
        """Fill time gaps by interpolating between known events."""
        if len(gap.related_memory_ids) < 2:
            return FillResult(
                gap_id=gap.id, success=False, strategy="temporal",
                reason="Need at least 2 temporal anchors",
            )

        # Get the bounding episodes
        episodes = []
        for mid in gap.related_memory_ids[:2]:
            row = self.storage.fetch_one("episodic_memory", mid)
            if row:
                episodes.append(row)

        if len(episodes) < 2:
            return FillResult(
                gap_id=gap.id, success=False, strategy="temporal",
                reason="Could not find bounding episodes",
            )

        before = min(episodes, key=lambda e: e["created_at"])
        after = max(episodes, key=lambda e: e["created_at"])

        content = (
            f"Temporal gap between: [{before['content'][:80]}] and [{after['content'][:80]}]. "
            f"No recorded events for this period."
        )

        ep = self.episodic.store(
            content=content,
            importance=0.3,
            tags=["cortex_inferred", "temporal_fill"],
            metadata={"gap_id": gap.id, "fill_strategy": "temporal"},
            source="gap_fill",
        )

        return FillResult(
            gap_id=gap.id, success=True, strategy="temporal",
            content=content, confidence=0.4, memory_id=ep.id,
        )

    def _fill_by_cross_memory(self, gap: Gap) -> FillResult:
        """Use one memory type to fill gaps in another."""
        # Try each memory type as a source
        for source_recall, source_name in [
            (self.episodic.recall, "episodic"),
            (self.semantic.recall, "semantic"),
            (self.procedural.recall, "procedural"),
        ]:
            results = source_recall(gap.topic, top_k=3)
            if results:
                content = "; ".join(r.content[:100] for r in results)
                confidence = 0.6

                fact = self.semantic.store(
                    content=f"Cross-memory fill from {source_name}: {content}",
                    importance=gap.priority * 0.6,
                    category="inferred",
                    confidence=confidence,
                    tags=["cortex_inferred", "cross_memory_fill"],
                    metadata={
                        "gap_id": gap.id,
                        "fill_strategy": "cross_memory",
                        "source_type": source_name,
                    },
                    source="gap_fill",
                )

                return FillResult(
                    gap_id=gap.id, success=True, strategy="cross_memory",
                    content=content, confidence=confidence, memory_id=fact.id,
                )

        return FillResult(
            gap_id=gap.id, success=False, strategy="cross_memory",
            reason="No memories found in any type for this topic",
        )

    # ── Gap Verification ────────────────────────────────────

    def confirm_fill(self, gap_id: str, correct: bool) -> None:
        """User confirms or corrects a gap fill."""
        gap = self._gap_cache.get(gap_id)
        if not gap:
            return

        gap.confirmed = correct

        if correct:
            # Boost confidence of the filled memory
            if gap.fill_memory_id:
                self.semantic.update_confidence(gap.fill_memory_id, min(1.0, gap.fill_confidence + 0.15))
        else:
            # Reduce confidence or remove
            if gap.fill_memory_id:
                self.semantic.update_confidence(gap.fill_memory_id, max(0.0, gap.fill_confidence - 0.3))

        if self.self_improvement:
            self.self_improvement.log_fill_outcome(gap_id, correct, gap.fill_confidence)

    def request_fill(self, gap_id: str) -> str | None:
        """
        Queue a gap for external data. Returns a question to ask the user,
        or None if the gap can't be expressed as a question.
        """
        gap = self._gap_cache.get(gap_id)
        if not gap:
            return None

        if gap.gap_type == GapType.ASKABLE:
            return f"I have a knowledge gap about '{gap.topic}': {gap.description}. Can you help fill this in?"
        elif gap.gap_type == GapType.SEARCHABLE:
            return f"I need to search for information about '{gap.topic}': {gap.description}"
        else:
            return f"I'm missing information about '{gap.topic}'. What can you tell me?"

    # ── Gap Report ──────────────────────────────────────────

    def gap_report(self) -> GapReport:
        """Generate a comprehensive gap analysis with coverage map."""
        gaps = self.find_gaps(top_k=50)

        # Count by type
        by_type: dict[str, int] = defaultdict(int)
        for g in gaps:
            by_type[g.gap_type.value] += 1

        filled = sum(1 for g in gaps if g.filled)

        # Fill accuracy
        fill_accuracy = 0.0
        if self.self_improvement:
            fill_accuracy = self.self_improvement.get_fill_accuracy()

        # Coverage clusters
        clusters = self._build_coverage_clusters()

        # Timeline gaps
        timeline = self._build_timeline_gaps()

        # Recommendations
        recommendations = []
        if by_type.get("consolidatable", 0) > 3:
            recommendations.append(f"Run consolidation — {by_type['consolidatable']} topics have fragmented memories.")
        if by_type.get("askable", 0) > 2:
            recommendations.append(f"Ask the user about {by_type['askable']} topics with missing context.")
        if by_type.get("inferrable", 0) > 2:
            recommendations.append(f"Run fill_gaps() to auto-infer {by_type['inferrable']} relationships.")
        if filled == 0 and gaps:
            recommendations.append("No gaps have been filled yet. Try engine.fill_gaps().")
        if fill_accuracy < 0.5 and filled > 5:
            recommendations.append(f"Fill accuracy is low ({fill_accuracy:.0%}). Review inferred memories.")

        return GapReport(
            total_gaps=len(gaps),
            filled_gaps=filled,
            fill_accuracy=fill_accuracy,
            gaps_by_type=dict(by_type),
            top_priority_gaps=gaps[:10],
            coverage_clusters=clusters,
            timeline_gaps=timeline,
            recommendations=recommendations,
        )

    def _build_coverage_clusters(self) -> list[CoverageCluster]:
        """Build topic clusters with coverage metrics."""
        clusters: list[CoverageCluster] = []

        # Gather all content across memory types
        all_items: list[dict] = []
        for table in ["semantic_memory", "episodic_memory", "procedural_memory"]:
            all_items.extend(self.storage.fetch_all(table))

        # Token frequency as rough topic detection
        token_memories: dict[str, list[dict]] = defaultdict(list)
        for item in all_items:
            for token in tokenize(item["content"])[:8]:
                token_memories[token].append(item)

        # Build clusters from tokens with enough memories
        for token, items in token_memories.items():
            if len(items) < 2:
                continue

            confidences = [it.get("confidence", 0.5) for it in items]
            importances = [it.get("importance", 0.5) for it in items]

            # Count gaps for this topic
            gap_count = sum(1 for g in self._gap_cache.values() if token in g.topic.lower())

            coverage = min(1.0, len(items) / 10.0)

            clusters.append(CoverageCluster(
                topic=token,
                coverage=round(coverage, 2),
                memory_count=len(items),
                gap_count=gap_count,
                avg_confidence=round(sum(confidences) / len(confidences), 2),
                avg_importance=round(sum(importances) / len(importances), 2),
            ))

        clusters.sort(key=lambda c: c.memory_count, reverse=True)
        return clusters[:30]

    def _build_timeline_gaps(self) -> list[dict[str, Any]]:
        """Find temporal gaps in episodic memory."""
        rows = self.storage.execute(
            "SELECT id, content, created_at FROM episodic_memory ORDER BY created_at"
        )
        gaps = []
        for i in range(1, len(rows)):
            gap_hours = (rows[i]["created_at"] - rows[i - 1]["created_at"]) / 3600
            if gap_hours > 24:
                gaps.append({
                    "before_id": rows[i - 1]["id"],
                    "after_id": rows[i]["id"],
                    "gap_hours": round(gap_hours, 1),
                    "before_time": rows[i - 1]["created_at"],
                    "after_time": rows[i]["created_at"],
                })
        return gaps
