"""
Meta-Cognition Layer — confidence scoring, uncertainty detection, knowledge gaps.

Provides self-awareness about what the agent knows and doesn't know, how
confident it should be in its memories, and where gaps exist.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cortex.embeddings import embed_text, cosine_similarity, tokenize
from cortex.storage import Storage

if TYPE_CHECKING:
    from cortex.memories.episodic import EpisodicMemory
    from cortex.memories.semantic import SemanticMemory
    from cortex.memories.procedural import ProceduralMemory


@dataclass
class ConfidenceAssessment:
    """How confident the agent is about a topic."""
    topic: str
    confidence: float      # [0, 1]
    coverage: float        # [0, 1] — how much is known
    memory_count: int      # number of relevant memories
    sources: list[str]     # which memory types contributed
    evidence: list[str]    # top relevant memory contents
    gaps: list[str]        # identified gaps


@dataclass
class KnowledgeGap:
    """A detected gap in knowledge."""
    id: str
    topic: str
    description: str
    gap_type: str          # inferrable, searchable, askable, consolidatable
    priority: float        # [0, 1]
    related_memories: list[str]
    detected_at: float = field(default_factory=time.time)


@dataclass
class CoverageMap:
    """Overview of knowledge coverage by topic cluster."""
    clusters: dict[str, float]            # topic → coverage [0,1]
    total_memories: int
    well_covered: list[str]               # topics with >0.7 coverage
    sparse: list[str]                     # topics with <0.3 coverage
    temporal_coverage: dict[str, int]     # time period → memory count


@dataclass
class SelfEvaluation:
    """Overall assessment of memory quality."""
    total_memories: int
    avg_confidence: float
    avg_importance: float
    coverage_score: float
    diversity_score: float
    recency_score: float
    overall_quality: float
    recommendations: list[str]


class MetaCognitionEngine:
    """Self-awareness layer: confidence, uncertainty, and knowledge gaps."""

    def __init__(
        self,
        storage: Storage,
        episodic: "EpisodicMemory",
        semantic: "SemanticMemory",
        procedural: "ProceduralMemory",
    ) -> None:
        self.storage = storage
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

    # ── Confidence Assessment ───────────────────────────────

    def assess_confidence(self, topic: str, detail: bool = False) -> ConfidenceAssessment:
        """
        How confident is the agent about a topic?

        Looks across all memory types for relevant memories and scores
        confidence based on quantity, quality, recency, and agreement.
        """
        query_vec = embed_text(topic)

        # Gather evidence from all memory types
        ep_results = self.episodic.recall(topic, top_k=10, min_retention=0.0)
        sem_results = self.semantic.recall(topic, top_k=10)
        proc_results = self.procedural.recall(topic, top_k=5)

        sources = []
        evidence = []
        all_importances = []
        all_confidences = []

        if ep_results:
            sources.append("episodic")
            for ep in ep_results[:3]:
                evidence.append(ep.content[:100])
                all_importances.append(ep.importance)

        if sem_results:
            sources.append("semantic")
            for fact in sem_results[:3]:
                evidence.append(fact.content[:100])
                all_importances.append(fact.importance)
                all_confidences.append(fact.confidence)

        if proc_results:
            sources.append("procedural")
            for skill in proc_results[:2]:
                evidence.append(skill.content[:100])
                all_importances.append(skill.importance)

        total_count = len(ep_results) + len(sem_results) + len(proc_results)

        # Confidence formula:
        #   - More memories → higher confidence (log scale)
        #   - Higher average semantic confidence → higher
        #   - More diverse sources → higher
        quantity_score = min(1.0, math.log1p(total_count) / math.log1p(20))
        quality_score = (sum(all_confidences) / len(all_confidences)) if all_confidences else 0.5
        diversity_score = len(sources) / 3.0
        importance_avg = (sum(all_importances) / len(all_importances)) if all_importances else 0.0

        confidence = (
            0.35 * quantity_score
            + 0.30 * quality_score
            + 0.20 * diversity_score
            + 0.15 * importance_avg
        )

        # Coverage: ratio of memory count to expected coverage
        coverage = min(1.0, total_count / 10.0)

        # Detect gaps
        gaps = self._detect_topic_gaps(topic, ep_results, sem_results, proc_results)

        assessment = ConfidenceAssessment(
            topic=topic,
            confidence=round(confidence, 3),
            coverage=round(coverage, 3),
            memory_count=total_count,
            sources=sources,
            evidence=evidence if detail else evidence[:3],
            gaps=[g.description for g in gaps],
        )

        # Persist assessment
        self.storage.insert("meta_assessments", {
            "topic": topic,
            "confidence": assessment.confidence,
            "coverage": assessment.coverage,
            "memory_count": total_count,
            "timestamp": time.time(),
        })

        return assessment

    # ── Knowledge Gap Detection ─────────────────────────────

    def find_knowledge_gaps(self, top_k: int = 10) -> list[KnowledgeGap]:
        """
        Scan all memories to identify topics with sparse coverage,
        broken inference chains, and temporal gaps.
        """
        gaps: list[KnowledgeGap] = []

        # 1. Find sparse topics from semantic memory categories
        gaps.extend(self._find_sparse_topics())

        # 2. Find broken inference chains
        gaps.extend(self._find_broken_chains())

        # 3. Find temporal gaps in episodic memory
        gaps.extend(self._find_temporal_gaps())

        # 4. Find mentioned-but-unknown topics
        gaps.extend(self._find_mentioned_unknowns())

        # Sort by priority and deduplicate
        seen_topics = set()
        unique_gaps = []
        for g in sorted(gaps, key=lambda x: x.priority, reverse=True):
            if g.topic.lower() not in seen_topics:
                seen_topics.add(g.topic.lower())
                unique_gaps.append(g)

        return unique_gaps[:top_k]

    def _find_sparse_topics(self) -> list[KnowledgeGap]:
        """Find categories with very few semantic memories."""
        gaps = []
        categories = self.semantic.get_categories()
        for cat in categories:
            count = self.storage.count("semantic_memory", "category = ?", (cat,))
            if count < 3:
                gaps.append(KnowledgeGap(
                    id=f"gap_sparse_{cat}",
                    topic=cat,
                    description=f"Topic '{cat}' has only {count} memories — sparse coverage",
                    gap_type="searchable",
                    priority=0.6,
                    related_memories=[],
                ))
        return gaps

    def _find_broken_chains(self) -> list[KnowledgeGap]:
        """Find edges in the memory graph that point to missing nodes."""
        gaps = []
        edges = self.storage.fetch_all("memory_edges")
        for edge in edges:
            target = edge["target_id"]
            # Check if target exists in any memory table
            exists = (
                self.storage.fetch_one("episodic_memory", target)
                or self.storage.fetch_one("semantic_memory", target)
                or self.storage.fetch_one("procedural_memory", target)
            )
            if not exists:
                gaps.append(KnowledgeGap(
                    id=f"gap_chain_{edge['source_id']}_{target}",
                    topic=edge.get("relation", "unknown"),
                    description=f"Broken inference chain: {edge['source_id']} → {target} (missing)",
                    gap_type="inferrable",
                    priority=0.7,
                    related_memories=[edge["source_id"]],
                ))
        return gaps

    def _find_temporal_gaps(self) -> list[KnowledgeGap]:
        """Find large time gaps in episodic memory."""
        gaps = []
        rows = self.storage.execute(
            "SELECT id, content, created_at FROM episodic_memory ORDER BY created_at"
        )
        if len(rows) < 2:
            return gaps

        for i in range(1, len(rows)):
            gap_hours = (rows[i]["created_at"] - rows[i - 1]["created_at"]) / 3600
            if gap_hours > 72:  # >3 days gap
                gaps.append(KnowledgeGap(
                    id=f"gap_temporal_{i}",
                    topic="temporal_gap",
                    description=f"No episodes for {gap_hours:.0f} hours between events",
                    gap_type="askable",
                    priority=min(1.0, gap_hours / 168),  # normalize to 1 week
                    related_memories=[rows[i - 1]["id"], rows[i]["id"]],
                ))
        return gaps

    def _find_mentioned_unknowns(self) -> list[KnowledgeGap]:
        """Find topics mentioned in metadata/content but not deeply covered."""
        gaps = []
        # Collect all tags from episodic memory
        ep_rows = self.storage.fetch_all("episodic_memory")
        tag_counts: Counter = Counter()
        for row in ep_rows:
            tags = json.loads(row.get("tags") or "[]")
            tag_counts.update(tags)

        # Check which tags lack semantic coverage
        for tag, count in tag_counts.items():
            if count >= 2:  # mentioned multiple times
                sem_results = self.semantic.recall(tag, top_k=3)
                if len(sem_results) < 2:
                    gaps.append(KnowledgeGap(
                        id=f"gap_mentioned_{tag}",
                        topic=tag,
                        description=f"Topic '{tag}' mentioned {count} times in episodes but lacks semantic knowledge",
                        gap_type="consolidatable",
                        priority=min(1.0, count / 5.0) * 0.8,
                        related_memories=[],
                    ))
        return gaps

    def _detect_topic_gaps(self, topic, ep_results, sem_results, proc_results) -> list[KnowledgeGap]:
        """Detect gaps specific to a queried topic."""
        gaps = []
        if not sem_results:
            gaps.append(KnowledgeGap(
                id=f"gap_no_semantic_{topic}",
                topic=topic,
                description=f"No semantic (factual) knowledge about '{topic}'",
                gap_type="searchable",
                priority=0.7,
                related_memories=[],
            ))
        if not proc_results:
            gaps.append(KnowledgeGap(
                id=f"gap_no_procedural_{topic}",
                topic=topic,
                description=f"No procedural (how-to) knowledge about '{topic}'",
                gap_type="searchable",
                priority=0.4,
                related_memories=[],
            ))
        return gaps

    # ── Coverage Map ────────────────────────────────────────

    def get_coverage_map(self) -> CoverageMap:
        """Build a knowledge coverage map across all memory types."""
        # Cluster by semantic categories + frequent tokens
        clusters: dict[str, float] = {}

        # From semantic categories
        for cat in self.semantic.get_categories():
            count = self.storage.count("semantic_memory", "category = ?", (cat,))
            clusters[cat] = min(1.0, count / 10.0)

        # From frequent tokens across all memories
        all_content = []
        for table in ["episodic_memory", "semantic_memory", "procedural_memory"]:
            rows = self.storage.execute(f"SELECT content FROM {table}")
            all_content.extend(r["content"] for r in rows)

        token_counts = Counter()
        for content in all_content:
            tokens = tokenize(content)
            token_counts.update(tokens)

        # Top tokens as topic clusters
        for token, count in token_counts.most_common(20):
            if token not in clusters and count >= 3:
                clusters[token] = min(1.0, count / 15.0)

        total = sum(
            self.storage.count(t) for t in
            ["episodic_memory", "semantic_memory", "procedural_memory"]
        )

        well_covered = [t for t, c in clusters.items() if c >= 0.7]
        sparse = [t for t, c in clusters.items() if c < 0.3]

        # Temporal coverage (by month)
        temporal: dict[str, int] = defaultdict(int)
        rows = self.storage.execute(
            "SELECT created_at FROM episodic_memory ORDER BY created_at"
        )
        for r in rows:
            month = time.strftime("%Y-%m", time.gmtime(r["created_at"]))
            temporal[month] += 1

        return CoverageMap(
            clusters=clusters,
            total_memories=total,
            well_covered=well_covered,
            sparse=sparse,
            temporal_coverage=dict(temporal),
        )

    # ── Self-Evaluation ─────────────────────────────────────

    def self_evaluate(self) -> SelfEvaluation:
        """Overall quality assessment of the memory system."""
        now = time.time()

        # Gather stats
        sem_rows = self.storage.fetch_all("semantic_memory")
        ep_rows = self.storage.fetch_all("episodic_memory")
        proc_rows = self.storage.fetch_all("procedural_memory")
        all_rows = sem_rows + ep_rows + proc_rows
        total = len(all_rows)

        if total == 0:
            return SelfEvaluation(
                total_memories=0,
                avg_confidence=0, avg_importance=0,
                coverage_score=0, diversity_score=0, recency_score=0,
                overall_quality=0,
                recommendations=["Memory is empty. Start storing information."],
            )

        # Average confidence (semantic only)
        confidences = [r.get("confidence", 0.5) for r in sem_rows]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Average importance
        importances = [r.get("importance", 0.5) for r in all_rows]
        avg_importance = sum(importances) / len(importances)

        # Coverage: how many categories have good coverage
        coverage_map = self.get_coverage_map()
        coverage_score = (
            len(coverage_map.well_covered) /
            max(1, len(coverage_map.clusters))
        )

        # Diversity: ratio of memory types used
        type_counts = [len(ep_rows), len(sem_rows), len(proc_rows)]
        active_types = sum(1 for c in type_counts if c > 0)
        diversity_score = active_types / 3.0

        # Recency: what fraction of memories were accessed in last 7 days
        week_ago = now - 7 * 86400
        recent = sum(1 for r in all_rows if r.get("last_accessed", 0) > week_ago)
        recency_score = recent / total if total > 0 else 0.0

        overall = (
            0.25 * avg_confidence
            + 0.20 * avg_importance
            + 0.20 * coverage_score
            + 0.15 * diversity_score
            + 0.20 * recency_score
        )

        recommendations = []
        if len(sem_rows) < 5:
            recommendations.append("Store more semantic facts to build knowledge base.")
        if len(proc_rows) == 0:
            recommendations.append("Add procedural skills for actionable knowledge.")
        if coverage_map.sparse:
            recommendations.append(
                f"Sparse topics: {', '.join(coverage_map.sparse[:5])}. Consider filling gaps."
            )
        if recency_score < 0.3:
            recommendations.append("Many memories haven't been accessed recently. Consider consolidation.")
        if avg_confidence < 0.5:
            recommendations.append("Average confidence is low. Verify and update stored facts.")

        return SelfEvaluation(
            total_memories=total,
            avg_confidence=round(avg_confidence, 3),
            avg_importance=round(avg_importance, 3),
            coverage_score=round(coverage_score, 3),
            diversity_score=round(diversity_score, 3),
            recency_score=round(recency_score, 3),
            overall_quality=round(overall, 3),
            recommendations=recommendations,
        )
