"""
CortexEngine — the main orchestrator for the CORTEX cognitive memory system.

Coordinates all 7 layers:
1. Working Memory (ring buffer)
2. Episodic Memory (timestamped events)
3. Semantic Memory (facts + knowledge)
4. Procedural Memory (skills + patterns)
5. Self-Improvement Engine (error tracking + learning)
6. Self-Evolution Engine (genetic algorithm for retrieval)
7. Meta-Cognition Layer (confidence + knowledge gaps)

Plus: Consolidation, Forgetting, Hippocampal Index, Gap Filling, Obsidian.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cortex.storage import Storage
from cortex.memories.working import WorkingMemory
from cortex.memories.episodic import EpisodicMemory
from cortex.memories.semantic import SemanticMemory
from cortex.memories.procedural import ProceduralMemory
from cortex.consolidation import ConsolidationEngine, ConsolidationReport
from cortex.self_improvement import SelfImprovementEngine, ImprovementReport
from cortex.evolution import EvolutionEngine, Strategy
from cortex.metacognition import MetaCognitionEngine, ConfidenceAssessment, SelfEvaluation
from cortex.hippocampus import HippocampalIndex, SearchResult
from cortex.gap_filling import GapFillingEngine, Gap, FillResult, GapReport
from cortex.embeddings import EmbeddingBackend, EmbeddingConfig, configure as configure_embeddings, get_backend_info
from cortex.obsidian import ObsidianIntegration


class CortexEngine:
    """
    Main entry point for the CORTEX cognitive memory architecture.

    Usage::

        engine = CortexEngine("cortex.db")
        engine.store("Paris is the capital of France", memory_type="semantic")
        engine.remember("Had a meeting about Project X")
        results = engine.recall("What do I know about France?")
        engine.consolidate()
        engine.evolve()
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        working_capacity: int = 20,
        obsidian_vault: str | None = None,
        obsidian_sync: bool = False,
        obsidian_para: bool = True,
        obsidian_export_inferred: bool = True,
        embedding_backend: EmbeddingBackend | str | None = None,
    ) -> None:
        # Configure embedding backend
        if embedding_backend is not None:
            if isinstance(embedding_backend, str):
                embedding_backend = EmbeddingBackend(embedding_backend)
            self._embedding_config = configure_embeddings(
                EmbeddingConfig(backend=embedding_backend)
            )
        else:
            self._embedding_config = None  # use defaults (auto-detect on demand)

        # Storage
        self._storage = Storage(db_path)

        # Layer 1: Working Memory
        self.working = WorkingMemory(capacity=working_capacity)

        # Layer 2-4: Long-term memories
        self.episodic = EpisodicMemory(self._storage)
        self.semantic = SemanticMemory(self._storage)
        self.procedural = ProceduralMemory(self._storage)

        # Hippocampal Index (hybrid search)
        self.hippocampus = HippocampalIndex(self._storage)

        # Consolidation engine
        self.consolidation = ConsolidationEngine(
            self._storage, self.working, self.episodic, self.semantic, self.procedural
        )

        # Layer 5: Self-Improvement
        self.improvement = SelfImprovementEngine(self._storage, self.procedural)

        # Layer 6: Self-Evolution
        self.evolution = EvolutionEngine(self._storage)

        # Layer 7: Meta-Cognition
        self.metacognition = MetaCognitionEngine(
            self._storage, self.episodic, self.semantic, self.procedural
        )

        # Gap Filling Engine
        self.gap_filling = GapFillingEngine(
            self._storage,
            self.episodic, self.semantic, self.procedural,
            metacognition=self.metacognition,
            self_improvement=self.improvement,
            consolidation=self.consolidation,
        )

        # Obsidian Integration
        self._obsidian_vault = obsidian_vault
        self._obsidian_sync_on_consolidate = obsidian_sync
        self.obsidian = ObsidianIntegration(
            self._storage,
            self.episodic, self.semantic, self.procedural,
            self.hippocampus,
            gap_filling=self.gap_filling,
            para_enabled=obsidian_para,
            export_inferred=obsidian_export_inferred,
        )

    # ══════════════════════════════════════════════════════════
    #  CORE API: Store / Remember / Recall
    # ══════════════════════════════════════════════════════════

    def store(
        self,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Store information into a specific memory layer.

        Parameters
        ----------
        content : str
            The information to store.
        memory_type : str
            One of 'working', 'episodic', 'semantic', 'procedural'.
        importance : float
            How important this memory is [0, 1].
        tags : list[str] | None
            Tags for categorisation.
        metadata : dict | None
            Additional metadata.

        Returns
        -------
        str
            Memory ID (or slot number for working memory).
        """
        if memory_type == "working":
            item = self.working.push(content, metadata)
            return str(item.slot)

        elif memory_type == "episodic":
            ep = self.episodic.store(content, importance, tags, metadata,
                                     source=kwargs.get("source", "user"))
            return ep.id

        elif memory_type == "semantic":
            fact = self.semantic.store(
                content, importance,
                category=kwargs.get("category", "general"),
                confidence=kwargs.get("confidence", 1.0),
                tags=tags, metadata=metadata,
                source=kwargs.get("source", "user"),
            )
            return fact.id

        elif memory_type == "procedural":
            skill = self.procedural.store(
                content, importance=importance,
                pattern=kwargs.get("pattern", ""),
                trigger=kwargs.get("trigger", ""),
                tags=tags, metadata=metadata,
            )
            return skill.id

        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")

    def remember(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Alias for store() — reads more naturally for episodic events."""
        return self.store(content, memory_type, importance, tags, metadata)

    def learn_skill(
        self,
        content: str,
        pattern: str = "",
        trigger: str = "",
        importance: float = 0.6,
        tags: list[str] | None = None,
    ) -> str:
        """Store a procedural skill."""
        return self.store(
            content,
            memory_type="procedural",
            importance=importance,
            tags=tags,
            pattern=pattern,
            trigger=trigger,
        )

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Recall memories relevant to a query using the hippocampal index.

        Uses the best evolved retrieval strategy for scoring.
        """
        # Get evolved weights
        strategy_params = self.evolution.get_active_params()
        weights = {
            "similarity": strategy_params.get("similarity_weight", 1.5),
            "bm25": 1.0,
            "temporal": strategy_params.get("recency_weight", 0.3),
            "graph": strategy_params.get("cross_memory_weight", 0.5),
            "importance": strategy_params.get("importance_weight", 0.8),
        }

        results = self.hippocampus.search(query, top_k, memory_types, weights)

        # Push to working memory for immediate context
        for r in results[:3]:
            self.working.push(r.content, {"recalled_from": r.memory_type, "query": query})

        return results

    # ══════════════════════════════════════════════════════════
    #  CONSOLIDATION
    # ══════════════════════════════════════════════════════════

    def consolidate(self) -> ConsolidationReport:
        """
        Run the sleep/dream consolidation cycle.

        Moves working→episodic, episodic→semantic, detects procedural patterns,
        and prunes forgotten memories.
        """
        report = self.consolidation.consolidate()

        # Invalidate search cache after consolidation
        self.hippocampus.invalidate_cache()

        # Obsidian sync if enabled
        if self._obsidian_sync_on_consolidate and self._obsidian_vault:
            self.obsidian.sync(self._obsidian_vault)

        return report

    # ══════════════════════════════════════════════════════════
    #  SELF-IMPROVEMENT
    # ══════════════════════════════════════════════════════════

    def log_error(
        self,
        description: str,
        context: str = "",
        category: str = "unknown",
        severity: float = 0.5,
    ) -> str:
        """Log an error for self-improvement tracking."""
        error = self.improvement.log_error(description, context, category, severity)
        return error.id

    def log_correction(
        self,
        correct: str,
        error_id: str = "",
        wrong: str = "",
        pattern: str = "",
    ) -> str:
        """Log a correction pattern."""
        correction = self.improvement.log_correction(correct, error_id, wrong, pattern)
        return correction.id

    def get_improvement_report(self) -> ImprovementReport:
        """Get comprehensive self-improvement metrics."""
        return self.improvement.get_improvement_report()

    # ══════════════════════════════════════════════════════════
    #  EVOLUTION
    # ══════════════════════════════════════════════════════════

    def evolve(self, generations: int = 1) -> list[dict[str, Any]]:
        """Run N generations of retrieval strategy evolution."""
        return self.evolution.evolve(generations)

    def feedback(self, query: str, result_id: str, useful: bool) -> None:
        """Provide feedback on a retrieval result (for evolution fitness)."""
        self.evolution.record_feedback(query, result_id, useful)

    def get_best_strategy(self) -> Strategy | None:
        """Get the current best retrieval strategy."""
        return self.evolution.get_best_strategy()

    def get_evolution_history(self) -> list[dict[str, Any]]:
        """Get the full history of strategy evolution."""
        return self.evolution.get_evolution_history()

    # ══════════════════════════════════════════════════════════
    #  META-COGNITION
    # ══════════════════════════════════════════════════════════

    def assess_confidence(self, topic: str) -> ConfidenceAssessment:
        """Assess confidence about a topic."""
        return self.metacognition.assess_confidence(topic)

    def find_knowledge_gaps(self, top_k: int = 10) -> list[Any]:
        """Find knowledge gaps (delegates to metacognition)."""
        return self.metacognition.find_knowledge_gaps(top_k)

    def self_evaluate(self) -> SelfEvaluation:
        """Overall assessment of memory quality."""
        return self.metacognition.self_evaluate()

    def reflect(self) -> dict[str, Any]:
        """
        Run a full meta-cognition reflection: evaluate, find gaps, recommend.
        """
        evaluation = self.self_evaluate()
        gaps = self.find_knowledge_gaps(top_k=5)
        confidence_samples = []

        # Sample confidence across top topics
        categories = self.semantic.get_categories()
        for cat in categories[:5]:
            assessment = self.assess_confidence(cat)
            confidence_samples.append({
                "topic": cat,
                "confidence": assessment.confidence,
                "memory_count": assessment.memory_count,
            })

        return {
            "evaluation": evaluation,
            "knowledge_gaps": gaps,
            "confidence_samples": confidence_samples,
            "recommendations": evaluation.recommendations,
        }

    # ══════════════════════════════════════════════════════════
    #  GAP FILLING
    # ══════════════════════════════════════════════════════════

    def find_gaps(self, top_k: int = 20) -> list[Gap]:
        """Find and prioritise knowledge gaps."""
        return self.gap_filling.find_gaps(top_k)

    def fill_gaps(self, max_fills: int = 10) -> list[FillResult]:
        """Auto-fill the highest-priority fillable gaps."""
        return self.gap_filling.fill_gaps(max_fills)

    def fill_gap(self, gap_id: str, strategy: str = "auto") -> FillResult:
        """Fill a specific gap."""
        return self.gap_filling.fill_gap(gap_id, strategy)

    def gap_report(self) -> GapReport:
        """Full gap analysis with coverage map."""
        return self.gap_filling.gap_report()

    def request_fill(self, gap_id: str) -> str | None:
        """Get a question to ask the user to fill a gap."""
        return self.gap_filling.request_fill(gap_id)

    # ══════════════════════════════════════════════════════════
    #  RELATIONSHIPS
    # ══════════════════════════════════════════════════════════

    def link(self, source_id: str, target_id: str, relation: str, weight: float = 1.0) -> None:
        """Create a relationship between two memories."""
        self.hippocampus.add_edge(source_id, target_id, relation, weight)

    def get_related(self, memory_id: str, max_depth: int = 2) -> list[dict[str, Any]]:
        """Get memories related to a given memory via graph traversal."""
        return self.hippocampus.get_neighbors(memory_id, max_depth)

    # ══════════════════════════════════════════════════════════
    #  STATS & INFO
    # ══════════════════════════════════════════════════════════

    def embedding_info(self) -> dict[str, Any]:
        """Get information about the active embedding backend."""
        return get_backend_info()

    def stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return {
            "working_memory": {
                "capacity": self.working.capacity,
                "used": len(self.working),
            },
            "episodic_memory": self.episodic.count(),
            "semantic_memory": self.semantic.count(),
            "procedural_memory": self.procedural.count(),
            "total_memories": (
                self.episodic.count() + self.semantic.count() + self.procedural.count()
            ),
            "evolution_generation": self.evolution.get_current_generation(),
            "edges": self._storage.count("memory_edges"),
            "errors_logged": self._storage.count("error_log"),
            "corrections": self._storage.count("corrections"),
        }

    def close(self) -> None:
        """Close the database connection."""
        self._storage.close()

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"CortexEngine(memories={s['total_memories']}, "
            f"working={s['working_memory']['used']}/{s['working_memory']['capacity']}, "
            f"gen={s['evolution_generation']})"
        )

    def __enter__(self) -> "CortexEngine":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
