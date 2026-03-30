"""Tests for the consolidation engine."""

import time
import pytest

from cortex.storage import Storage
from cortex.memories.working import WorkingMemory
from cortex.memories.episodic import EpisodicMemory
from cortex.memories.semantic import SemanticMemory
from cortex.memories.procedural import ProceduralMemory
from cortex.consolidation import ConsolidationEngine


class TestConsolidation:
    def setup_method(self):
        self.storage = Storage(":memory:")
        self.working = WorkingMemory(capacity=20)
        self.episodic = EpisodicMemory(self.storage)
        self.semantic = SemanticMemory(self.storage)
        self.procedural = ProceduralMemory(self.storage)
        self.engine = ConsolidationEngine(
            self.storage, self.working, self.episodic, self.semantic, self.procedural
        )

    def test_flush_working_to_episodic(self):
        self.working.push("Item 1")
        self.working.push("Item 2")
        report = self.engine.consolidate()
        assert report.working_to_episodic == 2
        assert len(self.working) == 0
        assert self.episodic.count() >= 2

    def test_promote_to_semantic(self):
        # Store an episode with high importance (above threshold)
        ep = self.episodic.store("Very important fact about quantum computing", importance=0.85)
        report = self.engine.consolidate()
        assert report.episodic_to_semantic >= 1
        # Should now exist in semantic memory
        sem_results = self.semantic.recall("quantum computing", top_k=3)
        assert len(sem_results) >= 1

    def test_promote_frequent_access(self):
        ep = self.episodic.store("Frequently accessed info", importance=0.3)
        # Simulate multiple accesses
        for _ in range(5):
            self.episodic.get(ep.id)
        report = self.engine.consolidate()
        assert report.episodic_to_semantic >= 1

    def test_promote_to_procedural(self):
        self.episodic.store(
            "When the user asks for a summary, then use the summarize function first",
            importance=0.6,
        )
        report = self.engine.consolidate()
        assert report.episodic_to_procedural >= 1
        assert self.procedural.count() >= 1

    def test_prune_old_memories(self):
        self.episodic.store("Old forgotten memory", importance=0.01)
        self.storage.execute(
            "UPDATE episodic_memory SET last_accessed = ?",
            (time.time() - 365 * 86400,),
        )
        report = self.engine.consolidate()
        assert report.pruned_episodic >= 1

    def test_consolidation_report(self):
        self.working.push("Test")
        report = self.engine.consolidate()
        assert report.duration_ms >= 0
        assert report.timestamp > 0

    def test_consolidate_fragments(self):
        # Store multiple related episodes
        for i in range(5):
            self.episodic.store(
                f"Meeting about Project Alpha - discussion point {i + 1}",
                importance=0.5,
                tags=["project_alpha"],
            )

        result = self.engine.consolidate_fragments("Project Alpha")
        assert result is not None
        assert "Project Alpha" in result or "project" in result.lower()
        assert self.semantic.count() >= 1
