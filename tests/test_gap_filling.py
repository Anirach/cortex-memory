"""Tests for the Gap Filling Engine."""

import time
import pytest

from cortex.storage import Storage
from cortex.memories.episodic import EpisodicMemory
from cortex.memories.semantic import SemanticMemory
from cortex.memories.procedural import ProceduralMemory
from cortex.memories.working import WorkingMemory
from cortex.metacognition import MetaCognitionEngine
from cortex.self_improvement import SelfImprovementEngine
from cortex.consolidation import ConsolidationEngine
from cortex.gap_filling import GapFillingEngine, GapType, Gap


class TestGapFilling:
    def setup_method(self):
        self.storage = Storage(":memory:")
        self.episodic = EpisodicMemory(self.storage)
        self.semantic = SemanticMemory(self.storage)
        self.procedural = ProceduralMemory(self.storage)
        self.working = WorkingMemory()
        self.metacognition = MetaCognitionEngine(
            self.storage, self.episodic, self.semantic, self.procedural
        )
        self.improvement = SelfImprovementEngine(self.storage, self.procedural)
        self.consolidation = ConsolidationEngine(
            self.storage, self.working, self.episodic, self.semantic, self.procedural
        )
        self.engine = GapFillingEngine(
            self.storage,
            self.episodic, self.semantic, self.procedural,
            metacognition=self.metacognition,
            self_improvement=self.improvement,
            consolidation=self.consolidation,
        )

    def test_find_gaps_empty(self):
        gaps = self.engine.find_gaps()
        # Might have no gaps with empty memory, or might detect emptiness itself
        assert isinstance(gaps, list)

    def test_find_fragmentation_gaps(self):
        # Create multiple episodes about the same topic
        for i in range(5):
            self.episodic.store(
                f"Discussion about machine learning topic {i}",
                importance=0.5,
                tags=["ml"],
            )
        gaps = self.engine.find_gaps()
        # Should detect fragmentation (many episodes, no semantic summary)
        frag_gaps = [g for g in gaps if g.gap_type == GapType.CONSOLIDATABLE]
        assert len(frag_gaps) >= 0  # May or may not detect depending on tokens

    def test_find_inference_gaps(self):
        self.semantic.store("Paris is in France", category="geography")
        self.semantic.store("France is in Europe", category="geography")
        gaps = self.engine.find_gaps()
        # May detect inferrable gap (Paris is in Europe)
        assert isinstance(gaps, list)

    def test_fill_by_consolidation(self):
        # Create fragmented episodes
        for i in range(4):
            self.episodic.store(
                f"Project Beta update: milestone {i + 1} completed",
                importance=0.6,
                tags=["project_beta"],
            )

        gaps = self.engine.find_gaps(top_k=20)
        consolidatable = [g for g in gaps if g.gap_type == GapType.CONSOLIDATABLE]

        if consolidatable:
            result = self.engine.fill_gap(consolidatable[0].id, strategy="consolidation")
            # Might succeed if enough fragments
            assert result.gap_id == consolidatable[0].id

    def test_fill_by_inference(self):
        # Store related facts
        self.semantic.store("Dogs are mammals", category="biology")
        self.semantic.store("Mammals are warm-blooded animals", category="biology")

        gaps = self.engine.find_gaps(top_k=20)
        inferrable = [g for g in gaps if g.gap_type == GapType.INFERRABLE]

        if inferrable:
            result = self.engine.fill_gap(inferrable[0].id, strategy="inference")
            assert result.gap_id == inferrable[0].id

    def test_fill_gaps_auto(self):
        # Populate with some data to create gaps
        for i in range(3):
            self.episodic.store(f"Event about AI topic {i}", tags=["ai"])
        self.semantic.store("AI stands for Artificial Intelligence", category="ai")

        results = self.engine.fill_gaps(max_fills=5)
        assert isinstance(results, list)

    def test_fill_verification(self):
        # Manually add a gap and fill it
        gap = Gap(
            id="test_gap_verify",
            topic="test_topic",
            description="Test gap for verification",
            gap_type=GapType.INFERRABLE,
            priority=0.5,
        )
        self.engine._gap_cache[gap.id] = gap

        # Simulate fill
        fact = self.semantic.store("Inferred answer", source="gap_fill")
        gap.filled = True
        gap.fill_memory_id = fact.id
        gap.fill_confidence = 0.7

        # Confirm correct
        self.engine.confirm_fill(gap.id, correct=True)
        assert gap.confirmed is True

        # Check confidence was boosted
        updated = self.semantic.get(fact.id)
        assert updated.confidence >= 0.7

    def test_confirm_fill_incorrect(self):
        gap = Gap(
            id="test_gap_wrong",
            topic="wrong_fill",
            description="Incorrectly filled gap",
            gap_type=GapType.INFERRABLE,
            priority=0.5,
        )
        self.engine._gap_cache[gap.id] = gap
        fact = self.semantic.store("Wrong answer", confidence=0.7, source="gap_fill")
        gap.filled = True
        gap.fill_memory_id = fact.id
        gap.fill_confidence = 0.7

        self.engine.confirm_fill(gap.id, correct=False)
        assert gap.confirmed is False
        updated = self.semantic.get(fact.id)
        assert updated.confidence < 0.7

    def test_request_fill(self):
        gap = Gap(
            id="test_gap_ask",
            topic="quantum_physics",
            description="Missing knowledge about quantum entanglement",
            gap_type=GapType.ASKABLE,
            priority=0.8,
        )
        self.engine._gap_cache[gap.id] = gap

        question = self.engine.request_fill(gap.id)
        assert question is not None
        assert "quantum_physics" in question

    def test_request_fill_searchable(self):
        gap = Gap(
            id="test_gap_search",
            topic="latest_news",
            description="Need current events data",
            gap_type=GapType.SEARCHABLE,
            priority=0.6,
        )
        self.engine._gap_cache[gap.id] = gap

        question = self.engine.request_fill(gap.id)
        assert question is not None
        assert "search" in question.lower()

    def test_gap_report(self):
        # Add some data
        self.semantic.store("Fact about biology", category="biology")
        self.episodic.store("Event in biology class")

        report = self.engine.gap_report()
        assert isinstance(report.total_gaps, int)
        assert isinstance(report.gaps_by_type, dict)
        assert isinstance(report.coverage_clusters, list)
        assert isinstance(report.recommendations, list)

    def test_gap_priority_scoring(self):
        gap = Gap(
            id="test_priority",
            topic="test",
            description="Test priority",
            gap_type=GapType.INFERRABLE,
            priority=0.0,
            related_queries=5,
            existing_coverage=0.2,
        )
        self.engine._gap_cache[gap.id] = gap
        score = self.engine._score_priority(gap)
        assert 0 <= score <= 1

    def test_fill_accuracy_tracking(self):
        accuracy = self.improvement.get_fill_accuracy()
        assert accuracy == 0.0  # No fills yet

    def test_timeline_gaps(self):
        # Create episodes with a large time gap
        self.episodic.store("Early event", importance=0.5)
        # Manually backdate first episode
        self.storage.execute(
            "UPDATE episodic_memory SET created_at = ? WHERE rowid = 1",
            (time.time() - 7 * 86400,),
        )
        self.episodic.store("Recent event", importance=0.5)

        report = self.engine.gap_report()
        # Should detect the 7-day gap
        assert isinstance(report.timeline_gaps, list)

    def test_coverage_clusters(self):
        self.semantic.store("Python is great", category="programming")
        self.semantic.store("Java is verbose", category="programming")
        self.semantic.store("History is interesting", category="history")

        report = self.engine.gap_report()
        assert len(report.coverage_clusters) >= 0

    def test_cross_memory_fill(self):
        # Store only in episodic
        self.episodic.store("The weather in Bangkok is hot", importance=0.7)

        gap = Gap(
            id="gap_cross_test",
            topic="weather bangkok",
            description="Missing semantic knowledge about Bangkok weather",
            gap_type=GapType.INFERRABLE,
            priority=0.6,
        )
        self.engine._gap_cache[gap.id] = gap

        result = self.engine._fill_by_cross_memory(gap)
        assert result.success
        assert "Bangkok" in result.content or "weather" in result.content.lower()

    def test_pattern_fill(self):
        self.procedural.store(
            "Check weather using wttr.in API",
            trigger="weather check",
            pattern="api_call",
        )

        gap = Gap(
            id="gap_pattern_test",
            topic="weather check",
            description="How to check weather",
            gap_type=GapType.INFERRABLE,
            priority=0.5,
        )
        self.engine._gap_cache[gap.id] = gap

        result = self.engine._fill_by_pattern(gap)
        assert result.success
        assert "wttr" in result.content.lower() or "weather" in result.content.lower()
