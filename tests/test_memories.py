"""Unit tests for each memory type."""

import time
import pytest

from cortex.storage import Storage
from cortex.memories.working import WorkingMemory
from cortex.memories.episodic import EpisodicMemory
from cortex.memories.semantic import SemanticMemory
from cortex.memories.procedural import ProceduralMemory


# ── Working Memory ──────────────────────────────────────────

class TestWorkingMemory:
    def test_push_and_retrieve(self):
        wm = WorkingMemory(capacity=5)
        wm.push("Hello")
        wm.push("World")
        assert len(wm) == 2
        items = wm.get_all()
        assert items[0].content == "Hello"
        assert items[1].content == "World"

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.push(f"Item {i}")
        assert len(wm) == 3
        contents = [it.content for it in wm.get_all()]
        assert contents == ["Item 2", "Item 3", "Item 4"]

    def test_get_recent(self):
        wm = WorkingMemory(capacity=10)
        for i in range(7):
            wm.push(f"Item {i}")
        recent = wm.get_recent(3)
        assert len(recent) == 3
        assert recent[0].content == "Item 4"

    def test_search(self):
        wm = WorkingMemory(capacity=10)
        wm.push("The cat sat on the mat")
        wm.push("The dog ran in the park")
        results = wm.search("cat")
        assert len(results) == 1
        assert "cat" in results[0].content

    def test_clear(self):
        wm = WorkingMemory(capacity=5)
        wm.push("A")
        wm.push("B")
        evicted = wm.clear()
        assert len(evicted) == 2
        assert len(wm) == 0


# ── Episodic Memory ────────────────────────────────────────

class TestEpisodicMemory:
    def setup_method(self):
        self.storage = Storage(":memory:")
        self.em = EpisodicMemory(self.storage)

    def test_store_and_recall(self):
        self.em.store("Had a meeting with John", importance=0.7, tags=["work"])
        results = self.em.recall("meeting John", top_k=5)
        assert len(results) >= 1
        assert "John" in results[0].content

    def test_importance_clamping(self):
        ep = self.em.store("Test", importance=5.0)
        assert ep.importance == 1.0
        ep2 = self.em.store("Test2", importance=-1.0)
        assert ep2.importance == 0.0

    def test_get_by_id(self):
        ep = self.em.store("Unique event")
        retrieved = self.em.get(ep.id)
        assert retrieved is not None
        assert retrieved.content == "Unique event"

    def test_delete(self):
        ep = self.em.store("To be deleted")
        self.em.delete(ep.id)
        assert self.em.get(ep.id) is None

    def test_count(self):
        assert self.em.count() == 0
        self.em.store("One")
        self.em.store("Two")
        assert self.em.count() == 2

    def test_prune_forgotten(self):
        # Store an episode with very low importance
        self.em.store("Ancient memory", importance=0.01)
        # Manually set last_accessed to far past
        self.storage.execute(
            "UPDATE episodic_memory SET last_accessed = ?", (time.time() - 365 * 86400,)
        )
        pruned = self.em.prune_forgotten(threshold=0.5)
        assert pruned == 1


# ── Semantic Memory ─────────────────────────────────────────

class TestSemanticMemory:
    def setup_method(self):
        self.storage = Storage(":memory:")
        self.sm = SemanticMemory(self.storage)

    def test_store_and_recall(self):
        self.sm.store("Paris is the capital of France", category="geography")
        results = self.sm.recall("capital France", top_k=5)
        assert len(results) >= 1
        assert "Paris" in results[0].content

    def test_category_filter(self):
        self.sm.store("Python is a language", category="programming")
        self.sm.store("Paris is in France", category="geography")
        results = self.sm.recall("language", top_k=5, category="programming")
        assert all(f.category == "programming" for f in results)

    def test_get_categories(self):
        self.sm.store("Fact A", category="science")
        self.sm.store("Fact B", category="history")
        cats = self.sm.get_categories()
        assert "science" in cats
        assert "history" in cats

    def test_update_confidence(self):
        fact = self.sm.store("Uncertain claim", confidence=0.5)
        self.sm.update_confidence(fact.id, 0.9)
        updated = self.sm.get(fact.id)
        assert updated.confidence == 0.9

    def test_boost_importance(self):
        fact = self.sm.store("Important fact", importance=0.5)
        self.sm.boost_importance(fact.id, 0.3)
        updated = self.sm.get(fact.id)
        assert updated.importance >= 0.7


# ── Procedural Memory ──────────────────────────────────────

class TestProceduralMemory:
    def setup_method(self):
        self.storage = Storage(":memory:")
        self.pm = ProceduralMemory(self.storage)

    def test_store_and_recall(self):
        self.pm.store(
            "When user asks for weather, use wttr.in API",
            trigger="weather request",
            pattern="api_call",
        )
        results = self.pm.recall("weather", top_k=5)
        assert len(results) >= 1
        assert "wttr" in results[0].content

    def test_record_outcome(self):
        skill = self.pm.store("Test skill")
        self.pm.record_outcome(skill.id, success=True)
        self.pm.record_outcome(skill.id, success=True)
        self.pm.record_outcome(skill.id, success=False)
        updated = self.pm.get(skill.id)
        assert updated.success_count == 2
        assert updated.failure_count == 1
        assert updated.success_rate == pytest.approx(2 / 3, abs=0.01)

    def test_reliability(self):
        skill = self.pm.store("Reliable skill")
        # Laplace smoothing with 0 observations: (0+1)/(0+2) = 0.5
        assert skill.reliability == pytest.approx(0.5, abs=0.01)
        self.pm.record_outcome(skill.id, success=True)
        updated = self.pm.get(skill.id)
        # (1+1)/(1+2) = 0.667
        assert updated.reliability == pytest.approx(2 / 3, abs=0.01)

    def test_get_by_trigger(self):
        self.pm.store("Skill A", trigger="email request")
        self.pm.store("Skill B", trigger="calendar query")
        results = self.pm.get_by_trigger("email")
        assert len(results) >= 1
        assert "email" in results[0].trigger
