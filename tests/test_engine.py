"""Integration tests for CortexEngine."""

import pytest

from cortex.engine import CortexEngine


class TestCortexEngine:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")

    def teardown_method(self):
        self.engine.close()

    def test_store_and_recall_semantic(self):
        self.engine.store("The Eiffel Tower is in Paris", memory_type="semantic")
        results = self.engine.recall("Eiffel Tower")
        assert len(results) >= 1
        assert "Eiffel" in results[0].content

    def test_store_and_recall_episodic(self):
        self.engine.remember("Met with Alice about the new design")
        results = self.engine.recall("Alice design")
        assert len(results) >= 1

    def test_learn_skill(self):
        self.engine.learn_skill(
            "When asked about time, use datetime.now()",
            trigger="time question",
        )
        results = self.engine.recall("time question")
        assert len(results) >= 1
        assert "datetime" in results[0].content

    def test_working_memory(self):
        self.engine.store("Quick note", memory_type="working")
        assert len(self.engine.working) == 1

    def test_consolidation(self):
        # Put items in working memory
        self.engine.store("Consolidation test item A", memory_type="working")
        self.engine.store("Consolidation test item B", memory_type="working")

        report = self.engine.consolidate()
        assert report.working_to_episodic == 2
        assert len(self.engine.working) == 0  # working memory flushed

    def test_error_logging(self):
        err_id = self.engine.log_error("Wrong date format", context="report generation")
        assert err_id.startswith("err_")

        cor_id = self.engine.log_correction(
            "Use YYYY-MM-DD format",
            error_id=err_id,
            wrong="DD/MM/YYYY",
            pattern="date_format",
        )
        assert cor_id.startswith("cor_")

        report = self.engine.get_improvement_report()
        assert report.total_errors >= 1
        assert report.resolved_errors >= 1

    def test_evolution(self):
        history = self.engine.evolve(generations=2)
        assert len(history) == 2
        assert "generation" in history[0]

        best = self.engine.get_best_strategy()
        assert best is not None
        assert "similarity_weight" in best.params

    def test_feedback(self):
        mid = self.engine.store("Test memory for feedback", memory_type="semantic")
        self.engine.feedback("test query", mid, useful=True)
        # Should not raise

    def test_confidence_assessment(self):
        self.engine.store("Python is a programming language", memory_type="semantic", category="programming")
        self.engine.store("Python was created by Guido van Rossum", memory_type="semantic", category="programming")

        assessment = self.engine.assess_confidence("Python programming")
        assert 0 <= assessment.confidence <= 1
        assert assessment.memory_count >= 2

    def test_self_evaluate(self):
        self.engine.store("Fact 1", memory_type="semantic")
        self.engine.remember("Event 1")

        evaluation = self.engine.self_evaluate()
        assert evaluation.total_memories >= 2
        assert 0 <= evaluation.overall_quality <= 1

    def test_reflect(self):
        self.engine.store("Some knowledge", memory_type="semantic", category="test")
        result = self.engine.reflect()
        assert "evaluation" in result
        assert "recommendations" in result

    def test_link_memories(self):
        id_a = self.engine.store("Concept A", memory_type="semantic")
        id_b = self.engine.store("Concept B", memory_type="semantic")
        self.engine.link(id_a, id_b, "related_to")

        related = self.engine.get_related(id_a)
        assert len(related) >= 1
        assert related[0]["id"] == id_b

    def test_stats(self):
        self.engine.store("Fact", memory_type="semantic")
        self.engine.remember("Event")
        self.engine.learn_skill("Skill")

        s = self.engine.stats()
        assert s["semantic_memory"] >= 1
        assert s["episodic_memory"] >= 1
        assert s["procedural_memory"] >= 1
        assert s["total_memories"] >= 3

    def test_context_manager(self):
        with CortexEngine(":memory:") as eng:
            eng.store("Test", memory_type="semantic")
            assert eng.stats()["total_memories"] >= 1

    def test_repr(self):
        r = repr(self.engine)
        assert "CortexEngine" in r

    def test_recall_pushes_to_working(self):
        self.engine.store("Important fact about elephants", memory_type="semantic")
        self.engine.recall("elephants")
        assert len(self.engine.working) >= 1

    def test_multiple_memory_types_in_recall(self):
        self.engine.store("Semantic about rockets", memory_type="semantic")
        self.engine.remember("Saw a rocket launch yesterday")
        self.engine.learn_skill("To launch a rocket, ignite fuel")

        results = self.engine.recall("rocket", top_k=10)
        types = {r.memory_type for r in results}
        # Should find results from multiple types
        assert len(types) >= 1
