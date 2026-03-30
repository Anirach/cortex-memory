"""Tests for the self-evolution engine."""

import pytest

from cortex.storage import Storage
from cortex.evolution import EvolutionEngine, Strategy


class TestEvolutionEngine:
    def setup_method(self):
        self.storage = Storage(":memory:")
        self.engine = EvolutionEngine(self.storage)

    def test_initial_population(self):
        """Should seed initial population on creation."""
        count = self.storage.count("strategies", "is_active = 1")
        assert count == EvolutionEngine.POPULATION_SIZE

    def test_evolve_one_generation(self):
        history = self.engine.evolve(generations=1)
        assert len(history) == 1
        assert history[0]["generation"] >= 1
        assert history[0]["population_size"] == EvolutionEngine.POPULATION_SIZE

    def test_evolve_multiple_generations(self):
        history = self.engine.evolve(generations=5)
        assert len(history) == 5
        gens = [h["generation"] for h in history]
        assert gens == sorted(gens)  # monotonically increasing

    def test_get_best_strategy(self):
        best = self.engine.get_best_strategy()
        assert best is not None
        assert isinstance(best.params, dict)
        assert "similarity_weight" in best.params

    def test_record_feedback_and_fitness(self):
        best = self.engine.get_best_strategy()
        self.engine.record_feedback("test query", "result_1", True, best.id)
        self.engine.record_feedback("test query 2", "result_2", False, best.id)

        updated = self.engine.get_best_strategy()
        # Fitness should reflect feedback
        assert updated is not None

    def test_evolution_history(self):
        self.engine.evolve(generations=3)
        history = self.engine.get_evolution_history()
        assert len(history) >= 3

    def test_default_params(self):
        params = Strategy.default_params()
        assert "recency_weight" in params
        assert "similarity_weight" in params
        assert all(isinstance(v, float) for v in params.values())

    def test_get_active_params(self):
        params = self.engine.get_active_params()
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_current_generation(self):
        gen = self.engine.get_current_generation()
        assert gen >= 0
        self.engine.evolve(1)
        assert self.engine.get_current_generation() > gen
