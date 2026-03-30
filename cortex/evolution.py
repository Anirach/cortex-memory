"""
Self-Evolution Engine — genetic algorithm for retrieval strategies.

Each "strategy" is a set of weights that control how memories are ranked
during retrieval. The engine mutates, crosses, and selects strategies based
on measured fitness (did the retrieved memory actually help?).
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from cortex.storage import Storage


@dataclass
class Strategy:
    """A retrieval strategy 'gene' — a set of tunable parameters."""
    id: str
    generation: int
    params: dict[str, float]
    fitness: float = 0.0
    evaluations: int = 0
    created_at: float = field(default_factory=time.time)
    is_active: bool = True

    # Default parameter ranges
    PARAM_RANGES: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "recency_weight": (0.0, 2.0),
        "frequency_weight": (0.0, 2.0),
        "importance_weight": (0.0, 2.0),
        "similarity_weight": (0.0, 3.0),
        "decay_rate": (0.001, 0.1),
        "consolidation_threshold": (0.3, 0.9),
        "cross_memory_weight": (0.0, 1.5),
    }, repr=False)

    @staticmethod
    def random_params() -> dict[str, float]:
        """Generate random strategy parameters."""
        ranges = Strategy.PARAM_RANGES.__func__(None)  # type: ignore
        return {
            k: random.uniform(lo, hi)
            for k, (lo, hi) in ranges.items()
        }

    @staticmethod
    def default_params() -> dict[str, float]:
        """Sensible default strategy."""
        return {
            "recency_weight": 0.8,
            "frequency_weight": 0.5,
            "importance_weight": 1.0,
            "similarity_weight": 1.5,
            "decay_rate": 0.01,
            "consolidation_threshold": 0.6,
            "cross_memory_weight": 0.3,
        }


# Fix the PARAM_RANGES default for random_params
_PARAM_RANGES = {
    "recency_weight": (0.0, 2.0),
    "frequency_weight": (0.0, 2.0),
    "importance_weight": (0.0, 2.0),
    "similarity_weight": (0.0, 3.0),
    "decay_rate": (0.001, 0.1),
    "consolidation_threshold": (0.3, 0.9),
    "cross_memory_weight": (0.0, 1.5),
}


class EvolutionEngine:
    """Genetic algorithm that evolves memory retrieval strategies."""

    POPULATION_SIZE = 10
    MUTATION_RATE = 0.2
    MUTATION_STRENGTH = 0.15
    CROSSOVER_RATE = 0.6
    ELITE_COUNT = 2  # top N strategies survive unchanged

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self._ensure_population()

    def _ensure_population(self) -> None:
        """Seed initial population if empty."""
        count = self.storage.count("strategies", "is_active = 1")
        if count == 0:
            # Create initial population with one default + random strategies
            self._create_strategy(Strategy.default_params(), generation=0)
            for _ in range(self.POPULATION_SIZE - 1):
                params = {
                    k: random.uniform(lo, hi)
                    for k, (lo, hi) in _PARAM_RANGES.items()
                }
                self._create_strategy(params, generation=0)

    def _create_strategy(self, params: dict[str, float], generation: int) -> Strategy:
        strat = Strategy(
            id=f"strat_{uuid.uuid4().hex[:8]}",
            generation=generation,
            params=params,
        )
        self.storage.insert("strategies", {
            "id": strat.id,
            "generation": strat.generation,
            "params": json.dumps(strat.params),
            "fitness": strat.fitness,
            "evaluations": strat.evaluations,
            "created_at": strat.created_at,
            "is_active": 1,
        })
        return strat

    # ── Core Evolution ──────────────────────────────────────

    def evolve(self, generations: int = 1) -> list[dict[str, Any]]:
        """
        Run N generations of evolution. Returns history of each generation.
        """
        history = []
        for _ in range(generations):
            gen_result = self._evolve_one_generation()
            history.append(gen_result)
        return history

    def _evolve_one_generation(self) -> dict[str, Any]:
        population = self._get_active_population()
        if not population:
            self._ensure_population()
            population = self._get_active_population()

        # Sort by fitness (descending)
        population.sort(key=lambda s: s.fitness, reverse=True)
        current_gen = max(s.generation for s in population) if population else 0
        new_gen = current_gen + 1

        best_fitness = population[0].fitness if population else 0
        avg_fitness = sum(s.fitness for s in population) / len(population) if population else 0

        # Elitism: keep top strategies
        elites = population[:self.ELITE_COUNT]
        new_population: list[dict[str, float]] = [s.params for s in elites]

        # Fill rest with crossover + mutation
        while len(new_population) < self.POPULATION_SIZE:
            if random.random() < self.CROSSOVER_RATE and len(population) >= 2:
                # Tournament selection
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)
                child_params = self._crossover(parent_a.params, parent_b.params)
            else:
                parent = self._tournament_select(population)
                child_params = dict(parent.params)

            # Mutation
            if random.random() < self.MUTATION_RATE:
                child_params = self._mutate(child_params)

            new_population.append(child_params)

        # Deactivate old population
        for s in population:
            self.storage.execute(
                "UPDATE strategies SET is_active = 0 WHERE id = ?", (s.id,)
            )

        # Create new population
        for params in new_population:
            self._create_strategy(params, new_gen)

        # Log history
        gen_result = {
            "generation": new_gen,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "best_params": population[0].params if population else {},
            "population_size": len(new_population),
        }
        self.storage.insert("evolution_history", {
            "generation": new_gen,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "best_params": json.dumps(gen_result["best_params"]),
            "population_size": len(new_population),
            "timestamp": time.time(),
        })

        return gen_result

    def _tournament_select(self, population: list[Strategy], k: int = 3) -> Strategy:
        """Select best from k random individuals."""
        contestants = random.sample(population, min(k, len(population)))
        return max(contestants, key=lambda s: s.fitness)

    def _crossover(self, a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
        """Uniform crossover between two parameter sets."""
        child = {}
        for key in a:
            child[key] = a[key] if random.random() < 0.5 else b.get(key, a[key])
        return child

    def _mutate(self, params: dict[str, float]) -> dict[str, float]:
        """Randomly adjust one or more parameters."""
        mutated = dict(params)
        for key in mutated:
            if random.random() < 0.4:  # mutate each gene with 40% chance
                lo, hi = _PARAM_RANGES.get(key, (0.0, 2.0))
                delta = random.gauss(0, self.MUTATION_STRENGTH * (hi - lo))
                mutated[key] = max(lo, min(hi, mutated[key] + delta))
        return mutated

    # ── Fitness & Feedback ──────────────────────────────────

    def record_feedback(
        self,
        query: str,
        result_id: str,
        useful: bool,
        strategy_id: str | None = None,
    ) -> None:
        """Record whether a retrieval result was useful."""
        if strategy_id is None:
            best = self.get_best_strategy()
            strategy_id = best.id if best else ""

        self.storage.insert("retrieval_feedback", {
            "query": query,
            "result_id": result_id,
            "strategy_id": strategy_id,
            "useful": 1 if useful else 0,
            "timestamp": time.time(),
        })

        # Update strategy fitness
        if strategy_id:
            self._update_fitness(strategy_id)

    def _update_fitness(self, strategy_id: str) -> None:
        """Recalculate fitness from feedback."""
        feedbacks = self.storage.fetch_all(
            "retrieval_feedback", "strategy_id = ?", (strategy_id,)
        )
        if not feedbacks:
            return

        useful_count = sum(1 for f in feedbacks if f["useful"])
        total = len(feedbacks)
        fitness = useful_count / total if total > 0 else 0.0

        self.storage.execute(
            "UPDATE strategies SET fitness = ?, evaluations = ? WHERE id = ?",
            (fitness, total, strategy_id),
        )

    # ── Queries ─────────────────────────────────────────────

    def get_best_strategy(self) -> Strategy | None:
        """Return the highest-fitness active strategy."""
        rows = self.storage.fetch_all(
            "strategies", "is_active = 1", ()
        )
        if not rows:
            return None
        rows.sort(key=lambda r: r["fitness"], reverse=True)
        return self._from_row(rows[0])

    def get_active_params(self) -> dict[str, float]:
        """Get the parameters of the best active strategy."""
        best = self.get_best_strategy()
        return best.params if best else Strategy.default_params()

    def get_evolution_history(self) -> list[dict[str, Any]]:
        """Return the full evolution history."""
        rows = self.storage.fetch_all("evolution_history")
        result = []
        for r in rows:
            result.append({
                "generation": r["generation"],
                "best_fitness": r["best_fitness"],
                "avg_fitness": r["avg_fitness"],
                "best_params": json.loads(r.get("best_params", "{}")),
                "population_size": r.get("population_size", 0),
                "timestamp": r.get("timestamp", 0),
            })
        return result

    def get_current_generation(self) -> int:
        rows = self.storage.execute(
            "SELECT MAX(generation) as gen FROM strategies WHERE is_active = 1"
        )
        return rows[0]["gen"] if rows and rows[0]["gen"] is not None else 0

    # ── Internal ────────────────────────────────────────────

    def _get_active_population(self) -> list[Strategy]:
        rows = self.storage.fetch_all("strategies", "is_active = 1")
        return [self._from_row(r) for r in rows]

    def _from_row(self, row: dict) -> Strategy:
        return Strategy(
            id=row["id"],
            generation=row["generation"],
            params=json.loads(row["params"]),
            fitness=row.get("fitness", 0.0),
            evaluations=row.get("evaluations", 0),
            created_at=row.get("created_at", 0),
            is_active=bool(row.get("is_active", True)),
        )
