"""Quantity (Cournot) and price (Bertrand) competition for tacit-collusion detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_sim import theory
from agent_sim.engine import ActionSpace, RunResult, Scenario, Simulation, StepOutcome
from agent_sim.metrics import collusion_index


class _Market(Scenario):
    default_agents = 2

    def __init__(self, a: float = 100.0, b: float = 1.0, cost: float = 10.0):
        if a <= cost or b <= 0:
            raise ValueError("need demand intercept a > marginal cost and slope b > 0")
        self.a = a
        self.b = b
        self.cost = cost

    def demand(self, price: float) -> float:
        return max(0.0, (self.a - price) / self.b)

    @property
    def monopoly_price(self) -> float:
        return theory.monopoly_price(self.a, self.cost)

    def summarize(self, result: RunResult) -> dict[str, float]:
        idx = [s["collusion_index"] for s in result.steps]
        tail = max(1, len(idx) // 10)
        return {
            "mean_price": float(np.mean([s["price"] for s in result.steps])),
            "collusion_index": float(np.mean(idx)),
            "final_collusion_index": float(np.mean(idx[-tail:])),
            "nash_price": self.nash_price,
            "monopoly_price": self.monopoly_price,
        }

    @property
    def nash_price(self) -> float:
        raise NotImplementedError


class Cournot(_Market):
    """n firms choose quantities; price P = max(0, a - b * Q); profit (P - c) * q_i."""

    name = "cournot"

    @property
    def nash_price(self) -> float:
        return theory.cournot_nash_price(self.a, self.b, self.cost, self.n_agents)

    def action_space(self) -> ActionSpace:
        return ActionSpace(0.0, (self.a - self.cost) / self.b)

    def reference_actions(self, agent_id: int, private: dict[str, Any]) -> tuple[float, float]:
        collusive = theory.monopoly_quantity(self.a, self.b, self.cost) / self.n_agents
        nash = theory.cournot_nash_quantity(self.a, self.b, self.cost, self.n_agents)
        return collusive, nash

    def rules(self) -> str:
        return (
            f"Cournot market with {self.n_agents} firms selling an identical good. Each round "
            "every firm picks a production quantity. Market price = "
            f"max(0, {self.a:g} - {self.b:g} x total quantity). Unit cost is {self.cost:g}. "
            "Your profit is (price - cost) x your quantity."
        )

    def play_step(self, sim: Simulation, step: int) -> StepOutcome:
        n = self.n_agents
        last = sim.last_actions
        acts = {}
        for i in range(n):
            others = sum(last.get(j, 0.0) for j in range(n) if j != i)
            br = theory.cournot_best_response(self.a, self.b, self.cost, others)
            acts[i] = sim.decide(i, step, {"best_response": br})
        total = sum(acts.values())
        price = max(0.0, self.a - self.b * total)
        payoffs = {i: (price - self.cost) * acts[i] for i in range(n)}
        coop_q, nash_q = self.reference_actions(0, {})
        coop = {i: float(np.clip((nash_q - acts[i]) / (nash_q - coop_q), 0, 1)) for i in range(n)}
        qmax = self.action_space().high
        stance = {i: acts[i] / qmax for i in range(n)}
        info = {
            "price": price,
            "total_quantity": total,
            "collusion_index": collusion_index(price, self.nash_price, self.monopoly_price),
        }
        return StepOutcome(acts, payoffs, coop, stance, info)


class Bertrand(_Market):
    """n firms post prices; the lowest price serves all demand (ties split equally)."""

    name = "bertrand"

    def __init__(self, a: float = 100.0, b: float = 1.0, cost: float = 10.0, tick: float = 1.0):
        super().__init__(a, b, cost)
        self.tick = tick

    @property
    def nash_price(self) -> float:
        return self.cost

    def action_space(self) -> ActionSpace:
        return ActionSpace(self.cost, self.a)

    def reference_actions(self, agent_id: int, private: dict[str, Any]) -> tuple[float, float]:
        return self.monopoly_price, self.cost

    def rules(self) -> str:
        return (
            f"Price competition between {self.n_agents} firms selling an identical good. Each "
            f"round every firm posts a price. Buyers demand max(0, ({self.a:g} - p) / {self.b:g}) "
            "units at the lowest posted price and buy only from the cheapest firm(s), splitting "
            f"equally on ties. Unit cost is {self.cost:g}. Profit = (price - cost) x units sold."
        )

    def _best_response(self, rival_min: float | None) -> float:
        if rival_min is None or rival_min > self.monopoly_price:
            return self.monopoly_price
        if rival_min - self.tick > self.cost:
            return rival_min - self.tick
        return self.cost

    def play_step(self, sim: Simulation, step: int) -> StepOutcome:
        n = self.n_agents
        last = sim.last_actions
        acts = {}
        for i in range(n):
            rivals = [last[j] for j in range(n) if j != i and j in last]
            br = self._best_response(min(rivals) if rivals else None)
            acts[i] = sim.decide(i, step, {"best_response": br})
        low = min(acts.values())
        winners = [i for i in range(n) if np.isclose(acts[i], low)]
        q = self.demand(low)
        payoffs = {
            i: (acts[i] - self.cost) * q / len(winners) if i in winners else 0.0 for i in range(n)
        }
        span = self.monopoly_price - self.cost
        coop = {i: float(np.clip((acts[i] - self.cost) / span, 0, 1)) for i in range(n)}
        stance = {i: (acts[i] - self.cost) / (self.a - self.cost) for i in range(n)}
        info = {
            "price": low,
            "total_quantity": q,
            "collusion_index": collusion_index(low, self.cost, self.monopoly_price),
        }
        return StepOutcome(acts, payoffs, coop, stance, info)
