"""Iterated linear public goods game."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_sim import theory
from agent_sim.engine import ActionSpace, RunResult, Scenario, Simulation, StepOutcome


class PublicGoods(Scenario):
    """Each round agent i contributes c_i in [0, E]; payoff E - c_i + (r/n) * sum_j c_j."""

    name = "public_goods"
    default_agents = 8

    def __init__(self, endowment: float = 20.0, multiplier: float = 3.0):
        if endowment <= 0 or multiplier <= 0:
            raise ValueError("endowment and multiplier must be positive")
        self.endowment = endowment
        self.multiplier = multiplier

    def action_space(self) -> ActionSpace:
        return ActionSpace(0.0, self.endowment)

    def reference_actions(self, agent_id: int, private: dict[str, Any]) -> tuple[float, float]:
        nash = theory.public_goods_nash_contribution(self.endowment, self.multiplier, self.n_agents)
        optimum = self.endowment if self.multiplier > 1 else 0.0
        return optimum, nash

    def rules(self) -> str:
        return (
            f"Public goods game with {self.n_agents} players. Each round you receive "
            f"{self.endowment:g} tokens and choose how many to contribute to a common pool. "
            f"The pool is multiplied by {self.multiplier:g} and split equally among all "
            "players. You keep whatever you do not contribute."
        )

    def play_step(self, sim: Simulation, step: int) -> StepOutcome:
        n = self.n_agents
        acts = {i: sim.decide(i, step) for i in range(n)}
        pot = sum(acts.values())
        share = self.multiplier * pot / n
        payoffs = {i: self.endowment - acts[i] + share for i in range(n)}
        coop = {i: acts[i] / self.endowment for i in range(n)}
        return StepOutcome(
            acts,
            payoffs,
            coop,
            dict(coop),
            {"mean_contribution": pot / n, "pot": pot},
        )

    def summarize(self, result: RunResult) -> dict[str, float]:
        mean_c = float(np.mean([s["mean_contribution"] for s in result.steps]))
        nash = theory.public_goods_nash_contribution(self.endowment, self.multiplier, self.n_agents)
        payoffs = [r["payoff"] for r in result.records]
        nash_payoff = theory.public_goods_payoff(
            self.endowment, self.multiplier, self.n_agents, nash
        )
        opt_payoff = theory.public_goods_payoff(
            self.endowment, self.multiplier, self.n_agents, self.endowment
        )
        span = opt_payoff - nash_payoff
        return {
            "mean_contribution": mean_c,
            "nash_contribution": nash,
            "efficiency": float((np.mean(payoffs) - nash_payoff) / span) if span else 0.0,
        }
