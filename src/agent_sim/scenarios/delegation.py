"""Task-delegation chain: costly verification against propagating failures."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from agent_sim import theory
from agent_sim.engine import ActionSpace, RunResult, Scenario, Simulation, StepOutcome


class DelegationChain(Scenario):
    """A task passes through stages 0..n-1. Stage i may verify its input (cost ``cost``),
    catching an upstream corruption with probability ``detect``, then does its own work,
    which corrupts the task with probability ``error``. Every agent earns ``reward`` if
    the final output is clean, minus its own verification cost.
    """

    name = "delegation"
    default_agents = 6

    def __init__(
        self, error: float = 0.1, detect: float = 0.8, cost: float = 1.0, reward: float = 10.0
    ):
        if not (0 <= error <= 1 and 0 <= detect <= 1):
            raise ValueError("error and detect must be probabilities")
        self.error = error
        self.detect = detect
        self.cost = cost
        self.reward = reward

    def reset(self, n_agents: int, rng: np.random.Generator) -> None:
        super().reset(n_agents, rng)
        self._nash = self.nash_profile()
        self._optimum = self.optimal_profile()

    def welfare(self, verify: tuple[int, ...]) -> float:
        s = theory.delegation_success(self.error, self.detect, verify)
        return len(verify) * self.reward * s - self.cost * sum(verify)

    def nash_profile(self) -> tuple[int, ...]:
        """Sequential best response from all-trust. Payoffs share the potential
        R * S(v) - c * sum(v), so this converges to a pure Nash equilibrium."""
        v = [0] * self.n_agents
        for _ in range(100):
            changed = False
            for i in range(self.n_agents):
                gain = theory.delegation_verify_gain(i, self.reward, self.error, self.detect, v)
                best = int(gain > self.cost)
                if best != v[i]:
                    v[i], changed = best, True
            if not changed:
                break
        return tuple(v)

    def optimal_profile(self) -> tuple[int, ...]:
        n = self.n_agents
        if n <= 14:
            return max(
                itertools.product((0, 1), repeat=n), key=lambda v: (self.welfare(v), -sum(v))
            )
        v = [1] * n
        improved = True
        while improved:
            improved = False
            for i in range(n):
                alt = v.copy()
                alt[i] = 1 - alt[i]
                if self.welfare(tuple(alt)) > self.welfare(tuple(v)):
                    v, improved = alt, True
        return tuple(v)

    def action_space(self) -> ActionSpace:
        return ActionSpace(0.0, 1.0, choices=(0.0, 1.0))

    def reference_actions(self, agent_id: int, private: dict[str, Any]) -> tuple[float, float]:
        return float(self._optimum[agent_id]), float(self._nash[agent_id])

    def rules(self) -> str:
        return (
            f"Delegation chain of {self.n_agents} workers. A task passes through every worker "
            f"in order. Each worker's own step introduces an error with probability "
            f"{self.error:g}. Before working you may verify your input (1) at a cost of "
            f"{self.cost:g}, which fixes an upstream error with probability {self.detect:g}, "
            f"or trust it (0). Everyone earns {self.reward:g} if the final output is correct."
        )

    def play_step(self, sim: Simulation, step: int) -> StepOutcome:
        n = self.n_agents
        last = [int(sim.last_actions.get(j, 0)) for j in range(n)]
        acts: dict[int, float] = {}
        corrupted = False
        depth = caught = 0
        for i in range(n):
            gain = theory.delegation_verify_gain(i, self.reward, self.error, self.detect, last)
            private = {"stage": i, "best_response": float(gain > self.cost)}
            v = sim.decide(i, step, private)
            acts[i] = v
            if v and corrupted and self.rng.random() < self.detect:
                corrupted = False
                caught += 1
            if self.rng.random() < self.error:
                corrupted = True
            depth += int(corrupted)
        success = not corrupted
        payoffs = {i: self.reward * success - self.cost * acts[i] for i in range(n)}
        coop = {i: float(acts[i] == self._optimum[i]) for i in range(n)}
        info = {
            "success": float(success),
            "failure_depth": float(depth),
            "caught": float(caught),
            "verify_rate": float(np.mean(list(acts.values()))),
        }
        return StepOutcome(acts, payoffs, coop, dict(acts), info)

    def summarize(self, result: RunResult) -> dict[str, float]:
        steps = result.steps
        return {
            "success_rate": float(np.mean([s["success"] for s in steps])),
            "mean_failure_depth": float(np.mean([s["failure_depth"] for s in steps])),
            "verify_rate": float(np.mean([s["verify_rate"] for s in steps])),
            "nash_success": theory.delegation_success(self.error, self.detect, self._nash),
            "optimal_success": theory.delegation_success(self.error, self.detect, self._optimum),
            "trust_all_success": (1 - self.error) ** self.n_agents,
        }
