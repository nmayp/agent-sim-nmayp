"""Information cascade: sequential binary guessing with private signals (BHW 1992)."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_sim import theory
from agent_sim.engine import ActionSpace, RunResult, Scenario, Simulation, StepOutcome


class InformationCascade(Scenario):
    """Each step is a fresh round: a hidden state in {0, 1} is drawn, every agent gets a
    private signal equal to the state with probability ``accuracy`` and guesses in turn,
    seeing the guesses of earlier movers it is connected to. Payoff 1 for a correct guess.
    """

    name = "cascade"
    default_agents = 10

    def __init__(self, accuracy: float = 0.7, shuffle_order: bool = False):
        if not 0.5 <= accuracy <= 1:
            raise ValueError("accuracy must be in [0.5, 1]")
        self.accuracy = accuracy
        self.shuffle_order = shuffle_order

    def action_space(self) -> ActionSpace:
        return ActionSpace(0.0, 1.0, choices=(0.0, 1.0))

    def reference_actions(self, agent_id: int, private: dict[str, Any]) -> tuple[float, float]:
        signal = private.get("signal", 0)
        guesses = private.get("observed_guesses", ())
        d = sum(1 if g == 1 else -1 for g in guesses)
        herd = signal if d == 0 else (1 if d > 0 else 0)
        return float(signal), float(herd)

    def rules(self) -> str:
        return (
            f"Sequential guessing game with {self.n_agents} players. A hidden state is 0 or 1 "
            f"with equal probability. You receive a private signal that equals the state with "
            f"probability {self.accuracy:g}. Players guess one at a time and you can see the "
            "guesses of connected players who moved before you. You earn 1 for a correct guess. "
            "Your cooperative action is to report your own signal; the herding action follows "
            "the majority of guesses you can see."
        )

    def play_step(self, sim: Simulation, step: int) -> StepOutcome:
        n = self.n_agents
        truth = int(self.rng.integers(2))
        order = list(self.rng.permutation(n)) if self.shuffle_order else list(range(n))
        guesses: dict[int, float] = {}
        signals: dict[int, int] = {}
        moved: list[int] = []
        d = 0
        cascade_dir = 0
        for i in order:
            i = int(i)
            if cascade_dir == 0 and abs(d) >= 2:
                cascade_dir = 1 if d > 0 else -1
            signals[i] = truth if self.rng.random() < self.accuracy else 1 - truth
            neigh = set(sim.neighbors[i])
            seen = [j for j in moved if j in neigh]
            private = {"signal": signals[i], "observed_guesses": [int(guesses[j]) for j in seen]}
            guesses[i] = sim.decide(i, step, private, visible={j: guesses[j] for j in seen})
            d += 1 if guesses[i] == 1 else -1
            moved.append(i)

        payoffs = {i: float(guesses[i] == truth) for i in range(n)}
        coop = {i: float(guesses[i] == signals[i]) for i in range(n)}
        seq = [guesses[i] for i in order]
        herd_pos = n
        for k in range(n - 1, -1, -1):
            if seq[k] != seq[-1]:
                break
            herd_pos = k
        truth_dir = 1 if truth == 1 else -1
        info = {
            "truth": float(truth),
            "accuracy": float(np.mean(list(payoffs.values()))),
            "cascade": float(cascade_dir != 0),
            "cascade_correct": float(cascade_dir == truth_dir),
            "cascade_wrong": float(cascade_dir == -truth_dir),
            "consensus_position": herd_pos / n,
        }
        return StepOutcome(guesses, payoffs, coop, dict(guesses), info)

    def summarize(self, result: RunResult) -> dict[str, float]:
        steps = result.steps
        right = sum(s["cascade_correct"] for s in steps)
        wrong = sum(s["cascade_wrong"] for s in steps)
        exact = theory.cascade_probabilities(self.accuracy, self.n_agents)
        out = {
            "guess_accuracy": float(np.mean([s["accuracy"] for s in steps])),
            "signal_accuracy": self.accuracy,
            "cascade_rate": float(np.mean([s["cascade"] for s in steps])),
            "cascade_correct_rate": right / len(steps),
            "cascade_wrong_rate": wrong / len(steps),
            "mean_consensus_position": float(np.mean([s["consensus_position"] for s in steps])),
            "bhw_cascade_correct": exact["correct"],
            "bhw_cascade_wrong": exact["wrong"],
        }
        out["cascade_correctness"] = right / (right + wrong) if right + wrong else float("nan")
        return out
