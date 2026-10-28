"""Agent policies: scripted game-theoretic baselines and an LLM-backed policy.

Scripted policies are scenario-agnostic: every scenario exposes a cooperative
(collective-optimum) action and a defect (one-shot Nash) action, and reports
each agent's cooperation score in [0, 1]. Scenario-specific rules
(best response, Bayesian herding) read the ``private`` dict.
"""

from __future__ import annotations

import re

import numpy as np

from agent_sim.engine import Decision, Observation
from agent_sim.providers import Provider


class AlwaysCooperate:
    name = "always_cooperate"

    def decide(self, obs: Observation, rng: np.random.Generator) -> Decision:
        return Decision(obs.cooperate_action)


class AlwaysDefect:
    name = "always_defect"

    def decide(self, obs: Observation, rng: np.random.Generator) -> Decision:
        return Decision(obs.defect_action)


class RandomPolicy:
    name = "random"

    def decide(self, obs: Observation, rng: np.random.Generator) -> Decision:
        return Decision(obs.action_space.sample(rng))


class TitForTat:
    """N-player TFT: cooperate first, then cooperate iff neighbours mostly cooperated."""

    name = "tit_for_tat"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def decide(self, obs: Observation, rng: np.random.Generator) -> Decision:
        c = obs.neighbor_cooperation
        cooperate = c is None or c >= self.threshold
        return Decision(obs.cooperate_action if cooperate else obs.defect_action)


class GrimTrigger:
    """Cooperate until any neighbour falls below the threshold once, then defect forever."""

    name = "grim_trigger"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._triggered: set[int] = set()

    def decide(self, obs: Observation, rng: np.random.Generator) -> Decision:
        c = obs.neighbor_cooperation
        if c is not None and c < self.threshold:
            self._triggered.add(obs.agent_id)
        triggered = obs.agent_id in self._triggered
        return Decision(obs.defect_action if triggered else obs.cooperate_action)


class BestResponse:
    """Myopic best response to last round, as computed by the scenario."""

    name = "best_response"

    def decide(self, obs: Observation, rng: np.random.Generator) -> Decision:
        return Decision(obs.private.get("best_response", obs.defect_action))


class BayesianHerder:
    """Bikhchandani-Hirshleifer-Welch rational guesser for the cascade scenario.

    Uses the public count rule: with d = (#guesses for 1) - (#guesses for 0)
    among observed predecessors, follow the majority when |d| >= 2; when |d| == 1
    and the private signal disagrees the agent is indifferent and breaks the tie
    by a coin flip (``tie="coin"``) or by its own signal (``tie="signal"``).
    """

    name = "bayesian"

    def __init__(self, tie: str = "coin"):
        if tie not in ("coin", "signal"):
            raise ValueError("tie must be 'coin' or 'signal'")
        self.tie = tie

    def decide(self, obs: Observation, rng: np.random.Generator) -> Decision:
        signal = obs.private.get("signal")
        if signal is None:
            return Decision(obs.defect_action)
        guesses = list(obs.private.get("observed_guesses", ()))
        d = sum(1 if g == 1 else -1 for g in guesses)
        lean = 1 if d > 0 else 0
        if abs(d) >= 2:
            return Decision(float(lean))
        if abs(d) == 1 and lean != signal and self.tie == "coin":
            return Decision(float(rng.integers(2)))
        return Decision(float(signal))


SYSTEM_PROMPT = (
    "You are one agent in a repeated multi-agent economic game. Each round you choose "
    "one numeric action. Reply with a line 'ACTION: <number>' and optionally a line "
    "'MESSAGE: <text>' that your neighbours will read next round."
)

_ACTION_RE = re.compile(r"ACTION:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_MESSAGE_RE = re.compile(r"MESSAGE:\s*(.+)", re.IGNORECASE)
_ANY_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class LLMPolicy:
    """Renders the observation as a prompt, calls a provider, parses the reply."""

    def __init__(
        self,
        provider: Provider,
        reveal_benchmarks: bool = True,
        fallback: str = "random",
        system: str = SYSTEM_PROMPT,
    ):
        if fallback not in ("random", "defect", "cooperate"):
            raise ValueError("fallback must be random, defect or cooperate")
        self.provider = provider
        self.reveal_benchmarks = reveal_benchmarks
        self.fallback = fallback
        self.system = system
        self.name = f"llm[{provider.name}]"

    def render(self, obs: Observation) -> str:
        space = obs.action_space
        lines = [
            obs.rules,
            "",
            f"You are agent {obs.agent_id} of {obs.n_agents}. Round {obs.step}.",
        ]
        if space.discrete:
            lines.append(f"Allowed choices: {', '.join(f'{c:g}' for c in space.choices or ())}")
        lines.append(f"Allowed range: [{space.low:g}, {space.high:g}]")
        if self.reveal_benchmarks:
            lines.append(f"Collective-optimum action: {obs.cooperate_action:g}")
            lines.append(f"Individually-rational action: {obs.defect_action:g}")
        for key, value in sorted(obs.private.items()):
            if key not in ("best_response",):
                lines.append(f"Private info - {key}: {value}")
        if obs.neighbor_cooperation is not None:
            lines.append(f"Neighbours' mean cooperation last round: {obs.neighbor_cooperation:.3f}")
        if obs.neighbor_actions:
            acts = ", ".join(f"agent {j}: {a:g}" for j, a in sorted(obs.neighbor_actions.items()))
            lines.append(f"Neighbours' visible actions: {acts}")
            mean = float(np.mean(list(obs.neighbor_actions.values())))
            lines.append(f"Neighbours' mean action: {mean:.4f}")
        if obs.history:
            hist = "; ".join(f"action {a:g} -> payoff {p:.2f}" for a, p in obs.history)
            lines.append(f"Your recent rounds: {hist}")
        for msg in obs.inbox:
            lines.append(f"Message from agent {msg.sender}: {msg.content}")
        lines.append("Choose your action now.")
        return "\n".join(lines)

    def parse(self, text: str) -> tuple[float | None, str | None]:
        m = _ACTION_RE.search(text)
        value = float(m.group(1)) if m else None
        if value is None:
            nums = _ANY_NUM_RE.findall(text)
            value = float(nums[-1]) if nums else None
        msg = _MESSAGE_RE.search(text)
        return value, (msg.group(1).strip() if msg else None)

    def decide(self, obs: Observation, rng: np.random.Generator) -> Decision:
        raw = self.provider.complete(self.system, self.render(obs))
        value, message = self.parse(raw)
        if value is not None:
            return Decision(value, message=message, raw=raw)
        if self.fallback == "defect":
            value = obs.defect_action
        elif self.fallback == "cooperate":
            value = obs.cooperate_action
        else:
            value = obs.action_space.sample(rng)
        return Decision(value, message=message, raw=raw, parsed=False)


SCRIPTED = {
    "always_cooperate": AlwaysCooperate,
    "always_defect": AlwaysDefect,
    "random": RandomPolicy,
    "tit_for_tat": TitForTat,
    "grim_trigger": GrimTrigger,
    "best_response": BestResponse,
    "bayesian": BayesianHerder,
}
