"""LLM provider protocol, a deterministic offline mock and an optional Anthropic backend."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np


class Provider(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


_NUM = r"(-?\d+(?:\.\d+)?)"


def _find(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


class MockProvider:
    """Deterministic stand-in for an LLM.

    The reply is a pure function of (seed, persona, prompt): it reads the
    benchmark lines the LLM policy writes into its prompt and answers in the
    same ``ACTION:``/``MESSAGE:`` format a real model is asked to use, so the
    full prompt -> text -> parse path is exercised offline.

    Personas: ``reciprocal`` (conditional cooperator), ``cooperative``,
    ``selfish``, ``conformist`` (copies the neighbour average action).
    """

    PERSONAS = ("reciprocal", "cooperative", "selfish", "conformist")

    def __init__(self, persona: str = "reciprocal", noise: float = 0.1, seed: int = 0):
        if persona not in self.PERSONAS:
            raise ValueError(f"unknown persona {persona!r}")
        self.persona = persona
        self.noise = noise
        self.seed = seed
        self.name = f"mock-{persona}"
        self.calls = 0

    def _rng(self, prompt: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{self.seed}|{self.persona}|{prompt}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        rng = self._rng(prompt)
        rng_range = re.search(rf"Allowed range: \[{_NUM}, {_NUM}\]", prompt)
        lo, hi = (float(rng_range.group(1)), float(rng_range.group(2))) if rng_range else (0, 1)
        coop = _find(rf"Collective-optimum action: {_NUM}", prompt)
        nash = _find(rf"Individually-rational action: {_NUM}", prompt)
        neigh_coop = _find(rf"Neighbours' mean cooperation last round: {_NUM}", prompt)
        neigh_act = _find(rf"Neighbours' mean action: {_NUM}", prompt)
        coop = (lo + hi) / 2 if coop is None else coop
        nash = lo if nash is None else nash

        if rng.random() < self.noise:
            action, intent = float(rng.uniform(lo, hi)), "I am experimenting this round."
        elif self.persona == "cooperative":
            action, intent = coop, "I will keep choosing the collective optimum."
        elif self.persona == "selfish":
            action, intent = nash, "I am looking after my own payoff."
        elif self.persona == "conformist":
            action = neigh_act if neigh_act is not None else coop
            intent = "I will do what the group does."
        else:
            reciprocate = neigh_coop is None or neigh_coop >= 0.5
            action = coop if reciprocate else nash
            intent = "I cooperate as long as you do." if reciprocate else "You defected, so will I."
        return f"Reasoning: persona={self.persona}.\nACTION: {action:.4f}\nMESSAGE: {intent}"


class AnthropicProvider:
    """Anthropic API via the ``anthropic`` SDK (``pip install agent-sim[anthropic]``)."""

    def __init__(self, model: str | None = None, max_tokens: int = 4000, effort: str = "low"):
        model = model or os.environ.get("AGENT_SIM_MODEL")
        if not model:
            raise RuntimeError("pass a model or set AGENT_SIM_MODEL")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'anthropic' extra to use AnthropicProvider") from exc
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.name = f"anthropic-{model}"

    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover - network
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text")


def make_provider(spec: str, seed: int = 0) -> Provider:
    """``mock``, ``mock:<persona>`` or ``anthropic[:<model>]``."""
    kind, _, arg = spec.partition(":")
    if kind == "mock":
        return MockProvider(persona=arg or "reciprocal", seed=seed)
    if kind == "anthropic":
        return AnthropicProvider(model=arg) if arg else AnthropicProvider()
    raise ValueError(f"unknown provider {spec!r}")
