"""Closed-form game-theoretic benchmarks for the built-in scenarios (see docs/theory.md)."""

from __future__ import annotations

from collections.abc import Sequence

# Public goods -------------------------------------------------------------


def public_goods_nash_contribution(endowment: float, multiplier: float, n: int) -> float:
    """Dominant-strategy contribution: the marginal per-capita return r/n decides it."""
    return endowment if multiplier / n > 1 else 0.0


def public_goods_payoff(endowment: float, multiplier: float, n: int, contributions: float) -> float:
    """Per-agent payoff when every agent contributes the same amount."""
    return endowment - contributions + multiplier * contributions


# Cournot / Bertrand -------------------------------------------------------


def cournot_nash_quantity(a: float, b: float, c: float, n: int) -> float:
    return max(0.0, (a - c) / (b * (n + 1)))


def cournot_nash_price(a: float, b: float, c: float, n: int) -> float:
    return (a + n * c) / (n + 1)


def monopoly_quantity(a: float, b: float, c: float) -> float:
    return max(0.0, (a - c) / (2 * b))


def monopoly_price(a: float, c: float) -> float:
    return (a + c) / 2


def cournot_best_response(a: float, b: float, c: float, others_total: float) -> float:
    return max(0.0, (a - c - b * others_total) / (2 * b))


# Information cascade ------------------------------------------------------


def cascade_probabilities(p: float, n: int, tie: str = "coin") -> dict[str, float]:
    """Exact probability that a BHW cascade starts within a sequence of ``n`` agents.

    The public count difference d (truth = +) is a Markov chain on {-2..2},
    absorbing at +-2. A cascade "starts" if some agent acts at |d| = 2, i.e. the
    chain is absorbed within the first n-1 guesses. Returns probabilities of a
    correct cascade, a wrong cascade and no cascade.
    """
    if not 0.5 <= p <= 1:
        raise ValueError("signal accuracy p must be in [0.5, 1]")
    q = 1 - p
    if tie == "coin":
        up_from_plus, back_from_plus = p + q / 2, q / 2
        down_from_minus, back_from_minus = q + p / 2, p / 2
    elif tie == "signal":
        up_from_plus, back_from_plus = p, q
        down_from_minus, back_from_minus = q, p
    else:
        raise ValueError("tie must be 'coin' or 'signal'")
    dist = {-1: 0.0, 0: 1.0, 1: 0.0}
    correct = wrong = 0.0
    for _ in range(max(0, n - 1)):
        new = {-1: 0.0, 0: 0.0, 1: 0.0}
        new[1] += dist[0] * p
        new[-1] += dist[0] * q
        correct += dist[1] * up_from_plus
        new[0] += dist[1] * back_from_plus
        wrong += dist[-1] * down_from_minus
        new[0] += dist[-1] * back_from_minus
        dist = new
    return {"correct": correct, "wrong": wrong, "none": 1 - correct - wrong}


def cascade_limit_correct(p: float, tie: str = "coin") -> float:
    """Probability that the eventual (n -> infinity) cascade is correct."""
    if tie == "coin":
        return p * (p + 1) / (2 * (1 - p + p * p))
    return p * p / (p * p + (1 - p) ** 2)


# Delegation chain ---------------------------------------------------------


def delegation_corruption_probs(error: float, detect: float, verify: Sequence[int]) -> list[float]:
    """pi_i = P(output of stage i is corrupted) given each stage's verify choice."""
    pis: list[float] = []
    prev = 0.0
    for v in verify:
        survive = prev * (1 - detect) if v else prev
        prev = 1 - (1 - survive) * (1 - error)
        pis.append(prev)
    return pis


def delegation_success(error: float, detect: float, verify: Sequence[int]) -> float:
    pis = delegation_corruption_probs(error, detect, verify)
    return 1 - pis[-1] if pis else 1.0


def delegation_verify_gain(
    stage: int, reward: float, error: float, detect: float, verify: Sequence[int]
) -> float:
    """Expected reward gain to every agent when ``stage`` switches from trust to verify."""
    on = list(verify)
    off = list(verify)
    on[stage], off[stage] = 1, 0
    return reward * (delegation_success(error, detect, on) - delegation_success(error, detect, off))
