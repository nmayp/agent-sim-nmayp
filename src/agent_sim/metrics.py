"""Population-level metrics. Definitions are in docs/theory.md."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def gini(values: Sequence[float]) -> float:
    """Gini coefficient; negative values are shifted so the minimum is zero."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    if x.min() < 0:
        x = x - x.min()
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    ranks = np.arange(1, n + 1)
    return float((2 * np.sum(ranks * x)) / (n * total) - (n + 1) / n)


def cooperation_rate(scores: Sequence[float]) -> float:
    return float(np.mean(scores)) if len(scores) else 0.0


def collusion_index(price: float, competitive: float, monopoly: float) -> float:
    """0 at the competitive (Nash) price, 1 at the joint-monopoly price."""
    if monopoly == competitive:
        return 0.0
    return float((price - competitive) / (monopoly - competitive))


def polarization(stances: Sequence[float]) -> float:
    """4 * variance of stances in [0, 1]: 0 at consensus, 1 for an even 0/1 split."""
    x = np.asarray(stances, dtype=float)
    if x.size < 2:
        return 0.0
    return float(min(1.0, 4.0 * x.var()))


def consensus_time(series: Sequence[Sequence[float]], eps: float = 0.05) -> int | None:
    """First step from which the stance range stays within ``eps`` until the end."""
    t_star: int | None = None
    for t, stances in enumerate(series):
        spread = float(np.ptp(np.asarray(stances, dtype=float))) if len(stances) else 0.0
        if spread <= eps:
            if t_star is None:
                t_star = t
        else:
            t_star = None
    return t_star
