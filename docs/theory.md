# Theory: benchmarks and metrics

Closed forms: `agent_sim.theory` (checked by `tests/test_baselines.py`).

## Model

- Discrete steps `t = 0..T-1`; `n` agents on graph `G`: `complete`, `ring`, `small_world` (WS, `k=4`, `p=0.1`), `scale_free` (BA, `m=2`).
- Each action maps to a cooperation score `s_i in [0,1]` and a stance `x_i in [0,1]`.
- `SeedSequence(seed)`: separate streams for topology, scenario, each agent.
- Baselines: `always_cooperate`, `always_defect`, `random`, `tit_for_tat` / `grim_trigger` (neighbour mean cooperation >= 0.5), `best_response`, `bayesian` (BHW).

## Public goods (`public_goods`, E=20, r=3)

    u_i = E - c_i + (r/n) * sum_j c_j
- Nash (1 < r < n): `c_i = 0`, payoff `E`. Optimum (r > 1): `c_i = E`, payoff `rE`.
- `efficiency = (mean payoff - Nash) / (optimum - Nash)`.

## Cournot / Bertrand (a=100, b=1, c=10)

    P(Q) = a - bQ;  BR(Q_-i) = max(0, (a - c - b Q_-i) / (2b))
- Cournot Nash: `q* = (a-c)/(b(n+1))`, `p_N = (a+nc)/(n+1)` (duopoly: 30, 40).
- Monopoly: `Q_M = (a-c)/(2b)`, `p_M = (a+c)/2 = 55`.
- Bertrand Nash: `p = c` (within one `tick`, default 1).
- Collusion index `CI = (p - p_N)/(p_M - p_N)`: 0 = Nash, 1 = monopoly.

## Information cascade (`cascade`, p=0.7)

- `d` = observed 1-guesses minus 0-guesses. `|d| >= 2`: cascade (follow majority); `|d| = 1` against signal: tie rule; else follow signal.
- As `n -> inf`:

      P(correct) = p(1+p) / (2(1 - p + p^2))   (coin, 0.753 at p=0.7)
      P(correct) = p^2 / (p^2 + (1-p)^2)       (signal, 0.845)
- Metrics: `cascade_rate`, `cascade_correct_rate` / `cascade_wrong_rate` (vs `bhw_*`).

## Delegation chain (`delegation`, e=0.1, delta=0.8, k=1, R=10)

    pi_i = 1 - (1 - pi_{i-1}(1 - v_i delta))(1 - e),  pi_{-1} = 0
    S(v) = 1 - pi_{n-1};  u_i = R S(v) - k v_i
- Potential game; best response from all-trust gives Nash. Optimum maximises `n R S(v) - k sum v` (exhaustive for `n <= 14`).

## Population metrics

- Cooperation rate: mean `s_i(t)`; final = last `max(1, T/10)` steps.
- Polarization: `min(1, 4 Var_i x_i(t))`.
- Consensus time: first `t*` with `max x - min x <= 0.05` for all `t >= t*`.
- Gini: `G = 2 sum_i i U_(i) / (n sum U) - (n+1)/n` (shifted if any `U < 0`).

## References

- Bikhchandani, Hirshleifer, Welch (1992). Informational cascades. *JPE* 100(5).
- Calvano et al. (2020). AI, algorithmic pricing, and collusion. *AER* 110(10).
- Monderer, Shapley (1996). Potential games. *GEB* 14(1).
