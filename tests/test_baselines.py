"""Scripted baselines must reproduce known game-theoretic outcomes."""

import numpy as np
import pytest

from agent_sim import theory
from agent_sim.runner import SimConfig, run


def summary(scenario, policy, **kw):
    params = kw.pop("params", {})
    return run(SimConfig(scenario, policy=policy, scenario_params=params, **kw))


# public goods -------------------------------------------------------------


def test_public_goods_all_defect_is_nash():
    r = summary("public_goods", "always_defect", n_agents=8, steps=20)
    assert r.summary["mean_contribution"] == 0.0
    assert r.summary["mean_payoff_per_step"] == pytest.approx(20.0)  # keep endowment
    assert r.summary["payoff_gini"] == 0.0


def test_public_goods_tft_population_sustains_full_cooperation():
    r = summary("public_goods", "tit_for_tat", n_agents=8, steps=20, topology="small_world")
    assert r.summary["cooperation_rate"] == 1.0
    assert r.summary["mean_payoff_per_step"] == pytest.approx(3.0 * 20.0)  # r * E
    assert r.summary["efficiency"] == pytest.approx(1.0)


def test_public_goods_tft_collapses_against_defector_majority():
    r = summary("public_goods", "tit_for_tat*3,always_defect*5", n_agents=8, steps=10)
    contributions = [s["mean_contribution"] for s in r.steps]
    assert contributions[0] == pytest.approx(3 * 20 / 8)
    assert all(c == 0 for c in contributions[1:])
    assert r.summary["payoff_per_step[always_defect]"] > r.summary["payoff_per_step[tit_for_tat]"]


def test_public_goods_nash_flips_when_mpcr_exceeds_one():
    assert theory.public_goods_nash_contribution(20, 3, 8) == 0
    assert theory.public_goods_nash_contribution(20, 9, 8) == 20
    r = summary("public_goods", "always_defect", n_agents=4, steps=5, params={"multiplier": 5.0})
    assert r.summary["mean_contribution"] == 20.0


# cournot / bertrand -------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5])
def test_cournot_nash_players_hit_nash_price(n):
    r = summary("cournot", "always_defect", n_agents=n, steps=5)
    assert r.summary["mean_price"] == pytest.approx(theory.cournot_nash_price(100, 1, 10, n))
    assert r.summary["collusion_index"] == pytest.approx(0.0)


def test_cournot_tft_sustains_monopoly_price():
    r = summary("cournot", "tit_for_tat", n_agents=2, steps=20)
    assert r.summary["mean_price"] == pytest.approx(55.0)
    assert r.summary["collusion_index"] == pytest.approx(1.0)


def test_cournot_duopoly_best_response_converges_to_nash():
    r = summary("cournot", "best_response", n_agents=2, steps=40)
    q = [s["total_quantity"] for s in r.steps]
    assert q[0] == pytest.approx(2 * 45.0)  # both start at monopoly output
    assert q[-1] == pytest.approx(60.0, abs=1e-6)
    assert r.summary["final_collusion_index"] == pytest.approx(0.0, abs=1e-6)


def test_cournot_best_response_unstable_with_many_firms():
    # Theocharis (1960): simultaneous best-response dynamics diverge for n >= 4.
    r = summary("cournot", "best_response", n_agents=4, steps=40)
    q = np.array([s["total_quantity"] for s in r.steps])
    assert abs(q[-1] - 4 * theory.cournot_nash_quantity(100, 1, 10, 4)) > 10


def test_bertrand_undercutting_drives_price_to_cost():
    r = summary("bertrand", "best_response", n_agents=2, steps=120)
    assert r.steps[0]["price"] == pytest.approx(55.0)
    assert r.steps[-1]["price"] <= 10.0 + 1.0
    assert r.summary["final_collusion_index"] <= 1.0 / 45.0 + 1e-9


def test_bertrand_nash_and_tft():
    assert summary("bertrand", "always_defect", steps=5).summary["collusion_index"] == 0.0
    tft = summary("bertrand", "tit_for_tat", steps=5).summary
    assert tft["collusion_index"] == 1.0


def test_tacit_collusion_detection_between_llm_personas():
    coop = summary("cournot", "llm=mock:cooperative", steps=40).summary["collusion_index"]
    selfish = summary("cournot", "llm=mock:selfish", steps=40).summary["collusion_index"]
    assert coop > 0.6 and abs(selfish) < 0.3


# information cascade -------------------------------------------------------


@pytest.mark.parametrize("tie", ["coin", "signal"])
@pytest.mark.parametrize("p", [0.6, 0.75])
def test_bayesian_cascade_rates_match_markov_chain(tie, p):
    n, rounds = 6, 4000
    r = summary(
        "cascade", f"bayesian={tie}", n_agents=n, steps=rounds, seed=11, params={"accuracy": p}
    )
    exact = theory.cascade_probabilities(p, n, tie=tie)
    se = np.sqrt(0.25 / rounds)
    assert r.summary["cascade_correct_rate"] == pytest.approx(exact["correct"], abs=4 * se)
    assert r.summary["cascade_wrong_rate"] == pytest.approx(exact["wrong"], abs=4 * se)


def test_cascade_chain_converges_to_bhw_limit():
    for p in (0.55, 0.7, 0.9):
        for tie in ("coin", "signal"):
            got = theory.cascade_probabilities(p, 400, tie)["correct"]
            assert got == pytest.approx(theory.cascade_limit_correct(p, tie), abs=1e-6)
    assert theory.cascade_limit_correct(0.5) == pytest.approx(0.5)


def test_signal_followers_reveal_information():
    r = summary("cascade", "always_cooperate", n_agents=8, steps=3000, params={"accuracy": 0.7})
    assert r.summary["guess_accuracy"] == pytest.approx(0.7, abs=0.02)
    assert r.summary["cooperation_rate"] == 1.0


def test_herding_cuts_late_mover_accuracy_below_full_information():
    bayes = summary("cascade", "bayesian", n_agents=20, steps=1500, params={"accuracy": 0.7})
    # with 20 independent signals a majority vote is right ~95% of the time;
    # cascades lock in after ~2 signals, so herders end up far below that.
    assert bayes.summary["guess_accuracy"] < 0.85


# delegation chain ---------------------------------------------------------


def test_trust_all_matches_independent_failures():
    params = {"cost": 100.0, "error": 0.1}
    r = summary("delegation", "always_defect", n_agents=6, steps=4000, params=params)
    assert r.summary["verify_rate"] == 0.0
    assert r.summary["success_rate"] == pytest.approx(0.9**6, abs=0.03)
    assert r.summary["nash_success"] == pytest.approx(0.9**6)


def test_optimal_profile_success_matches_theory():
    r = summary("delegation", "always_cooperate", n_agents=6, steps=4000, seed=2)
    assert r.summary["success_rate"] == pytest.approx(r.summary["optimal_success"], abs=0.03)
    assert r.summary["optimal_success"] >= r.summary["nash_success"]


def test_delegation_nash_profile_has_no_profitable_deviation():
    from agent_sim.scenarios import DelegationChain

    for cost in (0.2, 1.0, 3.0):
        sc = DelegationChain(cost=cost)
        sc.reset(7, np.random.default_rng(0))
        v = list(sc.nash_profile())
        for i in range(7):
            gain = theory.delegation_verify_gain(i, sc.reward, sc.error, sc.detect, v)
            assert (gain - cost) * (1 if v[i] else -1) >= -1e-12
        assert v[0] == 0  # the first stage has nothing to verify


def test_delegation_closed_form():
    assert theory.delegation_success(0.0, 0.5, [0, 0, 0]) == 1.0
    assert theory.delegation_success(0.2, 1.0, [0, 1]) == pytest.approx(0.8)
    assert theory.delegation_success(0.2, 0.0, [1, 1]) == pytest.approx(0.64)
