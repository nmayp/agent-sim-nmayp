import json

import numpy as np

from agent_sim.cli import main
from agent_sim.engine import ActionSpace, Observation
from agent_sim.policies import LLMPolicy
from agent_sim.providers import MockProvider
from agent_sim.runner import SimConfig, run


class Garbage:
    name = "garbage"

    def complete(self, system, prompt):
        return "I refuse to pick a number"


def _obs(**kw):
    base = dict(
        scenario="public_goods",
        agent_id=0,
        step=0,
        n_agents=2,
        action_space=ActionSpace(0, 20),
        cooperate_action=20.0,
        defect_action=0.0,
        neighbors=(1,),
        neighbor_cooperation=None,
        neighbor_actions={},
        history=(),
        private={},
        inbox=(),
        rules="rules text",
    )
    base.update(kw)
    return Observation(**base)


def test_mock_provider_is_deterministic_and_parseable():
    obs = _obs()
    p1 = LLMPolicy(MockProvider("cooperative", noise=0.0, seed=3))
    p2 = LLMPolicy(MockProvider("cooperative", noise=0.0, seed=3))
    rng = np.random.default_rng(0)
    d1, d2 = p1.decide(obs, rng), p2.decide(obs, rng)
    assert d1.action == d2.action == 20.0 and d1.raw == d2.raw
    assert d1.message


def test_reciprocal_mock_retaliates():
    policy = LLMPolicy(MockProvider("reciprocal", noise=0.0))
    rng = np.random.default_rng(0)
    assert policy.decide(_obs(neighbor_cooperation=0.9), rng).action == 20.0
    assert policy.decide(_obs(neighbor_cooperation=0.1), rng).action == 0.0


def test_prompt_contains_state():
    policy = LLMPolicy(MockProvider())
    text = policy.render(_obs(neighbor_actions={1: 5.0}, history=((3.0, 21.0),)))
    assert "rules text" in text and "agent 1: 5" in text and "payoff 21.00" in text
    hidden = LLMPolicy(MockProvider(), reveal_benchmarks=False).render(_obs())
    assert "Collective-optimum" not in hidden


def test_unparseable_reply_falls_back_and_is_counted():
    policy = LLMPolicy(Garbage(), fallback="defect")
    d = policy.decide(_obs(), np.random.default_rng(0))
    assert d.action == 0.0 and not d.parsed

    from agent_sim.engine import Simulation
    from agent_sim.scenarios import PublicGoods

    sim = Simulation(PublicGoods(), [LLMPolicy(Garbage()), LLMPolicy(Garbage())], seed=0)
    assert sim.run(5).summary["parse_failures"] == 10


def test_cooperative_llm_population_cooperates():
    r = run(SimConfig("public_goods", policy="llm", provider="mock:cooperative", steps=30))
    assert r.summary["cooperation_rate"] > 0.9


def test_cli_run_and_sweep(tmp_path, capsys):
    out = tmp_path / "run"
    rc = main(
        [
            "run",
            "public_goods",
            "--agents",
            "8",
            "--topology",
            "small_world",
            "--steps",
            "50",
            "--out",
            str(out),
            "--json",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["cooperation_rate"] == 1.0
    assert (out / "run.json").exists() and (out / "agents.csv").exists()

    sweep_path = tmp_path / "sweep.json"
    rc = main(
        [
            "sweep",
            "cournot",
            "--policy",
            "random",
            "--steps",
            "5",
            "--grid",
            "cost=5,15",
            "--seeds",
            "0-1",
            "--out",
            str(sweep_path),
        ]
    )
    assert rc == 0
    assert len(json.loads(sweep_path.read_text())) == 4


def test_cli_reports_bad_params(capsys):
    assert main(["run", "cascade", "--param", "nope=1"]) == 2
    assert "error" in capsys.readouterr().err
