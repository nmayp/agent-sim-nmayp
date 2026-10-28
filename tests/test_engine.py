import json

import networkx as nx
import pytest

from agent_sim import metrics
from agent_sim.engine import ActionSpace, MessageBus, Simulation
from agent_sim.policies import RandomPolicy
from agent_sim.runner import SimConfig, build, expand_grid, parse_mix, run, sweep, write_rows
from agent_sim.scenarios import SCENARIOS, PublicGoods
from agent_sim.topology import TOPOLOGIES, build_topology


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("policy", ["random", "llm", "tit_for_tat*2,random*2,llm*2"])
def test_same_seed_same_trajectory(scenario, policy):
    cfg = SimConfig(scenario, n_agents=6, topology="small_world", steps=25, seed=7, policy=policy)
    a, b = run(cfg), run(cfg)
    assert a.records == b.records
    assert a.steps == b.steps
    assert json.dumps(a.summary, sort_keys=True) == json.dumps(b.summary, sort_keys=True)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_different_seed_different_trajectory(scenario):
    base = SimConfig(scenario, n_agents=6, steps=25, policy="random")
    a = run(base)
    b = run(SimConfig(scenario, n_agents=6, steps=25, policy="random", seed=1))
    assert a.records != b.records


def test_topology_seed_is_part_of_run_seed():
    g1 = build(SimConfig("public_goods", topology="scale_free", n_agents=20, seed=3)).graph
    g2 = build(SimConfig("public_goods", topology="scale_free", n_agents=20, seed=3)).graph
    g3 = build(SimConfig("public_goods", topology="scale_free", n_agents=20, seed=4)).graph
    assert nx.utils.graphs_equal(g1, g2)
    assert not nx.utils.graphs_equal(g1, g3)


@pytest.mark.parametrize("name", TOPOLOGIES)
def test_topologies_are_connected_with_expected_size(name):
    g = build_topology(name, 30, seed=1)
    assert g.number_of_nodes() == 30
    assert nx.is_connected(g)


def test_topology_shapes():
    assert all(d == 2 for _, d in build_topology("ring", 10).degree())
    assert build_topology("complete", 10).number_of_edges() == 45
    ba = build_topology("scale_free", 200, seed=0, m=2)
    assert ba.number_of_edges() == 2 * (200 - 2)
    assert max(d for _, d in ba.degree()) > 15  # hubs
    ws = build_topology("small_world", 100, seed=0, k=4, p=0.1)
    assert nx.average_clustering(ws) > 0.3
    with pytest.raises(ValueError):
        build_topology("star", 5)


def test_message_bus_delivers_next_step_to_neighbours_only():
    bus = MessageBus({0: (1,), 1: (0, 2), 2: (1,)})
    assert bus.send(1, "hello", step=0) == 2
    assert bus.inbox(0) == ()
    bus.deliver()
    assert [m.content for m in bus.inbox(0)] == ["hello"]
    assert [m.sender for m in bus.inbox(2)] == [1]
    assert bus.inbox(1) == ()
    bus.deliver()
    assert bus.inbox(0) == ()
    assert bus.send(0, "   ", step=1) == 0


def test_llm_messages_flow_over_topology():
    cfg = SimConfig("public_goods", n_agents=6, topology="ring", steps=10, policy="llm")
    result = run(cfg)
    assert result.summary["messages_sent"] == 10 * 6 * 2


def test_action_space_clipping():
    cont = ActionSpace(0, 10)
    assert cont.clip(-3) == 0 and cont.clip(12) == 10 and cont.clip(float("nan")) == 0
    disc = ActionSpace(0, 1, choices=(0.0, 1.0))
    assert disc.clip(0.7) == 1.0 and disc.clip(0.2) == 0.0


def test_graph_size_mismatch_rejected():
    with pytest.raises(ValueError):
        Simulation(PublicGoods(), [RandomPolicy()] * 3, graph=nx.complete_graph(4))


def test_metrics():
    assert metrics.gini([5, 5, 5, 5]) == pytest.approx(0.0)
    assert metrics.gini([0, 0, 0, 10]) == pytest.approx(0.75)
    assert metrics.gini([-1, -1, -1, 9]) == pytest.approx(0.75)
    assert metrics.polarization([0, 0, 1, 1]) == pytest.approx(1.0)
    assert metrics.polarization([0.3] * 5) == 0.0
    assert metrics.collusion_index(55, 40, 55) == 1.0
    assert metrics.consensus_time([[0, 1], [1, 1], [0, 1], [1, 1], [1, 1]]) == 3
    assert metrics.consensus_time([[0, 1]]) is None


def test_parse_mix():
    assert parse_mix("random", 3) == ["random"] * 3
    assert parse_mix("a*2,b", 3) == ["a", "a", "b"]
    with pytest.raises(ValueError):
        parse_mix("a*2,b*2", 3)


def test_sweep_and_exports(tmp_path):
    base = SimConfig("cournot", steps=10, policy="random")
    grid = {"cost": [10.0, 20.0], "seed": [0, 1]}
    assert len(expand_grid(base, grid)) == 4
    rows = sweep(base, grid)
    assert {r["param.cost"] for r in rows} == {10.0, 20.0}
    assert all("collusion_index" in r for r in rows)
    csv_path = write_rows(rows, tmp_path / "s.csv")
    assert csv_path.read_text().count("\n") == 5
    json_path = write_rows(rows, tmp_path / "s.json")
    assert len(json.loads(json_path.read_text())) == 4

    result = run(base)
    result.to_json(tmp_path / "run.json")
    result.to_csv(tmp_path / "agents.csv")
    payload = json.loads((tmp_path / "run.json").read_text())
    assert payload["config"]["seed"] == 0 and len(payload["steps"]) == 10
    assert (tmp_path / "agents.csv").read_text().count("\n") == 1 + 10 * 2
