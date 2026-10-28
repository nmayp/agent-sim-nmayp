"""Discrete-time simulation engine: agents, message bus, scenario loop."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from agent_sim import metrics
from agent_sim.topology import build_topology, describe, neighbor_map

if TYPE_CHECKING:
    import networkx as nx


@dataclass(frozen=True)
class Message:
    sender: int
    recipient: int
    step: int
    content: str


class MessageBus:
    """Topology-restricted bus. Messages sent during step t are readable at t+1."""

    def __init__(self, neighbors: dict[int, tuple[int, ...]], max_len: int = 280):
        self.neighbors = neighbors
        self.max_len = max_len
        self._pending: list[Message] = []
        self._inboxes: dict[int, list[Message]] = {i: [] for i in neighbors}
        self.sent = 0

    def send(self, sender: int, content: str, step: int) -> int:
        content = content.strip()[: self.max_len]
        if not content:
            return 0
        targets = self.neighbors.get(sender, ())
        for r in targets:
            self._pending.append(Message(sender, r, step, content))
        self.sent += len(targets)
        return len(targets)

    def deliver(self) -> None:
        self._inboxes = {i: [] for i in self.neighbors}
        for msg in self._pending:
            self._inboxes[msg.recipient].append(msg)
        self._pending = []

    def inbox(self, agent_id: int) -> tuple[Message, ...]:
        return tuple(self._inboxes.get(agent_id, ()))


@dataclass(frozen=True)
class ActionSpace:
    low: float
    high: float
    choices: tuple[float, ...] | None = None

    @property
    def discrete(self) -> bool:
        return self.choices is not None

    def clip(self, x: float) -> float:
        if not np.isfinite(x):
            x = self.low
        if self.choices is not None:
            return min(self.choices, key=lambda c: (abs(c - x), c))
        return float(min(max(x, self.low), self.high))

    def sample(self, rng: np.random.Generator) -> float:
        if self.choices is not None:
            return float(self.choices[int(rng.integers(len(self.choices)))])
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class Observation:
    scenario: str
    agent_id: int
    step: int
    n_agents: int
    action_space: ActionSpace
    cooperate_action: float
    defect_action: float
    neighbors: tuple[int, ...]
    neighbor_cooperation: float | None
    neighbor_actions: dict[int, float]
    history: tuple[tuple[float, float], ...]
    private: dict[str, Any]
    inbox: tuple[Message, ...]
    rules: str


@dataclass
class Decision:
    action: float
    message: str | None = None
    raw: str | None = None
    parsed: bool = True


class Policy(Protocol):
    name: str

    def decide(self, obs: Observation, rng: np.random.Generator) -> Decision: ...


class Agent:
    def __init__(self, agent_id: int, policy: Policy, rng: np.random.Generator, memory: int = 5):
        self.id = agent_id
        self.policy = policy
        self.rng = rng
        self.history: deque[tuple[float, float]] = deque(maxlen=memory)
        self.total_payoff = 0.0

    def act(self, obs: Observation) -> Decision:
        decision = self.policy.decide(obs, self.rng)
        decision.action = obs.action_space.clip(float(decision.action))
        return decision

    def reward(self, action: float, payoff: float) -> None:
        self.history.append((action, payoff))
        self.total_payoff += payoff


@dataclass
class StepOutcome:
    actions: dict[int, float]
    payoffs: dict[int, float]
    cooperation: dict[int, float]
    stance: dict[int, float]
    info: dict[str, float] = field(default_factory=dict)


class Scenario(ABC):
    """A game played repeatedly. Subclasses drive ``sim.decide`` inside ``play_step``."""

    name: str = "scenario"
    default_agents: int = 4

    def reset(self, n_agents: int, rng: np.random.Generator) -> None:
        self.n_agents = n_agents
        self.rng = rng

    @abstractmethod
    def action_space(self) -> ActionSpace: ...

    @abstractmethod
    def reference_actions(self, agent_id: int, private: dict[str, Any]) -> tuple[float, float]:
        """(cooperative / collective-optimum action, defect / Nash action)."""

    @abstractmethod
    def rules(self) -> str: ...

    @abstractmethod
    def play_step(self, sim: Simulation, step: int) -> StepOutcome: ...

    def summarize(self, result: RunResult) -> dict[str, float]:
        return {}

    def params(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and k != "rng"}


@dataclass
class RunResult:
    config: dict[str, Any]
    records: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_csv(self, path: str | Path) -> Path:
        return _write_csv(path, self.records)

    def steps_to_csv(self, path: str | Path) -> Path:
        return _write_csv(path, self.steps)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.config, "summary": self.summary, "steps": self.steps}
        path.write_text(json.dumps(payload, indent=2, default=_json_default))
        return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


class Simulation:
    """Wires a scenario, a population of policies and a topology under one seed.

    All randomness flows from ``np.random.SeedSequence(seed)``: one child stream
    for the topology, one for the scenario and one per agent, so a run is a pure
    function of (config, seed) as long as policies only use the rng they are given.
    """

    def __init__(
        self,
        scenario: Scenario,
        policies: list[Policy],
        topology: str = "complete",
        seed: int = 0,
        topology_params: dict[str, Any] | None = None,
        graph: nx.Graph | None = None,
    ):
        n = len(policies)
        if n < 1:
            raise ValueError("need at least one agent")
        self.seed = seed
        ss = np.random.SeedSequence(seed)
        topo_ss, scen_ss, *agent_ss = ss.spawn(2 + n)
        self.topology = topology if graph is None else "custom"
        self.topology_params = dict(topology_params or {})
        self.graph = (
            graph
            if graph is not None
            else build_topology(
                topology, n, seed=int(topo_ss.generate_state(1)[0]), **self.topology_params
            )
        )
        if self.graph.number_of_nodes() != n:
            raise ValueError("graph size does not match number of agents")
        self.neighbors = neighbor_map(self.graph)
        self.bus = MessageBus(self.neighbors)
        self.agents = [
            Agent(i, p, np.random.default_rng(agent_ss[i])) for i, p in enumerate(policies)
        ]
        self.scenario = scenario
        self.scenario.reset(n, np.random.default_rng(scen_ss))
        self.last_actions: dict[int, float] = {}
        self.last_cooperation: dict[int, float] = {}
        self.parse_failures = 0

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def decide(
        self,
        agent_id: int,
        step: int,
        private: dict[str, Any] | None = None,
        visible: dict[int, float] | None = None,
    ) -> float:
        """Ask one agent for an action and route its message onto the bus.

        ``visible`` overrides which neighbour actions the agent sees (sequential
        scenarios pass actions taken earlier in the same step).
        """
        private = dict(private or {})
        neigh = self.neighbors[agent_id]
        if visible is None:
            visible = {j: self.last_actions[j] for j in neigh if j in self.last_actions}
        coops = [self.last_cooperation[j] for j in neigh if j in self.last_cooperation]
        coop, defect = self.scenario.reference_actions(agent_id, private)
        agent = self.agents[agent_id]
        obs = Observation(
            scenario=self.scenario.name,
            agent_id=agent_id,
            step=step,
            n_agents=self.n_agents,
            action_space=self.scenario.action_space(),
            cooperate_action=coop,
            defect_action=defect,
            neighbors=neigh,
            neighbor_cooperation=float(np.mean(coops)) if coops else None,
            neighbor_actions=visible,
            history=tuple(agent.history),
            private=private,
            inbox=self.bus.inbox(agent_id),
            rules=self.scenario.rules(),
        )
        decision = agent.act(obs)
        if not decision.parsed:
            self.parse_failures += 1
        if decision.message:
            self.bus.send(agent_id, decision.message, step)
        return decision.action

    def run(self, steps: int) -> RunResult:
        records: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        stances: list[list[float]] = []
        for t in range(steps):
            out = self.scenario.play_step(self, t)
            for agent in self.agents:
                i = agent.id
                agent.reward(out.actions[i], out.payoffs[i])
                records.append(
                    {
                        "step": t,
                        "agent": i,
                        "policy": agent.policy.name,
                        "action": out.actions[i],
                        "payoff": out.payoffs[i],
                        "cooperation": out.cooperation[i],
                        "stance": out.stance[i],
                    }
                )
            coop_mean = float(np.mean(list(out.cooperation.values())))
            stance_vec = [out.stance[i] for i in range(self.n_agents)]
            stances.append(stance_vec)
            step_rows.append(
                {
                    "step": t,
                    "cooperation": coop_mean,
                    "mean_payoff": float(np.mean(list(out.payoffs.values()))),
                    "polarization": metrics.polarization(stance_vec),
                    **out.info,
                }
            )
            self.last_actions = dict(out.actions)
            self.last_cooperation = dict(out.cooperation)
            self.bus.deliver()

        totals = [a.total_payoff for a in self.agents]
        coop_series = [row["cooperation"] for row in step_rows]
        tail = max(1, steps // 10)
        summary: dict[str, Any] = {
            "cooperation_rate": float(np.mean(coop_series)) if coop_series else 0.0,
            "final_cooperation_rate": float(np.mean(coop_series[-tail:])) if coop_series else 0.0,
            "mean_payoff_per_step": float(np.mean(totals) / steps) if steps else 0.0,
            "payoff_gini": metrics.gini(totals),
            "polarization": (
                float(np.mean([r["polarization"] for r in step_rows])) if steps else 0.0
            ),
            "final_polarization": step_rows[-1]["polarization"] if steps else 0.0,
            "consensus_time": metrics.consensus_time(stances),
            "messages_sent": self.bus.sent,
            "parse_failures": self.parse_failures,
        }
        result = RunResult(self.config(steps), records, step_rows, summary)
        summary.update(self.scenario.summarize(result))
        by_policy: dict[str, list[float]] = {}
        for agent in self.agents:
            by_policy.setdefault(agent.policy.name, []).append(agent.total_payoff / max(steps, 1))
        for name, vals in sorted(by_policy.items()):
            summary[f"payoff_per_step[{name}]"] = float(np.mean(vals))
        return result

    def config(self, steps: int) -> dict[str, Any]:
        return {
            "scenario": self.scenario.name,
            "scenario_params": self.scenario.params(),
            "n_agents": self.n_agents,
            "topology": self.topology,
            "topology_params": self.topology_params,
            "topology_stats": describe(self.graph),
            "policies": [a.policy.name for a in self.agents],
            "steps": steps,
            "seed": self.seed,
        }
