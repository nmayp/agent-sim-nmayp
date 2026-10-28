"""Run configuration, population building and parameter sweeps."""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from agent_sim.engine import Policy, RunResult, Simulation
from agent_sim.policies import SCRIPTED, LLMPolicy
from agent_sim.providers import make_provider
from agent_sim.scenarios import SCENARIOS, make_scenario


@dataclass(frozen=True)
class SimConfig:
    scenario: str
    n_agents: int | None = None
    topology: str = "complete"
    steps: int = 50
    seed: int = 0
    policy: str = "tit_for_tat"
    provider: str = "mock"
    scenario_params: dict[str, Any] = field(default_factory=dict)
    topology_params: dict[str, Any] = field(default_factory=dict)

    def resolved_agents(self) -> int:
        if self.n_agents is not None:
            return self.n_agents
        return SCENARIOS[self.scenario].default_agents


def parse_mix(spec: str, n: int) -> list[str]:
    """``"tit_for_tat"`` or ``"tit_for_tat*4,always_defect*4"`` -> one name per agent."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) == 1 and "*" not in parts[0]:
        return [parts[0]] * n
    names: list[str] = []
    for part in parts:
        name, _, count = part.partition("*")
        names.extend([name] * (int(count) if count else 1))
    if len(names) != n:
        raise ValueError(f"policy mix {spec!r} covers {len(names)} agents, expected {n}")
    return names


def make_policy(name: str, provider: str = "mock", seed: int = 0, index: int = 0) -> Policy:
    """Scripted names from ``SCRIPTED``, or ``llm`` / ``llm=<provider spec>``."""
    base, _, arg = name.partition("=")
    if base == "llm":
        return LLMPolicy(make_provider(arg or provider, seed=seed * 1_000_003 + index))
    if base == "bayesian" and arg:
        return SCRIPTED["bayesian"](tie=arg)
    if base not in SCRIPTED:
        raise ValueError(f"unknown policy {name!r}; choose from llm, {', '.join(SCRIPTED)}")
    return SCRIPTED[base]()


def build(config: SimConfig) -> Simulation:
    n = config.resolved_agents()
    scenario = make_scenario(config.scenario, **config.scenario_params)
    names = parse_mix(config.policy, n)
    policies = [make_policy(nm, config.provider, config.seed, i) for i, nm in enumerate(names)]
    return Simulation(
        scenario,
        policies,
        topology=config.topology,
        seed=config.seed,
        topology_params=config.topology_params,
    )


def run(config: SimConfig) -> RunResult:
    result = build(config).run(config.steps)
    result.config["policy_spec"] = config.policy
    return result


_CONFIG_FIELDS = {"n_agents", "topology", "steps", "seed", "policy", "provider"}


def expand_grid(base: SimConfig, grid: dict[str, list[Any]]) -> list[SimConfig]:
    """Cartesian product of ``grid`` applied to ``base``. Keys are SimConfig fields,
    ``topology.<param>`` or scenario parameters."""
    keys = list(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        updates: dict[str, Any] = {}
        sparams = dict(base.scenario_params)
        tparams = dict(base.topology_params)
        for k, v in zip(keys, values, strict=True):
            if k in _CONFIG_FIELDS:
                updates[k] = v
            elif k.startswith("topology."):
                tparams[k.removeprefix("topology.")] = v
            else:
                sparams[k] = v
        configs.append(replace(base, scenario_params=sparams, topology_params=tparams, **updates))
    return configs


def sweep(base: SimConfig, grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows = []
    for cfg in expand_grid(base, grid):
        result = run(cfg)
        row: dict[str, Any] = {k: v for k, v in asdict(cfg).items() if not isinstance(v, dict)}
        row["n_agents"] = cfg.resolved_agents()
        row.update({f"param.{k}": v for k, v in cfg.scenario_params.items()})
        row.update({f"topology.{k}": v for k, v in cfg.topology_params.items()})
        row.update(result.summary)
        rows.append(row)
    return rows


def write_rows(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2))
        return path
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def coerce(value: str) -> Any:
    """Parse a CLI value: int, float, bool, else string."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value
