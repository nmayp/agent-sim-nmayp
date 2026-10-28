"""Command-line interface: ``agent-sim run|sweep|list``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from agent_sim.policies import SCRIPTED
from agent_sim.runner import SimConfig, coerce, run, sweep, write_rows
from agent_sim.scenarios import SCENARIOS
from agent_sim.topology import TOPOLOGIES


def _kv(items: list[str]) -> dict[str, Any]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {item!r}")
        out[key] = coerce(value)
    return out


def _grid(items: list[str]) -> dict[str, list[Any]]:
    out = {}
    for item in items:
        key, sep, values = item.partition("=")
        if not sep:
            raise SystemExit(f"expected key=v1,v2,..., got {item!r}")
        out[key] = [coerce(v) for v in values.split(",")]
    return out


def _seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("scenario", choices=sorted(SCENARIOS))
    p.add_argument("--agents", type=int, default=None, help="population size")
    p.add_argument("--topology", choices=TOPOLOGIES, default="complete")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument(
        "--policy",
        default="tit_for_tat",
        help="policy name, llm[=provider], or a mix like 'tit_for_tat*4,always_defect*4'",
    )
    p.add_argument("--provider", default="mock", help="mock[:persona] or anthropic[:model]")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--topology-param", action="append", default=[], metavar="KEY=VALUE")


def _config(args: argparse.Namespace, seed: int) -> SimConfig:
    return SimConfig(
        scenario=args.scenario,
        n_agents=args.agents,
        topology=args.topology,
        steps=args.steps,
        seed=seed,
        policy=args.policy,
        provider=args.provider,
        scenario_params=_kv(args.param),
        topology_params=_kv(args.topology_param),
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def cmd_run(args: argparse.Namespace) -> int:
    result = run(_config(args, args.seed))
    if args.out:
        out = Path(args.out)
        result.to_json(out / "run.json")
        result.to_csv(out / "agents.csv")
        result.steps_to_csv(out / "steps.csv")
    if args.json:
        print(json.dumps(result.summary, indent=2))
    else:
        width = max(len(k) for k in result.summary)
        for key, value in result.summary.items():
            print(f"{key:<{width}}  {_fmt(value)}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    grid = _grid(args.grid)
    grid["seed"] = _seeds(args.seeds)
    rows = sweep(_config(args, 0), grid)
    path = write_rows(rows, args.out)
    print(f"wrote {len(rows)} runs to {path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print("scenarios: ", ", ".join(sorted(SCENARIOS)))
    print("topologies:", ", ".join(TOPOLOGIES))
    print("policies:  ", ", ".join([*SCRIPTED, "llm"]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-sim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one simulation and print summary metrics")
    _common(p_run)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--out", help="directory for run.json, agents.csv and steps.csv")
    p_run.add_argument("--json", action="store_true", help="print the summary as JSON")
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser("sweep", help="grid sweep over parameters and seeds")
    _common(p_sweep)
    p_sweep.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2")
    p_sweep.add_argument("--seeds", default="0-4", help="'0-9' or '1,2,3'")
    p_sweep.add_argument("--out", default="results/sweep.csv", help=".csv or .json")
    p_sweep.set_defaults(func=cmd_sweep)

    p_list = sub.add_parser("list", help="list scenarios, topologies and policies")
    p_list.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
