# agent-sim

Multi-agent sandbox for measuring emergent behavior (cooperation, collusion, herding, failure cascades) in LLM agent populations, with scripted game-theoretic baselines.

## Install

```bash
pip install -e .            # core (numpy, networkx)
pip install -e ".[anthropic]"  # optional Anthropic-backed agents
```

## Usage

```bash
agent-sim run public_goods --agents 8 --topology small_world --steps 50
```

## Options

- `--agents`: Number of interacting agents in the population (default: `8`)
- `--topology`: Network structure for agent interactions (`small_world`, `scale_free`, `complete`)
- `--steps`: Total simulation time steps / rounds (default: `50`)
- `--seed`: Optional random seed for deterministic reproduction
