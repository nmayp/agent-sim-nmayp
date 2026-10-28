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
