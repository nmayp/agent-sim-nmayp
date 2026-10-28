"""Built-in scenarios."""

from agent_sim.engine import Scenario
from agent_sim.scenarios.cascade import InformationCascade
from agent_sim.scenarios.delegation import DelegationChain
from agent_sim.scenarios.duopoly import Bertrand, Cournot
from agent_sim.scenarios.public_goods import PublicGoods

SCENARIOS: dict[str, type[Scenario]] = {
    cls.name: cls for cls in (PublicGoods, Cournot, Bertrand, InformationCascade, DelegationChain)
}


def make_scenario(name: str, **params) -> Scenario:
    try:
        cls = SCENARIOS[name]
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}; choose from {', '.join(SCENARIOS)}") from None
    return cls(**params)


__all__ = [
    "SCENARIOS",
    "Bertrand",
    "Cournot",
    "DelegationChain",
    "InformationCascade",
    "PublicGoods",
    "make_scenario",
]
