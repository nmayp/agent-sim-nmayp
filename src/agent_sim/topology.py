"""Communication topologies: who can observe and message whom."""

from __future__ import annotations

import networkx as nx

TOPOLOGIES = ("complete", "ring", "small_world", "scale_free")


def build_topology(
    name: str,
    n: int,
    seed: int = 0,
    *,
    k: int = 4,
    p: float = 0.1,
    m: int = 2,
) -> nx.Graph:
    """Build an undirected communication graph over nodes ``0..n-1``.

    ``k``/``p`` parameterise the Watts-Strogatz small world (k nearest ring
    neighbours, rewiring probability p); ``m`` is the Barabasi-Albert
    attachment count. Degenerate sizes fall back to the complete graph.
    """
    if n < 1:
        raise ValueError("topology needs at least one node")
    if name not in TOPOLOGIES:
        raise ValueError(f"unknown topology {name!r}; choose from {', '.join(TOPOLOGIES)}")
    if name == "complete" or n <= 3:
        return nx.complete_graph(n)
    if name == "ring":
        return nx.cycle_graph(n)
    if name == "small_world":
        k_eff = min(k, n - 1)
        k_eff -= k_eff % 2
        if k_eff < 2:
            return nx.cycle_graph(n)
        return nx.connected_watts_strogatz_graph(n, k_eff, p, tries=200, seed=seed)
    m_eff = max(1, min(m, n - 1))
    return nx.barabasi_albert_graph(n, m_eff, seed=seed)


def neighbor_map(graph: nx.Graph) -> dict[int, tuple[int, ...]]:
    return {int(v): tuple(sorted(int(u) for u in graph.neighbors(v))) for v in graph.nodes}


def describe(graph: nx.Graph) -> dict[str, float]:
    degrees = [d for _, d in graph.degree()]
    n = graph.number_of_nodes()
    stats: dict[str, float] = {
        "nodes": float(n),
        "edges": float(graph.number_of_edges()),
        "mean_degree": float(sum(degrees) / n) if n else 0.0,
        "clustering": float(nx.average_clustering(graph)) if n > 2 else 0.0,
    }
    if n > 1 and nx.is_connected(graph):
        stats["avg_path_length"] = float(nx.average_shortest_path_length(graph))
    return stats
