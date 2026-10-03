"""
receptive_field.py — K-hop receptive field analysis for GCN.

Demonstrates the fundamental GCN property:
    A K-layer GCN aggregates information from up to K hops away.

    1 layer  →  1-hop neighbors
    2 layers →  2-hop neighbors (neighbors of neighbors)
    K layers →  K-hop neighbors

This is a core concept from Hamilton (2020) Chapter 5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import torch

logger = logging.getLogger(__name__)

_NODE_COLOR_DEFAULT = "#CCCCCC"
_NODE_COLOR_TARGET = "#E63946"
_HOP_COLORS = ["#2A9D8F", "#F4A261", "#457B9D", "#A8DADC"]


def compute_k_hop_neighbors(
    A: torch.Tensor,
    node: int,
    max_k: int = 3,
) -> dict[int, set[int]]:
    """Compute the k-hop neighborhood sets for a given node.

    Parameters
    ----------
    A : torch.Tensor, shape (N, N)
        Binary adjacency matrix (without self-loops).
    node : int
        Target node index.
    max_k : int
        Maximum number of hops to compute.

    Returns
    -------
    dict[int, set[int]]
        Mapping from hop distance k → set of node indices reachable
        in exactly k hops.  Does not include the target node itself
        or nodes reachable in fewer hops.

    Examples
    --------
    >>> hop_sets = compute_k_hop_neighbors(A, node=0, max_k=2)
    >>> hop_sets[1]  # direct neighbors of node 0
    >>> hop_sets[2]  # nodes 2 hops from node 0 (not already in hop 1)
    """
    A_np = A.cpu().numpy()
    G = nx.from_numpy_array(A_np)
    G.remove_edges_from(nx.selfloop_edges(G))

    hop_sets: dict[int, set[int]] = {}
    visited: set[int] = {node}

    current_frontier: set[int] = {node}

    for k in range(1, max_k + 1):
        next_frontier: set[int] = set()
        for v in current_frontier:
            for neighbor in G.neighbors(v):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
        visited |= next_frontier
        hop_sets[k] = next_frontier
        current_frontier = next_frontier

        if not current_frontier:
            logger.debug("Node %d has no %d-hop neighbors", node, k)
            break

    logger.info(
        "Receptive field of node %d (max_k=%d): %s",
        node, max_k,
        {k: sorted(v) for k, v in hop_sets.items()},
    )
    return hop_sets


def visualize_receptive_field(
    A: torch.Tensor,
    node: int,
    labels: Optional[torch.Tensor] = None,
    node_ids: Optional[list[int]] = None,
    max_k: int = 3,
    *,
    figsize: tuple[int, int] = (10, 7),
    save_path: Optional[str | Path] = None,
    seed: int = 42,
) -> matplotlib.figure.Figure:
    """Visualize the K-hop receptive field of a node in the graph.

    Nodes are highlighted by their distance from the target node:
        - Red   : target node itself
        - Teal  : 1-hop neighbors
        - Orange: 2-hop neighbors
        - Blue  : 3-hop neighbors
        - Gray  : outside receptive field

    Parameters
    ----------
    A : torch.Tensor, shape (N, N)
        Binary adjacency matrix.
    node : int
        Target node to analyse.
    labels : torch.Tensor, optional
        True class labels (displayed in title).
    node_ids : list[int], optional
        Node identifier labels.
    max_k : int
        Maximum number of hops to visualize.
    figsize : tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        Save path.
    seed : int
        Layout seed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    N = int(A.shape[0])
    if node_ids is None:
        node_ids = list(range(N))

    hop_sets = compute_k_hop_neighbors(A, node, max_k=max_k)

    A_np = A.cpu().numpy()
    G = nx.from_numpy_array(A_np)
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.spring_layout(G, seed=seed)

    # Assign colors based on hop distance
    node_colors: list[str] = []
    node_sizes: list[int] = []
    for i in range(N):
        if i == node:
            node_colors.append(_NODE_COLOR_TARGET)
            node_sizes.append(900)
        else:
            hop = None
            for k, s in hop_sets.items():
                if i in s:
                    hop = k
                    break
            if hop is not None:
                node_colors.append(_HOP_COLORS[(hop - 1) % len(_HOP_COLORS)])
                node_sizes.append(700)
            else:
                node_colors.append(_NODE_COLOR_DEFAULT)
                node_sizes.append(500)

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, edge_color="#888888", width=1.5)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        linewidths=1.5, edgecolors="#333333", alpha=0.9,
    )
    nx.draw_networkx_labels(
        G, pos,
        labels={i: str(node_ids[i]) for i in range(N)},
        ax=ax, font_size=10, font_color="white", font_weight="bold",
    )

    # Legend
    legend_entries = [
        matplotlib.patches.Patch(facecolor=_NODE_COLOR_TARGET,  label=f"Target node ({node_ids[node]})"),
    ]
    for k in range(1, max_k + 1):
        count = len(hop_sets.get(k, set()))
        legend_entries.append(
            matplotlib.patches.Patch(
                facecolor=_HOP_COLORS[(k - 1) % len(_HOP_COLORS)],
                label=f"{k}-hop neighbors ({count} nodes)",
            )
        )
    legend_entries.append(
        matplotlib.patches.Patch(facecolor=_NODE_COLOR_DEFAULT, label="Outside receptive field")
    )
    ax.legend(handles=legend_entries, loc="upper right", framealpha=0.9)

    ax.set_title(
        f"Receptive Field of Node {node_ids[node]}\n"
        f"(K-layer GCN aggregates K-hop information)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Receptive field figure saved to %s", save_path)

    return fig
