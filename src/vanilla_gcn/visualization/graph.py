"""
graph.py — Graph visualization utilities for Vanilla GCN.

Uses NetworkX for graph layout and Matplotlib for rendering.

Functions
---------
visualize_graph      — Show original graph with true labels.
visualize_predictions — Show graph with true vs predicted labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Class colours shared across visualisation functions
_CLASS_COLORS = [
    "#E63946",  # class 0 — vivid red
    "#2A9D8F",  # class 1 — teal
    "#F4A261",  # class 2 — warm orange
    "#457B9D",  # class 3 — steel blue
    "#A8DADC",  # class 4 — light teal
    "#6A0572",  # class 5 — deep purple
]


def _adjacency_to_networkx(A: torch.Tensor) -> nx.Graph:
    """Convert a PyTorch adjacency tensor to a NetworkX graph."""
    A_np = A.numpy() if not isinstance(A, np.ndarray) else A
    G = nx.from_numpy_array(A_np.astype(float))
    return G


def _color_list(labels: torch.Tensor) -> list[str]:
    """Map integer labels to HTML colour strings."""
    return [_CLASS_COLORS[int(lbl) % len(_CLASS_COLORS)] for lbl in labels]


def visualize_graph(
    A: torch.Tensor,
    labels: torch.Tensor,
    node_ids: Optional[list[int]] = None,
    *,
    title: str = "Graph — True Labels",
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str | Path] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    seed: int = 42,
) -> matplotlib.figure.Figure:
    """Visualize the graph with nodes coloured by true class label.

    Parameters
    ----------
    A : torch.Tensor, shape (N, N)
        Adjacency matrix (raw, without self-loops for clean visualization).
    labels : torch.Tensor, shape (N,)
        True integer class labels.
    node_ids : list[int], optional
        Human-readable node identifiers.  Defaults to ``[0, 1, ..., N-1]``.
    title : str
        Plot title.
    figsize : tuple[int, int]
        Matplotlib figure size.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.  If ``None``, a new figure is created.
    seed : int
        Random seed for the spring layout algorithm.

    Returns
    -------
    matplotlib.figure.Figure
    """
    N = int(A.shape[0])
    if node_ids is None:
        node_ids = list(range(N))

    G = _adjacency_to_networkx(A)

    # Remove self-loops for cleaner graph display
    G.remove_edges_from(nx.selfloop_edges(G))

    pos = nx.spring_layout(G, seed=seed)
    colors = _color_list(labels)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color="#888888", width=1.5)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=700, alpha=0.9,
        linewidths=1.5, edgecolors="#333333",
    )
    nx.draw_networkx_labels(
        G, pos, labels={i: str(node_ids[i]) for i in range(N)},
        ax=ax, font_size=11, font_color="white", font_weight="bold",
    )

    # Legend
    unique_classes = sorted(set(int(l) for l in labels))
    handles = [
        matplotlib.patches.Patch(
            facecolor=_CLASS_COLORS[c % len(_CLASS_COLORS)],
            edgecolor="#333333",
            label=f"Class {c}",
        )
        for c in unique_classes
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Graph figure saved to %s", save_path)

    if standalone:
        plt.tight_layout()

    return fig


def visualize_predictions(
    A: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    node_ids: Optional[list[int]] = None,
    *,
    title: str = "Graph — Predictions vs Truth",
    figsize: tuple[int, int] = (16, 6),
    save_path: Optional[str | Path] = None,
    seed: int = 42,
) -> matplotlib.figure.Figure:
    """Side-by-side plot: BEFORE GCN (true labels) vs AFTER GCN (predictions).

    Nodes are coloured by their class.  Incorrectly predicted nodes in
    the right panel are marked with a red border.

    Parameters
    ----------
    A : torch.Tensor, shape (N, N)
        Adjacency matrix.
    true_labels : torch.Tensor, shape (N,)
        Ground truth labels.
    pred_labels : torch.Tensor, shape (N,)
        Predicted labels from the trained GCN.
    node_ids : list[int], optional
        Node ID labels.
    title : str
        Overall figure title.
    figsize : tuple[int, int]
        Matplotlib figure size.
    save_path : str or Path, optional
        Save path for the figure.
    seed : int
        Layout seed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    N = int(A.shape[0])
    if node_ids is None:
        node_ids = list(range(N))

    G = _adjacency_to_networkx(A)
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.spring_layout(G, seed=seed)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    # --- BEFORE GCN ---
    colors_true = _color_list(true_labels)
    nx.draw_networkx_edges(G, pos, ax=ax_before, alpha=0.4, edge_color="#888888")
    nx.draw_networkx_nodes(
        G, pos, ax=ax_before, node_color=colors_true, node_size=700,
        linewidths=1.5, edgecolors="#333333",
    )
    nx.draw_networkx_labels(
        G, pos, labels={i: str(node_ids[i]) for i in range(N)},
        ax=ax_before, font_size=11, font_color="white", font_weight="bold",
    )
    ax_before.set_title("BEFORE GCN\n(True Labels)", fontsize=13, fontweight="bold")
    ax_before.axis("off")

    # --- AFTER GCN ---
    colors_pred = _color_list(pred_labels)
    correct = (true_labels == pred_labels).numpy()
    edge_colors = ["#333333" if c else "#FF0000" for c in correct]
    linewidths_list = [1.5 if c else 3.5 for c in correct]

    nx.draw_networkx_edges(G, pos, ax=ax_after, alpha=0.4, edge_color="#888888")
    nx.draw_networkx_nodes(
        G, pos, ax=ax_after, node_color=colors_pred, node_size=700,
        linewidths=linewidths_list, edgecolors=edge_colors,
    )
    nx.draw_networkx_labels(
        G, pos, labels={i: str(node_ids[i]) for i in range(N)},
        ax=ax_after, font_size=11, font_color="white", font_weight="bold",
    )

    # Annotations for correct/incorrect
    n_correct = int(correct.sum())
    n_total = N
    ax_after.set_title(
        f"AFTER GCN\n(Predictions) — {n_correct}/{n_total} correct",
        fontsize=13, fontweight="bold",
    )
    ax_after.axis("off")

    # Legend
    unique_classes = sorted(set(int(l) for l in true_labels))
    handles = [
        matplotlib.patches.Patch(
            facecolor=_CLASS_COLORS[c % len(_CLASS_COLORS)],
            edgecolor="#333333",
            label=f"Class {c}",
        )
        for c in unique_classes
    ]
    handles += [
        matplotlib.patches.Patch(facecolor="white", edgecolor="#FF0000", linewidth=2,
                                 label="Incorrect prediction"),
    ]
    ax_after.legend(handles=handles, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Prediction figure saved to %s", save_path)

    return fig
