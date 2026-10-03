"""
layers.py — Layer-by-layer embedding visualization for Vanilla GCN.

Shows how node representations evolve from the raw input features X
through each GCN layer to the final logits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import torch

from vanilla_gcn.visualization.embeddings import _to_2d, _CLASS_COLORS

logger = logging.getLogger(__name__)


def visualize_layer_embeddings(
    intermediate: dict[str, torch.Tensor],
    labels: torch.Tensor,
    node_ids: Optional[list[int]] = None,
    *,
    figsize_per_panel: tuple[int, int] = (5, 4),
    save_path: Optional[str | Path] = None,
    annotate: bool = True,
) -> matplotlib.figure.Figure:
    """Visualize node embeddings at each GCN layer side-by-side.

    Shows X → H^(1) → H^(2) → ... → H^(K) in a multi-panel plot.
    PCA is applied to reduce each embedding to 2D for visualization.

    Parameters
    ----------
    intermediate : dict[str, torch.Tensor]
        Dictionary mapping layer names to embedding tensors.
        Keys: ``'input'``, ``'layer_1'``, ..., ``'logits'``.
        Produced by ``model(X, A_tilde, return_intermediate=True)``.
    labels : torch.Tensor, shape (N,)
        True class labels.
    node_ids : list[int], optional
        Node ID labels.
    figsize_per_panel : tuple[int, int]
        Width and height of each subplot panel.
    save_path : str or Path, optional
        If provided, save the figure here.
    annotate : bool
        Annotate nodes with their IDs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    keys = list(intermediate.keys())
    n_panels = len(keys)
    N = labels.shape[0]
    if node_ids is None:
        node_ids = list(range(N))

    label_np = labels.cpu().numpy()
    unique_classes = sorted(set(label_np))

    fig_w = figsize_per_panel[0] * n_panels
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        "Layer-by-Layer Node Representations\n"
        r"$X \rightarrow H^{(1)} \rightarrow H^{(2)} \rightarrow \cdots \rightarrow \text{logits}$",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for idx, (key, ax) in enumerate(zip(keys, axes)):
        emb = intermediate[key]
        Z = _to_2d(emb)

        for c in unique_classes:
            mask = label_np == c
            color = _CLASS_COLORS[c % len(_CLASS_COLORS)]
            ax.scatter(
                Z[mask, 0], Z[mask, 1],
                c=color, s=120, label=f"Class {c}",
                edgecolors="#333333", linewidths=0.8, alpha=0.9, zorder=3,
            )

        if annotate:
            for i in range(N):
                ax.annotate(
                    str(node_ids[i]),
                    (Z[i, 0], Z[i, 1]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color="#333333",
                )

        # Title for each panel
        if key == "input":
            panel_title = r"$H^{(0)} = X$" + f"\n(raw features, dim={emb.shape[1]})"
        elif key == "logits":
            panel_title = f"Logits\n(output, dim={emb.shape[1]})"
        else:
            layer_num = key.replace("layer_", "")
            panel_title = rf"$H^{{({layer_num})}}$" + f"\n(dim={emb.shape[1]})"

        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("PC 1" if emb.shape[1] > 2 else "Dim 1", fontsize=9)
        ax.set_ylabel("PC 2" if emb.shape[1] > 2 else "Dim 2", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=8, loc="best", framealpha=0.8)

        # Arrow between panels
        if idx < n_panels - 1:
            fig.text(
                (idx + 1) / n_panels - 0.01, 0.5, "→",
                ha="center", va="center",
                fontsize=20, transform=fig.transFigure,
            )

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Layer-by-layer figure saved to %s", save_path)

    return fig
