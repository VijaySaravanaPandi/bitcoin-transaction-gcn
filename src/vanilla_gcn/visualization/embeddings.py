"""
embeddings.py — Node embedding visualization using PCA.

Reduces high-dimensional node representations to 2D using PCA from
scikit-learn and visualizes node positions coloured by class label.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

_CLASS_COLORS = [
    "#E63946", "#2A9D8F", "#F4A261",
    "#457B9D", "#A8DADC", "#6A0572",
]


def _to_2d(embeddings: torch.Tensor) -> np.ndarray:
    """Reduce embeddings to 2D via PCA (or return as-is if already 2D)."""
    Z = embeddings.detach().cpu().numpy()
    n, d = Z.shape
    if d == 1:
        Z = np.hstack([Z, np.zeros((n, 1))])
    elif d > 2:
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(Z)
        explained = pca.explained_variance_ratio_.sum() * 100
        logger.debug("PCA explained variance: %.1f%%", explained)
    return Z


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    node_ids: Optional[list[int]] = None,
    *,
    title: str = "Node Embeddings (PCA)",
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str | Path] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    annotate: bool = True,
) -> matplotlib.figure.Figure:
    """Visualize node embeddings in 2D (reduced by PCA if needed).

    Parameters
    ----------
    embeddings : torch.Tensor, shape (N, D)
        Node embedding matrix.
    labels : torch.Tensor, shape (N,)
        True integer class labels.
    node_ids : list[int], optional
        Node ID annotations.  Default: ``[0, …, N-1]``.
    title : str
        Plot title.
    figsize : tuple[int, int]
        Matplotlib figure size.
    save_path : str or Path, optional
        If provided, save the figure here.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    annotate : bool
        If ``True``, annotate each point with its node ID.

    Returns
    -------
    matplotlib.figure.Figure
    """
    N = embeddings.shape[0]
    if node_ids is None:
        node_ids = list(range(N))

    Z = _to_2d(embeddings)
    label_np = labels.cpu().numpy()
    unique_classes = sorted(set(label_np))

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for c in unique_classes:
        mask = label_np == c
        color = _CLASS_COLORS[c % len(_CLASS_COLORS)]
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=color, s=150, label=f"Class {c}",
            edgecolors="#333333", linewidths=1.0, zorder=3, alpha=0.9,
        )

    if annotate:
        for i in range(N):
            ax.annotate(
                str(node_ids[i]),
                (Z[i, 0], Z[i, 1]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color="#333333",
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("PC 1" if embeddings.shape[1] > 2 else "Dim 1")
    ax.set_ylabel("PC 2" if embeddings.shape[1] > 2 else "Dim 2")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    if standalone:
        plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Embedding figure saved to %s", save_path)

    return fig


def visualize_training_curves(
    train_loss: list[float],
    val_loss: list[float],
    train_acc: list[float],
    val_acc: list[float],
    epochs: list[int],
    *,
    figsize: tuple[int, int] = (14, 5),
    save_path: Optional[str | Path] = None,
) -> matplotlib.figure.Figure:
    """Plot training and validation loss + accuracy curves.

    Parameters
    ----------
    train_loss, val_loss : list[float]
        Per-epoch loss values.
    train_acc, val_acc : list[float]
        Per-epoch accuracy values (0–100%).
    epochs : list[int]
        Epoch indices.
    figsize : tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("GCN Training Curves", fontsize=15, fontweight="bold")

    # Loss
    ax_loss.plot(epochs, train_loss, label="Train Loss", color="#E63946", linewidth=2)
    ax_loss.plot(epochs, val_loss, label="Val Loss", color="#2A9D8F", linewidth=2, linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3, linestyle="--")

    # Accuracy
    ax_acc.plot(epochs, train_acc, label="Train Acc", color="#E63946", linewidth=2)
    ax_acc.plot(epochs, val_acc, label="Val Acc", color="#2A9D8F", linewidth=2, linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(0, 105)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Training curves saved to %s", save_path)

    return fig
