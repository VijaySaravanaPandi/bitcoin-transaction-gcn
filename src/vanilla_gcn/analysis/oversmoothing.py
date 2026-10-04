"""
oversmoothing.py — Over-smoothing analysis for Vanilla GCN.

Over-smoothing is the phenomenon where stacking too many GCN layers causes
all node representations to converge to the same value, making them
indistinguishable regardless of the input features.

This module:
1. Trains GCN models of varying depths (1, 2, 3, 5, 8, 10 layers).
2. Measures the cosine similarity between node embeddings as depth increases.
3. Plots the similarity curve to illustrate the over-smoothing effect.

Reference:
    Li et al. (2018) "Deeper Insights into Graph Convolutional Networks
    for Semi-Supervised Classification"
    Hamilton (2020) "Graph Representation Learning", Chapter 7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from vanilla_gcn.data.preprocessing import prepare_graph
from vanilla_gcn.models.vanilla_gcn import VanillaGCN
from vanilla_gcn.seed import set_seed
from vanilla_gcn.training.trainer import train_gcn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Similarity metric
# ---------------------------------------------------------------------------


def mean_pairwise_cosine_similarity(
    Z: torch.Tensor,
    *,
    max_pairs: int = 100_000,
    seed: int = 42,
) -> float:
    """Estimate mean pairwise cosine similarity from sampled node pairs.

    A value near 1.0 indicates over-smoothing (all embeddings are similar).
    A lower value indicates diverse node representations.

    Parameters
    ----------
    Z : torch.Tensor, shape (N, D)
        Node embedding matrix.

    Returns
    -------
    float
        Mean pairwise cosine similarity, in the range [-1, 1].
    """
    if Z.shape[0] < 2:
        return 1.0
    Z_norm = F.normalize(Z, p=2, dim=1)  # (N, D)
    N = Z.shape[0]
    pair_count = min(max_pairs, N * (N - 1) // 2)
    generator = torch.Generator(device=Z.device).manual_seed(seed)
    src = torch.randint(N, (pair_count,), generator=generator, device=Z.device)
    dst = torch.randint(N, (pair_count,), generator=generator, device=Z.device)
    distinct = src != dst
    if not distinct.any():
        return 1.0
    return float((Z_norm[src[distinct]] * Z_norm[dst[distinct]]).sum(dim=1).mean())


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass
class OverSmoothingResult:
    """Results from an over-smoothing depth sweep.

    Attributes
    ----------
    depths : list[int]
        GCN layer depths tested.
    similarities : list[float]
        Mean pairwise cosine similarity at each depth.
    train_accs : list[float]
        Final training accuracy at each depth.
    val_accs : list[float]
        Final validation accuracy at each depth.
    """

    depths: list[int] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Over-Smoothing Experiment Results", "-" * 40]
        lines.append(f"{'Depth':>6} | {'Similarity':>10} | {'Train Acc':>10} | {'Val Acc':>8}")
        lines.append("-" * 40)
        for d, s, ta, va in zip(self.depths, self.similarities, self.train_accs, self.val_accs):
            lines.append(f"{d:>6} | {s:>10.4f} | {ta:>9.1f}% | {va:>7.1f}%")
        return "\n".join(lines)


def run_oversmoothing_experiment(
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    depths: Optional[list[int]] = None,
    hidden_dim: int = 16,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    seed: int = 42,
) -> OverSmoothingResult:
    """Train GCNs of different depths and measure embedding similarity.

    Parameters
    ----------
    A : torch.Tensor, shape (N, N)
        Raw adjacency matrix.
    X : torch.Tensor, shape (N, F)
        Node features.
    y : torch.Tensor, shape (N,)
        Labels.
    train_mask, val_mask : torch.Tensor
        Split masks.
    depths : list[int], optional
        Layer depths to test.  Default: ``[1, 2, 3, 5, 8, 10]``.
    hidden_dim : int
        Hidden layer size for all models.
    epochs : int
        Training epochs per model.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularisation.
    seed : int
        Reproducibility seed.

    Returns
    -------
    OverSmoothingResult
        Similarity and accuracy at each depth.
    """
    if depths is None:
        depths = [1, 2, 3, 5, 8, 10]

    num_classes = int(y.max().item()) + 1
    input_dim = int(X.shape[1])

    _, A_tilde, X, y = prepare_graph(A, X, y)

    result = OverSmoothingResult()

    for depth in depths:
        logger.info("Over-smoothing experiment: depth=%d", depth)
        set_seed(seed)

        model = VanillaGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=depth,
            debug=False,
        )

        history = train_gcn(
            model, A_tilde, X, y, train_mask, val_mask,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            log_every=epochs,  # only log at end
        )

        # Extract final embeddings (before logits layer)
        model.eval()
        with torch.no_grad():
            out = model(X, A_tilde, return_intermediate=True)
            keys = out.keys()
            # Get the penultimate layer (last hidden, not logits)
            hidden_keys = [k for k in keys if k != "logits" and k != "input"]
            if hidden_keys:
                Z = out[hidden_keys[-1]]
            else:
                Z = out["input"]

        sim = mean_pairwise_cosine_similarity(Z)

        result.depths.append(depth)
        result.similarities.append(sim)
        result.train_accs.append(history.train_acc[-1])
        result.val_accs.append(history.val_acc[-1])

        logger.info(
            "  depth=%d | similarity=%.4f | train_acc=%.1f%% | val_acc=%.1f%%",
            depth, sim, history.train_acc[-1], history.val_acc[-1],
        )

    logger.info("\n%s", result.summary())
    return result


def plot_oversmoothing(
    result: OverSmoothingResult,
    *,
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[str | Path] = None,
) -> matplotlib.figure.Figure:
    """Plot the over-smoothing experiment results.

    Parameters
    ----------
    result : OverSmoothingResult
        Output from :func:`run_oversmoothing_experiment`.
    figsize : tuple[int, int]
        Figure size.
    save_path : str or Path, optional
        If provided, save the figure here.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_sim, ax_acc) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        "GCN Over-Smoothing Analysis\n"
        "As depth increases, node embeddings converge → classification degrades",
        fontsize=13, fontweight="bold",
    )

    # Similarity
    ax_sim.plot(
        result.depths, result.similarities,
        marker="o", color="#E63946", linewidth=2.5, markersize=8, zorder=3,
    )
    ax_sim.set_xlabel("Number of GCN Layers (K)", fontsize=11)
    ax_sim.set_ylabel("Mean Pairwise Cosine Similarity", fontsize=11)
    ax_sim.set_title("Embedding Similarity vs Depth\n(higher = more over-smoothed)", fontsize=11)
    ax_sim.set_ylim(0, 1.05)
    ax_sim.axhline(y=1.0, color="#888888", linestyle="--", alpha=0.5, label="Perfect over-smoothing")
    ax_sim.legend(fontsize=9)
    ax_sim.grid(True, alpha=0.3, linestyle="--")
    for d, s in zip(result.depths, result.similarities):
        ax_sim.annotate(f"{s:.2f}", (d, s), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)

    # Accuracy
    ax_acc.plot(
        result.depths, result.train_accs,
        marker="s", color="#2A9D8F", linewidth=2.5, markersize=8, label="Train Acc", zorder=3,
    )
    ax_acc.plot(
        result.depths, result.val_accs,
        marker="^", color="#F4A261", linewidth=2.5, markersize=8,
        linestyle="--", label="Val Acc", zorder=3,
    )
    ax_acc.set_xlabel("Number of GCN Layers (K)", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_title("Accuracy vs Depth\n(may degrade with excessive depth)", fontsize=11)
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Over-smoothing figure saved to %s", save_path)

    return fig
