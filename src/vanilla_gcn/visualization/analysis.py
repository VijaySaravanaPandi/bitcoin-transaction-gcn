"""Plots for fraud metrics, risk scores, clusters, and link analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _save(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_risk_ranking(
    ranking: pd.DataFrame,
    *,
    top_k: int = 25,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot illicit probability and uncertainty for the highest-risk nodes."""
    frame = ranking.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(frame["node_id"].astype(str), frame["illicit_probability"], color="#E63946")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted illicit probability")
    ax.set_ylabel("Node ID")
    ax.set_title("Highest-Risk Nodes")
    return _save(fig, save_path)


def plot_calibration_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot confidence versus observed illicit frequency."""
    target = (labels == 0).astype(np.int64)
    bins = np.linspace(0, 1, n_bins + 1)
    confidence, observed = [], []
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (probabilities >= lower) & (probabilities < upper)
        if upper == 1:
            selected |= probabilities == upper
        if selected.any():
            confidence.append(float(probabilities[selected].mean()))
            observed.append(float(target[selected].mean()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(confidence, observed, "o-", color="#E63946", label="Model")
    ax.set_xlabel("Mean predicted illicit probability")
    ax.set_ylabel("Observed illicit frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, save_path)


def plot_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a binary confusion matrix for a selected node mask."""
    predictions = logits.detach().argmax(dim=1).cpu().numpy()[mask.cpu().numpy()]
    truth = labels.detach().cpu().numpy()[mask.cpu().numpy()]
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (truth, predictions), 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Reds")
    fig.colorbar(image, ax=ax)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Illicit", "Licit"])
    ax.set_yticks([0, 1], labels=["Illicit", "Licit"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")
    return _save(fig, save_path)


def plot_cluster_summary(
    cluster_summary: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot labelled-node illicit rate by embedding cluster."""
    frame = cluster_summary.sort_values("illicit_rate")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(frame["cluster"].astype(str), frame["illicit_rate"], color="#457B9D")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Embedding cluster")
    ax.set_ylabel("Illicit rate among labelled nodes")
    ax.set_title("Illicit Concentration by Embedding Cluster")
    return _save(fig, save_path)


def plot_link_metrics(
    metrics: dict[str, float],
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot scalar link-prediction metrics."""
    names = ["ROC-AUC", "Average Precision"]
    values = [metrics.get("roc_auc", 0.0), metrics.get("average_precision", 0.0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values, color=["#2A9D8F", "#F4A261"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Embedding Link Prediction")
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center")
    return _save(fig, save_path)
