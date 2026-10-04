"""Fraud-risk ranking, calibration, and embedding cluster analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def risk_ranking(
    logits: torch.Tensor,
    node_ids: list[Any],
    labels: torch.Tensor | None = None,
    top_k: int = 100,
) -> pd.DataFrame:
    """Rank nodes by predicted illicit probability and uncertainty."""
    probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
    illicit_probability = probabilities[:, 0]
    predictions = probabilities.argmax(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    frame = pd.DataFrame(
        {
            "node_id": node_ids,
            "illicit_probability": illicit_probability,
            "predicted_class": predictions,
            "entropy": entropy,
        }
    )
    if labels is not None:
        frame["true_class"] = labels.detach().cpu().numpy()
    return frame.sort_values(
        ["illicit_probability", "entropy"], ascending=[False, False]
    ).head(top_k).reset_index(drop=True)


def expected_calibration_error(
    illicit_probability: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE for class-0 illicit probabilities."""
    target = (labels == 0).astype(np.int64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (illicit_probability >= lower) & (illicit_probability < upper)
        if upper == 1.0:
            selected |= illicit_probability == upper
        if selected.any():
            confidence = illicit_probability[selected].mean()
            accuracy = target[selected].mean()
            ece += selected.mean() * abs(confidence - accuracy)
    return float(ece)


def cluster_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_clusters: int = 8,
    sample_size: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    """Cluster hidden embeddings and summarize illicit concentration."""
    matrix = embeddings.detach().cpu().numpy().astype(np.float32)
    true_labels = labels.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(matrix), size=min(sample_size, len(matrix)), replace=False)
    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, batch_size=2048, n_init=3
    )
    clusterer.fit(matrix[sample])
    assignments = clusterer.predict(matrix)

    rows = []
    for cluster_id in range(n_clusters):
        selected = assignments == cluster_id
        labelled = selected & (true_labels >= 0)
        illicit = labelled & (true_labels == 0)
        rows.append(
            {
                "cluster": cluster_id,
                "nodes": int(selected.sum()),
                "labelled_nodes": int(labelled.sum()),
                "illicit_nodes": int(illicit.sum()),
                "illicit_rate": float(illicit.sum() / max(labelled.sum(), 1)),
            }
        )

    sample_assignments = assignments[sample]
    metrics = {
        "silhouette": float(
            silhouette_score(matrix[sample], sample_assignments)
            if len(np.unique(sample_assignments)) > 1 else float("nan")
        ),
        "adjusted_rand_index": float(
            adjusted_rand_score(true_labels[sample], sample_assignments)
        ),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(true_labels[sample], sample_assignments)
        ),
    }
    return assignments, pd.DataFrame(rows), metrics
