"""Embedding-based link prediction utilities."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def sample_negative_edges(
    num_nodes: int,
    positive_edges: torch.Tensor,
    *,
    count: int,
    seed: int = 42,
) -> torch.Tensor:
    """Sample non-edge pairs with deterministic rejection sampling."""
    rng = np.random.default_rng(seed)
    positive = {(int(src), int(dst)) for src, dst in positive_edges.T.tolist()}
    negatives: list[tuple[int, int]] = []
    while len(negatives) < count:
        src = int(rng.integers(0, num_nodes))
        dst = int(rng.integers(0, num_nodes))
        if src != dst and (src, dst) not in positive:
            positive.add((src, dst))
            negatives.append((src, dst))
    return torch.tensor(negatives, dtype=torch.long).T


def evaluate_link_prediction(
    embeddings: torch.Tensor,
    positive_edges: torch.Tensor,
    *,
    negative_count: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Score observed versus sampled edges using embedding dot products."""
    embeddings = torch.nn.functional.normalize(embeddings.detach(), dim=1)
    positive_edges = positive_edges[:, positive_edges[0] != positive_edges[1]]
    if negative_count is None:
        negative_count = positive_edges.shape[1]
    negative_edges = sample_negative_edges(
        embeddings.shape[0], positive_edges, count=negative_count, seed=seed
    )

    def score(edges: torch.Tensor) -> np.ndarray:
        return (embeddings[edges[0]] * embeddings[edges[1]]).sum(dim=1).cpu().numpy()

    scores = np.concatenate([score(positive_edges), score(negative_edges)])
    labels = np.concatenate([
        np.ones(positive_edges.shape[1], dtype=np.int64),
        np.zeros(negative_edges.shape[1], dtype=np.int64),
    ])
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "positive_edges": float(positive_edges.shape[1]),
        "negative_edges": float(negative_edges.shape[1]),
    }
