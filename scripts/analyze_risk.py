#!/usr/bin/env python3
"""Rank fraud risk and analyze learned Elliptic++ embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vanilla_gcn.analysis.risk import (  # noqa: E402
    cluster_embeddings,
    expected_calibration_error,
    risk_ranking,
)
from vanilla_gcn.config import load_config  # noqa: E402
from vanilla_gcn.data.loader import load_elliptic_transactions  # noqa: E402
from vanilla_gcn.data.preprocessing import prepare_graph  # noqa: E402
from vanilla_gcn.models.vanilla_gcn import VanillaGCN  # noqa: E402
from vanilla_gcn.seed import set_seed  # noqa: E402
from vanilla_gcn.utils.checkpointing import load_checkpoint  # noqa: E402
from vanilla_gcn.visualization.analysis import (  # noqa: E402
    plot_calibration_curve,
    plot_cluster_summary,
    plot_confusion_matrix,
    plot_risk_ranking,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/vanilla_gcn.pt")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    data = load_elliptic_transactions(
        raw_dir=cfg.data.raw_dir,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
        split_strategy=cfg.data.split_strategy,
        temporal_train_end=cfg.data.temporal_train_end,
        temporal_validation_end=cfg.data.temporal_validation_end,
        add_graph_features=cfg.data.add_graph_features,
    )
    _, A_tilde, X, y = prepare_graph(data.adjacency, data.features, data.labels)
    model = VanillaGCN(
        input_dim=data.num_features,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=data.num_classes,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )
    model, _, _, _, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    with torch.no_grad():
        outputs = model(X, A_tilde, return_intermediate=True)
        logits = outputs["logits"]
        hidden_keys = [key for key in outputs.keys() if key not in {"input", "logits"}]
        embeddings = outputs[hidden_keys[-1]] if hidden_keys else outputs["input"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = data.labels.clone()
    if data.labelled_mask is not None:
        labels[~data.labelled_mask] = -1

    ranking = risk_ranking(logits, data.node_ids, labels, top_k=args.top_k)
    ranking.to_csv(output_dir / "risk_ranking.csv", index=False)
    plot_risk_ranking(ranking, save_path=output_dir / "risk_ranking.png")

    test_mask = data.test_mask.cpu().numpy()
    probabilities = torch.softmax(logits, dim=1)[:, 0].detach().cpu().numpy()
    labelled_test = labels.cpu().numpy()[test_mask]
    ece = expected_calibration_error(probabilities[test_mask], labelled_test)
    plot_calibration_curve(
        probabilities[test_mask], labelled_test,
        save_path=output_dir / "calibration_curve.png",
    )
    plot_confusion_matrix(
        logits, data.labels, data.test_mask,
        save_path=output_dir / "test_confusion_matrix.png",
    )

    assignments, cluster_summary, cluster_metrics = cluster_embeddings(
        embeddings, labels, n_clusters=args.clusters, seed=cfg.seed
    )
    cluster_summary.to_csv(output_dir / "embedding_clusters.csv", index=False)
    np.save(output_dir / "cluster_assignments.npy", assignments)
    plot_cluster_summary(
        cluster_summary, save_path=output_dir / "embedding_clusters.png"
    )

    print(f"Saved risk ranking: {output_dir / 'risk_ranking.csv'}")
    print(f"Saved cluster summary: {output_dir / 'embedding_clusters.csv'}")
    print(f"Test expected calibration error: {ece:.4f}")
    print("Cluster metrics:", cluster_metrics)


if __name__ == "__main__":
    main()
