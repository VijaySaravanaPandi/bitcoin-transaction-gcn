#!/usr/bin/env python3
"""Evaluate whether learned embeddings recover observed graph links."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vanilla_gcn.analysis.link_prediction import evaluate_link_prediction  # noqa: E402
from vanilla_gcn.config import load_config  # noqa: E402
from vanilla_gcn.data.loader import load_elliptic_actors, load_elliptic_transactions  # noqa: E402
from vanilla_gcn.data.preprocessing import prepare_graph  # noqa: E402
from vanilla_gcn.models.vanilla_gcn import VanillaGCN  # noqa: E402
from vanilla_gcn.seed import set_seed  # noqa: E402
from vanilla_gcn.utils.checkpointing import load_checkpoint  # noqa: E402
from vanilla_gcn.visualization.analysis import plot_link_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/vanilla_gcn.pt")
    parser.add_argument("--negative-count", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    loader = load_elliptic_actors if cfg.data.dataset == "elliptic_actors" else load_elliptic_transactions
    data = loader(
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
    _, A_tilde, X, _ = prepare_graph(data.adjacency, data.features, data.labels)
    model = VanillaGCN(
        input_dim=data.num_features,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=data.num_classes,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )
    model, _, _, _, _ = load_checkpoint(args.checkpoint, model)
    embeddings = model.get_embeddings(X, A_tilde)
    hidden_keys = [key for key in embeddings if key not in {"input", "logits"}]
    hidden = embeddings[hidden_keys[-1]] if hidden_keys else embeddings["input"]
    metrics = evaluate_link_prediction(
        hidden,
        data.edge_index,
        negative_count=args.negative_count,
        seed=cfg.seed,
    )
    output_dir = Path(cfg.paths.figures)
    plot_link_metrics(metrics, save_path=output_dir / "link_prediction.png")
    print("Link prediction metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
