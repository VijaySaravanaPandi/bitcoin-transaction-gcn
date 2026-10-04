#!/usr/bin/env python3
"""
scripts/train.py — Command-line training script for Vanilla GCN.

Usage
-----
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --seed 0

This script:
1. Loads configuration from YAML.
2. Creates the synthetic graph (replaceable with real data).
3. Preprocesses the graph (self-loops → degree matrix → symmetric normalization).
4. Builds and trains the VanillaGCN model.
5. Evaluates on train/val/test splits.
6. Saves a checkpoint to outputs/checkpoints/.
7. Saves training curve figures to outputs/figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim

from vanilla_gcn.config import load_config
from vanilla_gcn.data.loader import load_elliptic_actors, load_elliptic_transactions
from vanilla_gcn.data.preprocessing import prepare_graph
from vanilla_gcn.data.synthetic import create_synthetic_graph
from vanilla_gcn.models.vanilla_gcn import VanillaGCN
from vanilla_gcn.seed import set_seed
from vanilla_gcn.training.evaluation import evaluate_gcn
from vanilla_gcn.training.trainer import train_gcn
from vanilla_gcn.utils.checkpointing import save_checkpoint
from vanilla_gcn.utils.logging import setup_logging
from vanilla_gcn.visualization.embeddings import visualize_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Vanilla GCN on the synthetic graph dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in the config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(
        level=log_level,
        log_file=Path(cfg.paths.logs) / "train.log",
        force=True,
    )

    import logging
    logger = logging.getLogger("vanilla_gcn.train_script")
    logger.info("=" * 60)
    logger.info("Vanilla GCN Training")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info("Seed: %d", cfg.seed)
    logger.info("Epochs: %d", cfg.training.epochs)

    # ------------------------------------------------------------------
    # 2. Reproducibility
    # ------------------------------------------------------------------
    set_seed(cfg.seed)

    # ------------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------------
    dataset_name = getattr(cfg.data, "dataset", "elliptic")
    if dataset_name == "elliptic":
        logger.info("Loading Elliptic++ Transactions dataset from '%s' …", cfg.data.raw_dir)
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
    elif dataset_name == "elliptic_actors":
        logger.info("Loading Elliptic++ Actors dataset from '%s' …", cfg.data.raw_dir)
        data = load_elliptic_actors(
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
    else:
        logger.info("Loading synthetic graph …")
        data = create_synthetic_graph(
            seed=cfg.seed,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.validation_ratio,
            test_ratio=cfg.data.test_ratio,
        )
    logger.info("%s", data.summary())

    class_weights = None
    if cfg.training.class_weighted_loss:
        counts = torch.bincount(
            data.labels[data.train_mask], minlength=data.num_classes
        ).float()
        class_weights = counts.sum() / (data.num_classes * counts.clamp_min(1.0))
        logger.info("Using class weights: %s", class_weights.tolist())

    # ------------------------------------------------------------------
    # 4. Preprocessing
    # ------------------------------------------------------------------
    logger.info("Preprocessing: self-loops → degree matrix → symmetric normalization")
    A_hat, A_tilde, X, y = prepare_graph(data.adjacency, data.features, data.labels)

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    logger.info("Building VanillaGCN...")
    model = VanillaGCN(
        input_dim=data.num_features,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=data.num_classes,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        debug=cfg.debug.enabled,
    )
    model.print_summary()

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    logger.info("Starting training...")
    history = train_gcn(
        model=model,
        A_tilde=A_tilde,
        X=X,
        y=y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        epochs=cfg.training.epochs,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        class_weights=class_weights,
    )

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    logger.info("Evaluating model...")
    metrics = evaluate_gcn(
        model, A_tilde, X, y,
        data.train_mask, data.val_mask, data.test_mask,
        class_weights=class_weights,
    )
    logger.info("%s", metrics.summary())

    # ------------------------------------------------------------------
    # 8. Save checkpoint
    # ------------------------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    ckpt_path = Path(cfg.paths.checkpoints) / "vanilla_gcn.pt"
    save_checkpoint(model, optimizer, cfg, history, epoch=cfg.training.epochs, path=ckpt_path)

    # ------------------------------------------------------------------
    # 9. Save training curves figure
    # ------------------------------------------------------------------
    fig_path = Path(cfg.paths.figures) / "training_curves.png"
    Path(cfg.paths.figures).mkdir(parents=True, exist_ok=True)
    visualize_training_curves(
        history.train_loss, history.val_loss,
        history.train_acc, history.val_acc,
        history.epochs,
        save_path=fig_path,
    )
    logger.info("Training curves saved to %s", fig_path)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
