#!/usr/bin/env python3
"""
scripts/evaluate.py — Evaluate a saved Vanilla GCN checkpoint.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint outputs/checkpoints/vanilla_gcn.pt
    python scripts/evaluate.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vanilla_gcn.config import load_config
from vanilla_gcn.data.loader import load_elliptic_transactions
from vanilla_gcn.data.preprocessing import prepare_graph
from vanilla_gcn.models.vanilla_gcn import VanillaGCN
from vanilla_gcn.seed import set_seed
from vanilla_gcn.training.evaluation import evaluate_gcn
from vanilla_gcn.utils.checkpointing import load_checkpoint
from vanilla_gcn.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Vanilla GCN checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/vanilla_gcn.pt",
        help="Path to the .pt checkpoint file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO", force=True)

    import logging
    logger = logging.getLogger("vanilla_gcn.evaluate_script")

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    # Data
    dataset_name = getattr(cfg.data, "dataset", "elliptic")
    if dataset_name != "elliptic":
        raise ValueError(
            f"Evaluation currently supports the Elliptic++ transaction dataset, got {dataset_name!r}"
        )
    data = load_elliptic_transactions(
        raw_dir=cfg.data.raw_dir,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
    )
    _, A_tilde, X, y = prepare_graph(data.adjacency, data.features, data.labels)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        logger.error("Run 'python scripts/train.py' first to create a checkpoint.")
        sys.exit(1)

    model = VanillaGCN(
        input_dim=data.num_features,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=data.num_classes,
        num_layers=cfg.model.num_layers,
    )
    model, _, saved_cfg, history, epoch = load_checkpoint(ckpt_path, model)
    logger.info("Loaded checkpoint from epoch %d", epoch)

    # Evaluate
    metrics = evaluate_gcn(model, A_tilde, X, y,
                           data.train_mask, data.val_mask, data.test_mask)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(metrics.summary())
    print("=" * 50)


if __name__ == "__main__":
    main()
