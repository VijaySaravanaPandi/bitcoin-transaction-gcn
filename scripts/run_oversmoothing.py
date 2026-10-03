#!/usr/bin/env python3
"""
scripts/run_oversmoothing.py — Over-smoothing depth sweep experiment.

Usage
-----
    python scripts/run_oversmoothing.py
    python scripts/run_oversmoothing.py --config configs/default.yaml
    python scripts/run_oversmoothing.py --depths 1 2 3 5 8 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vanilla_gcn.analysis.oversmoothing import plot_oversmoothing, run_oversmoothing_experiment
from vanilla_gcn.config import load_config
from vanilla_gcn.data.preprocessing import prepare_graph
from vanilla_gcn.data.synthetic import create_synthetic_graph
from vanilla_gcn.seed import set_seed
from vanilla_gcn.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GCN over-smoothing depth-sweep experiment."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 8, 10],
        help="GCN depths to test (e.g. --depths 1 2 3 5)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO", force=True)

    import logging
    logger = logging.getLogger("vanilla_gcn.oversmoothing_script")

    cfg = load_config(args.config)
    epochs = args.epochs or cfg.training.epochs
    set_seed(cfg.seed)

    data = create_synthetic_graph(seed=cfg.seed)

    logger.info("Running over-smoothing experiment with depths: %s", args.depths)
    result = run_oversmoothing_experiment(
        A=data.adjacency,
        X=data.features,
        y=data.labels,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        depths=args.depths,
        hidden_dim=cfg.model.hidden_dim,
        epochs=epochs,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        seed=cfg.seed,
    )

    print("\n" + result.summary())

    fig_path = Path(cfg.paths.figures) / "oversmoothing.png"
    Path(cfg.paths.figures).mkdir(parents=True, exist_ok=True)
    plot_oversmoothing(result, save_path=fig_path)
    logger.info("Over-smoothing figure saved to %s", fig_path)


if __name__ == "__main__":
    main()
