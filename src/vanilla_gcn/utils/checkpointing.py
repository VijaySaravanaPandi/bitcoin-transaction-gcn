"""
checkpointing.py — Model checkpoint save/load utilities.

Saves and loads the full training state so experiments can be resumed
and models can be evaluated without retraining.

Checkpoint format (PyTorch .pt file)
--------------------------------------
{
    "epoch"              : int,
    "model_state_dict"   : dict,
    "optimizer_state_dict": dict,
    "config"             : dict,
    "history"            : {
        "train_loss"  : list[float],
        "val_loss"    : list[float],
        "train_acc"   : list[float],
        "val_acc"     : list[float],
        "best_val_acc": float,
        "best_epoch"  : int,
    },
}

Usage
-----
    from vanilla_gcn.utils.checkpointing import save_checkpoint, load_checkpoint

    save_checkpoint(model, optimizer, cfg, history, epoch=200,
                    path="outputs/checkpoints/model.pt")

    model, optimizer, loaded_cfg, loaded_history, epoch = load_checkpoint(
        "outputs/checkpoints/model.pt", model, optimizer
    )
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.optim as optim

from vanilla_gcn.config import GCNConfig
from vanilla_gcn.models.vanilla_gcn import VanillaGCN
from vanilla_gcn.training.trainer import TrainingHistory

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: VanillaGCN,
    optimizer: optim.Optimizer,
    config: GCNConfig,
    history: TrainingHistory,
    epoch: int,
    path: str | Path,
) -> None:
    """Save a complete training checkpoint to disk.

    Parameters
    ----------
    model : VanillaGCN
        Trained GCN model.
    optimizer : torch.optim.Optimizer
        Optimizer (saves momentum / adaptive state).
    config : GCNConfig
        Configuration used for this run.
    history : TrainingHistory
        Per-epoch training metrics.
    epoch : int
        Current epoch number (saved for resumption).
    path : str or Path
        Destination file path (e.g. ``"outputs/checkpoints/run.pt"``).

    Notes
    -----
    The checkpoint file is saved with ``torch.save`` which uses Python's
    ``pickle`` format.  Do not load checkpoints from untrusted sources.
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "num_classes": model.num_classes,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
            "seed": config.seed,
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
        },
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_acc": history.train_acc,
            "val_acc": history.val_acc,
            "best_val_acc": history.best_val_acc,
            "best_epoch": history.best_epoch,
        },
    }

    torch.save(checkpoint, ckpt_path)
    logger.info(
        "Checkpoint saved: epoch=%d, val_acc=%.1f%%, path=%s",
        epoch, history.best_val_acc, ckpt_path,
    )


def load_checkpoint(
    path: str | Path,
    model: VanillaGCN,
    optimizer: Optional[optim.Optimizer] = None,
) -> tuple[VanillaGCN, Optional[optim.Optimizer], dict[str, Any], TrainingHistory, int]:
    """Load a training checkpoint from disk.

    Parameters
    ----------
    path : str or Path
        Path to the ``.pt`` checkpoint file.
    model : VanillaGCN
        Model instance (must match the architecture saved in the checkpoint).
    optimizer : torch.optim.Optimizer, optional
        Optimizer to restore state into.  Pass ``None`` for inference-only.

    Returns
    -------
    model : VanillaGCN
        Model with loaded weights.
    optimizer : torch.optim.Optimizer or None
        Optimizer with restored state (or ``None``).
    config : dict
        Configuration dict stored in the checkpoint.
    history : TrainingHistory
        Restored training history.
    epoch : int
        Epoch at which the checkpoint was saved.

    Examples
    --------
    >>> model, opt, cfg, hist, epoch = load_checkpoint("outputs/checkpoints/run.pt", model, opt)
    >>> print(f"Loaded checkpoint from epoch {epoch}")
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model weights loaded from %s", ckpt_path)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored")

    hist_data = checkpoint.get("history", {})
    history = TrainingHistory(
        train_loss=hist_data.get("train_loss", []),
        val_loss=hist_data.get("val_loss", []),
        train_acc=hist_data.get("train_acc", []),
        val_acc=hist_data.get("val_acc", []),
        best_val_acc=hist_data.get("best_val_acc", 0.0),
        best_epoch=hist_data.get("best_epoch", 0),
    )

    epoch = int(checkpoint.get("epoch", 0))
    config = checkpoint.get("config", {})

    logger.info(
        "Checkpoint loaded: epoch=%d, best_val_acc=%.1f%%",
        epoch, history.best_val_acc,
    )
    return model, optimizer, config, history, epoch
