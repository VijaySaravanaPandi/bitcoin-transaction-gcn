"""
trainer.py — Training loop for Vanilla GCN.

Provides a clean, reusable training function that works with any
:class:`~vanilla_gcn.data.types.GraphData` object — synthetic or real-world.

Usage
-----
    from vanilla_gcn.training.trainer import train_gcn

    history = train_gcn(
        model=model,
        A_tilde=A_tilde,
        X=data.features,
        y=data.labels,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        epochs=200,
        lr=0.01,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from vanilla_gcn.models.vanilla_gcn import VanillaGCN
from vanilla_gcn.training.evaluation import accuracy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training history container
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Records per-epoch metrics from a training run.

    Attributes
    ----------
    train_loss : list[float]
        Cross-entropy loss on training nodes at each epoch.
    val_loss : list[float]
        Cross-entropy loss on validation nodes at each epoch.
    train_acc : list[float]
        Node classification accuracy on training nodes (0–100%).
    val_acc : list[float]
        Node classification accuracy on validation nodes (0–100%).
    epochs : list[int]
        Epoch indices (1-based).
    best_val_acc : float
        Best validation accuracy observed during training.
    best_epoch : int
        Epoch at which best validation accuracy was achieved.
    total_time_s : float
        Total wall-clock training time in seconds.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    total_time_s: float = 0.0

    def summary(self) -> str:
        """Return a concise training summary string."""
        if not self.train_loss:
            return "TrainingHistory (empty)"
        final_epoch = self.epochs[-1]
        return (
            f"TrainingHistory over {final_epoch} epochs\n"
            f"  Final train loss : {self.train_loss[-1]:.4f}\n"
            f"  Final val   loss : {self.val_loss[-1]:.4f}\n"
            f"  Final train acc  : {self.train_acc[-1]:.1f}%\n"
            f"  Final val   acc  : {self.val_acc[-1]:.1f}%\n"
            f"  Best val    acc  : {self.best_val_acc:.1f}% (epoch {self.best_epoch})\n"
            f"  Total time       : {self.total_time_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train_gcn(
    model: VanillaGCN,
    A_tilde: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    log_every: int = 10,
) -> TrainingHistory:
    """Train a :class:`VanillaGCN` model using full-batch gradient descent.

    Uses:
        - Loss:      ``nn.CrossEntropyLoss``   (includes log-softmax)
        - Optimiser: ``torch.optim.Adam``

    The loss is computed ONLY on training nodes (``train_mask``).
    Validation metrics are computed without gradient tracking.

    Parameters
    ----------
    model : VanillaGCN
        Untrained (or partially trained) GCN model.
    A_tilde : torch.Tensor, shape (N, N)
        Symmetrically normalized adjacency matrix (pre-computed).
    X : torch.Tensor, shape (N, F)
        Node feature matrix.
    y : torch.Tensor, shape (N,)
        Integer class labels for all N nodes.
    train_mask : torch.Tensor, shape (N,)
        Boolean mask; ``True`` for training nodes.
    val_mask : torch.Tensor, shape (N,)
        Boolean mask; ``True`` for validation nodes.
    epochs : int
        Number of training epochs.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularisation coefficient (Adam ``weight_decay``).
    log_every : int
        Print training progress every this many epochs.

    Returns
    -------
    TrainingHistory
        Per-epoch training and validation metrics.

    Notes
    -----
    Full-batch training is used (as in the original GCN paper) because the
    synthetic graph is small.  For large graphs, mini-batch sampling would
    be needed.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = TrainingHistory()
    start_time = time.perf_counter()

    logger.info("Starting GCN training: epochs=%d, lr=%.4f, wd=%.6f", epochs, lr, weight_decay)

    for epoch in range(1, epochs + 1):
        # ----------------------------------------------------------------
        # Training step
        # ----------------------------------------------------------------
        model.train()
        optimizer.zero_grad()

        logits: torch.Tensor = model(X, A_tilde)  # (N, C)
        train_loss = criterion(logits[train_mask], y[train_mask])
        train_loss.backward()
        optimizer.step()

        # ----------------------------------------------------------------
        # Validation step (no gradient)
        # ----------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            logits_eval: torch.Tensor = model(X, A_tilde)
            val_loss_val = criterion(logits_eval[val_mask], y[val_mask])

            train_acc_val = accuracy(logits_eval, y, train_mask)
            val_acc_val = accuracy(logits_eval, y, val_mask)

        # ----------------------------------------------------------------
        # Record
        # ----------------------------------------------------------------
        history.train_loss.append(train_loss.detach().item())
        history.val_loss.append(float(val_loss_val))
        history.train_acc.append(float(train_acc_val))
        history.val_acc.append(float(val_acc_val))
        history.epochs.append(epoch)

        if val_acc_val > history.best_val_acc:
            history.best_val_acc = float(val_acc_val)
            history.best_epoch = epoch

        if epoch % log_every == 0 or epoch == 1:
            logger.info(
                "Epoch %03d | Train Loss: %.4f | Train Acc: %.1f%% | "
                "Val Loss: %.4f | Val Acc: %.1f%%",
                epoch,
                float(train_loss),
                float(train_acc_val),
                float(val_loss_val),
                float(val_acc_val),
            )

    history.total_time_s = time.perf_counter() - start_time
    logger.info("Training complete. %s", history.summary())
    return history
