"""
trainer.py — Training loop for NumPy GCN.

Uses manual gradient computation + SGD-style parameter update.
Loss: cross-entropy on training nodes only (semi-supervised setting).

The softmax / cross-entropy backward:
    probs         = softmax(logits)
    loss          = -mean( log(probs[train, y[train]]) )
    dL/dlogits[i] = (probs[i] - one_hot(y[i])) / n_train   for i ∈ train
    dL/dlogits[i] = 0                                        for i ∉ train
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from vanilla_gcn_numpy.vanilla_gcn import VanillaGCNNumPy


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _softmax(Z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    Z_shifted = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def _cross_entropy(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Cross-entropy loss over masked nodes."""
    n = mask.sum()
    if n == 0:
        return 0.0
    log_p = np.log(np.clip(probs[mask, y[mask]], 1e-12, 1.0))
    return float(-log_p.mean())


def _cross_entropy_backward(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Gradient of cross-entropy w.r.t. logits.

    Only training nodes (mask=True) contribute gradient.
    """
    N, C = probs.shape
    dL_dlogits = np.zeros((N, C), dtype=np.float64)
    n = mask.sum()
    if n == 0:
        return dL_dlogits
    one_hot = np.eye(C, dtype=np.float64)[y[mask]]      # (n_train, C)
    dL_dlogits[mask] = (probs[mask] - one_hot) / n
    return dL_dlogits


# ---------------------------------------------------------------------------
# History dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistoryNumPy:
    """Per-epoch training metrics (NumPy version)."""
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    train_acc:  list[float] = field(default_factory=list)
    val_acc:    list[float] = field(default_factory=list)
    epochs:     list[int]   = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch:   int   = 0

    def summary(self) -> str:
        return (
            f"TrainingHistoryNumPy | epochs={len(self.epochs)} | "
            f"best_val_acc={self.best_val_acc:.1f}% @ epoch {self.best_epoch}"
        )


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def _accuracy(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    n = mask.sum()
    if n == 0:
        return 0.0
    preds = probs[mask].argmax(axis=1)
    return float((preds == y[mask]).mean() * 100.0)


def train_gcn_numpy(
    model: VanillaGCNNumPy,
    A_tilde: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    log_every: int = 50,
) -> TrainingHistoryNumPy:
    """Train a VanillaGCNNumPy model.

    Parameters
    ----------
    model       : VanillaGCNNumPy
    A_tilde     : (N, N)  symmetrically normalised adjacency
    X           : (N, F)  node features
    y           : (N,)    integer labels
    train_mask  : (N,)    boolean mask for training nodes
    val_mask    : (N,)    boolean mask for validation nodes
    epochs      : int     number of full-batch epochs
    lr          : float   learning rate
    weight_decay: float   L2 regularisation coefficient
    log_every   : int     print interval (0 = silent)

    Returns
    -------
    TrainingHistoryNumPy
    """
    history = TrainingHistoryNumPy()
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------------------
        logits = model.forward(X, A_tilde)      # (N, C)
        probs  = _softmax(logits)

        # ------------------------------------------------------------------
        # Loss and accuracy
        # ------------------------------------------------------------------
        train_loss = _cross_entropy(probs, y, train_mask)
        val_loss   = _cross_entropy(probs, y, val_mask)
        train_acc  = _accuracy(probs, y, train_mask)
        val_acc    = _accuracy(probs, y, val_mask)

        # ------------------------------------------------------------------
        # Backward pass
        # ------------------------------------------------------------------
        dL_dlogits = _cross_entropy_backward(probs, y, train_mask)
        model.backward(dL_dlogits)

        # ------------------------------------------------------------------
        # Parameter update (SGD + L2 regularisation)
        # ------------------------------------------------------------------
        model.step(lr=lr, weight_decay=weight_decay)

        # ------------------------------------------------------------------
        # Record
        # ------------------------------------------------------------------
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        history.epochs.append(epoch)

        if val_acc > history.best_val_acc:
            history.best_val_acc = val_acc
            history.best_epoch   = epoch

        if log_every > 0 and epoch % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>4d}/{epochs} | "
                f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                f"Train acc: {train_acc:.1f}% | Val acc: {val_acc:.1f}% | "
                f"[{elapsed:.1f}s]"
            )

    print(
        f"\nBest val acc: {history.best_val_acc:.1f}% @ epoch {history.best_epoch}"
    )
    return history
