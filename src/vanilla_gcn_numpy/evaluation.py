"""evaluation.py — Evaluation metrics (NumPy version)."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from vanilla_gcn_numpy.vanilla_gcn import VanillaGCNNumPy
from vanilla_gcn_numpy.trainer import _softmax, _cross_entropy


@dataclass
class EvaluationMetricsNumPy:
    train_loss: float
    val_loss:   float
    test_loss:  float
    train_acc:  float
    val_acc:    float
    test_acc:   float
    predictions: np.ndarray

    def summary(self) -> str:
        return (
            "Evaluation Metrics (NumPy GCN)\n"
            + "-" * 40 + "\n"
            + f"  Train  | loss={self.train_loss:.4f}  acc={self.train_acc:.1f}%\n"
            + f"  Val    | loss={self.val_loss:.4f}  acc={self.val_acc:.1f}%\n"
            + f"  Test   | loss={self.test_loss:.4f}  acc={self.test_acc:.1f}%"
        )


def accuracy_numpy(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> float:
    n = mask.sum()
    if n == 0:
        return 0.0
    return float((probs[mask].argmax(axis=1) == y[mask]).mean() * 100.0)


def evaluate_gcn_numpy(
    model: VanillaGCNNumPy,
    A_tilde: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> EvaluationMetricsNumPy:
    """Evaluate a trained NumPy GCN on all splits."""
    logits = model.forward(X, A_tilde)
    probs  = _softmax(logits)
    preds  = logits.argmax(axis=1)

    return EvaluationMetricsNumPy(
        train_loss=_cross_entropy(probs, y, train_mask),
        val_loss=_cross_entropy(probs, y, val_mask),
        test_loss=_cross_entropy(probs, y, test_mask),
        train_acc=accuracy_numpy(probs, y, train_mask),
        val_acc=accuracy_numpy(probs, y, val_mask),
        test_acc=accuracy_numpy(probs, y, test_mask),
        predictions=preds,
    )
