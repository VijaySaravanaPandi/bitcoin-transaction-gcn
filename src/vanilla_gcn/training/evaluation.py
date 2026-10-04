"""
evaluation.py — Evaluation utilities for Vanilla GCN.

Keeps evaluation logic separate from training to ensure clean
separation of concerns and to allow evaluation of checkpointed models.

Usage
-----
    from vanilla_gcn.training.evaluation import accuracy, evaluate_gcn

    metrics = evaluate_gcn(model, A_tilde, X, y, data.train_mask,
                           data.val_mask, data.test_mask)
    print(f"Test accuracy: {metrics['test_acc']:.1f}%")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from vanilla_gcn.models.vanilla_gcn import VanillaGCN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core accuracy helper
# ---------------------------------------------------------------------------


def accuracy(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute node classification accuracy for a subset of nodes.

    Parameters
    ----------
    logits : torch.Tensor, shape (N, C)
        Raw logit scores from the GCN output layer.
    y : torch.Tensor, shape (N,)
        True integer class labels.
    mask : torch.Tensor, shape (N,)
        Boolean mask; accuracy is computed over ``True`` entries only.

    Returns
    -------
    float
        Accuracy in percent (0.0 – 100.0).

    Examples
    --------
    >>> logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    >>> y = torch.tensor([0, 1])
    >>> mask = torch.tensor([True, True])
    >>> accuracy(logits, y, mask)
    100.0
    """
    if mask.sum() == 0:
        logger.warning("accuracy(): mask selects 0 nodes — returning 0.0")
        return 0.0

    with torch.no_grad():
        predictions = logits[mask].argmax(dim=1)
        correct = (predictions == y[mask]).sum().item()
        total = int(mask.sum().item())

    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvaluationMetrics:
    """Results from a full train/val/test evaluation pass.

    Attributes
    ----------
    train_loss : float
    val_loss : float
    test_loss : float
    train_acc : float  (percent)
    val_acc : float    (percent)
    test_acc : float   (percent)
    predictions : torch.Tensor  shape (N,)
        Predicted class for every node.
    """

    train_loss: float
    val_loss: float
    test_loss: float
    train_acc: float
    val_acc: float
    test_acc: float
    predictions: torch.Tensor
    probabilities: torch.Tensor | None = None
    fraud_metrics: dict[str, float] | None = None

    def summary(self) -> str:
        lines = [
            "Evaluation Metrics",
            f"  Train loss : {self.train_loss:.4f}  |  Train acc : {self.train_acc:.1f}%",
            f"  Val   loss : {self.val_loss:.4f}  |  Val   acc : {self.val_acc:.1f}%",
            f"  Test  loss : {self.test_loss:.4f}  |  Test  acc : {self.test_acc:.1f}%",
        ]
        if self.fraud_metrics:
            lines.append(
                "  Test fraud: "
                f"F1={self.fraud_metrics['test_f1']:.3f} | "
                f"Recall={self.fraud_metrics['test_recall']:.3f} | "
                f"PR-AUC={self.fraud_metrics['test_pr_auc']:.3f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def evaluate_gcn(
    model: VanillaGCN,
    A_tilde: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> EvaluationMetrics:
    """Evaluate a trained GCN on train, validation, and test splits.

    Parameters
    ----------
    model : VanillaGCN
        Trained GCN model in evaluation mode.
    A_tilde : torch.Tensor, shape (N, N)
        Symmetrically normalized adjacency.
    X : torch.Tensor, shape (N, F)
        Node features.
    y : torch.Tensor, shape (N,)
        True labels.
    train_mask, val_mask, test_mask : torch.Tensor, shape (N,)
        Boolean split masks.

    Returns
    -------
    EvaluationMetrics
        Metrics for all three splits plus the per-node predictions.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.eval()

    with torch.no_grad():
        logits: torch.Tensor = model(X, A_tilde)  # (N, C)
        predictions = logits.argmax(dim=1)         # (N,)
        probabilities = torch.softmax(logits, dim=1)

        train_loss = float(criterion(logits[train_mask], y[train_mask]))
        val_loss = float(criterion(logits[val_mask], y[val_mask]))
        test_loss = float(criterion(logits[test_mask], y[test_mask]))

        train_acc = accuracy(logits, y, train_mask)
        val_acc = accuracy(logits, y, val_mask)
        test_acc = accuracy(logits, y, test_mask)

    test_labels = y[test_mask].cpu().numpy()
    test_predictions = predictions[test_mask].cpu().numpy()
    test_probabilities = probabilities[test_mask, 0].cpu().numpy()
    fraud_metrics = {
        "test_precision": precision_score(test_labels, test_predictions, pos_label=0, zero_division=0),
        "test_recall": recall_score(test_labels, test_predictions, pos_label=0, zero_division=0),
        "test_f1": f1_score(test_labels, test_predictions, pos_label=0, zero_division=0),
        "test_macro_f1": f1_score(test_labels, test_predictions, average="macro", zero_division=0),
        "test_balanced_accuracy": balanced_accuracy_score(test_labels, test_predictions),
        "test_pr_auc": average_precision_score((test_labels == 0).astype(int), test_probabilities),
    }
    if len(set(test_labels.tolist())) == 2:
        fraud_metrics["test_roc_auc"] = roc_auc_score(
            (test_labels == 0).astype(int), test_probabilities
        )
    else:
        fraud_metrics["test_roc_auc"] = float("nan")

    metrics = EvaluationMetrics(
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        predictions=predictions,
        probabilities=probabilities,
        fraud_metrics=fraud_metrics,
    )

    logger.info("%s", metrics.summary())
    return metrics
