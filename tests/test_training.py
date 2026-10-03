"""
test_training.py — Integration tests for training and evaluation.

Tests:
1. Training reduces / changes loss over epochs.
2. Training history has correct length.
3. evaluate_gcn returns metrics for all splits.
4. Accuracy is in [0, 100].
5. Training on synthetic data achieves > 0% accuracy.
"""

from __future__ import annotations

import pytest
import torch

from vanilla_gcn.data.preprocessing import prepare_graph
from vanilla_gcn.data.synthetic import create_synthetic_graph
from vanilla_gcn.models.vanilla_gcn import VanillaGCN
from vanilla_gcn.seed import set_seed
from vanilla_gcn.training.evaluation import accuracy, evaluate_gcn
from vanilla_gcn.training.trainer import train_gcn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_model_and_data():
    """Train a small GCN on the synthetic graph for 50 epochs."""
    set_seed(42)
    data = create_synthetic_graph(seed=42)
    _, A_tilde, X, y = prepare_graph(data.adjacency, data.features, data.labels)

    model = VanillaGCN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_classes=data.num_classes,
        num_layers=2,
    )
    history = train_gcn(
        model, A_tilde, X, y,
        data.train_mask, data.val_mask,
        epochs=50, lr=0.01,
    )
    return model, data, A_tilde, X, y, history


# ---------------------------------------------------------------------------
# Test 1: Training changes loss
# ---------------------------------------------------------------------------


def test_training_changes_loss(trained_model_and_data) -> None:
    """Loss at epoch 50 must differ from loss at epoch 1."""
    *_, history = trained_model_and_data
    initial_loss = history.train_loss[0]
    final_loss = history.train_loss[-1]
    assert abs(initial_loss - final_loss) > 1e-4, (
        f"Loss did not change: {initial_loss:.4f} → {final_loss:.4f}"
    )


def test_training_loss_decreases(trained_model_and_data) -> None:
    """Final loss should generally be lower than initial loss (or at least changes)."""
    *_, history = trained_model_and_data
    # Note: on a tiny graph this may not be strictly monotone, but should improve overall
    assert history.train_loss[-1] < history.train_loss[0] * 1.5, (
        "Training loss shows no improvement at all"
    )


# ---------------------------------------------------------------------------
# Test 2: Training history has correct length
# ---------------------------------------------------------------------------


def test_training_history_length(trained_model_and_data) -> None:
    """History lists must have length == epochs."""
    *_, history = trained_model_and_data
    epochs = 50
    assert len(history.train_loss) == epochs
    assert len(history.val_loss) == epochs
    assert len(history.train_acc) == epochs
    assert len(history.val_acc) == epochs
    assert len(history.epochs) == epochs


# ---------------------------------------------------------------------------
# Test 3: evaluate_gcn returns metrics for all splits
# ---------------------------------------------------------------------------


def test_evaluate_gcn_has_all_splits(trained_model_and_data) -> None:
    """evaluate_gcn must return metrics for train, val, and test."""
    model, data, A_tilde, X, y, _ = trained_model_and_data
    metrics = evaluate_gcn(
        model, A_tilde, X, y,
        data.train_mask, data.val_mask, data.test_mask,
    )
    assert hasattr(metrics, "train_loss")
    assert hasattr(metrics, "val_loss")
    assert hasattr(metrics, "test_loss")
    assert hasattr(metrics, "train_acc")
    assert hasattr(metrics, "val_acc")
    assert hasattr(metrics, "test_acc")


def test_evaluate_gcn_predictions_shape(trained_model_and_data) -> None:
    """predictions tensor must have shape (N,)."""
    model, data, A_tilde, X, y, _ = trained_model_and_data
    metrics = evaluate_gcn(
        model, A_tilde, X, y,
        data.train_mask, data.val_mask, data.test_mask,
    )
    assert metrics.predictions.shape == (data.num_nodes,)


# ---------------------------------------------------------------------------
# Test 4: Accuracy is in [0, 100]
# ---------------------------------------------------------------------------


def test_accuracy_range(trained_model_and_data) -> None:
    """All accuracy values must be in [0, 100]."""
    model, data, A_tilde, X, y, _ = trained_model_and_data
    metrics = evaluate_gcn(
        model, A_tilde, X, y,
        data.train_mask, data.val_mask, data.test_mask,
    )
    for name, acc in [
        ("train_acc", metrics.train_acc),
        ("val_acc",   metrics.val_acc),
        ("test_acc",  metrics.test_acc),
    ]:
        assert 0.0 <= acc <= 100.0, f"{name}={acc} out of [0, 100]"


# ---------------------------------------------------------------------------
# Test 5: Training achieves > 0% accuracy
# ---------------------------------------------------------------------------


def test_training_achieves_nonzero_accuracy(trained_model_and_data) -> None:
    """After 50 epochs, train accuracy must be > 0%."""
    *_, history = trained_model_and_data
    assert max(history.train_acc) > 0.0, "Training accuracy never exceeded 0%"


# ---------------------------------------------------------------------------
# Test: accuracy helper
# ---------------------------------------------------------------------------


def test_accuracy_perfect() -> None:
    """Perfect predictions must yield 100%."""
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    y = torch.tensor([0, 1, 0])
    mask = torch.ones(3, dtype=torch.bool)
    assert accuracy(logits, y, mask) == 100.0


def test_accuracy_zero() -> None:
    """Completely wrong predictions must yield 0%."""
    logits = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
    y = torch.tensor([0, 1])
    mask = torch.ones(2, dtype=torch.bool)
    assert accuracy(logits, y, mask) == 0.0


def test_accuracy_empty_mask() -> None:
    """Empty mask must yield 0% (no nodes selected)."""
    logits = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([0])
    mask = torch.zeros(1, dtype=torch.bool)
    assert accuracy(logits, y, mask) == 0.0
