"""
test_reproducibility.py — Tests for seed-based reproducibility.

Tests:
1. Same seed → identical outputs.
2. Different seeds → different outputs.
3. save/load checkpoint → identical model outputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.optim as optim

from vanilla_gcn.data.preprocessing import prepare_graph
from vanilla_gcn.data.synthetic import create_synthetic_graph
from vanilla_gcn.models.vanilla_gcn import VanillaGCN
from vanilla_gcn.seed import set_seed
from vanilla_gcn.training.trainer import train_gcn
from vanilla_gcn.utils.checkpointing import load_checkpoint, save_checkpoint
from vanilla_gcn.config import load_config


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_model_and_run(seed: int, epochs: int = 20):
    """Create, seed, and train a small GCN; return model + data."""
    set_seed(seed)
    data = create_synthetic_graph(seed=seed)
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
        epochs=epochs, lr=0.01,
    )
    model.eval()
    with torch.no_grad():
        logits = model(X, A_tilde)
    return model, logits, data, A_tilde, X, y, history


# ---------------------------------------------------------------------------
# Test 1: Same seed → identical outputs
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_outputs() -> None:
    """Running twice with seed=42 must produce bit-identical logits."""
    _, logits_a, *_ = _make_model_and_run(seed=42, epochs=10)
    _, logits_b, *_ = _make_model_and_run(seed=42, epochs=10)
    assert torch.allclose(logits_a, logits_b, atol=1e-6), (
        "Same-seed runs produced different logits"
    )


# ---------------------------------------------------------------------------
# Test 2: Different seeds → different outputs
# ---------------------------------------------------------------------------


def test_different_seeds_produce_different_outputs() -> None:
    """Seeds 42 and 123 should produce at least some different initial weights."""
    set_seed(42)
    data = create_synthetic_graph(seed=42)
    _, A_tilde, X, y = prepare_graph(data.adjacency, data.features, data.labels)

    set_seed(42)
    model_a = VanillaGCN(
        input_dim=data.num_features, hidden_dim=16,
        num_classes=data.num_classes, num_layers=2,
    )

    set_seed(123)
    model_b = VanillaGCN(
        input_dim=data.num_features, hidden_dim=16,
        num_classes=data.num_classes, num_layers=2,
    )

    # Initial weights should differ between seeds
    w_a = model_a.layers[0].weight
    w_b = model_b.layers[0].weight
    assert not torch.allclose(w_a, w_b), (
        "Different seeds produced identical initial weights (unexpected)"
    )


# ---------------------------------------------------------------------------
# Test 3: save/load checkpoint → identical outputs
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_identical_outputs() -> None:
    """After save + load, model must produce bit-identical outputs."""
    model, logits_before, data, A_tilde, X, y, history = _make_model_and_run(
        seed=42, epochs=20
    )

    cfg = load_config()  # use defaults
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model.pt"
        save_checkpoint(model, optimizer, cfg, history, epoch=20, path=ckpt_path)

        # Create a new model with the same architecture
        model2 = VanillaGCN(
            input_dim=data.num_features,
            hidden_dim=16,
            num_classes=data.num_classes,
            num_layers=2,
        )
        optimizer2 = optim.Adam(model2.parameters(), lr=0.01)
        model2, optimizer2, _, _, _ = load_checkpoint(ckpt_path, model2, optimizer2)

        model2.eval()
        with torch.no_grad():
            logits_after = model2(X, A_tilde)

    assert torch.allclose(logits_before, logits_after, atol=1e-6), (
        "Checkpoint save/load produced different logits"
    )


# ---------------------------------------------------------------------------
# Test: set_seed does not raise
# ---------------------------------------------------------------------------


def test_set_seed_no_error() -> None:
    """set_seed must run without error for various seeds."""
    for seed in [0, 1, 42, 999, 2**31 - 1]:
        set_seed(seed)  # should not raise
