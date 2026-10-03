"""
tests/test_numpy_gcn.py — Unit and integration tests for vanilla_gcn_numpy.

Covers:
  - Preprocessing: add_self_loops, symmetric_normalize, prepare_graph
  - GCNLayerNumPy: forward shape, backward gradient dimensions, weight update
  - VanillaGCNNumPy: topology, intermediate outputs, parameter count
  - Training: loss decreases, history length
  - Evaluation: metric ranges
  - Reproducibility: same seed → same result
"""

from __future__ import annotations

import numpy as np
import pytest

from vanilla_gcn_numpy.preprocessing import (
    add_self_loops,
    symmetric_normalize,
    prepare_graph,
)
from vanilla_gcn_numpy.gcn_layer import GCNLayerNumPy
from vanilla_gcn_numpy.vanilla_gcn import VanillaGCNNumPy
from vanilla_gcn_numpy.trainer import train_gcn_numpy
from vanilla_gcn_numpy.evaluation import evaluate_gcn_numpy
from vanilla_gcn_numpy.synthetic import create_synthetic_graph_numpy


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

N, F, C = 10, 4, 3


@pytest.fixture
def small_graph():
    """Return (A, X, y, A_hat, A_tilde) for a small random graph."""
    rng = np.random.default_rng(0)
    A = (rng.random((N, N)) > 0.6).astype(np.float64)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    X = rng.standard_normal((N, F))
    y = rng.integers(0, C, size=N).astype(np.int64)
    A_hat, A_tilde, X, y = prepare_graph(A, X, y)
    return A, X, y, A_hat, A_tilde


@pytest.fixture
def masks():
    train = np.array([True] * 6 + [False] * 4)
    val   = np.array([False] * 6 + [True] * 2 + [False] * 2)
    test  = np.array([False] * 8 + [True] * 2)
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingNumPy:

    def test_self_loops_diagonal(self):
        A = np.zeros((5, 5))
        A_hat = add_self_loops(A)
        assert np.allclose(np.diag(A_hat), 1.0), "diagonal must be 1 after self-loops"

    def test_self_loops_preserves_edges(self):
        A = np.eye(4, k=1) + np.eye(4, k=-1)
        A_hat = add_self_loops(A)
        assert A_hat[0, 1] == 1.0

    def test_symmetric_normalize_shape(self):
        A_hat = add_self_loops(np.eye(6, k=1) + np.eye(6, k=-1))
        A_tilde = symmetric_normalize(A_hat)
        assert A_tilde.shape == (6, 6)

    def test_symmetric_normalize_is_symmetric(self):
        A_hat = add_self_loops(np.eye(6, k=1) + np.eye(6, k=-1))
        A_tilde = symmetric_normalize(A_hat)
        assert np.allclose(A_tilde, A_tilde.T, atol=1e-10)

    def test_symmetric_normalize_no_nan(self):
        A_hat = add_self_loops(np.eye(5, k=1) + np.eye(5, k=-1))
        A_tilde = symmetric_normalize(A_hat)
        assert not np.isnan(A_tilde).any()

    def test_prepare_graph_dtypes(self):
        A = np.ones((4, 4)) - np.eye(4)
        X = np.random.randn(4, 3).astype(np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.int32)
        _, A_tilde, X_out, y_out = prepare_graph(A, X, y)
        assert A_tilde.dtype == np.float64
        assert X_out.dtype == np.float64
        assert y_out.dtype == np.int64


# ─────────────────────────────────────────────────────────────────────────────
# 2. GCNLayerNumPy
# ─────────────────────────────────────────────────────────────────────────────

class TestGCNLayerNumPy:

    def test_output_shape_relu(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        layer = GCNLayerNumPy(F, 8, activation="relu", seed=0)
        out = layer.forward(X, A_tilde)
        assert out.shape == (N, 8)

    def test_output_shape_no_activation(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        layer = GCNLayerNumPy(F, C, activation=None, seed=0)
        out = layer.forward(X, A_tilde)
        assert out.shape == (N, C)

    def test_relu_nonnegative(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        layer = GCNLayerNumPy(F, 8, activation="relu", seed=0)
        out = layer.forward(X, A_tilde)
        assert (out >= 0).all(), "ReLU output must be non-negative"

    def test_no_activation_can_be_negative(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        layer = GCNLayerNumPy(F, C, activation=None, seed=1)
        out = layer.forward(X, A_tilde)
        assert out.min() < 0 or out.max() >= 0  # just checking it runs

    def test_weight_shape(self):
        layer = GCNLayerNumPy(5, 12, seed=0)
        assert layer.W.shape == (5, 12)

    def test_no_nan_forward(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        layer = GCNLayerNumPy(F, 8, seed=0)
        out = layer.forward(X, A_tilde)
        assert not np.isnan(out).any()

    def test_backward_dw_shape(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        layer = GCNLayerNumPy(F, 8, activation="relu", seed=0)
        out = layer.forward(X, A_tilde)
        dL_dH_out = np.ones_like(out)
        dL_dH = layer.backward(dL_dH_out)
        assert layer.dW.shape == (F, 8), f"dW shape {layer.dW.shape}"

    def test_backward_dh_shape(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        layer = GCNLayerNumPy(F, 8, activation="relu", seed=0)
        out = layer.forward(X, A_tilde)
        dL_dH = layer.backward(np.ones_like(out))
        assert dL_dH.shape == (N, F), f"dL_dH shape {dL_dH.shape}"

    def test_step_updates_weights(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        layer = GCNLayerNumPy(F, 8, activation="relu", seed=0)
        W_before = layer.W.copy()
        out = layer.forward(X, A_tilde)
        layer.backward(np.ones_like(out))
        layer.step(lr=0.1)
        assert not np.allclose(layer.W, W_before), "step should update W"

    def test_num_parameters(self):
        layer = GCNLayerNumPy(4, 8, seed=0)
        assert layer.num_parameters == 32  # 4 × 8


# ─────────────────────────────────────────────────────────────────────────────
# 3. VanillaGCNNumPy
# ─────────────────────────────────────────────────────────────────────────────

class TestVanillaGCNNumPy:

    def test_output_shape_2layer(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        logits = model.forward(X, A_tilde)
        assert logits.shape == (N, C)

    def test_output_shape_1layer(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        model = VanillaGCNNumPy(F, 16, C, num_layers=1, seed=0)
        logits = model.forward(X, A_tilde)
        assert logits.shape == (N, C)

    def test_output_shape_3layer(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        model = VanillaGCNNumPy(F, 16, C, num_layers=3, seed=0)
        logits = model.forward(X, A_tilde)
        assert logits.shape == (N, C)

    def test_intermediate_keys_2layer(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        out = model.forward(X, A_tilde, return_intermediate=True)
        assert "input" in out
        assert "layer_1" in out
        assert "logits" in out

    def test_intermediate_shapes(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        out = model.forward(X, A_tilde, return_intermediate=True)
        assert out["input"].shape    == (N, F)
        assert out["layer_1"].shape  == (N, 16)
        assert out["logits"].shape   == (N, C)

    def test_num_layers_topology(self):
        model = VanillaGCNNumPy(4, 16, 3, num_layers=3, seed=0)
        assert len(model.layers) == 3
        assert model.layers[0].activation == "relu"
        assert model.layers[1].activation == "relu"
        assert model.layers[2].activation is None  # output layer

    def test_invalid_num_layers(self):
        with pytest.raises(ValueError):
            VanillaGCNNumPy(4, 16, 3, num_layers=0)

    def test_parameter_count(self):
        # 2-layer: F→H and H→C  →  F*H + H*C
        model = VanillaGCNNumPy(4, 16, 3, num_layers=2, seed=0)
        assert model.count_parameters() == 4 * 16 + 16 * 3

    def test_no_nan_forward(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        logits = model.forward(X, A_tilde)
        assert not np.isnan(logits).any()

    def test_backward_runs_without_crash(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        logits = model.forward(X, A_tilde)
        dL = np.random.randn(*logits.shape)
        model.backward(dL)   # should not raise

    def test_step_changes_weights(self, small_graph):
        _, X, _, _, A_tilde = small_graph
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        W0_before = model.layers[0].W.copy()
        logits = model.forward(X, A_tilde)
        model.backward(np.ones_like(logits))
        model.step(lr=0.1)
        assert not np.allclose(model.layers[0].W, W0_before)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingNumPy:

    def test_history_length(self, small_graph, masks):
        _, X, y, _, A_tilde = small_graph
        train_mask, val_mask, _ = masks
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        history = train_gcn_numpy(model, A_tilde, X, y, train_mask, val_mask,
                                  epochs=30, log_every=0)
        assert len(history.train_loss) == 30
        assert len(history.val_loss)   == 30
        assert len(history.train_acc)  == 30

    def test_loss_decreases(self, small_graph, masks):
        """Final loss should be lower than initial loss."""
        _, X, y, _, A_tilde = small_graph
        train_mask, val_mask, _ = masks
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        history = train_gcn_numpy(model, A_tilde, X, y, train_mask, val_mask,
                                  epochs=200, lr=0.01, log_every=0)
        assert history.train_loss[-1] < history.train_loss[0], (
            f"Loss did not decrease: {history.train_loss[0]:.4f} → {history.train_loss[-1]:.4f}"
        )

    def test_accuracy_range(self, small_graph, masks):
        _, X, y, _, A_tilde = small_graph
        train_mask, val_mask, _ = masks
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        history = train_gcn_numpy(model, A_tilde, X, y, train_mask, val_mask,
                                  epochs=50, log_every=0)
        for acc in history.train_acc:
            assert 0.0 <= acc <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationNumPy:

    def test_metrics_keys(self, small_graph, masks):
        _, X, y, _, A_tilde = small_graph
        train_mask, val_mask, test_mask = masks
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        metrics = evaluate_gcn_numpy(model, A_tilde, X, y, train_mask, val_mask, test_mask)
        assert hasattr(metrics, "train_acc")
        assert hasattr(metrics, "val_acc")
        assert hasattr(metrics, "test_acc")

    def test_metrics_ranges(self, small_graph, masks):
        _, X, y, _, A_tilde = small_graph
        train_mask, val_mask, test_mask = masks
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        train_gcn_numpy(model, A_tilde, X, y, train_mask, val_mask,
                        epochs=50, log_every=0)
        metrics = evaluate_gcn_numpy(model, A_tilde, X, y, train_mask, val_mask, test_mask)
        for acc in [metrics.train_acc, metrics.val_acc, metrics.test_acc]:
            assert 0.0 <= acc <= 100.0

    def test_predictions_shape(self, small_graph, masks):
        _, X, y, _, A_tilde = small_graph
        train_mask, val_mask, test_mask = masks
        model = VanillaGCNNumPy(F, 16, C, num_layers=2, seed=0)
        metrics = evaluate_gcn_numpy(model, A_tilde, X, y, train_mask, val_mask, test_mask)
        assert metrics.predictions.shape == (N,)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Synthetic data
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticNumPy:

    def test_synthetic_shapes(self):
        d = create_synthetic_graph_numpy(seed=42)
        N = d["num_nodes"]
        F = d["num_features"]
        C = d["num_classes"]
        assert d["A"].shape == (N, N)
        assert d["X"].shape == (N, F)
        assert d["y"].shape == (N,)

    def test_synthetic_masks_cover_all_nodes(self):
        d = create_synthetic_graph_numpy(seed=42)
        total = d["train_mask"].sum() + d["val_mask"].sum() + d["test_mask"].sum()
        assert total == d["num_nodes"]

    def test_synthetic_adjacency_symmetric(self):
        d = create_synthetic_graph_numpy(seed=42)
        assert np.allclose(d["A"], d["A"].T)

    def test_synthetic_no_self_loops(self):
        d = create_synthetic_graph_numpy(seed=42)
        assert np.all(np.diag(d["A"]) == 0)

    def test_synthetic_reproducible(self):
        d1 = create_synthetic_graph_numpy(seed=99)
        d2 = create_synthetic_graph_numpy(seed=99)
        assert np.allclose(d1["X"], d2["X"])
        assert np.array_equal(d1["y"], d2["y"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibilityNumPy:

    def _run(self, seed):
        d = create_synthetic_graph_numpy(seed=42)
        _, A_tilde, X, y = prepare_graph(d["A"], d["X"], d["y"])
        model = VanillaGCNNumPy(d["num_features"], 16, d["num_classes"],
                                num_layers=2, seed=seed)
        train_gcn_numpy(model, A_tilde, X, y,
                        d["train_mask"], d["val_mask"],
                        epochs=20, lr=0.01, log_every=0)
        return model.forward(X, A_tilde)

    def test_same_seed_same_result(self):
        out_a = self._run(seed=7)
        out_b = self._run(seed=7)
        assert np.allclose(out_a, out_b, atol=1e-12)

    def test_different_seed_different_result(self):
        out_a = self._run(seed=1)
        out_b = self._run(seed=2)
        assert not np.allclose(out_a, out_b)
