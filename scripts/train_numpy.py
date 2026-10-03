"""
scripts/train_numpy.py — CLI entry point for the Pure NumPy GCN.

Usage examples
--------------
# Default: synthetic data, 2 layers, 200 epochs
python scripts/train_numpy.py

# Custom hyperparameters
python scripts/train_numpy.py --epochs 300 --lr 0.005 --hidden 32 --layers 3

# Your own data (npy files)
python scripts/train_numpy.py --adj data/raw/adj.npy --features data/raw/X.npy --labels data/raw/y.npy

# Your own data (npz)
python scripts/train_numpy.py --npz data/raw/my_graph.npz

# Save weights
python scripts/train_numpy.py --save outputs/checkpoints/gcn_numpy.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

# Allow running from repo root without pip install
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from vanilla_gcn_numpy.preprocessing import prepare_graph
from vanilla_gcn_numpy.vanilla_gcn import VanillaGCNNumPy
from vanilla_gcn_numpy.trainer import train_gcn_numpy
from vanilla_gcn_numpy.evaluation import evaluate_gcn_numpy
from vanilla_gcn_numpy.synthetic import create_synthetic_graph_numpy


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Vanilla GCN (pure NumPy, no PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data source ──
    data = p.add_argument_group("data")
    data.add_argument("--adj",      type=str, default=None,
                      help=".npy file for adjacency matrix (N×N)")
    data.add_argument("--features", type=str, default=None,
                      help=".npy file for feature matrix (N×F)")
    data.add_argument("--labels",   type=str, default=None,
                      help=".npy file for labels (N,)")
    data.add_argument("--npz",      type=str, default=None,
                      help=".npz archive containing 'adj', 'features', 'labels' arrays")
    data.add_argument("--train-ratio", type=float, default=0.6)
    data.add_argument("--val-ratio",   type=float, default=0.2)

    # ── Model ──
    model = p.add_argument_group("model")
    model.add_argument("--hidden", type=int, default=16,
                       help="Hidden layer dimension")
    model.add_argument("--layers", type=int, default=2,
                       help="Number of GCN layers (depth K)")

    # ── Training ──
    train = p.add_argument_group("training")
    train.add_argument("--epochs",       type=int,   default=200)
    train.add_argument("--lr",           type=float, default=0.01)
    train.add_argument("--weight-decay", type=float, default=5e-4)
    train.add_argument("--seed",         type=int,   default=42)
    train.add_argument("--log-every",    type=int,   default=50,
                       help="Print every N epochs. 0 = silent.")

    # ── Output ──
    out = p.add_argument_group("output")
    out.add_argument("--save", type=str, default=None,
                     help="Path to save model weights (.npz)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    return arr


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, X, y, train_mask, val_mask, test_mask)."""

    if args.npz:
        print(f"Loading from NPZ: {args.npz}")
        d = np.load(args.npz)
        keys = list(d.files)
        A = d[keys[0]].astype(np.float64)
        X = d[keys[1]].astype(np.float64)
        y = d[keys[2]].astype(np.int64)

    elif args.adj and args.features and args.labels:
        print(f"Loading from npy files:")
        print(f"  adj={args.adj}  features={args.features}  labels={args.labels}")
        A = _load_npy(args.adj).astype(np.float64)
        X = _load_npy(args.features).astype(np.float64)
        y = _load_npy(args.labels).astype(np.int64)

    else:
        print("No data files specified -> using built-in synthetic graph.")
        raw = create_synthetic_graph_numpy(seed=args.seed)
        return (raw["A"], raw["X"], raw["y"],
                raw["train_mask"], raw["val_mask"], raw["test_mask"])

    # Build masks for custom data
    N = len(y)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    n_train = max(1, int(N * args.train_ratio))
    n_val   = max(1, int(N * args.val_ratio))

    train_mask = np.zeros(N, dtype=bool);  train_mask[idx[:n_train]] = True
    val_mask   = np.zeros(N, dtype=bool);  val_mask[idx[n_train:n_train+n_val]] = True
    test_mask  = np.zeros(N, dtype=bool);  test_mask[idx[n_train+n_val:]] = True

    return A, X, y, train_mask, val_mask, test_mask


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    np.random.seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    A, X, y, train_mask, val_mask, test_mask = load_data(args)
    N, F = X.shape
    C    = int(y.max()) + 1

    print("\n" + "=" * 60)
    print("VANILLA GCN — Pure NumPy")
    print("=" * 60)
    print(f"  Nodes     : {N}")
    print(f"  Features  : {F}")
    print(f"  Classes   : {C}")
    print(f"  Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    _, A_tilde, X, y = prepare_graph(A, X, y)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VanillaGCNNumPy(
        input_dim=F,
        hidden_dim=args.hidden,
        num_classes=C,
        num_layers=args.layers,
        seed=args.seed,
    )
    print(f"\n{model.architecture_summary()}\n")

    # ── Training ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    history = train_gcn_numpy(
        model=model,
        A_tilde=A_tilde,
        X=X,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTraining time: {elapsed:.2f}s")

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics = evaluate_gcn_numpy(
        model, A_tilde, X, y, train_mask, val_mask, test_mask
    )
    print("\n" + metrics.summary())

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            **{f"layer_{i}_W": layer.W for i, layer in enumerate(model.layers)},
            config_input_dim=np.array(F),
            config_hidden_dim=np.array(args.hidden),
            config_num_classes=np.array(C),
            config_num_layers=np.array(args.layers),
            test_acc=np.array(metrics.test_acc),
        )
        print(f"\nModel weights saved -> {save_path}")


if __name__ == "__main__":
    main()
