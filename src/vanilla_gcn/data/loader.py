"""
loader.py — Generic graph data loader for Vanilla GCN.

This module defines the dataset interface that allows the GCN model to
work with any real-world dataset without modifying the model architecture.

Future loaders (CSV, NPZ, NetworkX, citation networks, etc.) should be
added here as separate functions that all return a :class:`GraphData` object.

See also
--------
docs/dataset_interface.md — detailed instructions for adding new datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from vanilla_gcn.data.features import add_transaction_graph_features
from vanilla_gcn.data.types import GraphData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic numpy loader — the primary interface for real-world datasets
# ---------------------------------------------------------------------------


def load_graph_from_numpy(
    A: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    name: str = "custom",
) -> GraphData:
    """Load a graph from NumPy arrays into a :class:`GraphData` object.

    This is the primary entry point for replacing the synthetic dataset
    with a real-world graph.  The GCN model, trainer, and evaluation code
    only depend on :class:`GraphData` — they do not know or care where the
    data comes from.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Binary symmetric adjacency matrix.  Off-diagonal entry A[i,j] = 1
        means there is an edge between node i and node j.
    X : np.ndarray, shape (N, F)
        Node feature matrix.  Row i is the feature vector for node i.
    y : np.ndarray, shape (N,)
        Integer class labels in the range [0, C-1].
    train_ratio : float
        Fraction of nodes assigned to the training set.
    val_ratio : float
        Fraction of nodes assigned to the validation set.
    test_ratio : float
        Fraction of nodes assigned to the test set.
    seed : int
        Random seed for the node split.
    name : str
        Human-readable dataset name stored in :class:`GraphData`.

    Returns
    -------
    GraphData
        Fully populated graph data container ready for GCN training.

    Examples
    --------
    Replace the synthetic graph with your own data:

    >>> import numpy as np
    >>> from vanilla_gcn.data.loader import load_graph_from_numpy
    >>> A = np.load("my_adjacency.npy")   # shape (N, N)
    >>> X = np.load("my_features.npy")    # shape (N, F)
    >>> y = np.load("my_labels.npy")      # shape (N,)
    >>> data = load_graph_from_numpy(A, X, y, name="my_dataset")

    Then pass ``data`` directly to the training pipeline — no other changes
    are required.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "train_ratio + val_ratio + test_ratio must equal 1.0"
    )
    n = A.shape[0]
    num_classes = int(y.max()) + 1

    # Split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train: n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Edge index from adjacency
    src, dst = np.nonzero(A)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    data = GraphData(
        adjacency=torch.tensor(A, dtype=torch.float32),
        features=torch.tensor(X, dtype=torch.float32),
        labels=torch.tensor(y, dtype=torch.long),
        node_ids=list(range(n)),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
        name=name,
    )

    logger.info("Loaded graph '%s': N=%d, F=%d, C=%d", name, n, X.shape[1], num_classes)
    return data


# ---------------------------------------------------------------------------
# Split utility
# ---------------------------------------------------------------------------


def split_nodes(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random train/val/test boolean masks.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes N.
    train_ratio, val_ratio, test_ratio : float
        Split ratios (must sum to 1.0).
    seed : int
        Random seed.

    Returns
    -------
    train_mask, val_mask, test_mask : torch.Tensor
        Boolean tensors of shape ``(N,)``.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train: n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# NPZ loader (future-ready)
# ---------------------------------------------------------------------------


def load_graph_from_npz(
    path: str | Path,
    *,
    adjacency_key: str = "adjacency",
    features_key: str = "features",
    labels_key: str = "labels",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> GraphData:
    """Load a graph from an NPZ archive.

    Expected NPZ keys (configurable via keyword arguments):
    - ``adjacency`` : shape (N, N)
    - ``features``  : shape (N, F)
    - ``labels``    : shape (N,)

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file.
    adjacency_key, features_key, labels_key : str
        Keys for the respective arrays inside the NPZ file.
    train_ratio, val_ratio, test_ratio : float
        Node split ratios.
    seed : int
        Random seed.

    Returns
    -------
    GraphData
        Populated graph data object.
    """
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    archive = np.load(npz_path, allow_pickle=False)
    A = archive[adjacency_key].astype(np.float32)
    X = archive[features_key].astype(np.float32)
    y = archive[labels_key].astype(np.int64)

    return load_graph_from_numpy(
        A, X, y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        name=npz_path.stem,
    )


# ---------------------------------------------------------------------------
# Elliptic++ Transactions loader
# ---------------------------------------------------------------------------


def load_elliptic_transactions(
    raw_dir: str | Path = "data/raw",
    *,
    features_file: str = "txs_features.csv",
    classes_file: str = "txs_classes.csv",
    edgelist_file: str = "txs_edgelist.csv",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    split_strategy: str = "random",
    temporal_train_end: int = 30,
    temporal_validation_end: int = 39,
    add_graph_features: bool = False,
) -> "GraphData":
    """Load the Elliptic++ Transactions dataset from CSV files.

    Reads the three CSV files that make up the Elliptic++ transaction graph:

    * ``txs_features.csv``  — 203 769 rows × 183 feature columns + txId column
    * ``txs_classes.csv``   — txId, class  (1 = illicit, 2 = licit, unknown)
    * ``txs_edgelist.csv``  — txId1, txId2  (directed money-flow edges)

    Class mapping (after loading):
        illicit (class-1 in CSV)  →  label 0
        licit   (class-2 in CSV)  →  label 1
        unknown (class-3 / "unknown" string) → label -1  (excluded from masks)

    Only labelled nodes (illicit + licit) participate in train/val/test masks.
    All 203 769 nodes are included in the graph for message passing.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing the three CSV files.
    features_file, classes_file, edgelist_file : str
        Filenames within *raw_dir*.
    train_ratio, val_ratio, test_ratio : float
        Split ratios for *labelled* nodes (must sum to 1.0).
    seed : int
        Random seed for the node split.
    split_strategy : str
        ``"random"`` or ``"temporal"``. Temporal splits use Elliptic++ time
        steps and ignore the ratio arguments.
    temporal_train_end, temporal_validation_end : int
        Inclusive time-step boundaries for temporal train and validation sets.
    add_graph_features : bool
        Append log-scaled in-degree, out-degree, and total-degree features.

    Returns
    -------
    GraphData
        Populated graph data object ready for GCN training.

    Notes
    -----
    The adjacency matrix is kept as a sparse COO tensor so the full
    Elliptic++ graph can be loaded without allocating an N×N dense matrix.
    """
    import pandas as pd

    raw_dir = Path(raw_dir)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "train_ratio + val_ratio + test_ratio must equal 1.0"
    )
    if split_strategy not in {"random", "temporal"}:
        raise ValueError("split_strategy must be 'random' or 'temporal'")

    # ------------------------------------------------------------------
    # 1. Features
    # ------------------------------------------------------------------
    feat_path = raw_dir / features_file
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {feat_path}\n"
            "Download from https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l"
        )
    logger.info("Reading features from %s …", feat_path)
    feat_df = pd.read_csv(feat_path, header=0)
    # First column is the transaction id; remaining 183 columns are features.
    tx_ids = feat_df.iloc[:, 0].astype(int).values  # (N,)
    X_np = feat_df.iloc[:, 1:].values.astype(np.float32)  # (N, 183)
    node_times = X_np[:, 0].astype(np.int64)
    # Missing feature values occur in the released CSV; zero is the neutral
    # value for these normalized numeric features and keeps training finite.
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(tx_ids)
    tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    # ------------------------------------------------------------------
    # 2. Labels
    # ------------------------------------------------------------------
    cls_path = raw_dir / classes_file
    if not cls_path.exists():
        raise FileNotFoundError(f"Classes file not found: {cls_path}")
    logger.info("Reading classes from %s …", cls_path)
    cls_df = pd.read_csv(cls_path)
    cls_df.columns = cls_df.columns.str.strip()

    # Map raw class values → integer labels
    # CSV values: "1" = illicit → 0, "2" = licit → 1, "unknown" → -1
    raw_class = cls_df.set_index("txId")["class"].astype(str).str.strip()
    y_np = np.full(n, -1, dtype=np.int64)
    for tx_id, cls_val in raw_class.items():
        idx = tx_id_to_idx.get(int(tx_id))
        if idx is None:
            continue
        if cls_val == "1":
            y_np[idx] = 0   # illicit
        elif cls_val == "2":
            y_np[idx] = 1   # licit
        # else: unknown → remains -1

    # ------------------------------------------------------------------
    # 3. Edges → adjacency
    # ------------------------------------------------------------------
    edge_path = raw_dir / edgelist_file
    if not edge_path.exists():
        raise FileNotFoundError(f"Edge list file not found: {edge_path}")
    logger.info("Reading edges from %s …", edge_path)
    edge_df = pd.read_csv(edge_path)
    edge_df.columns = edge_df.columns.str.strip()

    src_col, dst_col = edge_df.columns[0], edge_df.columns[1]
    src_ids = edge_df[src_col].astype(int).values
    dst_ids = edge_df[dst_col].astype(int).values

    # Filter edges where both endpoints are known
    valid = np.array(
        [(s in tx_id_to_idx and d in tx_id_to_idx) for s, d in zip(src_ids, dst_ids)]
    )
    src_idx = np.array([tx_id_to_idx[s] for s, v in zip(src_ids, valid) if v])
    dst_idx = np.array([tx_id_to_idx[d] for d, v in zip(dst_ids, valid) if v])

    if add_graph_features:
        X_np = add_transaction_graph_features(X_np, src_idx, dst_idx, n)

    # Build symmetric sparse adjacency; keep it sparse for the full graph.
    all_src = np.concatenate([src_idx, dst_idx])
    all_dst = np.concatenate([dst_idx, src_idx])
    edge_values = np.ones(len(all_src), dtype=np.float32)
    adjacency = torch.sparse_coo_tensor(
        torch.tensor(np.stack([all_src, all_dst]), dtype=torch.long),
        torch.tensor(edge_values, dtype=torch.float32),
        size=(n, n),
    ).coalesce()

    # ------------------------------------------------------------------
    # 4. Train / val / test masks (labelled nodes only)
    # ------------------------------------------------------------------
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    if split_strategy == "temporal":
        labelled = y_np >= 0
        train_mask[torch.from_numpy(labelled & (node_times <= temporal_train_end))] = True
        val_mask[torch.from_numpy(
            labelled & (node_times > temporal_train_end) &
            (node_times <= temporal_validation_end)
        )] = True
        test_mask[torch.from_numpy(labelled & (node_times > temporal_validation_end))] = True
    else:
        labelled_idx = np.where(y_np >= 0)[0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(labelled_idx)
        n_lab = len(perm)
        n_train = int(n_lab * train_ratio)
        n_val = int(n_lab * val_ratio)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train: n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True

    # Replace -1 labels with 0 for tensor safety (masked out during training)
    y_safe = y_np.copy()
    y_safe[y_safe < 0] = 0

    # ------------------------------------------------------------------
    # 5. Assemble GraphData
    # ------------------------------------------------------------------
    edge_index = torch.tensor(np.stack([all_src, all_dst], axis=0), dtype=torch.long)

    data = GraphData(
        adjacency=adjacency,
        features=torch.tensor(X_np, dtype=torch.float32),
        labels=torch.tensor(y_safe, dtype=torch.long),
        node_ids=list(map(int, tx_ids)),
        node_times=torch.tensor(node_times, dtype=torch.long),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        labelled_mask=torch.from_numpy(y_np >= 0),
        num_classes=2,  # illicit (0) and licit (1)
        name="elliptic_transactions",
    )

    logger.info(
        "Loaded Elliptic++ Transactions: N=%d, F=%d, labelled=%d "
        "(illicit=%d, licit=%d), edges=%d",
        n, X_np.shape[1], int((y_np >= 0).sum()),
        int((y_np == 0).sum()), int((y_np == 1).sum()),
        len(src_idx),
    )
    return data


def load_elliptic_actors(
    raw_dir: str | Path = "data/raw",
    *,
    features_file: str = "wallets_features.csv",
    classes_file: str = "wallets_classes.csv",
    edgelist_file: str = "AddrAddr_edgelist.csv",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    split_strategy: str = "random",
    temporal_train_end: int = 30,
    temporal_validation_end: int = 39,
    add_graph_features: bool = False,
) -> GraphData:
    """Load the Elliptic++ wallet-address interaction graph.

    This loader uses the homogeneous address-address graph. The AddrTx and
    TxAddr files describe a different relation and are intentionally not
    merged into this graph.
    """
    import pandas as pd

    raw_dir = Path(raw_dir)
    if split_strategy not in {"random", "temporal"}:
        raise ValueError("split_strategy must be 'random' or 'temporal'")

    feat_path = raw_dir / features_file
    cls_path = raw_dir / classes_file
    edge_path = raw_dir / edgelist_file
    for path in (feat_path, cls_path, edge_path):
        if not path.exists():
            raise FileNotFoundError(f"Actor dataset file not found: {path}")

    feat_df = pd.read_csv(feat_path, header=0)
    node_ids = feat_df.iloc[:, 0].astype(str).str.strip().to_numpy()
    X_np = np.nan_to_num(
        feat_df.iloc[:, 1:].to_numpy(dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    node_times = X_np[:, 0].astype(np.int64)
    n = len(node_ids)
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    cls_df = pd.read_csv(cls_path)
    cls_df.columns = cls_df.columns.astype(str).str.strip()
    id_column = cls_df.columns[0]
    class_column = "class" if "class" in cls_df.columns else cls_df.columns[1]
    y_np = np.full(n, -1, dtype=np.int64)
    for node_id, cls_val in cls_df.set_index(id_column)[class_column].items():
        idx = node_id_to_idx.get(str(node_id).strip())
        if idx is not None and str(cls_val).strip() == "1":
            y_np[idx] = 0
        elif idx is not None and str(cls_val).strip() == "2":
            y_np[idx] = 1

    edge_df = pd.read_csv(edge_path)
    edge_df.columns = edge_df.columns.astype(str).str.strip()
    src_ids = edge_df.iloc[:, 0].astype(str).str.strip().to_numpy()
    dst_ids = edge_df.iloc[:, 1].astype(str).str.strip().to_numpy()
    valid = np.array(
        [(src in node_id_to_idx and dst in node_id_to_idx)
         for src, dst in zip(src_ids, dst_ids)],
        dtype=bool,
    )
    src_idx = np.array([node_id_to_idx[src] for src in src_ids[valid]], dtype=np.int64)
    dst_idx = np.array([node_id_to_idx[dst] for dst in dst_ids[valid]], dtype=np.int64)
    all_src = np.concatenate([src_idx, dst_idx])
    all_dst = np.concatenate([dst_idx, src_idx])
    adjacency = torch.sparse_coo_tensor(
        torch.tensor(np.stack([all_src, all_dst]), dtype=torch.long),
        torch.ones(len(all_src), dtype=torch.float32),
        size=(n, n),
    ).coalesce()
    if add_graph_features:
        X_np = add_transaction_graph_features(X_np, src_idx, dst_idx, n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    if split_strategy == "temporal":
        labelled = y_np >= 0
        train_mask[torch.from_numpy(labelled & (node_times <= temporal_train_end))] = True
        val_mask[torch.from_numpy(
            labelled & (node_times > temporal_train_end) &
            (node_times <= temporal_validation_end)
        )] = True
        test_mask[torch.from_numpy(labelled & (node_times > temporal_validation_end))] = True
    else:
        labelled_idx = np.flatnonzero(y_np >= 0)
        perm = np.random.default_rng(seed).permutation(labelled_idx)
        n_train = int(len(perm) * train_ratio)
        n_val = int(len(perm) * val_ratio)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True

    y_safe = y_np.copy()
    y_safe[y_safe < 0] = 0
    return GraphData(
        adjacency=adjacency,
        features=torch.tensor(X_np, dtype=torch.float32),
        labels=torch.tensor(y_safe, dtype=torch.long),
        node_ids=list(map(str, node_ids)),
        node_times=torch.tensor(node_times, dtype=torch.long),
        edge_index=torch.tensor(np.stack([all_src, all_dst]), dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        labelled_mask=torch.from_numpy(y_np >= 0),
        num_classes=2,
        name="elliptic_actors",
    )
