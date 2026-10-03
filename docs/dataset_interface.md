# Dataset Interface

This document explains exactly how to replace the synthetic graph with a
real-world dataset. The GCN model, trainer, and evaluation code are
completely agnostic to the data source — they only depend on three tensors:

```
A  ∈ R^(N × N)    adjacency matrix
X  ∈ R^(N × F)    node feature matrix
y  ∈ Z^(N,)       integer class labels  (range [0, C-1])
```

---

## Step-by-Step: Load Your Own Dataset

### Option A — From NumPy arrays (most common)

```python
import numpy as np
from vanilla_gcn.data.loader import load_graph_from_numpy

# 1. Load your arrays
A = np.load("data/raw/my_adjacency.npy")   # shape (N, N), float or int
X = np.load("data/raw/my_features.npy")    # shape (N, F), float
y = np.load("data/raw/my_labels.npy")      # shape (N,),  int

# 2. Wrap in GraphData
data = load_graph_from_numpy(
    A, X, y,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=42,
    name="my_dataset",
)

# 3. Preprocess  (self-loops → degree → symmetric normalization)
from vanilla_gcn.data.preprocessing import prepare_graph
A_hat, A_tilde, X, y = prepare_graph(data.adjacency, data.features, data.labels)

# 4. Build model  (only input_dim and num_classes change)
from vanilla_gcn.models.vanilla_gcn import VanillaGCN
model = VanillaGCN(
    input_dim=data.num_features,   # F
    hidden_dim=64,
    num_classes=data.num_classes,  # C
    num_layers=2,
)

# 5. Train  (identical API)
from vanilla_gcn.training.trainer import train_gcn
history = train_gcn(
    model, A_tilde, X, y,
    data.train_mask, data.val_mask,
    epochs=200, lr=0.01,
)

# 6. Evaluate
from vanilla_gcn.training.evaluation import evaluate_gcn
metrics = evaluate_gcn(model, A_tilde, X, y,
                       data.train_mask, data.val_mask, data.test_mask)
print(metrics.summary())
```

**Nothing else changes.** The model, training loop, and evaluation code
are completely unchanged.

---

### Option B — From an NPZ archive

```python
from vanilla_gcn.data.loader import load_graph_from_npz

data = load_graph_from_npz(
    "data/raw/cora.npz",
    adjacency_key="adj",     # key name inside the .npz
    features_key="features",
    labels_key="labels",
)
```

---

### Option C — From a NetworkX graph

```python
import networkx as nx
import numpy as np
from vanilla_gcn.data.loader import load_graph_from_numpy

G = nx.karate_club_graph()
A = nx.to_numpy_array(G, dtype=np.float32)          # (N, N)
X = np.eye(G.number_of_nodes(), dtype=np.float32)   # identity features
y = np.array([G.nodes[i]["club"] == "Officer"
              for i in G.nodes], dtype=np.int64)

data = load_graph_from_numpy(A, X, y, name="karate")
```

---

### Option D — From an edge list (CSV)

```python
import pandas as pd
import numpy as np
from vanilla_gcn.data.loader import load_graph_from_numpy

edges = pd.read_csv("data/raw/edges.csv")   # columns: src, dst
features = pd.read_csv("data/raw/feats.csv")
labels = pd.read_csv("data/raw/labels.csv")

N = features.shape[0]
A = np.zeros((N, N), dtype=np.float32)
for _, row in edges.iterrows():
    A[int(row.src), int(row.dst)] = 1.0
    A[int(row.dst), int(row.src)] = 1.0

X = features.values.astype(np.float32)
y = labels["label"].values.astype(np.int64)

data = load_graph_from_numpy(A, X, y, name="csv_graph")
```

---

## Tensor Specifications

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| A | (N, N) | float32 | Binary (0/1) preferred; symmetric for undirected |
| X | (N, F) | float32 | Any real-valued features; normalize if needed |
| y | (N,) | int64 | Values in [0, C-1]; no gaps |

---

## What the Model Receives

The model **never** receives raw data. The training pipeline always
passes preprocessed tensors:

```python
model.forward(
    X      : Tensor (N, F),   # node features
    A_tilde: Tensor (N, N),   # symmetrically normalized adjacency
)
```

The model returns logits of shape `(N, C)`.

---

## Adding a New Loader

To add support for a new format (e.g. JSON, HDF5, citation network):

1. Add a function to `src/vanilla_gcn/data/loader.py`:

```python
def load_graph_from_json(path: str | Path, ...) -> GraphData:
    """Load a graph from a JSON file."""
    # ... parse the format ...
    return load_graph_from_numpy(A, X, y, name=name)
```

2. Export it in `src/vanilla_gcn/data/__init__.py`.

3. Write a test in `tests/`.

No other changes needed — the model, trainer, and evaluator remain unchanged.

---

## Supported Future Datasets (without code changes)

| Dataset | Format | How to load |
|---------|--------|-------------|
| Cora | NPZ | `load_graph_from_npz` |
| Citeseer | NPZ | `load_graph_from_npz` |
| Karate club | NetworkX | Option C above |
| OGB graphs | NumPy arrays | Option A above |
| Custom CSV | CSV edge list | Option D above |
| Knowledge graphs | Processed triples → A, X, y | Option A above |
| Social networks | Adjacency + features | Option A above |
