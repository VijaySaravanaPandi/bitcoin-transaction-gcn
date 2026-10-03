# Data Directory

This directory holds graph datasets used with Vanilla GCN.

## Structure

```
data/
├── raw/          ← Original, unmodified dataset files
│   └── .gitkeep
└── processed/    ← Preprocessed / cached tensors
    └── .gitkeep
```

## Policy

- **Never commit real datasets** unless explicitly requested.
- Raw data goes in `data/raw/`, processed tensors go in `data/processed/`.
- All paths are configurable via `configs/default.yaml` → `paths.data_raw`.
- Always use `pathlib.Path` — never hardcode absolute paths.

## Replacing the Synthetic Dataset

See `docs/dataset_interface.md` for step-by-step instructions on loading
your own graph. The short version:

```python
from vanilla_gcn.data.loader import load_graph_from_numpy
import numpy as np

A = np.load("data/raw/my_adjacency.npy")  # (N, N)
X = np.load("data/raw/my_features.npy")   # (N, F)
y = np.load("data/raw/my_labels.npy")     # (N,)

data = load_graph_from_numpy(A, X, y, name="my_dataset")
```

Then pass `data` directly to the training pipeline — no model changes needed.
