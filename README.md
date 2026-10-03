## Two Implementations — Same Architecture

| | 🔴 PyTorch Version | 🔵 NumPy-Only Version |
|---|---|---|
| **Location** | `src/vanilla_gcn/` | `src/vanilla_gcn_numpy/` |
| **Dependencies** | `torch`, `numpy` | `numpy` only |
| **Autograd** | PyTorch (`loss.backward()`) | Manual chain-rule backprop |
| **GCN equation** | `A_tilde @ H @ W` | `A_tilde @ H @ W` (identical) |
| **Preprocessing** | Same math → `torch.Tensor` | Same math → `np.ndarray` |
| **Checkpointing** | `torch.save` `.pt` format | `np.savez` `.npz` format |

> **Both versions implement exactly:** $H^{(k)} = \sigma\!\left(\tilde{A}\,H^{(k-1)}\,W^{(k)}\right)$

---

## Quick Start — Google Colab

Open [`notebooks/GCN_Colab_Main.ipynb`](notebooks/GCN_Colab_Main.ipynb) in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/vanilla-gcn/blob/main/notebooks/GCN_Colab_Main.ipynb)

The notebook is **fully self-contained** — all code is defined inline. It:
- Installs all dependencies automatically
- Lets you upload your own dataset (CSV, NPZ, NumPy `.npy`, or Google Drive)
- Trains both versions and compares them side-by-side
- Shows training curves, PCA embeddings, and over-smoothing analysis
- Downloads all outputs to your machine

---



---

## Mathematical Formulation

The core GCN layer equation:

$$H^{(k)} = \sigma\!\left(\tilde{A}\, H^{(k-1)}\, W^{(k)}\right)$$

where:

| Symbol | Definition |
|--------|-----------|
| $\hat{A} = A + I$ | Adjacency with self-loops |
| $\hat{D}_{ii} = \sum_j \hat{A}_{ij}$ | Degree matrix of $\hat{A}$ |
| $\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ | Symmetrically normalized adjacency |
| $H^{(0)} = X$ | Input node features |
| $W^{(k)}$ | Learnable weight matrix at layer $k$ |
| $\sigma$ | Non-linear activation (ReLU for hidden, none for output) |

The graph convolution is implemented literally as `A_tilde @ H @ W`.

---

## Architecture

```
Input X  (N × F)
    ↓
GCN Layer 1:  H¹ = ReLU( Ã X  W¹ )    (N × hidden_dim)
    ↓
GCN Layer 2:  H² = ReLU( Ã H¹ W² )   (N × hidden_dim)
    ↓
Output Layer: logits = Ã H² W^K       (N × num_classes)
    ↓
ŷ_u = argmax_c logits_{u,c}
```

---

## Repository Structure

```
vanilla-gcn/
├── README.md
├── LICENSE                 MIT
├── .gitignore
├── .editorconfig
├── .python-version         3.11
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
│
├── configs/
│   ├── default.yaml        All hyperparameters
│   └── experiments.yaml    Named experiment configs
│
├── data/
│   ├── raw/               Original datasets (gitignored)
│   ├── processed/         Processed tensors (gitignored)
│   └── README.md
│
├── notebooks/
│   ├── 01_gcn_fundamentals.ipynb   Step-by-step GCN tutorial
│   ├── 02_gcn_training.ipynb       Training, checkpoints, embeddings
│   └── 03_gcn_over_smoothing.ipynb Over-smoothing analysis
│
├── src/vanilla_gcn/
│   ├── config.py           YAML → typed dataclasses
│   ├── seed.py             set_seed(seed)
│   ├── data/               GraphData, synthetic, preprocessing, loader
│   ├── models/             GCNLayer, VanillaGCN
│   ├── training/           train_gcn, evaluate_gcn
│   ├── visualization/      graph, embeddings, layers
│   ├── analysis/           receptive field, over-smoothing
│   └── utils/              logging, checkpointing
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_oversmoothing.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_gcn_layer.py
│   ├── test_model.py
│   ├── test_training.py
│   └── test_reproducibility.py
│
├── outputs/
│   ├── figures/            Training curves, graph plots
│   ├── embeddings/         Saved embedding tensors
│   ├── checkpoints/        Model .pt files
│   └── logs/               Training logs
│
└── docs/
    ├── mathematics.md      Full mathematical derivation
    ├── architecture.md     Software architecture
    └── dataset_interface.md How to add real-world datasets
```

---

## Installation

```bash
# Clone
git clone <your-repo-url>
cd vanilla-gcn

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Install runtime dependencies
pip install -r requirements.txt
pip install -e .

# Install dev dependencies (tests, linting, Jupyter)
pip install -r requirements-dev.txt
```

---

## Environment Setup

Python 3.11 is required. Runtime dependencies:

| Package | Purpose |
|---------|---------|
| `torch` | Tensors, autograd, nn.Module |
| `numpy` | Array operations |
| `pandas` | Data utilities |
| `matplotlib` | Visualizations |
| `networkx` | Graph layout |
| `scikit-learn` | PCA for embedding visualization |
| `pyyaml` | Config loading |

---

## Running the Notebook

```bash
make notebook
# or
jupyter lab notebooks/
```

Start with **`01_gcn_fundamentals.ipynb`** for the complete step-by-step tutorial.

---

## Running Training

```bash
# Default config
make train

# or directly
python scripts/train.py --config configs/default.yaml

# Override seed and epochs
python scripts/train.py --seed 0 --epochs 300
```

---

## Running Evaluation

```bash
make evaluate
# or
python scripts/evaluate.py --checkpoint outputs/checkpoints/vanilla_gcn.pt
```

---

## Running Tests

```bash
make test
# or with coverage
make test-cov
```

Expected output: all 30+ tests pass.

---

## Running the Over-Smoothing Experiment

```bash
make oversmoothing
# or
python scripts/run_oversmoothing.py --depths 1 2 3 5 8 10
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
seed: 42

model:
  hidden_dim: 16
  num_layers: 2
  dropout: 0.0

training:
  learning_rate: 0.01
  weight_decay: 0.0005
  epochs: 200

data:
  train_ratio: 0.6
  validation_ratio: 0.2
  test_ratio: 0.2
```

Load in Python:

```python
from vanilla_gcn.config import load_config
cfg = load_config("configs/default.yaml")
print(cfg.model.hidden_dim)  # 16
```

---

## Dataset Interface

The GCN model accepts any graph as `(A, X, y)` NumPy arrays:

```python
from vanilla_gcn.data.loader import load_graph_from_numpy
import numpy as np

A = np.load("data/raw/my_adjacency.npy")  # (N, N) float32
X = np.load("data/raw/my_features.npy")   # (N, F) float32
y = np.load("data/raw/my_labels.npy")     # (N,)   int64

data = load_graph_from_numpy(A, X, y, name="my_dataset")
# Then use the same train_gcn / evaluate_gcn pipeline
```

See [`docs/dataset_interface.md`](docs/dataset_interface.md) for full details including NetworkX, CSV, and NPZ loaders.

---

## Model Architecture

```python
from vanilla_gcn.models.vanilla_gcn import VanillaGCN

model = VanillaGCN(
    input_dim=4,     # F = number of input features
    hidden_dim=16,   # hidden dimension
    num_classes=3,   # C = number of output classes
    num_layers=2,    # K = GCN depth
)
model.print_summary()
```

### Intermediate outputs

```python
out = model(X, A_tilde, return_intermediate=True)
out['input']    # H^(0) = X           (N × F)
out['layer_1']  # H^(1)               (N × hidden_dim)
out['logits']   # Final output        (N × num_classes)
```

---

## Visualization

```python
from vanilla_gcn.visualization.graph import visualize_graph, visualize_predictions
from vanilla_gcn.visualization.embeddings import visualize_embeddings
from vanilla_gcn.visualization.layers import visualize_layer_embeddings

# Original graph
visualize_graph(data.adjacency, data.labels, data.node_ids)

# Before vs After
visualize_predictions(data.adjacency, data.labels, predictions, data.node_ids)

# PCA embeddings
visualize_embeddings(embeddings, data.labels, data.node_ids)

# Layer progression
visualize_layer_embeddings(model.get_embeddings(X, A_tilde), data.labels)
```

---

## Over-Smoothing Experiment

```python
from vanilla_gcn.analysis.oversmoothing import run_oversmoothing_experiment, plot_oversmoothing

result = run_oversmoothing_experiment(
    A=data.adjacency, X=data.features, y=data.labels,
    train_mask=data.train_mask, val_mask=data.val_mask,
    depths=[1, 2, 3, 5, 8, 10],
)
plot_oversmoothing(result)
```

---

## Saving / Loading Models

```python
from vanilla_gcn.utils.checkpointing import save_checkpoint, load_checkpoint

save_checkpoint(model, optimizer, cfg, history, epoch=200,
                path="outputs/checkpoints/model.pt")

model, optimizer, cfg_dict, history, epoch = load_checkpoint(
    "outputs/checkpoints/model.pt", model, optimizer
)
```

---

## Replacing the Synthetic Dataset

The model is **completely agnostic** to the data source. Replace:

```python
data = create_synthetic_graph()
```

with:

```python
data = load_graph_from_numpy(A_real, X_real, y_real, name="cora")
```

No changes to model, training, or evaluation code.

---

## Mathematical Notation

| Symbol | Description |
|--------|-------------|
| $G=(V,E)$ | Graph |
| $N = |V|$ | Number of nodes |
| $F$ | Number of input features |
| $C$ | Number of classes |
| $A \in \mathbb{R}^{N \times N}$ | Raw adjacency |
| $\hat{A} = A + I$ | Self-loop adjacency |
| $\hat{D}_{ii} = \sum_j \hat{A}_{ij}$ | Augmented degree |
| $\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ | Normalized adjacency |
| $X \in \mathbb{R}^{N \times F}$ | Node features |
| $H^{(k)} \in \mathbb{R}^{N \times d_k}$ | Node embeddings at layer $k$ |
| $W^{(k)} \in \mathbb{R}^{d_{k-1} \times d_k}$ | Weight matrix |
| $\sigma$ | Activation (ReLU) |

---

## Reference

- Hamilton, W.L. (2020). *Graph Representation Learning*. Synthesis Lectures on AI and ML.
- Kipf, T.N. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- Li, Q., Han, Z., & Wu, X.M. (2018). *Deeper Insights into GCNs for Semi-Supervised Classification*. AAAI.

---

## License

MIT — see [`LICENSE`](LICENSE).
