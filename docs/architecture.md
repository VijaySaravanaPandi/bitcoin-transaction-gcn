# Repository Architecture

This document describes the software architecture of the Vanilla GCN repository.

---

## ASCII Architecture Diagram

```
vanilla-gcn/
│
│  ┌─────────────────────────────────────────────────────────┐
│  │                    Entry Points                         │
│  │  notebooks/  ←  Educational interface (Jupyter)        │
│  │  scripts/    ←  CLI tools (train, evaluate, analysis)  │
│  └────────────────────────┬────────────────────────────────┘
│                           │  imports from
│                           ▼
│  ┌─────────────────────────────────────────────────────────┐
│  │                 src/vanilla_gcn/                        │
│  │                                                         │
│  │  ┌──────────────┐   ┌──────────────┐                   │
│  │  │  config.py   │   │   seed.py    │                   │
│  │  │  (YAML→types)│   │  (set_seed)  │                   │
│  │  └──────────────┘   └──────────────┘                   │
│  │                                                         │
│  │  ┌──────────────────────────────────────────────────┐  │
│  │  │  data/                                           │  │
│  │  │  ├── types.py       ← GraphData dataclass        │  │
│  │  │  ├── synthetic.py   ← Synthetic 12-node graph    │  │
│  │  │  ├── preprocessing.py ← Â, D̂, Ã pipeline        │  │
│  │  │  └── loader.py      ← Generic interface (numpy,  │  │
│  │  │                        NPZ, future loaders)      │  │
│  │  └────────────────────┬─────────────────────────────┘  │
│  │                       │                                  │
│  │                       ▼                                  │
│  │  ┌──────────────────────────────────────────────────┐  │
│  │  │  models/                                         │  │
│  │  │  ├── gcn_layer.py   ← GCNLayer: Ã @ H @ W       │  │
│  │  │  └── vanilla_gcn.py ← VanillaGCN (K layers)     │  │
│  │  └────────────────────┬─────────────────────────────┘  │
│  │                       │                                  │
│  │            ┌──────────┴──────────┐                      │
│  │            ▼                     ▼                      │
│  │  ┌──────────────────┐  ┌──────────────────────────┐    │
│  │  │  training/       │  │  visualization/           │    │
│  │  │  ├── trainer.py  │  │  ├── graph.py             │    │
│  │  │  └── evaluation.py  │  ├── embeddings.py        │    │
│  │  └──────────────────┘  │  └── layers.py            │    │
│  │                        └──────────────────────────┘    │
│  │                                                         │
│  │  ┌──────────────────┐  ┌──────────────────────────┐    │
│  │  │  analysis/       │  │  utils/                   │    │
│  │  │  ├── receptive_  │  │  ├── logging.py           │    │
│  │  │  │   field.py    │  │  └── checkpointing.py     │    │
│  │  │  └── oversmooth- │  └──────────────────────────┘    │
│  │  │      ing.py      │                                   │
│  │  └──────────────────┘                                   │
│  └─────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────────────────┐
│  │  tests/       ← pytest unit & integration tests        │
│  └─────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────────────────┐
│  │  outputs/     ← Generated (gitignored except .gitkeep) │
│  │  ├── figures/     ← Training curves, graph plots       │
│  │  ├── embeddings/  ← Saved embedding tensors            │
│  │  ├── checkpoints/ ← Model .pt files                    │
│  │  └── logs/        ← Training logs                      │
│  └─────────────────────────────────────────────────────────┘
```

---

## Layers of the System

### 1. Data Layer (`data/`)

Responsible for:
- Defining the `GraphData` container
- Generating or loading graph data
- Preprocessing (Â, D̂, Ã)

**Key design principle:** The model never sees the raw dataset format.
Everything is converted to `GraphData` before touching the model.

| Module | Responsibility |
|--------|---------------|
| `types.py` | `GraphData` dataclass with adjacency, features, labels, masks |
| `synthetic.py` | Deterministic 12-node 3-class community graph |
| `preprocessing.py` | `add_self_loops`, `compute_degree_matrix`, `symmetric_normalize`, `prepare_graph` |
| `loader.py` | `load_graph_from_numpy`, `load_graph_from_npz` (generic interfaces) |

### 2. Model Layer (`models/`)

Responsible for:
- The core GCN computation: `Ã @ H @ W`
- Stacking layers into a full model
- Supporting intermediate output extraction

| Module | Responsibility |
|--------|---------------|
| `gcn_layer.py` | `GCNLayer`: single GCN layer, explicit matrix ops |
| `vanilla_gcn.py` | `VanillaGCN`: K-layer model, intermediate outputs, summary |

**Critical design constraint:** No PyTorch Geometric / DGL dependency.
The graph convolution is literally `A_tilde @ H @ self.weight`.

### 3. Training Layer (`training/`)

Responsible for:
- Full-batch training loop
- CrossEntropyLoss + Adam optimiser
- Per-epoch metric tracking

| Module | Responsibility |
|--------|---------------|
| `trainer.py` | `train_gcn(...)` → `TrainingHistory` |
| `evaluation.py` | `accuracy(...)`, `evaluate_gcn(...)` → `EvaluationMetrics` |

### 4. Visualization Layer (`visualization/`)

Responsible for:
- Graph plots (before / after GCN)
- PCA-reduced embedding scatter plots
- Layer-by-layer progression plots
- Training curves

| Module | Responsibility |
|--------|---------------|
| `graph.py` | `visualize_graph`, `visualize_predictions` |
| `embeddings.py` | `visualize_embeddings`, `visualize_training_curves` |
| `layers.py` | `visualize_layer_embeddings` (X → H1 → H2 → logits) |

### 5. Analysis Layer (`analysis/`)

Responsible for:
- K-hop receptive field computation and visualization
- Over-smoothing depth sweep experiments

| Module | Responsibility |
|--------|---------------|
| `receptive_field.py` | `compute_k_hop_neighbors`, `visualize_receptive_field` |
| `oversmoothing.py` | `run_oversmoothing_experiment`, `plot_oversmoothing` |

### 6. Utilities Layer (`utils/`)

Responsible for:
- Centralised logging configuration
- Model checkpoint save / load

| Module | Responsibility |
|--------|---------------|
| `logging.py` | `setup_logging(level, log_file)` |
| `checkpointing.py` | `save_checkpoint`, `load_checkpoint` |

---

## Data Flow

```
Raw graph (A, X, y)
        │
        ▼
  prepare_graph(A, X, y)
        │
        ├── add_self_loops(A)         → Â
        ├── compute_degree_matrix(Â) → D̂
        └── symmetric_normalize(Â)  → Ã
                │
                ▼
        VanillaGCN.forward(X, Ã)
                │
         GCN Layer 1: H¹ = ReLU(Ã X  W¹)
                │
         GCN Layer 2: H² = ReLU(Ã H¹ W²)
                │
         Output Layer: logits = Ã H² W^K
                │
        ┌───────┴───────┐
        ▼               ▼
  CrossEntropyLoss   argmax → ŷ
  (train nodes only)
        │
        ▼
   Adam.step()
```

---

## Dependency Graph

```
config  ←── seed
   ↑          ↑
   │          │
  data ──────→ models
   │              │
   │         training ──→ utils
   │              │
   └──────→ visualization
              analysis
```

No circular imports. `data` does not import from `models`.
`models` does not import from `data` (it receives tensors only).

---

## Extension Points

To add a new dataset:
→ Add a loader function in `data/loader.py` that returns `GraphData`.

To add a new GNN variant:
→ Add a new file in `models/` (e.g. `gat.py`).
→ It can reuse `GCNLayer` or define its own layer.
→ Training/evaluation code works unchanged (expects logits).

To add new visualizations:
→ Add functions to `visualization/` or create new modules.
→ Import from notebooks/scripts as needed.
