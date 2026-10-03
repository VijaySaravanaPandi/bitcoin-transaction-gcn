"""
vanilla_gcn — Vanilla Graph Convolutional Network (GCN) package.

A clean, mathematically transparent implementation of the GCN formulation
from William L. Hamilton's "Graph Representation Learning" (Chapters 5 & 7).

Core equation:
    H^(k) = σ( Ã H^(k-1) W^(k) )

where:
    Ã = D̂^(-1/2) Â D̂^(-1/2)    (symmetrically normalized adjacency)
    Â = A + I                       (adjacency with self-loops)
    D̂_ii = Σ_j Â_ij               (degree matrix of Â)
    H^(0) = X                       (input node features)
    W^(k)                           (learnable weight matrix at layer k)
    σ                               (activation function)

This package does NOT depend on PyTorch Geometric or DGL.
The graph convolution is explicitly implemented as matrix multiplication.

Package layout
--------------
vanilla_gcn/
├── config.py           — YAML config loader
├── seed.py             — global reproducibility seed setter
├── data/               — graph data structures, synthetic data, preprocessing
├── models/             — GCNLayer, VanillaGCN
├── training/           — trainer, evaluation
├── visualization/      — graph, embedding, layer visualizations
├── analysis/           — receptive field, over-smoothing
└── utils/              — logging, checkpointing
"""

from vanilla_gcn.config import GCNConfig, load_config
from vanilla_gcn.seed import set_seed

__all__ = ["load_config", "GCNConfig", "set_seed"]
__version__ = "0.1.0"
