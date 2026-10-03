"""
vanilla_gcn_numpy — Pure NumPy implementation of Vanilla GCN.

Same architecture as the PyTorch version:
    H^(k) = σ( Ã H^(k-1) W^(k) )

No PyTorch / autograd. Gradients computed manually.
"""

from vanilla_gcn_numpy.preprocessing import (
    add_self_loops,
    compute_degree_matrix,
    compute_inverse_sqrt_degree,
    symmetric_normalize,
    prepare_graph,
)
prepare_graph_np = prepare_graph   # convenience alias
from vanilla_gcn_numpy.gcn_layer import GCNLayerNumPy
from vanilla_gcn_numpy.vanilla_gcn import VanillaGCNNumPy
from vanilla_gcn_numpy.trainer import train_gcn_numpy, TrainingHistoryNumPy
from vanilla_gcn_numpy.evaluation import accuracy_numpy, evaluate_gcn_numpy
from vanilla_gcn_numpy.synthetic import create_synthetic_graph_numpy

__all__ = [
    "add_self_loops",
    "compute_degree_matrix",
    "compute_inverse_sqrt_degree",
    "symmetric_normalize",
    "prepare_graph",
    "prepare_graph_np",
    "GCNLayerNumPy",
    "VanillaGCNNumPy",
    "train_gcn_numpy",
    "TrainingHistoryNumPy",
    "accuracy_numpy",
    "create_synthetic_graph_numpy",
    "evaluate_gcn_numpy",
]
