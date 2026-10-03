"""vanilla_gcn.data — Graph data structures, loaders, and preprocessing."""

from vanilla_gcn.data.types import GraphData
from vanilla_gcn.data.preprocessing import (
    add_self_loops,
    compute_degree_matrix,
    compute_inverse_sqrt_degree,
    symmetric_normalize,
    prepare_graph,
)
from vanilla_gcn.data.synthetic import create_synthetic_graph
from vanilla_gcn.data.loader import load_graph_from_numpy, split_nodes

__all__ = [
    # Data container
    "GraphData",
    # Preprocessing
    "add_self_loops",
    "compute_degree_matrix",
    "compute_inverse_sqrt_degree",
    "symmetric_normalize",
    "prepare_graph",
    # Synthetic
    "create_synthetic_graph",
    # Loader
    "load_graph_from_numpy",
    "split_nodes",
]
