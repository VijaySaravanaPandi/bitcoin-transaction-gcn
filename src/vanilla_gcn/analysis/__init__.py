"""vanilla_gcn.analysis — Receptive field and over-smoothing analysis."""

from vanilla_gcn.analysis.receptive_field import (
    compute_k_hop_neighbors,
    visualize_receptive_field,
)
from vanilla_gcn.analysis.oversmoothing import (
    run_oversmoothing_experiment,
    plot_oversmoothing,
)

__all__ = [
    "compute_k_hop_neighbors",
    "visualize_receptive_field",
    "run_oversmoothing_experiment",
    "plot_oversmoothing",
]
