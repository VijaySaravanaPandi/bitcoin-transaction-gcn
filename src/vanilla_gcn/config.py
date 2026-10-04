"""
config.py — Configuration loader for Vanilla GCN.

Reads YAML config files and exposes a typed dataclass so that all
hyperparameters are centrally defined and never hard-coded in source files.

Usage
-----
    from vanilla_gcn.config import load_config

    cfg = load_config("configs/default.yaml")
    print(cfg.model.hidden_dim)       # 16
    print(cfg.training.epochs)        # 200
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Hyperparameters for the GCN architecture."""

    hidden_dim: int = 16
    num_layers: int = 2
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Hyperparameters for model training."""

    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    epochs: int = 200
    class_weighted_loss: bool = True


@dataclass
class DataConfig:
    """Data-split ratios and dataset selection."""

    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2
    # "elliptic" loads transactions; "elliptic_actors" loads wallets;
    # "synthetic" uses the built-in synthetic community graph.
    dataset: str = "elliptic"
    # Directory containing txs_features.csv / txs_classes.csv / txs_edgelist.csv
    raw_dir: str = "data/raw"
    split_strategy: str = "random"
    temporal_train_end: int = 30
    temporal_validation_end: int = 39
    add_graph_features: bool = True


@dataclass
class DebugConfig:
    """Debug mode settings."""

    enabled: bool = True


@dataclass
class PathsConfig:
    """Filesystem paths (relative to project root)."""

    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    outputs: str = "outputs"
    checkpoints: str = "outputs/checkpoints"
    figures: str = "outputs/figures"
    embeddings: str = "outputs/embeddings"
    logs: str = "outputs/logs"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class GCNConfig:
    """Complete configuration for a Vanilla GCN run.

    Attributes
    ----------
    seed:
        Global random seed for reproducibility.
    model:
        Architecture hyperparameters.
    training:
        Training hyperparameters.
    data:
        Data-split ratios.
    debug:
        Debug mode settings.
    paths:
        Output directory paths.
    """

    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> GCNConfig:
    """Load a YAML configuration file and return a :class:`GCNConfig`.

    Parameters
    ----------
    path:
        Path to a YAML config file.  If *None* the built-in defaults are used.

    Returns
    -------
    GCNConfig
        Fully populated configuration object.

    Examples
    --------
    >>> cfg = load_config("configs/default.yaml")
    >>> cfg.seed
    42
    """
    defaults: dict[str, Any] = {
        "seed": 42,
        "model": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.0},
        "training": {"learning_rate": 0.01, "weight_decay": 0.0005, "epochs": 200,
                      "class_weighted_loss": True},
        "data": {"train_ratio": 0.6, "validation_ratio": 0.2, "test_ratio": 0.2,
                  "dataset": "elliptic", "raw_dir": "data/raw",
                  "split_strategy": "random", "temporal_train_end": 30,
                  "temporal_validation_end": 39, "add_graph_features": True},
        "debug": {"enabled": True},
        "paths": {
            "data_raw": "data/raw",
            "data_processed": "data/processed",
            "outputs": "outputs",
            "checkpoints": "outputs/checkpoints",
            "figures": "outputs/figures",
            "embeddings": "outputs/embeddings",
            "logs": "outputs/logs",
        },
    }

    raw: dict[str, Any] = defaults
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        raw = _merge(defaults, loaded)

    return GCNConfig(
        seed=int(raw["seed"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        data=DataConfig(**raw["data"]),
        debug=DebugConfig(**raw["debug"]),
        paths=PathsConfig(**raw["paths"]),
    )
