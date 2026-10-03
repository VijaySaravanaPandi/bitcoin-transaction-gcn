"""vanilla_gcn.training — Training and evaluation for Vanilla GCN."""

from vanilla_gcn.training.trainer import train_gcn, TrainingHistory
from vanilla_gcn.training.evaluation import accuracy, evaluate_gcn

__all__ = ["train_gcn", "TrainingHistory", "accuracy", "evaluate_gcn"]
