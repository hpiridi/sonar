"""SONAR: A Large-Scale Social Network Benchmark for Graph Anomaly Detection."""

from sonar.dataset import SONAR
from sonar.utils import dataset_summary, evaluate_detector

__version__ = "0.1.0"
__all__ = ["SONAR", "evaluate_detector", "dataset_summary"]
