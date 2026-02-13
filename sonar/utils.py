"""Evaluation metrics and helper utilities for SONAR."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def evaluate_detector(
    labels: torch.Tensor,
    scores: torch.Tensor,
    k: int | None = None,
) -> dict[str, float]:
    """Compute standard anomaly detection metrics.

    Wraps PyGOD evaluation functions to return a single results dict.

    Args:
        labels: Binary ground-truth labels (1 = anomaly).
        scores: Continuous anomaly scores from a detector.
        k: Number of top-scored nodes for Recall@k.
            Defaults to the number of true anomalies.

    Returns:
        Dictionary with keys ``roc_auc``, ``average_precision``, and
        ``recall_at_k``.
    """
    from pygod.metric import eval_average_precision, eval_recall_at_k, eval_roc_auc

    if k is None:
        k = int(labels.sum().item())

    roc_auc = float(eval_roc_auc(labels, scores))
    ap = float(eval_average_precision(labels, scores))
    recall = float(eval_recall_at_k(labels, scores, k=k))

    return {
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(ap, 4),
        "recall_at_k": round(recall, 4),
    }


def dataset_summary(data) -> dict:
    """Return a summary dict describing a PyG Data or HeteroData object.

    Args:
        data: A ``torch_geometric.data.Data`` or ``HeteroData`` object.

    Returns:
        Dictionary with node/edge/feature/anomaly counts.
    """
    from torch_geometric.data import HeteroData

    summary: dict = {}

    if isinstance(data, HeteroData):
        summary["type"] = "heterogeneous"
        summary["node_types"] = data.node_types
        summary["edge_types"] = [
            f"({s}, {r}, {d})" for s, r, d in data.edge_types
        ]
        summary["num_nodes"] = {nt: data[nt].num_nodes for nt in data.node_types}
        summary["num_edges"] = {
            f"({s}, {r}, {d})": data[(s, r, d)].edge_index.size(1)
            for s, r, d in data.edge_types
        }
        summary["num_features"] = {
            nt: data[nt].x.size(1)
            for nt in data.node_types
            if hasattr(data[nt], "x") and data[nt].x is not None
        }
    else:
        summary["type"] = "homogeneous"
        summary["num_nodes"] = data.num_nodes
        summary["num_edges"] = data.num_edges
        summary["num_features"] = data.x.size(1) if data.x is not None else 0
        if hasattr(data, "y_outlier") and data.y_outlier is not None:
            summary["num_anomalies"] = int(data.y_outlier.sum().item())
            summary["anomaly_ratio"] = round(
                summary["num_anomalies"] / summary["num_nodes"], 4
            )

    return summary
