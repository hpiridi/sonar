"""Tests for sonar.utils (evaluate_detector and dataset_summary)."""

import pytest
import torch


class TestDatasetSummary:
    def test_homogeneous_summary(self, homo_data):
        from sonar.utils import dataset_summary

        summary = dataset_summary(homo_data)
        assert summary["type"] == "homogeneous"
        assert summary["num_nodes"] == 100
        assert summary["num_edges"] == 200
        assert summary["num_features"] == 16
        assert summary["num_anomalies"] == 10
        assert summary["anomaly_ratio"] == 0.1

    def test_heterogeneous_summary(self, hetero_data):
        from sonar.utils import dataset_summary

        summary = dataset_summary(hetero_data)
        assert summary["type"] == "heterogeneous"
        assert "user" in summary["node_types"]
        assert "tweet" in summary["node_types"]
        assert "hashtag" in summary["node_types"]
        assert summary["num_nodes"]["user"] == 50
        assert summary["num_nodes"]["tweet"] == 30
        assert summary["num_nodes"]["hashtag"] == 20
        assert summary["num_features"]["user"] == 4
        assert summary["num_features"]["tweet"] == 772
        assert summary["num_features"]["hashtag"] == 16

    def test_homogeneous_no_anomalies(self):
        from torch_geometric.data import Data

        from sonar.utils import dataset_summary

        data = Data(
            x=torch.randn(50, 8),
            edge_index=torch.randint(0, 50, (2, 100)),
        )
        summary = dataset_summary(data)
        assert summary["type"] == "homogeneous"
        assert summary["num_nodes"] == 50
        assert summary["num_features"] == 8
        assert "num_anomalies" not in summary

    def test_homogeneous_no_features(self):
        from torch_geometric.data import Data

        from sonar.utils import dataset_summary

        data = Data(
            x=None,
            edge_index=torch.randint(0, 10, (2, 20)),
            num_nodes=10,
        )
        summary = dataset_summary(data)
        assert summary["num_features"] == 0

    def test_all_anomalies(self):
        from torch_geometric.data import Data

        from sonar.utils import dataset_summary

        data = Data(
            x=torch.randn(20, 4),
            edge_index=torch.randint(0, 20, (2, 30)),
            y_outlier=torch.ones(20, dtype=torch.long),
        )
        summary = dataset_summary(data)
        assert summary["num_anomalies"] == 20
        assert summary["anomaly_ratio"] == 1.0

    def test_zero_anomalies(self):
        from torch_geometric.data import Data

        from sonar.utils import dataset_summary

        data = Data(
            x=torch.randn(20, 4),
            edge_index=torch.randint(0, 20, (2, 30)),
            y_outlier=torch.zeros(20, dtype=torch.long),
        )
        summary = dataset_summary(data)
        assert summary["num_anomalies"] == 0
        assert summary["anomaly_ratio"] == 0.0


class TestEvaluateDetector:
    @pytest.mark.slow
    def test_perfect_separation(self, binary_labels, perfect_scores):
        from sonar.utils import evaluate_detector

        metrics = evaluate_detector(binary_labels, perfect_scores)
        assert "roc_auc" in metrics
        assert "average_precision" in metrics
        assert "recall_at_k" in metrics
        assert metrics["roc_auc"] == 1.0
        assert metrics["average_precision"] == 1.0
        assert metrics["recall_at_k"] == 1.0

    @pytest.mark.slow
    def test_custom_k(self, binary_labels, perfect_scores):
        from sonar.utils import evaluate_detector

        metrics = evaluate_detector(binary_labels, perfect_scores, k=5)
        assert "recall_at_k" in metrics
        assert metrics["recall_at_k"] == 1.0

    @pytest.mark.slow
    def test_return_types(self, binary_labels, random_scores):
        from sonar.utils import evaluate_detector

        metrics = evaluate_detector(binary_labels, random_scores)
        assert isinstance(metrics, dict)
        for key in ("roc_auc", "average_precision", "recall_at_k"):
            assert isinstance(metrics[key], float)
            assert 0.0 <= metrics[key] <= 1.0

    @pytest.mark.slow
    def test_default_k_equals_num_anomalies(self, binary_labels, perfect_scores):
        from sonar.utils import evaluate_detector

        metrics = evaluate_detector(binary_labels, perfect_scores)
        # Default k = sum of labels = 10
        assert metrics["recall_at_k"] == 1.0
