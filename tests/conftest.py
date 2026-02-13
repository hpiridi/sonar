"""Shared test fixtures for SONAR tests."""

import pytest
import torch
from torch_geometric.data import Data, HeteroData


@pytest.fixture
def homo_data():
    """A small homogeneous graph mimicking SONAR structure."""
    num_nodes = 100
    num_edges = 200
    num_features = 16

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y_outlier = torch.zeros(num_nodes, dtype=torch.long)
    y_outlier[:10] = 1  # 10% anomalies

    return Data(x=x, edge_index=edge_index, y_outlier=y_outlier)


@pytest.fixture
def hetero_data():
    """A small heterogeneous graph mimicking SONAR structure."""
    data = HeteroData()

    data["user"].x = torch.randn(50, 4)
    data["tweet"].x = torch.randn(30, 772)
    data["hashtag"].x = torch.randn(20, 16)

    data["user", "posts", "tweet"].edge_index = torch.randint(0, 30, (2, 40))
    data["user", "posts", "tweet"].edge_index[0] = torch.randint(0, 50, (40,))

    data["tweet", "contains", "hashtag"].edge_index = torch.tensor(
        [list(range(30)) + list(range(10)), list(range(20)) + list(range(20))]
    )

    return data


@pytest.fixture
def binary_labels():
    """Binary labels tensor: 90 normal, 10 anomalies."""
    labels = torch.zeros(100, dtype=torch.long)
    labels[:10] = 1
    return labels


@pytest.fixture
def perfect_scores(binary_labels):
    """Scores that perfectly separate anomalies from normals."""
    scores = torch.zeros(100, dtype=torch.float)
    scores[:10] = 1.0  # anomalies get high scores
    return scores


@pytest.fixture
def random_scores():
    """Random scores (no discrimination)."""
    return torch.rand(100)
