"""PyG InMemoryDataset loader for the SONAR benchmark.

Usage::

    from sonar import SONAR

    # Homogeneous graph with injected anomalies (PyGOD-ready)
    dataset = SONAR(root="./data", name="small", anomalies=True)
    data = dataset[0]
    print(data)  # Data(x=[36860, 16], edge_index=[2, 49865], y_outlier=[36860], ...)

    # Heterogeneous graph (original representation)
    dataset = SONAR(root="./data", name="small", representation="heterogeneous")
    data = dataset[0]
    print(data)  # HeteroData(user={x=[18430, 4]}, tweet={x=[18429, 772]}, ...)
"""

from __future__ import annotations

import os.path as osp
from typing import Callable

import torch
from torch_geometric.data import InMemoryDataset

_VALID_NAMES = {"small", "medium", "large"}
_VALID_REPRESENTATIONS = {"homogeneous", "heterogeneous"}

_RELEASE_URL = "https://github.com/hpiridi/sonar/releases/download/v0.1.0/"

# Mapping from (name, anomalies) to (filename, download_url)
_DOWNLOAD_URLS = {
    ("small", True): (
        "SONAR_small_anomalies.pt",
        _RELEASE_URL + "SONAR_small_anomalies.pt",
    ),
    ("small", False): (
        "SONAR_small_clean.pt",
        _RELEASE_URL + "SONAR_small_clean.pt",
    ),
}

_CONTACT = (
    "Medium and large SONAR graphs are available by contacting the authors:\n"
    "  Hari Prasad Piridi  — p20210102@hyderabad.bits-pilani.ac.in\n"
    "  Dipanjan Chakraborty — dipanjan@hyderabad.bits-pilani.ac.in\n"
    "Please include your affiliation and intended use."
)


class SONAR(InMemoryDataset):
    """SONAR: A Large-Scale Social Network Benchmark for Graph Anomaly Detection.

    A hashtag co-occurrence graph dataset derived from social network data,
    available in three scales (small, medium, large) with optional injected
    anomalies.

    +--------+---------+-----------+----------+-----------+
    | Scale  |  Nodes  |   Edges   | Features | Anomalies |
    +========+=========+===========+==========+===========+
    | Small  |  36,860 |    49,865 |       16 |     1,818 |
    +--------+---------+-----------+----------+-----------+
    | Medium | 846,496 | 1,112,995 |       16 |    41,830 |
    +--------+---------+-----------+----------+-----------+
    | Large  |7,410,001|10,204,721 |       16 |   365,861 |
    +--------+---------+-----------+----------+-----------+

    The homogeneous representation provides 16-dimensional node features
    suitable for PyGOD detectors. The heterogeneous representation preserves
    the original 3-type node schema (user, tweet, hashtag) and 7 edge types.

    Args:
        root: Root directory where the dataset should be saved.
        name: Dataset scale — ``"small"``, ``"medium"``, or ``"large"``.
        anomalies: If ``True`` (default), load the variant with injected
            anomalies. If ``False``, load the clean graph.
        representation: ``"homogeneous"`` (default, 16-dim) or
            ``"heterogeneous"`` (original multi-type graph). The heterogeneous
            representation is only available for clean graphs (``anomalies=False``).
        transform: A function/transform that takes in a Data object and returns
            a transformed version.
        pre_transform: A function/transform applied before saving.

    .. note::
        Only the **small** variant is auto-downloaded. Medium and
        large graphs are available by contacting the corresponding authors.

    Reference:
        Piridi et al. "SONAR: A Large-Scale Social Network Benchmark for
        Graph-Based Anomaly Detection." Submitted to SIGIR 2026.
    """

    def __init__(
        self,
        root: str,
        name: str = "small",
        anomalies: bool = True,
        representation: str = "homogeneous",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ) -> None:
        self.name = name.lower()
        self.anomalies = anomalies
        self.representation = representation.lower()

        if self.name not in _VALID_NAMES:
            raise ValueError(
                f"Invalid name '{name}'. Choose from: {sorted(_VALID_NAMES)}"
            )
        if self.representation not in _VALID_REPRESENTATIONS:
            raise ValueError(
                f"Invalid representation '{representation}'. "
                f"Choose from: {sorted(_VALID_REPRESENTATIONS)}"
            )
        if self.representation == "heterogeneous" and self.anomalies:
            raise ValueError(
                "Heterogeneous representation is only available for clean "
                "graphs (anomalies=False). Anomaly-injected graphs use the "
                "homogeneous 16-dim representation."
            )
        if self.name in ("medium", "large"):
            raise FileNotFoundError(
                f"The '{self.name}' dataset is not included in this package.\n"
                f"{_CONTACT}"
            )

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "SONAR", self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "SONAR", self.name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        key = (self.name, self.anomalies)
        if key in _DOWNLOAD_URLS:
            return [_DOWNLOAD_URLS[key][0]]
        return []

    @property
    def processed_file_names(self) -> list[str]:
        suffix = "anomalies" if self.anomalies else "clean"
        rep = self.representation
        return [f"SONAR_{self.name}_{suffix}_{rep}.pt"]

    def download(self) -> None:
        from torch_geometric.data import download_url

        key = (self.name, self.anomalies)
        if key not in _DOWNLOAD_URLS:
            raise FileNotFoundError(
                f"No download available for '{self.name}' "
                f"(anomalies={self.anomalies}).\n{_CONTACT}"
            )
        filename, url = _DOWNLOAD_URLS[key]
        download_url(url, self.raw_dir, filename=filename)

    def process(self) -> None:
        key = (self.name, self.anomalies)
        if key not in _DOWNLOAD_URLS:
            raise FileNotFoundError(
                f"No raw data for '{self.name}' (anomalies={self.anomalies}).\n"
                f"{_CONTACT}"
            )

        raw_path = osp.join(self.raw_dir, _DOWNLOAD_URLS[key][0])
        data = torch.load(raw_path, weights_only=False)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        suffix = "+anomalies" if self.anomalies else "clean"
        return f"SONAR({self.name}, {suffix}, {self.representation})"
