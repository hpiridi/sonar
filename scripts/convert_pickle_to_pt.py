#!/usr/bin/env python
"""Convert raw pickle graph files to PyTorch .pt format.

This script documents how the raw pickle files (from the data collection pipeline)
were converted to the .pt files used by the SONAR dataset loader.

Usage:
    python scripts/convert_pickle_to_pt.py graphfiles/GRAPH_HASHTAG_SMALL_with_anomalies.pickle SONAR_small_anomalies.pt
    python scripts/convert_pickle_to_pt.py graphfiles/GRAPH_HASHTAG_SMALL_clean.pickle SONAR_small_clean.pt
"""

import argparse
import pickle
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Convert pickle graph files to .pt format"
    )
    parser.add_argument("input", help="Path to input .pickle file")
    parser.add_argument("output", help="Path to output .pt file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading {input_path}...")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    print(f"  Type: {type(data).__name__}")
    if hasattr(data, "num_nodes"):
        print(f"  Nodes: {data.num_nodes}")
    if hasattr(data, "num_edges"):
        print(f"  Edges: {data.num_edges}")
    if hasattr(data, "x") and data.x is not None:
        print(f"  Features: {data.x.shape}")
    if hasattr(data, "y_outlier") and data.y_outlier is not None:
        print(f"  Anomalies: {int(data.y_outlier.sum().item())}")

    print(f"Saving {output_path}...")
    torch.save(data, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done. Output size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
