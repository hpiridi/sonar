import argparse
import inspect
import json
import pickle
import time
from pathlib import Path

import pygod.detector as detectors
from pygod.metric import eval_average_precision, eval_recall_at_k, eval_roc_auc

AVAILABLE_DETECTORS = [
    "AdONE", "ANOMALOUS", "AnomalyDAE", "CoLA", "CONAD", "DMGD",
    "DOMINANT", "DONE", "GAAN", "GADNR", "GAE", "GUIDE", "OCGNN",
    "ONE", "Radar", "SCAN",
]


def main():
    parser = argparse.ArgumentParser(description="Run a PyGOD outlier detector on a graph dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Path to a .pickle file containing a PyG Data object")
    group.add_argument(
        "--dataset-name",
        choices=["small", "medium", "large"],
        help="Load a SONAR dataset by name (small/medium/large)",
    )
    parser.add_argument("--algorithm", required=True, choices=AVAILABLE_DETECTORS, help="Detector algorithm name")
    parser.add_argument("--output", default="results.json", help="Path to save metrics JSON (default: results.json)")
    parser.add_argument("--epoch", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--contamination", type=float, default=0.1, help="Expected outlier fraction (default: 0.1)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index, -1 for CPU (default: 0)")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = full batch, default: 0)")
    args = parser.parse_args()

    # Load dataset
    if args.dataset_name:
        from sonar import SONAR

        dataset = SONAR(root="./data", name=args.dataset_name, anomalies=True)
        data = dataset[0]
        dataset_label = f"SONAR_{args.dataset_name}"
    else:
        with open(args.dataset, "rb") as f:
            data = pickle.load(f)
        dataset_label = str(Path(args.dataset).name)

    labels = data.y_outlier
    k = int(labels.sum().item())

    print(f"Dataset: {dataset_label}")
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.x.shape[1]}")
    print(f"  Ground truth anomalies: {k} / {data.num_nodes}")
    print(f"Algorithm: {args.algorithm}")
    print()

    # Build detector — some detectors (SCAN, Radar) don't accept gpu/epoch/contamination
    detector_cls = getattr(detectors, args.algorithm)
    sig_params = inspect.signature(detector_cls.__init__).parameters
    detector_kwargs = {}
    if "gpu" in sig_params:
        detector_kwargs["gpu"] = args.gpu
    if "verbose" in sig_params:
        detector_kwargs["verbose"] = 1
    if "epoch" in sig_params:
        detector_kwargs["epoch"] = args.epoch
    if "contamination" in sig_params:
        detector_kwargs["contamination"] = args.contamination
    if "batch_size" in sig_params and args.batch_size > 0:
        detector_kwargs["batch_size"] = args.batch_size
    detector = detector_cls(**detector_kwargs)

    # Fit
    t_start = time.time()
    detector.fit(data)
    t_fit = time.time() - t_start

    # Predict
    pred, score = detector.predict(data, return_pred=True, return_score=True)

    # Metrics
    roc_auc = eval_roc_auc(labels, score)
    ap = eval_average_precision(labels, score)
    recall_at_k = eval_recall_at_k(labels, score, k=k)

    results = {
        "dataset": dataset_label,
        "algorithm": args.algorithm,
        "epoch": args.epoch,
        "contamination": args.contamination,
        "gpu": args.gpu,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.x.shape[1],
        "num_anomalies": k,
        "roc_auc": round(float(roc_auc), 4),
        "average_precision": round(float(ap), 4),
        f"recall_at_{k}": round(float(recall_at_k), 4),
        "outliers_detected": int(pred.sum().item()),
        "threshold": round(float(detector.threshold_), 4),
        "fit_time_seconds": round(t_fit, 2),
    }

    print(f"\nResults:")
    print(f"  ROC-AUC:        {results['roc_auc']:.4f}")
    print(f"  Avg Precision:  {results['average_precision']:.4f}")
    print(f"  Recall@{k}:  {results[f'recall_at_{k}']:.4f}")
    print(f"  Outliers found: {results['outliers_detected']}")
    print(f"  Fit time:       {results['fit_time_seconds']}s")

    # Save — append to JSON array so multiple runs accumulate
    out_path = Path(args.output)
    existing = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    # Replace any previous entry for same algorithm+dataset
    existing = [r for r in existing if not (r.get("algorithm") == args.algorithm and r.get("dataset") == results["dataset"])]
    existing.append(results)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults appended to {args.output} ({len(existing)} total entries)")


if __name__ == "__main__":
    main()
