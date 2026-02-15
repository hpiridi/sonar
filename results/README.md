# Benchmark Results

Pre-computed benchmark results from running PyGOD detectors on SONAR graphs.

## Files

| File | Description |
|------|-------------|
| `small_detectors_results.json` | All 16 detectors on the small graph (36,860 nodes) |
| `medium_detectors_results.json` | Scalable detectors on the medium graph (846,496 nodes) |
| `large_detectors_results.json` | Scalable detectors on the large graph (7,410,001 nodes) |

## Format

Each JSON file contains an array of result objects with the following fields:

```json
{
  "dataset": "graph filename",
  "algorithm": "detector name",
  "epoch": 5,
  "contamination": 0.1,
  "device": "cuda:0",
  "num_nodes": 36860,
  "num_edges": 49865,
  "num_features": 16,
  "num_anomalies": 1818,
  "roc_auc": 0.7997,
  "average_precision": 0.4305,
  "recall_at_<k>": 0.4455,
  "outliers_detected": 3686,
  "threshold": 7.1174,
  "fit_time_seconds": 11.76
}
```

## Reproducing

```bash
# Run all detectors on the small dataset
bash run_all.sh

# Run a single detector
uv run python run_detector.py --dataset-name small --algorithm DOMINANT --epoch 5
```
