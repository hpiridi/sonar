# Contributing to SONAR

Thank you for your interest in contributing to SONAR! This guide covers the most common ways to contribute: adding new detectors to the benchmark, reporting issues, and improving the codebase.

## Development Setup

```bash
git clone https://github.com/hpiridi/sonar.git
cd sonar
pip install -e ".[dev]"
```

Verify your setup:

```bash
ruff check sonar/
pytest tests/ -m "not slow" -v
```

## Adding a New Detector

This is the most impactful contribution you can make. SONAR is designed to make benchmarking new detectors straightforward.

### Option A: PyGOD or PyOD Detectors

If your detector is already available in [PyGOD](https://docs.pygod.org/) or [PyOD](https://pyod.readthedocs.io/), you can benchmark it immediately:

```bash
# PyGOD detector
uv run python run_detector.py \
    --dataset-name small \
    --algorithm YOUR_DETECTOR \
    --epoch 5 \
    --output results/small_detectors_results.json

# Run on all three scales (small is auto-downloaded; medium/large require author contact)
uv run python run_detector.py --dataset-name small --algorithm YOUR_DETECTOR --epoch 5 --output results/small_detectors_results.json
uv run python run_detector.py --dataset-name medium --algorithm YOUR_DETECTOR --epoch 5 --output results/medium_detectors_results.json
uv run python run_detector.py --dataset-name large --algorithm YOUR_DETECTOR --epoch 5 --output results/large_detectors_results.json
```

The script automatically deduplicates results — re-running the same detector replaces its previous entry.

### Option B: Custom Detectors

For detectors not in PyGOD/PyOD, write a short script that:

1. Loads the SONAR dataset
2. Fits your detector
3. Produces anomaly scores (one per node)
4. Evaluates with `evaluate_detector()`

```python
from sonar import SONAR, evaluate_detector

dataset = SONAR(root="./data", name="small", anomalies=True)
data = dataset[0]

# --- Your detector here ---
# Must produce a 1-D tensor of anomaly scores, one per node.
# Higher score = more anomalous.
import torch
scores = torch.randn(data.num_nodes)  # replace with your detector
# ---

metrics = evaluate_detector(data.y_outlier, scores)
print(metrics)
# {'roc_auc': ..., 'average_precision': ..., 'recall_at_k': ...}
```

### Submitting Your Results

1. Fork the repository and create a branch: `git checkout -b add-detector-XYZ`
2. Run your detector on the small dataset (and medium/large if available)
3. Commit the updated `results/*.json` files
4. Open a pull request with:
   - Detector name and a brief description
   - A link to the paper or library implementing the detector
   - The command you used to produce the results
   - Hardware details (GPU model, RAM) for reproducibility

## Reporting Issues

When opening an issue, please include:

- Python version (`python --version`)
- PyTorch and PyG versions (`python -c "import torch; print(torch.__version__)"`)
- OS and hardware (GPU model if relevant)
- Full error traceback
- Minimal code to reproduce the problem

## Code Contributions

### Workflow

1. Fork the repo and create a feature branch from `master`
2. Make your changes
3. Run linting and tests:
   ```bash
   ruff check sonar/
   pytest tests/ -m "not slow" -v
   ```
4. Open a pull request against `master`

### Code Style

- Python 3.10+ (use modern type annotations, `X | Y` unions)
- Formatted and linted with [Ruff](https://docs.astral.sh/ruff/) (`line-length = 100`)
- Keep dependencies minimal — the core package (`sonar/`) should only require `torch` and `torch-geometric` for loading data
- Tests use pytest; mark slow tests (network/GPU) with `@pytest.mark.slow`

### What Makes a Good PR

- **Focused**: one logical change per PR
- **Tested**: add or update tests for any new functionality
- **Documented**: update the README if user-facing behavior changes

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE) (code) and [CC-BY-4.0](LICENSE-DATA) (data).
