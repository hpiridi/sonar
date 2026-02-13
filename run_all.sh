#!/usr/bin/env bash
set -e

DATASET="${1:-graphfiles/GRAPH_HASHTAG_SMALL_with_anomalies.pickle}"
OUTPUT="${2:-results/all_detectors_results.json}"
EPOCHS=5
GPU=0

echo "Dataset: $DATASET"
echo "Output:  $OUTPUT"
echo ""

DETECTORS=(
    ANOMALOUS
    AnomalyDAE
    CoLA
    CONAD
    DOMINANT
    DONE
    GAE
    OCGNN
    ONE
    AdONE
)

for det in "${DETECTORS[@]}"; do
    echo "============================================================"
    echo "Running $det ..."
    echo "============================================================"
    uv run python run_detector.py \
        --dataset "$DATASET" \
        --algorithm "$det" \
        --output "$OUTPUT" \
        --epoch "$EPOCHS" \
        --gpu "$GPU" || echo "FAILED: $det"
    echo ""
done

echo "============================================================"
echo "All done. Results in $OUTPUT"
