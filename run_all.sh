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
    DMGD
    DOMINANT
    DONE
    GAAN
    GADNR
    GAE
    GUIDE
    OCGNN
    ONE
    Radar
    SCAN
    AdONE
    LOF
    IF
    MLPAE
)

DATASET_LABEL="$(basename "$DATASET")"

for det in "${DETECTORS[@]}"; do
    # Skip if result already exists for this algorithm+dataset
    if [ -f "$OUTPUT" ] && python3 -c "
import json, sys
with open('$OUTPUT') as f:
    data = json.load(f)
sys.exit(0 if any(r.get('algorithm') == '$det' and r.get('dataset') == '$DATASET_LABEL' for r in data) else 1)
" 2>/dev/null; then
        echo "SKIP: $det (result already in $OUTPUT)"
        continue
    fi

    echo "============================================================"
    echo "Running $det ..."
    echo "============================================================"
    uv run --extra benchmark python run_detector.py \
        --dataset "$DATASET" \
        --algorithm "$det" \
        --output "$OUTPUT" \
        --epoch "$EPOCHS" \
        --gpu "$GPU" || echo "FAILED: $det"
    echo ""
done

echo "============================================================"
echo "All done. Results in $OUTPUT"
