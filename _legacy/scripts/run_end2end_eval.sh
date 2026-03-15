#!/bin/bash
# P4: End-to-end evaluation script
# This script ensures consistent evaluation across all paths using a unified manifest.
# 
# Usage:
#   ./scripts/run_end2end_eval.sh
#   ./scripts/run_end2end_eval.sh --n_samples 400
#
# All steps use the same manifest to ensure N_eval consistency.

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Default parameters
N_SAMPLES="${1:-400}"
MANIFEST_PATH="Output/debug/eval_manifest.json"
LABELS_PATH="Output/sim_spectrum/labels.json"
RAW_CURVES_DIR="Output/sim_spectrum/raw_curves"
BATCH_DIAGNOSIS_DIR="Output/batch_diagnosis"

echo "============================================================"
echo "End-to-End Evaluation Pipeline"
echo "============================================================"
echo "Working directory: $(pwd)"
echo "Manifest path: $MANIFEST_PATH"
echo "Expected samples: $N_SAMPLES"
echo "============================================================"

# Step 1: Build/update manifest
echo ""
echo "[Step 1/7] Building evaluation manifest..."
python tools/build_eval_manifest.py \
    --labels "$LABELS_PATH" \
    --curves_dir "$RAW_CURVES_DIR" \
    --out "$MANIFEST_PATH"

# Verify manifest exists and has expected sample count
if [ ! -f "$MANIFEST_PATH" ]; then
    echo "[ERROR] Manifest was not created!"
    exit 1
fi

MANIFEST_N=$(python -c "import json; print(json.load(open('$MANIFEST_PATH'))['n_samples'])")
echo "[INFO] Manifest contains $MANIFEST_N samples"

# Step 2: Run system-level comparison (compare_methods)
echo ""
echo "[Step 2/7] Running compare_methods with manifest..."
python compare_methods.py --manifest "$MANIFEST_PATH"

# Step 3: Run batch diagnosis
echo ""
echo "[Step 3/7] Running batch diagnosis with manifest..."
mkdir -p "$BATCH_DIAGNOSIS_DIR"
python brb_diagnosis_cli.py \
    --manifest "$MANIFEST_PATH" \
    --input_dir "$RAW_CURVES_DIR" \
    --output "$BATCH_DIAGNOSIS_DIR" \
    --labels "$LABELS_PATH"

# Step 4: Aggregate batch diagnosis
echo ""
echo "[Step 4/7] Aggregating batch diagnosis results..."
python tools/aggregate_batch_diagnosis.py \
    --manifest "$MANIFEST_PATH" \
    --input_dir "$BATCH_DIAGNOSIS_DIR" \
    --output "$BATCH_DIAGNOSIS_DIR/module_localization_report.json"

# Step 5: Evaluate module level
echo ""
echo "[Step 5/7] Running module-level evaluation..."
python tools/eval_module_localization.py --manifest "$MANIFEST_PATH"

# Step 6: Generate final summary
echo ""
echo "[Step 6/7] Generating final summary..."

SUMMARY_PATH="Output/debug/final_summary.md"
cat > "$SUMMARY_PATH" << EOF
# End-to-End Evaluation Summary

## Configuration
- Manifest: $MANIFEST_PATH
- N_eval: $MANIFEST_N
- Generated at: $(date -Iseconds)

## Results

See individual reports:
- System-level comparison: Output/compare_methods/
- Batch diagnosis: $BATCH_DIAGNOSIS_DIR/
- Module localization: Output/module_eval/

## Provenance
- Manifest hash: $(python -c "import json; print(json.load(open('$MANIFEST_PATH'))['manifest_hash'])")
- Labels hash: $(python -c "import json; print(json.load(open('$MANIFEST_PATH'))['labels_hash'])")
EOF

echo "[INFO] Final summary saved to: $SUMMARY_PATH"

# Step 7: Consistency check
echo ""
echo "[Step 7/7] Checking N_eval consistency..."

# Check aggregate report N_eval
if [ -f "$BATCH_DIAGNOSIS_DIR/metrics_provenance.json" ]; then
    AGG_N=$(python -c "import json; print(json.load(open('$BATCH_DIAGNOSIS_DIR/metrics_provenance.json'))['n_eval'])")
    echo "[INFO] Aggregate N_eval: $AGG_N"
    
    # Compare with manifest
    if [ "$AGG_N" != "$MANIFEST_N" ] && [ "$AGG_N" != "0" ]; then
        echo "[WARN] N_eval mismatch: manifest=$MANIFEST_N, aggregate=$AGG_N"
        echo "[WARN] This may be expected if not all samples were diagnosed"
    fi
fi

echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "============================================================"
