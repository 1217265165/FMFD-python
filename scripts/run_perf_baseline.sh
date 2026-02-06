#!/bin/bash
# ============================================================
# Performance Baseline Run
# ============================================================
# Creates a timestamped performance snapshot for comparison.
# Uses manifest_fault_300 for consistent evaluation across paths.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="Output/debug/perf_runs/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "V-C.2 Performance Baseline Run"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Ensure manifests exist
if [ ! -f "Output/debug/manifest_fault_300.json" ]; then
    echo "[Step 0] Building unified manifests..."
    python tools/build_unified_manifest.py \
        --labels Output/sim_spectrum/labels.json \
        --out Output/debug/
fi

# Path-A: compare_methods (full evaluation)
echo ""
echo "[Path-A] Running compare_methods..."
python compare_methods.py \
    --manifest Output/debug/manifest_fault_300.json \
    --eval_mode full \
    2>&1 | tee "$OUTPUT_DIR/path_a.log"

# Path-B: CLI batch + aggregate
echo ""
echo "[Path-B] Running brb_diagnosis_cli + aggregate..."
rm -rf Output/batch_diagnosis/*.json 2>/dev/null || true
mkdir -p Output/batch_diagnosis

python brb_diagnosis_cli.py \
    --manifest Output/debug/manifest_fault_300.json \
    --input_dir Output/sim_spectrum/raw_curves \
    --output Output/batch_diagnosis \
    --labels Output/sim_spectrum/labels.json \
    --verbose \
    2>&1 | tee "$OUTPUT_DIR/path_b_cli.log"

python tools/aggregate_batch_diagnosis.py \
    --manifest Output/debug/manifest_fault_300.json \
    --input_dir Output/batch_diagnosis \
    --output Output/batch_diagnosis/module_localization_report.json \
    2>&1 | tee "$OUTPUT_DIR/path_b_aggregate.log"

# Path-C: eval_module_localization
echo ""
echo "[Path-C] Running eval_module_localization..."
python tools/eval_module_localization.py \
    --manifest Output/debug/manifest_fault_300.json \
    2>&1 | tee "$OUTPUT_DIR/path_c.log"

# Consistency report
echo ""
echo "[Step 4] Generating consistency report..."
python tools/report_three_pipeline_consistency.py 2>&1 | tee "$OUTPUT_DIR/consistency.log"

# Copy key outputs
cp -f Output/debug/three_pipeline_consistency.md "$OUTPUT_DIR/" 2>/dev/null || true
cp -f Output/debug/three_pipeline_consistency.json "$OUTPUT_DIR/" 2>/dev/null || true
cp -f Output/batch_diagnosis/module_localization_report.json "$OUTPUT_DIR/" 2>/dev/null || true

# Create summary
echo ""
echo "============================================================"
echo "Performance Baseline Complete: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "Run 'python tools/report_perf_delta.py --baseline $OUTPUT_DIR --current <new_run_dir>'"
echo "to compare with a future run."
