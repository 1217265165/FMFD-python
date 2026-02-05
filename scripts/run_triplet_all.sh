#!/bin/bash
# ============================================================
# Run All Three Paths (Triplet Run)
# ============================================================
# This script runs all three evaluation paths with the same manifest
# to verify consistency.
#
# Paths:
#   Path-A: compare_methods (full evaluation mode)
#   Path-B: brb_diagnosis_cli + aggregate_batch_diagnosis
#   Path-C: eval_module_localization
#
# Output: Output/debug/triplet_runs/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "============================================================"
echo "V-C.1 Triplet Run - All Three Paths"
echo "============================================================"
echo "Working directory: $(pwd)"
echo "Timestamp: $(date -Iseconds)"

# Create output directory
OUTPUT_DIR="Output/debug/triplet_runs"
mkdir -p "$OUTPUT_DIR"

# Step 0: Build unified manifests
echo ""
echo "[Step 0] Building unified manifests..."
python tools/build_unified_manifest.py \
    --labels Output/sim_spectrum/labels.json \
    --out Output/debug/

echo ""
echo "[INFO] Manifests created:"
echo "  - Output/debug/manifest_all_400.json"
echo "  - Output/debug/manifest_fault_300.json"

# Step 1: Path-A (compare_methods with full evaluation)
echo ""
echo "============================================================"
echo "[Path-A] Running compare_methods (full evaluation mode)..."
echo "============================================================"
python compare_methods.py \
    --manifest Output/debug/manifest_fault_300.json \
    --eval_mode full \
    2>&1 | tee "$OUTPUT_DIR/path_a_compare.log"

# Step 2: Path-B (CLI batch + aggregate)
echo ""
echo "============================================================"
echo "[Path-B] Running brb_diagnosis_cli + aggregate..."
echo "============================================================"

# Clear previous batch diagnosis
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

# Step 3: Path-C (eval_module_localization)
echo ""
echo "============================================================"
echo "[Path-C] Running eval_module_localization..."
echo "============================================================"
python tools/eval_module_localization.py \
    --manifest Output/debug/manifest_fault_300.json \
    2>&1 | tee "$OUTPUT_DIR/path_c_eval.log"

# Step 4: Generate consistency report
echo ""
echo "============================================================"
echo "[Step 4] Generating consistency report..."
echo "============================================================"
python tools/report_three_pipeline_consistency.py \
    2>&1 | tee "$OUTPUT_DIR/consistency_report.log"

# Summary
echo ""
echo "============================================================"
echo "TRIPLET RUN COMPLETE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - $OUTPUT_DIR/path_a_compare.log"
echo "  - $OUTPUT_DIR/path_b_cli.log"
echo "  - $OUTPUT_DIR/path_b_aggregate.log"
echo "  - $OUTPUT_DIR/path_c_eval.log"
echo "  - $OUTPUT_DIR/consistency_report.log"
echo ""
echo "Consistency report: Output/debug/three_pipeline_consistency.md"
echo ""
