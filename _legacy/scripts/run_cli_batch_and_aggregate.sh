#!/bin/bash
# ============================================================
# Run CLI Batch and Aggregate (Path-B)
# ============================================================

echo "============================================================"
echo "Running CLI Batch and Aggregate Path"
echo "============================================================"

# Step 1: Build unified manifests
echo ""
echo "[Step 1] Building unified manifests..."
python tools/build_unified_manifest.py --labels Output/sim_spectrum/labels.json --out Output/debug/

# Step 2: Run brb_diagnosis_cli with manifest
echo ""
echo "[Step 2] Running brb_diagnosis_cli..."
python brb_diagnosis_cli.py \
  --input_dir Output/sim_spectrum/raw_curves \
  --output Output/batch_diagnosis \
  --labels Output/sim_spectrum/labels.json \
  --manifest Output/debug/manifest_fault_300.json \
  --verbose

# Step 3: Run aggregate_batch_diagnosis
echo ""
echo "[Step 3] Running aggregate_batch_diagnosis..."
python tools/aggregate_batch_diagnosis.py \
  --input_dir Output/batch_diagnosis \
  --output Output/batch_diagnosis/module_localization_report.json \
  --manifest Output/debug/manifest_fault_300.json

echo ""
echo "============================================================"
echo "CLI Batch and Aggregate Path Complete!"
echo "============================================================"
