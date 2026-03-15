# P4: End-to-end evaluation script (PowerShell)
# This script ensures consistent evaluation across all paths using a unified manifest.
# 
# Usage:
#   .\scripts\run_end2end_eval.ps1
#   .\scripts\run_end2end_eval.ps1 -NSamples 400
#
# All steps use the same manifest to ensure N_eval consistency.

param (
    [int]$NSamples = 400
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

# Default parameters
$ManifestPath = "Output/debug/eval_manifest.json"
$LabelsPath = "Output/sim_spectrum/labels.json"
$RawCurvesDir = "Output/sim_spectrum/raw_curves"
$BatchDiagnosisDir = "Output/batch_diagnosis"

Write-Host "============================================================"
Write-Host "End-to-End Evaluation Pipeline"
Write-Host "============================================================"
Write-Host "Working directory: $(Get-Location)"
Write-Host "Manifest path: $ManifestPath"
Write-Host "Expected samples: $NSamples"
Write-Host "============================================================"

# Step 1: Build/update manifest
Write-Host ""
Write-Host "[Step 1/7] Building evaluation manifest..."
python tools/build_eval_manifest.py `
    --labels $LabelsPath `
    --curves_dir $RawCurvesDir `
    --out $ManifestPath

# Verify manifest exists
if (-not (Test-Path $ManifestPath)) {
    Write-Host "[ERROR] Manifest was not created!"
    exit 1
}

$ManifestN = python -c "import json; print(json.load(open('$ManifestPath'))['n_samples'])"
Write-Host "[INFO] Manifest contains $ManifestN samples"

# Step 2: Run system-level comparison (compare_methods)
Write-Host ""
Write-Host "[Step 2/7] Running compare_methods with manifest..."
python compare_methods.py --manifest $ManifestPath

# Step 3: Run batch diagnosis
Write-Host ""
Write-Host "[Step 3/7] Running batch diagnosis with manifest..."
New-Item -ItemType Directory -Force -Path $BatchDiagnosisDir | Out-Null
python brb_diagnosis_cli.py `
    --manifest $ManifestPath `
    --input_dir $RawCurvesDir `
    --output $BatchDiagnosisDir `
    --labels $LabelsPath

# Step 4: Aggregate batch diagnosis
Write-Host ""
Write-Host "[Step 4/7] Aggregating batch diagnosis results..."
python tools/aggregate_batch_diagnosis.py `
    --manifest $ManifestPath `
    --input_dir $BatchDiagnosisDir `
    --output "$BatchDiagnosisDir/module_localization_report.json"

# Step 5: Evaluate module level
Write-Host ""
Write-Host "[Step 5/7] Running module-level evaluation..."
python tools/eval_module_localization.py --manifest $ManifestPath

# Step 6: Generate final summary
Write-Host ""
Write-Host "[Step 6/7] Generating final summary..."

$ManifestHash = python -c "import json; print(json.load(open('$ManifestPath'))['manifest_hash'])"
$LabelsHash = python -c "import json; print(json.load(open('$ManifestPath'))['labels_hash'])"
$Timestamp = Get-Date -Format "o"

$SummaryPath = "Output/debug/final_summary.md"
$SummaryContent = @"
# End-to-End Evaluation Summary

## Configuration
- Manifest: $ManifestPath
- N_eval: $ManifestN
- Generated at: $Timestamp

## Results

See individual reports:
- System-level comparison: Output/compare_methods/
- Batch diagnosis: $BatchDiagnosisDir/
- Module localization: Output/module_eval/

## Provenance
- Manifest hash: $ManifestHash
- Labels hash: $LabelsHash
"@

Set-Content -Path $SummaryPath -Value $SummaryContent
Write-Host "[INFO] Final summary saved to: $SummaryPath"

# Step 7: Consistency check
Write-Host ""
Write-Host "[Step 7/7] Checking N_eval consistency..."

$ProvenancePath = "$BatchDiagnosisDir/metrics_provenance.json"
if (Test-Path $ProvenancePath) {
    $AggN = python -c "import json; print(json.load(open('$ProvenancePath'))['n_eval'])"
    Write-Host "[INFO] Aggregate N_eval: $AggN"
    
    if ($AggN -ne $ManifestN -and $AggN -ne "0") {
        Write-Host "[WARN] N_eval mismatch: manifest=$ManifestN, aggregate=$AggN"
        Write-Host "[WARN] This may be expected if not all samples were diagnosed"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Evaluation complete!"
Write-Host "============================================================"
