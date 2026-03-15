@echo off
REM ============================================================
REM Run All Three Paths (Triplet Run)
REM ============================================================
REM This script runs all three evaluation paths with the same manifest
REM to verify consistency.
REM
REM Paths:
REM   Path-A: compare_methods (full evaluation mode)
REM   Path-B: brb_diagnosis_cli + aggregate_batch_diagnosis
REM   Path-C: eval_module_localization
REM
REM Output: Output/debug/triplet_runs/

setlocal enabledelayedexpansion

echo ============================================================
echo V-C.1 Triplet Run - All Three Paths
echo ============================================================
echo Working directory: %CD%
echo Timestamp: %DATE% %TIME%

REM Create output directory
set OUTPUT_DIR=Output\debug\triplet_runs
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Step 0: Build unified manifests
echo.
echo [Step 0] Building unified manifests...
python tools/build_unified_manifest.py --labels Output/sim_spectrum/labels.json --out Output/debug/

echo.
echo [INFO] Manifests created:
echo   - Output/debug/manifest_all_400.json
echo   - Output/debug/manifest_fault_300.json

REM Step 1: Path-A (compare_methods with full evaluation)
echo.
echo ============================================================
echo [Path-A] Running compare_methods (full evaluation mode)...
echo ============================================================
python compare_methods.py --manifest Output/debug/manifest_fault_300.json --eval_mode full > "%OUTPUT_DIR%\path_a_compare.log" 2>&1
type "%OUTPUT_DIR%\path_a_compare.log"

REM Step 2: Path-B (CLI batch + aggregate)
echo.
echo ============================================================
echo [Path-B] Running brb_diagnosis_cli + aggregate...
echo ============================================================

REM Clear previous batch diagnosis
if exist Output\batch_diagnosis\*.json del /Q Output\batch_diagnosis\*.json 2>nul
if not exist Output\batch_diagnosis mkdir Output\batch_diagnosis

python brb_diagnosis_cli.py ^
    --manifest Output/debug/manifest_fault_300.json ^
    --input_dir Output/sim_spectrum/raw_curves ^
    --output Output/batch_diagnosis ^
    --labels Output/sim_spectrum/labels.json ^
    --verbose > "%OUTPUT_DIR%\path_b_cli.log" 2>&1
type "%OUTPUT_DIR%\path_b_cli.log"

python tools/aggregate_batch_diagnosis.py ^
    --manifest Output/debug/manifest_fault_300.json ^
    --input_dir Output/batch_diagnosis ^
    --output Output/batch_diagnosis/module_localization_report.json > "%OUTPUT_DIR%\path_b_aggregate.log" 2>&1
type "%OUTPUT_DIR%\path_b_aggregate.log"

REM Step 3: Path-C (eval_module_localization)
echo.
echo ============================================================
echo [Path-C] Running eval_module_localization...
echo ============================================================
python tools/eval_module_localization.py --manifest Output/debug/manifest_fault_300.json > "%OUTPUT_DIR%\path_c_eval.log" 2>&1
type "%OUTPUT_DIR%\path_c_eval.log"

REM Step 4: Generate consistency report
echo.
echo ============================================================
echo [Step 4] Generating consistency report...
echo ============================================================
python tools/report_three_pipeline_consistency.py > "%OUTPUT_DIR%\consistency_report.log" 2>&1
type "%OUTPUT_DIR%\consistency_report.log"

REM Summary
echo.
echo ============================================================
echo TRIPLET RUN COMPLETE
echo ============================================================
echo.
echo Output files:
echo   - %OUTPUT_DIR%\path_a_compare.log
echo   - %OUTPUT_DIR%\path_b_cli.log
echo   - %OUTPUT_DIR%\path_b_aggregate.log
echo   - %OUTPUT_DIR%\path_c_eval.log
echo   - %OUTPUT_DIR%\consistency_report.log
echo.
echo Consistency report: Output\debug\three_pipeline_consistency.md
echo.

endlocal
