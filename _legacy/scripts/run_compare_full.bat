@echo off
REM ============================================================
REM Run Compare Full (Path-A)
REM ============================================================

echo ============================================================
echo Running Compare Full Path
echo ============================================================

REM Step 1: Build unified manifests
echo.
echo [Step 1] Building unified manifests...
python tools/build_unified_manifest.py --labels Output/sim_spectrum/labels.json --out Output/debug/

REM Step 2: Run baseline (optional - uncomment if needed)
REM echo.
REM echo [Step 2] Running baseline...
REM python run_baseline.py

REM Step 3: Run compare_methods with full evaluation
echo.
echo [Step 3] Running compare_methods with manifest...
python compare_methods.py --manifest Output/debug/manifest_all_400.json --eval_mode full

echo.
echo ============================================================
echo Compare Full Path Complete!
echo ============================================================
