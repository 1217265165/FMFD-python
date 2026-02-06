# P0.1: Sample Count Trace Documentation

This document traces the default sample counts and behaviors across all evaluation entry points.

## Entry Points and Default Sample Counts

### 1. `pipelines/simulate/run_simulation_brb.py`
- **Default n_samples**: 400 (from `default_paths.py:DEFAULT_N_SAMPLES`)
- **Output directory**: `Output/sim_spectrum/`
- **Overwrites old data**: Yes (regenerates all samples)
- **Implicit sampling**: No

### 2. `pipelines/compare_methods.py` (via `compare_methods.py`)
- **Default n_samples**: Uses all samples in `features_brb.csv` and `labels.json`
- **Output directory**: `Output/compare_methods/`
- **Overwrites old data**: Yes
- **Implicit sampling**: No (uses all available samples)
- **NEW: Manifest support**: Yes (`--manifest` flag)

### 3. `brb_diagnosis_cli.py`
- **Default n_samples**: Processes all CSVs in `--input_dir`
- **Output directory**: `Output/batch_diagnosis/`
- **Overwrites old data**: No (creates separate JSON per sample)
- **Implicit sampling**: No
- **NEW: Manifest support**: Yes (`--manifest` flag)

### 4. `tools/aggregate_batch_diagnosis.py`
- **Default n_samples**: Aggregates all `*_diagnosis.json` in input directory
- **Output directory**: Same as input (or specified)
- **Overwrites old data**: Yes (report files)
- **Implicit sampling**: No
- **NEW: Manifest support**: Yes (`--manifest` flag)

### 5. `tools/eval_module_localization.py`
- **Default n_samples**: Uses all samples in evaluation set
- **Output directory**: `Output/module_eval/`
- **Overwrites old data**: Yes
- **Implicit sampling**: No
- **NEW: Manifest support**: Yes (`--manifest` flag)

### 6. `tools/build_eval_manifest.py`
- **Default n_samples**: All valid samples in labels
- **Output directory**: `Output/debug/`
- **Overwrites old data**: Yes
- **Implicit sampling**: No (builds from full labels)

## Key Configuration Constants

From `pipelines/default_paths.py`:
```python
DEFAULT_N_SAMPLES = 400
DEFAULT_BALANCED = True
SPLIT = (0.6, 0.2, 0.2)  # train/val/test
SEED = 2025
```

## Unified Evaluation with Manifest

To ensure consistent sample counts across all evaluation paths:

1. **Build manifest first**:
   ```bash
   python tools/build_eval_manifest.py --labels Output/sim_spectrum/labels.json
   ```

2. **Use manifest in all evaluation commands**:
   ```bash
   python compare_methods.py --manifest Output/debug/eval_manifest.json
   python brb_diagnosis_cli.py --manifest Output/debug/eval_manifest.json --input_dir Output/sim_spectrum/raw_curves
   python tools/aggregate_batch_diagnosis.py --manifest Output/debug/eval_manifest.json
   python tools/eval_module_localization.py --manifest Output/debug/eval_manifest.json
   ```

## Potential Inconsistency Sources

1. **Generation vs Evaluation**: Simulation generates 400 samples, but evaluation may use a subset
2. **Missing curves/features**: Some samples may be filtered out if curve files or features are missing
3. **Right-click execution**: Running scripts via IDE may use different working directories

## Recommendations

1. Always use the end-to-end script: `scripts/run_end2end_eval.sh` or `scripts/run_end2end_eval.ps1`
2. Always pass `--manifest` to evaluation commands
3. Check `metrics_provenance.json` after each run to verify N_eval
