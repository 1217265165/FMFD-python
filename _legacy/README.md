# Legacy Code Archive

This directory contains deprecated code archived during the V-E.14 project cleanup.

## Archive Contents

### tools/ (31 scripts)
Development/debugging utilities that are NOT imported by the current pipeline.
Moved here to keep the active `tools/` directory clean (only 3 files remain).

### docs/ (6 markdown files)
Old audit reports and temporary design notes. Current documentation lives in `docs/`.

### scripts/ (9 shell/batch scripts)
Automation scripts for earlier pipeline versions. Current execution is documented in the root `README.md`.

### BRB/ (3 files)
Old per-fault-type BRB engines, replaced by the unified layered engine.

### feature_pool.py
Deprecated feature pool, replaced by `config/feature_definitions.json`.

### check_sim_labels.py
Simulation label validator — useful for debugging but not part of the core pipeline.

### pipelines/ (4 scripts + apply/)
Unused pipeline entry points not imported by any active module:
- `eval_system_level.py` — standalone system-level evaluation (functionality merged into `compare_methods.py`)
- `evaluate_module_level.py` — standalone module-level evaluation (functionality merged into `compare_methods.py`)
- `generate_features.py` — standalone feature generation (integrated into `simulate/run_simulation_brb.py`)
- `visualize_results.py` — standalone visualization (integrated into `compare_methods.py`)
- `apply/validate_against_labels.py` — label validation utility

## Archive Date
2026-03-02 (V-E.14 cleanup)
