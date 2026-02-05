# P0: Inference Entry Point Trace

This document traces the actual inference function calls for each of the three evaluation paths.

## Path 1: Right-Click Path (compare_methods)

```
run_baseline.py
    └── baseline/baseline.py::build_baseline()
    
run_simulation_brb.py
    └── pipelines/simulate/run_simulation_brb.py::main()
        └── Generates: Output/sim_spectrum/raw_curves/*.csv
        └── Generates: Output/sim_spectrum/labels.json
        └── Generates: Output/sim_spectrum/features_brb.csv

compare_methods.py
    └── pipelines/compare_methods.py::main()
        └── OursAdapter.predict(X_test)
            └── methods/ours_adapter.py::infer_system_and_modules()
                └── System: BRB/system_brb.py + RF gating prior
                └── Module: BRB/module_brb.py::hierarchical_module_infer_soft_gating()
```

## Path 2: Diagnosis CLI Path

```
brb_diagnosis_cli.py --input_dir ... --output ... --labels ...
    └── methods/ours_adapter.py::infer_system_and_modules()
        └── System: BRB/system_brb.py + RF gating prior  
        └── Module: BRB/module_brb.py::hierarchical_module_infer_soft_gating()
    └── Outputs: Output/batch_diagnosis/*_diagnosis.json

tools/aggregate_batch_diagnosis.py
    └── Reads: Output/batch_diagnosis/*_diagnosis.json
    └── Computes: sys_acc, mod_top1, mod_top3
    └── Outputs: module_localization_report.json
```

## Path 3: Right-Click eval_module_localization

```
tools/eval_module_localization.py
    └── methods/ours_adapter.py::infer_system_and_modules()
        └── System: BRB/system_brb.py + RF gating prior
        └── Module: BRB/module_brb.py::hierarchical_module_infer_soft_gating()
    └── (Fallback): module_level_infer_with_activation() if exception
```

## Unified Inference Entry Point

All three paths now use the **same** unified entry point:

```python
methods/ours_adapter.py::infer_system_and_modules(
    features: Dict[str, float],
    use_gating: bool = True,
    rf_classifier: Optional = None,
    allow_fallback: bool = False
) -> Dict
```

This function:
1. Runs BRB system-level inference
2. Optionally fuses with RF gating prior
3. Uses soft-gating module inference (hierarchical_module_infer_soft_gating)
4. Returns system_probs, fault_type_pred, module_topk

## Module Metrics Computation

All three paths use the unified metrics from:

```python
metrics/module_localization_metrics.py::compute_mod_topk()
```

## Verification

To verify consistency:
1. Run all three paths on the same dataset
2. Compare N_eval, sys_acc, mod_top1, mod_top3
3. Maximum allowed difference: ≤ 1 sample
