# Architecture Snapshot

Generated: 2026-02-05T11:14:57.672894

This document answers 8 critical questions about the current code architecture.

---

## 1. System-level inference entry point

**Answer**: `methods/ours_adapter.py::infer_system_and_modules()`

This is the ONLY entry point that all evaluation paths must use:
- `compare_methods.py` → `OursAdapter.predict()` → `infer_system_and_modules()`
- `brb_diagnosis_cli.py` → `infer_system_and_modules()`
- `eval_module_localization.py` → `infer_system_and_modules()`

---

## 2. RF artifact usage

**RF Artifact Path**: `/home/runner/work/FMFD-python/FMFD-python/artifacts/rf_system_classifier.joblib`
**Artifact Exists**: `True`
**Meta Exists**: `True`

**Behavior when RF not found**:
- If `allow_fallback=True`: Silent fallback to BRB-only (gating_status="fallback_brb_only")
- If `allow_fallback=False`: Raises `ValueError`

---

## 3. BRB system-level inference

**Answer**: Yes, BRB `system_level_infer()` is ALWAYS called.

From analysis: `system_level_infer() is called`

BRB output dimension: 4 classes (normal, amp_error, freq_error, ref_error)

---

## 4. Gating fusion method

**Gating Configuration**:
```json
{
  "_version": "1.0",
  "_description": "门控先验融合配置 - 所有入口必须加载此配置",
  "method": "gated",
  "linear_weight": 0.7,
  "logit_weight": 0.6,
  "gated": {
    "threshold": 0.55,
    "w_min": 0.3,
    "w_max": 0.85,
    "temperature": 1.0
  },
  "fallback": {
    "allow_fallback": true,
    "fallback_method": "brb_only"
  }
}
```

**Fusion Formula**: Linear weighted average
```python
fused = alpha * brb_probs + beta * rf_probs
# Default: alpha=0.6, beta=0.4
```

**Analysis**: `GatingPriorFusion is used for RF+BRB fusion`

---

## 5. Final system-level output for fault_type determination

**Answer**: `fused_probs` is used for fault_type prediction (when gating enabled)

Flow:
1. `rf_probs` from RF classifier (if available)
2. `brb_probs` from BRB system_level_infer()
3. `fused_probs` = GatingPriorFusion.fuse(rf_probs, brb_probs)
4. `fault_type_pred` = argmax(fused_probs)

When RF unavailable: `fault_type_pred` = argmax(brb_probs)

---

## 6. Module-level inference entry point

**Answer**: `BRB/module_brb.py::hierarchical_module_infer_soft_gating()`

Fallback: `BRB/module_brb.py::module_level_infer_with_activation()`

Analysis: `hierarchical_module_infer_soft_gating() is used`

---

## 7. Soft-gating multi-subgraph activation

**Answer**: YES, soft-gating exists.

**Trigger Condition**: When `top1_prob - top2_prob < delta` (delta=0.1 by default)

**Fusion Method**:
```python
# Activate top-2 fault hypotheses
score(module) = sum(P(fault_type) * score_type(module))
```

When top1-top2 diff >= delta: Only top1 hypothesis is activated.

---

## 8. Feature pool separation

**Answer**: **NOT IMPLEMENTED IN INFERENCE CODE**

Analysis: `No feature pool separation in inference`

Current state:
- All 22 features (X1-X22) are used uniformly
- No AMP_POOL / FREQ_POOL / REF_POOL separation in `infer_system_and_modules()`
- Feature pools exist only in documentation/planning

---

## Summary

| Question | Answer |
|----------|--------|
| System entry | `infer_system_and_modules()` |
| RF used? | Yes (with fallback) |
| BRB used? | Yes (always) |
| Fusion | Linear (alpha*BRB + beta*RF) |
| Final probs | fused_probs (or brb_probs fallback) |
| Module entry | hierarchical_module_infer_soft_gating() |
| Soft gating? | Yes (delta=0.1 trigger) |
| Feature pools? | NOT implemented |

---

## Truth Fields (from config/eval_truth.json)

- **System truth field**: `system_fault_class`
- **Module truth field**: `module_v2`
- **Module eval policy**: Exclude normal samples

---
