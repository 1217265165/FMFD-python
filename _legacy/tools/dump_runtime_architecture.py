#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump runtime architecture snapshot.

Answers the 8 critical architecture questions and outputs to
Output/debug/architecture_snapshot.md

Usage:
    python tools/dump_runtime_architecture.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_rf_artifact():
    """Check RF artifact status."""
    artifact_path = ROOT / "artifacts" / "rf_system_classifier.joblib"
    meta_path = ROOT / "artifacts" / "rf_meta.json"
    
    status = {
        "artifact_path": str(artifact_path),
        "artifact_exists": artifact_path.exists(),
        "meta_path": str(meta_path),
        "meta_exists": meta_path.exists(),
    }
    
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                status["meta"] = json.load(f)
        except Exception:
            pass
    
    return status


def check_gating_config():
    """Check gating prior configuration."""
    config_path = ROOT / "config" / "gating_prior.json"
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {"error": "Failed to load"}
    return {"error": "File not found", "path": str(config_path)}


def analyze_infer_system_and_modules():
    """Analyze the infer_system_and_modules function."""
    analysis = {
        "file": "methods/ours_adapter.py",
        "function": "infer_system_and_modules()",
        "system_level_backend": None,
        "module_level_backend": None,
        "gating_logic": None,
    }
    
    try:
        ours_path = ROOT / "methods" / "ours_adapter.py"
        with open(ours_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for RF usage
        if "load_rf_artifact" in content:
            analysis["rf_loading"] = "load_rf_artifact() is called"
        else:
            analysis["rf_loading"] = "RF artifact NOT loaded"
        
        # Check for BRB usage
        if "system_level_infer" in content:
            analysis["brb_system"] = "system_level_infer() is called"
        else:
            analysis["brb_system"] = "BRB system_level_infer NOT called"
        
        # Check for gating
        if "GatingPriorFusion" in content:
            analysis["gating_logic"] = "GatingPriorFusion is used for RF+BRB fusion"
        elif "fuse" in content.lower():
            analysis["gating_logic"] = "Some fusion logic exists"
        else:
            analysis["gating_logic"] = "No fusion detected"
        
        # Check for module-level soft gating
        if "hierarchical_module_infer_soft_gating" in content:
            analysis["module_soft_gating"] = "hierarchical_module_infer_soft_gating() is used"
        elif "module_level_infer_with_activation" in content:
            analysis["module_soft_gating"] = "module_level_infer_with_activation() fallback"
        else:
            analysis["module_soft_gating"] = "Unknown module inference"
        
        # Check for feature pool
        if "AMP_POOL" in content or "FREQ_POOL" in content or "REF_POOL" in content:
            analysis["feature_pool"] = "Feature pools are referenced"
        else:
            analysis["feature_pool"] = "No feature pool separation in inference"
        
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis


def generate_architecture_snapshot():
    """Generate the architecture snapshot markdown."""
    rf_status = check_rf_artifact()
    gating_config = check_gating_config()
    infer_analysis = analyze_infer_system_and_modules()
    
    md = f"""# Architecture Snapshot

Generated: {datetime.now().isoformat()}

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

**RF Artifact Path**: `{rf_status['artifact_path']}`
**Artifact Exists**: `{rf_status['artifact_exists']}`
**Meta Exists**: `{rf_status['meta_exists']}`

**Behavior when RF not found**:
- If `allow_fallback=True`: Silent fallback to BRB-only (gating_status="fallback_brb_only")
- If `allow_fallback=False`: Raises `ValueError`

---

## 3. BRB system-level inference

**Answer**: Yes, BRB `system_level_infer()` is ALWAYS called.

From analysis: `{infer_analysis.get('brb_system', 'Unknown')}`

BRB output dimension: 4 classes (normal, amp_error, freq_error, ref_error)

---

## 4. Gating fusion method

**Gating Configuration**:
```json
{json.dumps(gating_config, indent=2, ensure_ascii=False)}
```

**Fusion Formula**: Linear weighted average
```python
fused = alpha * brb_probs + beta * rf_probs
# Default: alpha=0.6, beta=0.4
```

**Analysis**: `{infer_analysis.get('gating_logic', 'Unknown')}`

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

Analysis: `{infer_analysis.get('module_soft_gating', 'Unknown')}`

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

Analysis: `{infer_analysis.get('feature_pool', 'Unknown')}`

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
"""
    return md


def main():
    print("=" * 60)
    print("Dump Runtime Architecture")
    print("=" * 60)
    
    out_dir = ROOT / "Output" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot = generate_architecture_snapshot()
    
    out_path = out_dir / "architecture_snapshot.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(snapshot)
    
    print(f"\nArchitecture snapshot saved to: {out_path}")
    print("\n" + "=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
