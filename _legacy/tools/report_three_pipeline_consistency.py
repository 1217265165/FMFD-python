#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report three-pipeline consistency.

Compares metrics from the three evaluation paths and generates a
consistency report.

Usage:
    python tools/report_three_pipeline_consistency.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_json_safe(path: Path) -> dict:
    """Load JSON file safely."""
    if not path.exists():
        return {"error": f"File not found: {path}"}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def compare_sample_sets(manifest1: dict, manifest2: dict, name1: str, name2: str) -> dict:
    """Compare sample sets between two manifests."""
    ids1 = set(manifest1.get("sample_ids", []))
    ids2 = set(manifest2.get("sample_ids", []))
    
    intersection = ids1 & ids2
    only_in_1 = ids1 - ids2
    only_in_2 = ids2 - ids1
    
    return {
        f"{name1}_count": len(ids1),
        f"{name2}_count": len(ids2),
        "intersection_count": len(intersection),
        f"only_in_{name1}_count": len(only_in_1),
        f"only_in_{name2}_count": len(only_in_2),
        f"only_in_{name1}_first20": sorted(only_in_1)[:20],
        f"only_in_{name2}_first20": sorted(only_in_2)[:20],
        "sample_sets_identical": ids1 == ids2,
    }


def generate_consistency_report(
    metrics_compare: dict,
    metrics_cli: dict,
    metrics_eval: dict,
    manifest_compare: dict,
    manifest_cli: dict,
    manifest_eval: dict,
) -> str:
    """Generate markdown consistency report."""
    
    now = datetime.now().isoformat()
    
    # Extract metrics
    def get_metric(d, key, default=None):
        if "summary" in d:
            return d["summary"].get(key, default)
        return d.get(key, default)
    
    compare_sys_acc = get_metric(metrics_compare, "sys_acc") or get_metric(metrics_compare, "sys_accuracy")
    cli_sys_acc = get_metric(metrics_cli, "sys_acc")
    eval_sys_acc = get_metric(metrics_eval, "sys_acc")
    
    compare_mod_top1 = get_metric(metrics_compare, "mod_top1") or get_metric(metrics_compare, "mod_top1_accuracy")
    cli_mod_top1 = get_metric(metrics_cli, "mod_top1")
    eval_mod_top1 = get_metric(metrics_eval, "mod_top1")
    
    compare_mod_top3 = get_metric(metrics_compare, "mod_top3") or get_metric(metrics_compare, "mod_top3_accuracy")
    cli_mod_top3 = get_metric(metrics_cli, "mod_top3")
    eval_mod_top3 = get_metric(metrics_eval, "mod_top3")
    
    # Compare sample sets
    compare_vs_cli = compare_sample_sets(manifest_compare, manifest_cli, "compare", "cli")
    cli_vs_eval = compare_sample_sets(manifest_cli, manifest_eval, "cli", "eval")
    
    # Check metric consistency (tolerance: 1 sample)
    def samples_diff(m1, m2, n):
        if m1 is None or m2 is None:
            return None
        return abs(m1 - m2) * n
    
    n_cli = manifest_cli.get("n_samples", 300)
    
    mod_top1_diff = samples_diff(cli_mod_top1, eval_mod_top1, n_cli)
    mod_top3_diff = samples_diff(cli_mod_top3, eval_mod_top3, n_cli)
    
    mod_top1_consistent = mod_top1_diff is None or mod_top1_diff <= 1
    mod_top3_consistent = mod_top3_diff is None or mod_top3_diff <= 1
    
    report = f"""# Three-Pipeline Consistency Report

Generated: {now}

---

## 1. Sample Set Comparison

### Compare vs CLI Batch

| Metric | Compare | CLI Batch |
|--------|---------|-----------|
| N_samples | {compare_vs_cli.get('compare_count', 'N/A')} | {compare_vs_cli.get('cli_count', 'N/A')} |
| Intersection | {compare_vs_cli.get('intersection_count', 'N/A')} | - |
| Sample sets identical? | {compare_vs_cli.get('sample_sets_identical', 'N/A')} | - |

**Only in compare (first 20)**: {compare_vs_cli.get('only_in_compare_first20', [])}

**Only in CLI (first 20)**: {compare_vs_cli.get('only_in_cli_first20', [])}

### CLI Batch vs Eval Module

| Metric | CLI Batch | Eval Module |
|--------|-----------|-------------|
| N_samples | {cli_vs_eval.get('cli_count', 'N/A')} | {cli_vs_eval.get('eval_count', 'N/A')} |
| Intersection | {cli_vs_eval.get('intersection_count', 'N/A')} | - |
| Sample sets identical? | {cli_vs_eval.get('sample_sets_identical', 'N/A')} | - |

---

## 2. System-Level Accuracy Comparison

| Path | sys_acc | N_eval | Include Normal? |
|------|---------|--------|-----------------|
| Compare | {compare_sys_acc if compare_sys_acc else 'N/A'} | {manifest_compare.get('n_samples', 'N/A')} | {manifest_compare.get('include_normal', 'N/A')} |
| CLI Batch | {cli_sys_acc if cli_sys_acc else 'N/A'} | {manifest_cli.get('n_samples', 'N/A')} | {manifest_cli.get('include_normal', 'N/A')} |
| Eval Module | {eval_sys_acc if eval_sys_acc else 'N/A'} | {manifest_eval.get('n_samples', 'N/A')} | {manifest_eval.get('include_normal', 'N/A')} |

**sys_acc Comparable?**: 
- Compare vs CLI: {'Yes' if manifest_compare.get('include_normal') == manifest_cli.get('include_normal') else 'No (different normal inclusion)'}
- CLI vs Eval: {'Yes' if manifest_cli.get('include_normal') == manifest_eval.get('include_normal') else 'No'}

---

## 3. Module-Level Accuracy Comparison

| Path | mod_top1 | mod_top3 |
|------|----------|----------|
| Compare | {compare_mod_top1 if compare_mod_top1 else 'N/A'} | {compare_mod_top3 if compare_mod_top3 else 'N/A'} |
| CLI Batch | {cli_mod_top1 if cli_mod_top1 else 'N/A'} | {cli_mod_top3 if cli_mod_top3 else 'N/A'} |
| Eval Module | {eval_mod_top1 if eval_mod_top1 else 'N/A'} | {eval_mod_top3 if eval_mod_top3 else 'N/A'} |

### Consistency Check (CLI vs Eval)

| Metric | Diff (samples) | Consistent (≤1 sample)? |
|--------|----------------|-------------------------|
| mod_top1 | {f'{mod_top1_diff:.1f}' if mod_top1_diff else 'N/A'} | {'✅ Yes' if mod_top1_consistent else '❌ No'} |
| mod_top3 | {f'{mod_top3_diff:.1f}' if mod_top3_diff else 'N/A'} | {'✅ Yes' if mod_top3_consistent else '❌ No'} |

---

## 4. Architecture Summary

From `architecture_snapshot.md`:

- **System-level entry**: `methods/ours_adapter.py::infer_system_and_modules()`
- **System-level backend**: RF+BRB Gating Fusion (alpha*BRB + beta*RF)
- **Module-level backend**: hierarchical_module_infer_soft_gating() with delta=0.1
- **Feature pools**: NOT implemented (X1-X22 used uniformly)

---

## 5. Truth Fields

From `config/eval_truth.json`:

- **System truth field**: `system_fault_class`
- **Module truth field**: `module_v2`
- **Module eval policy**: Exclude normal samples

---

## 6. Conclusion

### Sample Set Consistency
- Compare vs CLI: {'✅ Identical' if compare_vs_cli.get('sample_sets_identical') else '⚠️ Different'}
- CLI vs Eval: {'✅ Identical' if cli_vs_eval.get('sample_sets_identical') else '⚠️ Different'}

### Metric Consistency (CLI vs Eval)
- mod_top1: {'✅ Consistent' if mod_top1_consistent else '❌ Inconsistent'}
- mod_top3: {'✅ Consistent' if mod_top3_consistent else '❌ Inconsistent'}

### Overall Status
{'✅ All paths are consistent!' if (cli_vs_eval.get('sample_sets_identical') and mod_top1_consistent and mod_top3_consistent) else '⚠️ Some inconsistencies detected - see details above'}

---
"""
    return report


def main():
    print("=" * 60)
    print("Three-Pipeline Consistency Report")
    print("=" * 60)
    
    debug_dir = ROOT / "Output" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics files
    metrics_compare = load_json_safe(debug_dir / "metrics_compare_full.json")
    if "error" in metrics_compare:
        # Try alternative path
        metrics_compare = load_json_safe(ROOT / "Output" / "compare_methods" / "comparison_summary.json")
    
    metrics_cli = load_json_safe(ROOT / "Output" / "batch_diagnosis" / "module_localization_report.json")
    metrics_eval = load_json_safe(ROOT / "Output" / "module_eval" / "module_localization_results.json")
    
    # Load manifests
    manifest_compare = load_json_safe(debug_dir / "manifest_all_400.json")
    if "error" in manifest_compare:
        manifest_compare = load_json_safe(debug_dir / "eval_manifest_compare.json")
    
    manifest_cli = load_json_safe(debug_dir / "manifest_fault_300.json")
    if "error" in manifest_cli:
        manifest_cli = load_json_safe(debug_dir / "eval_manifest_cli.json")
    
    manifest_eval = load_json_safe(debug_dir / "manifest_fault_300.json")
    if "error" in manifest_eval:
        manifest_eval = load_json_safe(debug_dir / "eval_manifest_eval.json")
    
    # Generate report
    report = generate_consistency_report(
        metrics_compare, metrics_cli, metrics_eval,
        manifest_compare, manifest_cli, manifest_eval,
    )
    
    # Save report
    out_path = debug_dir / "three_pipeline_consistency.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved to: {out_path}")
    
    # Also save as JSON
    json_out = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "compare": metrics_compare,
            "cli": metrics_cli,
            "eval": metrics_eval,
        },
        "manifests": {
            "compare": {"n_samples": manifest_compare.get("n_samples"), "hash": manifest_compare.get("sample_id_hash")},
            "cli": {"n_samples": manifest_cli.get("n_samples"), "hash": manifest_cli.get("sample_id_hash")},
            "eval": {"n_samples": manifest_eval.get("n_samples"), "hash": manifest_eval.get("sample_id_hash")},
        },
    }
    json_path = debug_dir / "three_pipeline_consistency.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    
    print(f"JSON saved to: {json_path}")
    print("\n" + "=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
