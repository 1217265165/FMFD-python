#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Delta Report Tool

Compares two performance runs and generates a detailed delta report.

Usage:
    python tools/report_perf_delta.py --baseline <dir1> --current <dir2>
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime


def load_consistency_report(run_dir: Path) -> Dict[str, Any]:
    """Load the three_pipeline_consistency.json from a run directory."""
    json_path = run_dir / "three_pipeline_consistency.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_module_report(run_dir: Path) -> Dict[str, Any]:
    """Load the module_localization_report.json from a run directory."""
    json_path = run_dir / "module_localization_report.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def compute_delta(baseline: float, current: float) -> Tuple[float, str]:
    """Compute delta and format as string with direction."""
    delta = current - baseline
    if delta > 0:
        return delta, f"+{delta:.2f}%"
    elif delta < 0:
        return delta, f"{delta:.2f}%"
    else:
        return 0, "±0.00%"


def generate_delta_report(baseline_dir: Path, current_dir: Path, output_path: Path) -> Dict[str, Any]:
    """Generate a performance delta report comparing two runs."""
    
    baseline_consistency = load_consistency_report(baseline_dir)
    current_consistency = load_consistency_report(current_dir)
    
    baseline_module = load_module_report(baseline_dir)
    current_module = load_module_report(current_dir)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_dir": str(baseline_dir),
        "current_dir": str(current_dir),
        "metrics_comparison": {},
        "fault_type_breakdown": {},
        "improvement_summary": [],
        "regression_warnings": [],
    }
    
    # Compare key metrics
    metrics_to_compare = [
        ("sys_acc", "System Accuracy"),
        ("mod_top1", "Module Top-1"),
        ("mod_top3", "Module Top-3"),
    ]
    
    for metric_key, metric_name in metrics_to_compare:
        baseline_val = baseline_consistency.get("path_b", {}).get(metric_key, 0)
        current_val = current_consistency.get("path_b", {}).get(metric_key, 0)
        
        if isinstance(baseline_val, str):
            baseline_val = float(baseline_val.replace('%', '')) if '%' in str(baseline_val) else 0
        if isinstance(current_val, str):
            current_val = float(current_val.replace('%', '')) if '%' in str(current_val) else 0
        
        delta, delta_str = compute_delta(baseline_val, current_val)
        
        report["metrics_comparison"][metric_key] = {
            "name": metric_name,
            "baseline": baseline_val,
            "current": current_val,
            "delta": delta,
            "delta_str": delta_str,
        }
        
        if delta > 1:
            report["improvement_summary"].append(f"{metric_name}: {delta_str} improvement")
        elif delta < -1:
            report["regression_warnings"].append(f"{metric_name}: {delta_str} regression")
    
    # Generate markdown report
    md_lines = [
        "# Performance Delta Report",
        "",
        f"**Generated**: {report['timestamp']}",
        f"**Baseline**: {baseline_dir.name}",
        f"**Current**: {current_dir.name}",
        "",
        "## Metrics Comparison",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|--------|----------|---------|-------|",
    ]
    
    for metric_key, metric_info in report["metrics_comparison"].items():
        md_lines.append(
            f"| {metric_info['name']} | {metric_info['baseline']:.2f}% | "
            f"{metric_info['current']:.2f}% | {metric_info['delta_str']} |"
        )
    
    md_lines.extend([
        "",
        "## Summary",
        "",
    ])
    
    if report["improvement_summary"]:
        md_lines.append("### Improvements ✅")
        for item in report["improvement_summary"]:
            md_lines.append(f"- {item}")
        md_lines.append("")
    
    if report["regression_warnings"]:
        md_lines.append("### Regressions ⚠️")
        for item in report["regression_warnings"]:
            md_lines.append(f"- {item}")
        md_lines.append("")
    
    if not report["improvement_summary"] and not report["regression_warnings"]:
        md_lines.append("No significant changes detected (±1% threshold).")
    
    # Write outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print('\n'.join(md_lines))
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare two performance runs")
    parser.add_argument("--baseline", required=True, help="Baseline run directory")
    parser.add_argument("--current", required=True, help="Current run directory")
    parser.add_argument("--output", default="Output/debug/perf_delta_report",
                        help="Output path (without extension)")
    args = parser.parse_args()
    
    baseline_dir = Path(args.baseline)
    current_dir = Path(args.current)
    output_path = Path(args.output)
    
    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        return 1
    
    if not current_dir.exists():
        print(f"Error: Current directory not found: {current_dir}")
        return 1
    
    generate_delta_report(baseline_dir, current_dir, output_path)
    
    print(f"\nReport saved to: {output_path}.md and {output_path}.json")
    return 0


if __name__ == "__main__":
    exit(main())
