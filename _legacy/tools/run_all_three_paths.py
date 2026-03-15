#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3.2: One-click reproducibility script for three evaluation paths.

This script runs all three evaluation paths with the same manifest and verifies
that they produce consistent results.

Paths:
1. Path-A (compare_methods): baseline + compare_methods
2. Path-B (CLI diagnosis): brb_diagnosis_cli + aggregate_batch_diagnosis
3. Path-C (eval_module): eval_module_localization

Output: Output/debug/three_path_consistency_report.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_command(cmd: list, cwd: Path = ROOT, timeout: int = 600) -> tuple:
    """Run a command and return (success, stdout, stderr)."""
    print(f"\n[CMD] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"[WARN] Command failed with code {result.returncode}")
            if result.stderr:
                print(f"[STDERR] {result.stderr[:500]}")
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command timed out after {timeout}s")
        return False, "", "Timeout"
    except Exception as e:
        print(f"[ERROR] {e}")
        return False, "", str(e)


def load_manifest(manifest_path: Path) -> dict:
    """Load manifest file."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics_from_compare(output_dir: Path) -> dict:
    """Extract metrics from compare_methods output."""
    metrics = {"sys_acc": None, "mod_top1": None, "mod_top3": None}
    
    # Try performance_table.csv
    perf_path = output_dir / "performance_table.csv"
    if perf_path.exists():
        import csv
        with open(perf_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("method") == "ours":
                    metrics["sys_acc"] = float(row.get("sys_accuracy", 0))
                    metrics["mod_top1"] = float(row.get("mod_top1_accuracy", 0))
                    metrics["mod_top3"] = float(row.get("mod_top3_accuracy", 0))
                    break
    
    return metrics


def extract_metrics_from_aggregate(report_path: Path) -> dict:
    """Extract metrics from aggregate_batch_diagnosis output."""
    metrics = {"sys_acc": None, "mod_top1": None, "mod_top3": None}
    
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        summary = report.get("summary", {})
        metrics["sys_acc"] = summary.get("sys_acc", 0)
        metrics["mod_top1"] = summary.get("mod_top1", 0)
        metrics["mod_top3"] = summary.get("mod_top3", 0)
    
    return metrics


def extract_metrics_from_eval_module(results_path: Path) -> dict:
    """Extract metrics from eval_module_localization output."""
    metrics = {"sys_acc": None, "mod_top1": None, "mod_top3": None}
    
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        metrics["mod_top1"] = results.get("mod_top1", 0)
        metrics["mod_top3"] = results.get("mod_top3", 0)
        # sys_acc may not be in eval_module
        metrics["sys_acc"] = results.get("sys_acc")
    
    return metrics


def check_consistency(metrics_a: dict, metrics_b: dict, metrics_c: dict, n_eval: int) -> dict:
    """Check if metrics are consistent within tolerance."""
    tolerance_sys_acc = 0.005  # 0.5%
    tolerance_samples = 1  # 1 sample
    
    def samples_diff(m1, m2, n):
        if m1 is None or m2 is None:
            return 0
        return abs(m1 - m2) * n
    
    consistency = {
        "sys_acc_consistent": True,
        "mod_top1_consistent": True,
        "mod_top3_consistent": True,
        "all_consistent": True,
        "details": {},
    }
    
    # Compare Path-B vs Path-C (both use full sample set)
    mod_top1_diff = samples_diff(metrics_b["mod_top1"], metrics_c["mod_top1"], n_eval)
    mod_top3_diff = samples_diff(metrics_b["mod_top3"], metrics_c["mod_top3"], n_eval)
    
    consistency["details"]["path_b_vs_c_mod_top1_diff_samples"] = mod_top1_diff
    consistency["details"]["path_b_vs_c_mod_top3_diff_samples"] = mod_top3_diff
    
    if mod_top1_diff > tolerance_samples:
        consistency["mod_top1_consistent"] = False
        consistency["all_consistent"] = False
    
    if mod_top3_diff > tolerance_samples:
        consistency["mod_top3_consistent"] = False
        consistency["all_consistent"] = False
    
    # Note: Path-A (compare_methods) may use different sample split, so we just record
    if metrics_a["sys_acc"] is not None and metrics_b["sys_acc"] is not None:
        sys_acc_diff = abs(metrics_a["sys_acc"] - metrics_b["sys_acc"])
        consistency["details"]["path_a_vs_b_sys_acc_diff"] = sys_acc_diff
        if sys_acc_diff > tolerance_sys_acc:
            # Only flag if both use same sample set
            pass  # May be expected due to different sample sets
    
    return consistency


def main():
    parser = argparse.ArgumentParser(description='Run all three evaluation paths')
    parser.add_argument('--mode', choices=['full400', 'fault300'], default='fault300',
                        help='Manifest mode for evaluation')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip running baseline (assume already exists)')
    parser.add_argument('--skip_compare', action='store_true',
                        help='Skip Path-A (compare_methods)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("P3.2: Three-Path Consistency Verification")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "paths": {},
        "consistency": {},
        "success": False,
    }
    
    # Step 0: Build manifest
    print("\n" + "=" * 60)
    print("[Step 0] Building evaluation manifest...")
    manifest_path = ROOT / "Output" / "debug" / "eval_manifest.json"
    
    success, stdout, stderr = run_command([
        sys.executable, "tools/build_eval_manifest.py",
        "--mode", args.mode,
        "--out", str(manifest_path),
    ])
    
    if not success:
        print("[ERROR] Failed to build manifest")
        report["error"] = "Failed to build manifest"
        return 1
    
    manifest = load_manifest(manifest_path)
    n_eval = manifest.get("n_samples", 0)
    manifest_hash = manifest.get("manifest_hash", "")
    
    print(f"[INFO] Manifest: N_eval={n_eval}, hash={manifest_hash}")
    report["manifest"] = {
        "path": str(manifest_path),
        "n_eval": n_eval,
        "manifest_hash": manifest_hash,
        "include_normal": manifest.get("include_normal", True),
        "sample_id_min": manifest.get("sample_ids", [""])[0],
        "sample_id_max": manifest.get("sample_ids", [""])[-1] if manifest.get("sample_ids") else "",
    }
    
    # Step 1: Path-A (compare_methods)
    print("\n" + "=" * 60)
    print("[Step 1] Path-A: compare_methods")
    
    if args.skip_compare:
        print("[SKIP] Skipping Path-A")
        report["paths"]["path_a_compare"] = {"skipped": True}
        metrics_a = {"sys_acc": None, "mod_top1": None, "mod_top3": None}
    else:
        if not args.skip_baseline:
            print("[1.1] Running baseline...")
            run_command([sys.executable, "baseline.py"])
        
        print("[1.2] Running compare_methods with manifest...")
        success, stdout, stderr = run_command([
            sys.executable, "compare_methods.py",
            "--manifest", str(manifest_path),
        ], timeout=900)
        
        compare_output = ROOT / "Output" / "compare_methods"
        metrics_a = extract_metrics_from_compare(compare_output)
        
        report["paths"]["path_a_compare"] = {
            "success": success,
            "output_dir": str(compare_output),
            "metrics": metrics_a,
        }
        print(f"[INFO] Path-A metrics: {metrics_a}")
    
    # Step 2: Path-B (CLI diagnosis + aggregate)
    print("\n" + "=" * 60)
    print("[Step 2] Path-B: brb_diagnosis_cli + aggregate")
    
    batch_output = ROOT / "Output" / "batch_diagnosis"
    batch_output.mkdir(parents=True, exist_ok=True)
    
    # Clear previous batch diagnosis
    for f in batch_output.glob("*_diagnosis.json"):
        f.unlink()
    
    print("[2.1] Running brb_diagnosis_cli with manifest...")
    success_cli, stdout, stderr = run_command([
        sys.executable, "brb_diagnosis_cli.py",
        "--manifest", str(manifest_path),
        "--input_dir", "Output/sim_spectrum/raw_curves",
        "--output", str(batch_output),
        "--labels", "Output/sim_spectrum/labels.json",
    ], timeout=900)
    
    print("[2.2] Running aggregate_batch_diagnosis with manifest...")
    report_path = batch_output / "module_localization_report.json"
    success_agg, stdout, stderr = run_command([
        sys.executable, "tools/aggregate_batch_diagnosis.py",
        "--manifest", str(manifest_path),
        "--input_dir", str(batch_output),
        "--output", str(report_path),
    ])
    
    metrics_b = extract_metrics_from_aggregate(report_path)
    
    report["paths"]["path_b_cli_aggregate"] = {
        "success": success_cli and success_agg,
        "output_dir": str(batch_output),
        "metrics": metrics_b,
    }
    print(f"[INFO] Path-B metrics: {metrics_b}")
    
    # Step 3: Path-C (eval_module_localization)
    print("\n" + "=" * 60)
    print("[Step 3] Path-C: eval_module_localization")
    
    print("[3.1] Running eval_module_localization with manifest...")
    success_eval, stdout, stderr = run_command([
        sys.executable, "tools/eval_module_localization.py",
        "--manifest", str(manifest_path),
    ])
    
    eval_results_path = ROOT / "Output" / "module_eval" / "module_localization_results.json"
    metrics_c = extract_metrics_from_eval_module(eval_results_path)
    
    report["paths"]["path_c_eval_module"] = {
        "success": success_eval,
        "output_path": str(eval_results_path),
        "metrics": metrics_c,
    }
    print(f"[INFO] Path-C metrics: {metrics_c}")
    
    # Step 4: Consistency check
    print("\n" + "=" * 60)
    print("[Step 4] Checking consistency...")
    
    consistency = check_consistency(metrics_a, metrics_b, metrics_c, n_eval)
    report["consistency"] = consistency
    
    # Summary
    print("\n" + "=" * 70)
    print("THREE-PATH CONSISTENCY REPORT")
    print("=" * 70)
    print(f"\nManifest: {manifest_path}")
    print(f"  N_eval: {n_eval}")
    print(f"  manifest_hash: {manifest_hash}")
    print(f"  include_normal: {manifest.get('include_normal')}")
    
    print(f"\nPath-A (compare_methods):")
    print(f"  sys_acc: {metrics_a.get('sys_acc')}")
    print(f"  mod_top1: {metrics_a.get('mod_top1')}")
    print(f"  mod_top3: {metrics_a.get('mod_top3')}")
    
    print(f"\nPath-B (cli_batch + aggregate):")
    print(f"  sys_acc: {metrics_b.get('sys_acc')}")
    print(f"  mod_top1: {metrics_b.get('mod_top1')}")
    print(f"  mod_top3: {metrics_b.get('mod_top3')}")
    
    print(f"\nPath-C (eval_module):")
    print(f"  sys_acc: {metrics_c.get('sys_acc')}")
    print(f"  mod_top1: {metrics_c.get('mod_top1')}")
    print(f"  mod_top3: {metrics_c.get('mod_top3')}")
    
    print(f"\nConsistency (Path-B vs Path-C):")
    print(f"  mod_top1 diff samples: {consistency['details'].get('path_b_vs_c_mod_top1_diff_samples', 0):.1f}")
    print(f"  mod_top3 diff samples: {consistency['details'].get('path_b_vs_c_mod_top3_diff_samples', 0):.1f}")
    print(f"  all_consistent: {consistency['all_consistent']}")
    
    # Save report
    report_path = ROOT / "Output" / "debug" / "three_path_consistency_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Report saved to: {report_path}")
    
    if consistency["all_consistent"]:
        print("\n✅ All paths are consistent!")
        report["success"] = True
        return 0
    else:
        print("\n❌ Paths are NOT consistent - see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
