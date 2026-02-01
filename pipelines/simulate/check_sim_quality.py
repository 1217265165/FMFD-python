#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真质量检查与报告输出。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pipelines.simulate.sim_constraints import build_quality_report, load_baseline_stats


def _load_sim_data(sim_dir: Path) -> Tuple[np.ndarray, Dict[str, dict], List[Dict[str, object]]]:
    curves_path = sim_dir / "simulated_curves.npz"
    labels_path = sim_dir / "labels.json"
    features_path = sim_dir / "features_brb.csv"
    if not curves_path.exists():
        raise FileNotFoundError(f"Missing simulated_curves.npz at {curves_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.json at {labels_path}")

    curves_npz = np.load(curves_path, allow_pickle=True)
    curves = curves_npz["curves"]
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    feature_rows = []
    if features_path.exists():
        import csv

        with features_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            feature_rows = [row for row in reader]
    return curves, labels, feature_rows


def run_quality_check(
    sim_dir: Path,
    baseline_npz: Path,
    baseline_meta: Path | None = None,
    active_modules: List[str] | None = None,
    reject_records: List[Dict[str, object]] | None = None,
    prev_dir: Path | None = None,
) -> Tuple[dict, List[Dict[str, str]]]:
    curves, labels, feature_rows = _load_sim_data(sim_dir)
    baseline = load_baseline_stats(baseline_npz, baseline_meta)
    report, violations = build_quality_report(
        sim_dir=sim_dir,
        baseline=baseline,
        curves=curves,
        labels=labels,
        active_modules=active_modules or [],
        reject_records=reject_records or [],
        prev_dir=prev_dir,
        feature_rows=feature_rows,
    )
    return report, violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check simulation quality for RRS constraints.")
    parser.add_argument("--sim_dir", default="Output/sim_spectrum", help="Simulation output directory")
    parser.add_argument("--baseline_npz", default="Output/baseline_artifacts.npz")
    parser.add_argument("--baseline_meta", default="Output/baseline_meta.json")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    sim_dir = Path(args.sim_dir)
    baseline_npz = Path(args.baseline_npz)
    baseline_meta = Path(args.baseline_meta) if args.baseline_meta else None
    report, violations = run_quality_check(sim_dir, baseline_npz, baseline_meta)

    if violations:
        print("[ERROR] Simulation quality check failed.")
        for v in violations:
            print(f"  - {v.get('sample_id')}: {v.get('reason')}")
        print("Suggested adjustments: reduce beta/shift ranges or tighten mean offset correction.")
        return 1

    overall = report.get("overall", {})
    print("[INFO] Simulation quality check passed.")
    for cls, stats in overall.items():
        print(f"  {cls}: max_abs_dev_max={stats.get('max_abs_dev_max'):.3f} dB, "
              f"mean_offset_p95={stats.get('mean_offset_p95'):.3f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
