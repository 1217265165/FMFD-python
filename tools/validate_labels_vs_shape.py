#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate simulated labels against curve shape signatures.
Outputs Output/label_shape_audit.json and .md, and returns non-zero on failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _smooth(series: np.ndarray, window: int = 31) -> np.ndarray:
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(series, kernel, mode="same")


def _smooth_rows(series: np.ndarray, window: int = 31) -> np.ndarray:
    if series.ndim == 1:
        return _smooth(series, window=window)
    return np.vstack([_smooth(row, window=window) for row in series])


def _load_labels(path: Path) -> Dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_raw_curve(csv_path: Path) -> np.ndarray:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 1], data[:, 2]


def _max_contiguous(mask: np.ndarray) -> int:
    max_len = 0
    current = 0
    for val in mask:
        if val:
            current += 1
            max_len = max(max_len, current)
        else:
            current = 0
    return max_len


def validate(sim_dir: Path, output_dir: Path) -> Dict[str, object]:
    labels_path = sim_dir / "labels.json"
    curves_path = sim_dir / "simulated_curves.npz"
    raw_dir = sim_dir / "raw_curves"
    baseline_path = output_dir / "baseline_artifacts.npz"

    labels = _load_labels(labels_path)
    curves_npz = np.load(curves_path, allow_pickle=True)
    curves = curves_npz["curves"]
    frequency = curves_npz["frequency"]
    baseline = np.load(baseline_path, allow_pickle=True)
    rrs = baseline["rrs"]

    normal_indices = [
        int(sample_id.split("_")[1])
        for sample_id, row in labels.items()
        if row.get("system_fault_class") == "normal"
    ]
    normal_curves = curves[normal_indices] if normal_indices else np.empty((0, len(rrs)))
    normal_res = normal_curves - rrs if normal_curves.size else np.empty((0, len(rrs)))
    normal_hf = np.std(normal_res - _smooth_rows(normal_res, window=31), axis=1) if normal_res.size else np.array([])
    normal_hf_mean = float(np.mean(normal_hf)) if normal_hf.size else 0.0
    normal_std = np.std(normal_res, axis=1) if normal_res.size else np.array([])
    normal_std_mean = float(np.mean(normal_std)) if normal_std.size else 0.0

    checks: List[Dict[str, object]] = []

    # ref_error: mean offset should exceed threshold
    ref_indices = [
        int(sample_id.split("_")[1])
        for sample_id, row in labels.items()
        if row.get("system_fault_class") == "ref_error"
    ]
    if ref_indices:
        ref_res = curves[ref_indices] - rrs
        ref_mean = float(np.mean(np.abs(np.median(ref_res, axis=1))))
        normal_p95 = float(np.quantile(np.abs(normal_res), 0.95)) if normal_res.size else 0.05
        threshold = max(0.06, 0.6 * normal_p95)
        checks.append(
            {
                "check": "ref_error_mean_offset",
                "value": ref_mean,
                "threshold": threshold,
                "passed": abs(ref_mean) > threshold,
            }
        )

    # amp_error_ripple: HF energy higher than normal
    ripple_indices = [
        int(sample_id.split("_")[1])
        for sample_id, row in labels.items()
        if row.get("amp_error_subtype") == "amp_error_ripple"
    ]
    if ripple_indices:
        ripple_res = curves[ripple_indices] - rrs
        ripple_std = np.std(ripple_res, axis=1)
        ripple_mean = float(np.mean(ripple_std))
        threshold = normal_std_mean * 1.05 if normal_std_mean else 0.0
        checks.append(
            {
                "check": "amp_error_ripple_hf",
                "value": ripple_mean,
                "threshold": threshold,
                "passed": ripple_mean > threshold,
            }
        )

    # freq_error dense/hole: contiguous delta freq length threshold
    freq_checks = []
    for sample_id, row in labels.items():
        if row.get("system_fault_class") != "freq_error":
            continue
        if row.get("peak_track_type") not in {"dense", "hole"}:
            continue
        csv_path = raw_dir / f"{sample_id}.csv"
        if not csv_path.exists():
            continue
        freq_inj, peak_freq = _read_raw_curve(csv_path)
        delta = np.abs(peak_freq - freq_inj)
        mask = delta > 5e6
        max_len = _max_contiguous(mask)
        freq_checks.append(max_len)
    if freq_checks:
        checks.append(
            {
                "check": "freq_error_dense_hole_length",
                "value": float(np.mean(freq_checks)),
                "threshold": 8.0,
                "passed": float(np.mean(freq_checks)) >= 8.0,
            }
        )

    summary = {
        "sim_dir": str(sim_dir),
        "checks": checks,
        "passed": all(entry["passed"] for entry in checks) if checks else False,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate labels vs curve shapes")
    parser.add_argument("--sim_dir", default="Output/sim_spectrum")
    parser.add_argument("--output_dir", default="Output")
    args = parser.parse_args()
    sim_dir = Path(args.sim_dir)
    output_dir = Path(args.output_dir)
    report = validate(sim_dir, output_dir)
    json_path = output_dir / "label_shape_audit.json"
    md_path = output_dir / "label_shape_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = ["# Label Shape Audit", "", f"- sim_dir: {sim_dir}", ""]
    for entry in report.get("checks", []):
        md_lines.append(
            f"- {entry['check']}: value={entry['value']:.4f}, "
            f"threshold={entry['threshold']:.4f}, passed={entry['passed']}"
        )
    md_lines.append("")
    md_lines.append(f"Overall passed: {report.get('passed')}")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
