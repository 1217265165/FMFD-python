#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRB 结果评估与可视化脚本（自动使用仓库根下的默认路径）
------------------------------------------------------------
默认路径（相对仓库根 = 当前文件的上一级目录）：
- sim_csv:   Output/sim_spectrum/features_brb.csv
- detect_csv: Output/detection_results.csv
- out_dir:  Output/reports

依赖：pandas, numpy；若需画图，安装 matplotlib
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_rows(path: Optional[Path]) -> Optional[List[Dict[str, object]]]:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] file not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed: Dict[str, object] = {}
            for key, val in row.items():
                if val is None or val == "":
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(val)
                    except (ValueError, TypeError):
                        parsed[key] = val
            rows.append(parsed)
    return rows


def top_module_from_labels(label_faults: Any) -> Optional[str]:
    if isinstance(label_faults, str):
        try:
            val = json.loads(label_faults)
        except Exception:
            return None
    else:
        val = label_faults
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
        return val[0].get("module", None)
    return None


def norm_sys_label(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).lower().strip()
    if s.startswith("sys_"):
        s = s[4:]
    mapping = {
        "amp": "amp_error",
        "amplitude": "amp_error",
        "amplitude_error": "amp_error",
        "amp_error": "amp_error",
        "freq": "freq_error",
        "frequency": "freq_error",
        "frequency_error": "freq_error",
        "freq_error": "freq_error",
        "ref": "ref_error",
        "reflevel": "ref_error",
        "reference": "ref_error",
        "reference_error": "ref_error",
        "ref_level": "ref_error",
        "ref_error": "ref_error",
        "幅度失准": "amp_error",
        "幅度": "amp_error",
        "频率失准": "freq_error",
        "频率": "freq_error",
        "参考电平失准": "ref_error",
        "参考电平": "ref_error",
    }
    return mapping.get(s, s)


def eval_system_level(rows: List[Dict[str, object]]) -> Dict[str, Any]:
    if not rows:
        return {}
    sys_cols = [c for c in rows[0].keys() if c.startswith("sys_")]
    if not sys_cols or "label_system_fault_class" not in rows[0]:
        return {}
    sys_true: List[str] = []
    sys_pred: List[str] = []
    for row in rows:
        label_val = row.get("label_system_fault_class")
        if label_val is None:
            continue
        scores = {col: row.get(col, -np.inf) for col in sys_cols}
        pred = max(scores, key=scores.get).replace("sys_", "")
        sys_pred.append(norm_sys_label(pred))
        sys_true.append(norm_sys_label(label_val))
    if not sys_true:
        return {}
    overall = float(np.mean([p == t for p, t in zip(sys_pred, sys_true)]))
    per_class: Dict[str, float] = {}
    for cls in sorted(set(sys_true)):
        idx = [i for i, t in enumerate(sys_true) if t == cls]
        per_class[cls] = float(np.mean([sys_pred[i] == sys_true[i] for i in idx]))
    confusion: Dict[str, Dict[str, int]] = {}
    for t, p in zip(sys_true, sys_pred):
        confusion.setdefault(t, {})
        confusion[t][p] = confusion[t].get(p, 0) + 1
    return {"overall": overall, "per_class": per_class, "confusion": confusion}


def eval_module_level(rows: List[Dict[str, object]]) -> Dict[str, Any]:
    if not rows:
        return {}
    mod_cols = [c for c in rows[0].keys() if c.startswith("mod_")]
    if not mod_cols or "label_faults" not in rows[0]:
        return {}
    mod_true: List[str] = []
    mod_pred: List[str] = []
    for row in rows:
        label_faults = row.get("label_faults")
        if label_faults is None:
            continue
        pred = max({col: row.get(col, -np.inf) for col in mod_cols}, key=lambda k: row.get(k, -np.inf)).replace("mod_", "")
        true_mod = top_module_from_labels(label_faults)
        if true_mod is None:
            continue
        mod_pred.append(pred)
        mod_true.append(true_mod)
    if not mod_true:
        return {}
    overall = float(np.mean([p == t for p, t in zip(mod_pred, mod_true)]))
    per_mod: Dict[str, float] = {}
    for cls in sorted(set(mod_true)):
        idx = [i for i, t in enumerate(mod_true) if t == cls]
        per_mod[cls] = float(np.mean([mod_pred[i] == mod_true[i] for i in idx]))
    return {"overall": overall, "per_module": per_mod}


def save_confusion(conf: Dict[str, Dict[str, int]], out_path: Path):
    if not conf:
        return
    columns = sorted({pred for preds in conf.values() for pred in preds.keys()})
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true"] + columns)
        for true_label, preds in conf.items():
            writer.writerow([true_label] + [preds.get(col, 0) for col in columns])
    print(f"[INFO] confusion matrix saved: {out_path}")


def plot_bars(data: Dict[str, float], title: str, out_path: Path):
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip plot: {e}")
        return
    if not data:
        print(f"[WARN] empty data for plot: {title}")
        return
    keys = list(data.keys())
    vals = [data[k] for k in keys]
    plt.figure(figsize=(6, 3))
    plt.bar(keys, vals)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] plot saved: {out_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_csv", default=None, help="仿真+BRB 结果 (features_brb.csv)")
    ap.add_argument("--detect_csv", default=None, help="检测结果 (detection_results.csv)")
    ap.add_argument("--out_dir", default=None, help="报告输出目录")
    ap.add_argument("--run_dir", default=None, help="运行目录 (默认 Output/runs/run_001)")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        detect_csv = Path(args.detect_csv) if args.detect_csv else run_dir / "tables" / "detection_results.csv"
        out_dir = Path(args.out_dir) if args.out_dir else run_dir / "reports"
    else:
        detect_csv = Path(args.detect_csv) if args.detect_csv else repo_root / "Output" / "detection_results.csv"
        out_dir = Path(args.out_dir) if args.out_dir else repo_root / "Output" / "reports"
    sim_csv = Path(args.sim_csv) if args.sim_csv else repo_root / "Output" / "sim_spectrum" / "features_brb.csv"
    safe_mkdir(out_dir)

    print(f"[INFO] repo_root   = {repo_root}")
    print(f"[INFO] sim_csv     = {sim_csv}")
    print(f"[INFO] detect_csv  = {detect_csv}")
    print(f"[INFO] out_dir     = {out_dir}")

    sim_rows = load_rows(sim_csv)
    if sim_rows is not None:
        sys_res = eval_system_level(sim_rows)
        mod_res = eval_module_level(sim_rows)

        if sys_res.get("confusion") is not None:
            save_confusion(sys_res["confusion"], out_dir / "confusion_system.csv")

        summary = {
            "system_overall_acc": sys_res.get("overall"),
            "system_per_class": sys_res.get("per_class"),
            "module_overall_acc": mod_res.get("overall"),
            "module_per_module": mod_res.get("per_module"),
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] summary saved: {out_dir / 'summary.json'}")

        if sys_res.get("per_class"):
            plot_bars(sys_res["per_class"], "System-level accuracy by class", out_dir / "sys_per_class.png")
        if mod_res.get("per_module"):
            plot_bars(mod_res["per_module"], "Module-level accuracy (label_faults)", out_dir / "mod_per_module.png")

    detect_rows = load_rows(detect_csv)
    if detect_rows is not None:
        out_detect = out_dir / "detect_summary.csv"
        numeric_cols = [k for k, v in detect_rows[0].items() if isinstance(v, (int, float))]
        summary_rows: List[Dict[str, object]] = []
        for col in numeric_cols:
            vals = np.array([row.get(col, np.nan) for row in detect_rows], dtype=float)
            summary_rows.append(
                {
                    "column": col,
                    "mean": float(np.nanmean(vals)),
                    "std": float(np.nanstd(vals)),
                    "min": float(np.nanmin(vals)),
                    "max": float(np.nanmax(vals)),
                }
            )
        with out_detect.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["column", "mean", "std", "min", "max"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[INFO] detect summary saved: {out_detect}")


if __name__ == "__main__":
    main()
