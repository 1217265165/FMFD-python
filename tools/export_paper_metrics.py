#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export paper metrics and tables for section 3.3."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.baseline import align_to_frequency
from baseline.config import BASELINE_ARTIFACTS, BASELINE_META, BAND_RANGES
from BRB.module_brb import DISABLED_MODULES, MODULE_LABELS, module_level_infer
from BRB.system_brb import system_level_infer
from features.extract import extract_system_features
from methods.ours_adapter import OursAdapter
from pipelines.compare_methods import (
    SYS_LABEL_ORDER,
    calculate_accuracy,
    calculate_macro_f1,
    load_expected_features,
    prepare_dataset,
    select_feature_matrix,
    set_global_seed,
    stratified_split,
)
from pipelines.default_paths import PROJECT_ROOT, SEED, SIM_DIR
from tools.label_mapping import get_topk_modules


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _latest_file(pattern_root: Path, name: str) -> Optional[Path]:
    candidates = list(pattern_root.rglob(name))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_sim_dir() -> Path:
    sim_dir = PROJECT_ROOT / SIM_DIR
    if sim_dir.exists():
        return sim_dir
    fallback = PROJECT_ROOT / "Output"
    latest_labels = _latest_file(fallback, "labels.json")
    if latest_labels:
        return latest_labels.parent
    raise FileNotFoundError("No simulation output directory with labels.json found.")


def _load_labels(sim_dir: Path) -> Dict[str, Dict[str, object]]:
    labels_path = sim_dir / "labels.json"
    if not labels_path.exists():
        latest = _latest_file(sim_dir.parent, "labels.json")
        if latest:
            labels_path = latest
    if not labels_path.exists():
        raise FileNotFoundError("labels.json not found.")
    return json.loads(labels_path.read_text(encoding="utf-8"))


def _load_compare_dir() -> Path:
    compare_dir = PROJECT_ROOT / "Output" / "compare_methods"
    if compare_dir.exists():
        return compare_dir
    latest = _latest_file(PROJECT_ROOT / "Output", "comparison_table.csv")
    if latest:
        return latest.parent
    raise FileNotFoundError("compare_methods outputs not found.")


def _load_feature_usage(compare_dir: Path) -> Dict[str, int]:
    report_path = compare_dir / "feature_integrity_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    usage = report.get("method_feature_usage", {})
    summary = {}
    for method, payload in usage.items():
        expected = payload.get("expected_features", [])
        summary[method] = len(expected)
    return summary


def _load_baseline_artifacts(root: Path) -> Dict[str, np.ndarray]:
    artifacts_path = root / BASELINE_ARTIFACTS
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Baseline artifacts not found: {artifacts_path}")
    data = np.load(artifacts_path, allow_pickle=True)
    return {
        "frequency": data["frequency"],
        "rrs": data["rrs"],
        "upper": data["upper"],
        "lower": data["lower"],
    }


def _infer_sample(
    sample_path: Path,
    baseline: Dict[str, np.ndarray],
    band_ranges: List[List[float]],
    topk: int = 3,
) -> Dict[str, object]:
    raw = np.loadtxt(sample_path, delimiter=",", skiprows=1)
    freq = raw[:, 0]
    amp = raw[:, -1]
    frequency = baseline["frequency"]
    rrs = baseline["rrs"]
    bounds = (baseline["upper"], baseline["lower"])
    amp_aligned = align_to_frequency(frequency, freq, amp)
    features = extract_system_features(frequency, rrs, bounds, band_ranges, amp_aligned)
    sys_result = system_level_infer(features, mode="sub_brb")
    sys_probs = sys_result.get("probabilities", sys_result)
    mod_probs = module_level_infer(features, sys_result)
    topk_modules = get_topk_modules(mod_probs, k=topk, skip_disabled=True, disabled_modules=list(DISABLED_MODULES))
    return {
        "system_probs": {k: float(v) for k, v in sys_probs.items()},
        "system_pred": max(sys_probs, key=sys_probs.get) if sys_probs else "未知",
        "module_topk": [
            {"module": name, "probability": float(prob)} for name, prob in topk_modules
        ],
    }


def _choose_examples(labels: Dict[str, Dict[str, object]]) -> Dict[str, str]:
    examples = {}
    targets = {
        "normal": "normal",
        "amp": "amp_error",
        "freq": "freq_error",
        "ref": "ref_error",
    }
    for key, fault in targets.items():
        for sample_id, payload in labels.items():
            if payload.get("system_fault_class") == fault:
                examples[key] = sample_id
                break
    return examples


def _plot_acc_compare(rows: List[Dict[str, str]], output_path: Path) -> None:
    methods = [row["method"] for row in rows]
    acc = [_as_float(row.get("sys_accuracy")) for row in rows]
    f1 = [_as_float(row.get("sys_macro_f1")) for row in rows]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_runtime_compare(rows: List[Dict[str, str]], output_path: Path) -> None:
    methods = [row["method"] for row in rows]
    runtimes = [_as_float(row.get("infer_ms_per_sample")) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    ax.bar(x, runtimes, color="#4C72B0")
    ax.set_ylabel("Inference Time (ms/sample)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_fewshot_curve(rows: List[Dict[str, object]], output_path: Path) -> None:
    ratios = sorted({row["train_ratio"] for row in rows})
    means = []
    stds = []
    for ratio in ratios:
        vals = [row["system_accuracy"] for row in rows if row["train_ratio"] == ratio]
        means.append(float(np.mean(vals)) if vals else 0.0)
        stds.append(float(np.std(vals)) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar([r * 100 for r in ratios], means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("Train Ratio (%)")
    ax.set_ylabel("System Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _stratified_subsample_indices(y: np.ndarray, ratio: float, rng: np.random.RandomState) -> np.ndarray:
    indices: List[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if cls_idx.size == 0:
            continue
        n_select = max(1, int(round(cls_idx.size * ratio)))
        n_select = min(n_select, cls_idx.size)
        indices.extend(rng.choice(cls_idx, size=n_select, replace=False).tolist())
    rng.shuffle(indices)
    return np.array(indices, dtype=int)


def run_fewshot(
    sim_dir: Path,
    compare_dir: Path,
    ratios: List[float],
    repeats: int,
    seed: int,
) -> List[Dict[str, object]]:
    set_global_seed(seed)
    (
        X,
        y_sys,
        y_mod,
        feature_names,
        sample_ids,
        leak_columns,
        missing_in_features,
        missing_in_labels,
    ) = prepare_dataset(sim_dir, use_pool_features=True, strict_leakage=True)
    X_train, X_val, X_test, y_sys_train, y_sys_val, y_sys_test, *_ = stratified_split(
        X, y_sys, 0.6, 0.2, seed
    )
    expected_features = load_expected_features("ours", feature_names, compare_dir)
    X_train_sel, _ = select_feature_matrix(X_train, feature_names, expected_features)
    X_test_sel, _ = select_feature_matrix(X_test, feature_names, expected_features)

    results: List[Dict[str, object]] = []
    for ratio in ratios:
        for rep in range(repeats):
            rng = np.random.RandomState(seed + rep + int(ratio * 100))
            idx = _stratified_subsample_indices(y_sys_train, ratio, rng)
            model = OursAdapter()
            model.fit(X_train_sel[idx], y_sys_train[idx], None, {"feature_names": expected_features})
            pred = model.predict(X_test_sel, {"feature_names": expected_features})
            y_pred = pred["system_pred"]
            acc = calculate_accuracy(y_sys_test, y_pred)
            f1 = calculate_macro_f1(y_sys_test, y_pred, len(SYS_LABEL_ORDER))
            results.append(
                {
                    "method": "ours",
                    "train_ratio": ratio,
                    "seed": seed + rep,
                    "system_accuracy": float(acc),
                    "macro_f1": float(f1),
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Export paper metrics for section 3.3")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "Output" / "paper_v1"))
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    logs_dir = output_root / "logs"
    _ensure_dir(output_root)
    _ensure_dir(tables_dir)
    _ensure_dir(figures_dir)
    _ensure_dir(logs_dir)

    sim_dir = _find_sim_dir()
    labels = _load_labels(sim_dir)
    labels_path = sim_dir / "labels.json"

    compare_dir = _load_compare_dir()
    comparison_table = compare_dir / "comparison_table.csv"
    performance_table = compare_dir / "performance_table.csv"
    confusion_counts = compare_dir / "ours_confusion_matrix_counts.csv"
    predictions_path = compare_dir / "predictions_ours.csv"

    comparison_rows = _read_csv(comparison_table) if comparison_table.exists() else []
    feature_usage = _load_feature_usage(compare_dir)

    n_samples = len(labels)
    fault_modes_def = "system_fault_class ∈ {normal, amp_error, freq_error, ref_error}"

    tab_3_2_rows = []
    method_lookup: Dict[str, Dict[str, object]] = {}
    for row in comparison_rows:
        method = row.get("method")
        record = {
            "method": method,
            "fault_modes_definition": fault_modes_def,
            "n_samples": n_samples,
            "n_rules_total": int(float(row.get("n_rules", 0))) if row.get("n_rules") else 0,
            "n_params_total": int(float(row.get("n_params", 0))) if row.get("n_params") else 0,
            "system_feature_dim": feature_usage.get(method, int(float(row.get("n_features_used", 0))))
            if method
            else 0,
            "system_accuracy": _as_float(row.get("sys_accuracy")),
            "avg_infer_time_ms": _as_float(row.get("infer_ms_per_sample")),
        }
        tab_3_2_rows.append(record)
        if method:
            method_lookup[method] = record

    tab_3_2_path = tables_dir / "tab_3_2_benchmark.csv"
    _write_csv(
        tab_3_2_path,
        tab_3_2_rows,
        [
            "method",
            "fault_modes_definition",
            "n_samples",
            "n_rules_total",
            "n_params_total",
            "system_feature_dim",
            "system_accuracy",
            "avg_infer_time_ms",
        ],
    )

    tab_3_3_rows = []
    normal_fpr = None
    if confusion_counts.exists():
        counts_rows = _read_csv(confusion_counts)
        labels_order = SYS_LABEL_ORDER
        for row in counts_rows:
            label = row.get("True\\Pred")
            if label in labels_order:
                n_total = int(float(row.get("Total", 0)))
                correct = int(float(row.get(label, 0)))
                acc = correct / n_total if n_total else 0.0
                tab_3_3_rows.append(
                    {
                        "class": label,
                        "n": n_total,
                        "correct": correct,
                        "acc": acc,
                        "fpr_normal": "",
                    }
                )
        normal_row = next((r for r in tab_3_3_rows if r["class"] == "正常"), None)
        if normal_row:
            total_faults = sum(r["n"] for r in tab_3_3_rows if r["class"] != "正常")
            pred_normal = sum(
                int(float(row.get("正常", 0)))
                for row in counts_rows
                if row.get("True\\Pred") in labels_order and row.get("True\\Pred") != "正常"
            )
            normal_fpr = pred_normal / total_faults if total_faults else 0.0
            normal_row["fpr_normal"] = normal_fpr

    tab_3_3_path = tables_dir / "tab_3_3_system_perf.csv"
    _write_csv(tab_3_3_path, tab_3_3_rows, ["class", "n", "correct", "acc", "fpr_normal"])

    tab_3_4_rows = []
    severity_levels = sorted({payload.get("severity") for payload in labels.values() if payload.get("severity")})
    predictions = {}
    if predictions_path.exists():
        for row in _read_csv(predictions_path):
            predictions[row["sample_id"]] = row
    for severity in severity_levels:
        severity_ids = [
            sample_id
            for sample_id, payload in labels.items()
            if payload.get("severity") == severity
        ]
        correct = 0
        for sample_id in severity_ids:
            pred_row = predictions.get(sample_id)
            if not pred_row:
                continue
            true_class = labels.get(sample_id, {}).get("system_fault_class")
            pred_label_cn = pred_row.get("pred_label")
            pred_en = {
                "正常": "normal",
                "幅度失准": "amp_error",
                "频率失准": "freq_error",
                "参考电平失准": "ref_error",
            }.get(pred_label_cn, pred_label_cn)
            if pred_en == true_class:
                correct += 1
        n_total = len(severity_ids)
        acc = correct / n_total if n_total else 0.0
        tab_3_4_rows.append({"severity": severity, "n": n_total, "correct": correct, "acc": acc})

    tab_3_4_path = tables_dir / "tab_3_4_severity_perf.csv"
    _write_csv(tab_3_4_path, tab_3_4_rows, ["severity", "n", "correct", "acc"])

    baseline = _load_baseline_artifacts(PROJECT_ROOT)
    meta_path = PROJECT_ROOT / BASELINE_META
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        band_ranges = meta.get("band_ranges", BAND_RANGES)
    else:
        band_ranges = BAND_RANGES

    examples = _choose_examples(labels)
    tab_3_5_rows = []
    raw_dir = sim_dir / "raw_curves"
    for kind, sample_id in examples.items():
        sample_path = raw_dir / f"{sample_id}.csv"
        if not sample_path.exists():
            continue
        inf = _infer_sample(sample_path, baseline, band_ranges)
        topk_text = "; ".join(
            [f"{item['module']}:{item['probability']:.3f}" for item in inf["module_topk"]]
        )
        tab_3_5_rows.append(
            {
                "sample_id": sample_id,
                "case": kind,
                "true_class": labels[sample_id].get("system_fault_class"),
                "pred_class": inf["system_pred"],
                "system_probs": json.dumps(inf["system_probs"], ensure_ascii=False),
                "module_topk": topk_text or "NA",
                "note": "typical",
            }
        )

    misclassified = None
    if predictions:
        for sample_id, pred in predictions.items():
            true_class = labels.get(sample_id, {}).get("system_fault_class")
            pred_label_cn = pred.get("pred_label")
            pred_en = {
                "正常": "normal",
                "幅度失准": "amp_error",
                "频率失准": "freq_error",
                "参考电平失准": "ref_error",
            }.get(pred_label_cn, pred_label_cn)
            if true_class and pred_en != true_class:
                misclassified = sample_id
                break
    if misclassified:
        sample_path = raw_dir / f"{misclassified}.csv"
        if sample_path.exists():
            inf = _infer_sample(sample_path, baseline, band_ranges)
            topk_text = "; ".join(
                [f"{item['module']}:{item['probability']:.3f}" for item in inf["module_topk"]]
            )
            tab_3_5_rows.append(
                {
                    "sample_id": misclassified,
                    "case": "misclassified",
                    "true_class": labels[misclassified].get("system_fault_class"),
                    "pred_class": inf["system_pred"],
                    "system_probs": json.dumps(inf["system_probs"], ensure_ascii=False),
                    "module_topk": topk_text or "NA",
                    "note": "error",
                }
            )

    tab_3_5_path = tables_dir / "tab_3_5_examples.csv"
    _write_csv(
        tab_3_5_path,
        tab_3_5_rows,
        [
            "sample_id",
            "case",
            "true_class",
            "pred_class",
            "system_probs",
            "module_topk",
            "note",
        ],
    )

    fewshot_rows = run_fewshot(sim_dir, compare_dir, [0.1, 0.2, 0.4, 0.6, 0.8], 3, SEED)
    fewshot_path = tables_dir / "fewshot_results.csv"
    _write_csv(
        fewshot_path,
        fewshot_rows,
        ["method", "train_ratio", "seed", "system_accuracy", "macro_f1"],
    )

    acc_fig = figures_dir / "acc_compare.png"
    runtime_fig = figures_dir / "runtime_compare.png"
    fewshot_fig = figures_dir / "fewshot_curve.png"

    if comparison_rows:
        _plot_acc_compare(comparison_rows, acc_fig)
        _plot_runtime_compare(comparison_rows, runtime_fig)
    if fewshot_rows:
        _plot_fewshot_curve(fewshot_rows, fewshot_fig)

    def _ratio(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
        if numer is None or denom in (None, 0):
            return None
        return numer / denom

    def _lookup(method: str, key: str) -> Optional[float]:
        record = method_lookup.get(method)
        if not record:
            return None
        return _as_float(record.get(key)) if record.get(key) is not None else None

    rule_ratio = _ratio(_lookup("brb_p", "n_rules_total"), _lookup("ours", "n_rules_total"))
    speedup = _ratio(_lookup("brb_p", "avg_infer_time_ms"), _lookup("ours", "avg_infer_time_ms"))

    fewshot_summary = {}
    if fewshot_rows:
        for ratio in sorted({row["train_ratio"] for row in fewshot_rows}):
            vals = [row["system_accuracy"] for row in fewshot_rows if row["train_ratio"] == ratio]
            fewshot_summary[str(ratio)] = {
                "mean_accuracy": float(np.mean(vals)) if vals else 0.0,
                "std_accuracy": float(np.std(vals)) if vals else 0.0,
            }

    metrics_summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "labels": str(labels_path),
            "comparison_table": str(comparison_table),
            "performance_table": str(performance_table),
            "confusion_counts": str(confusion_counts),
            "predictions_ours": str(predictions_path),
            "fewshot_results": str(fewshot_path),
        },
        "dataset": {
            "n_samples": {"value": n_samples, "source": str(labels_path)},
            "fault_modes_definition": {"value": fault_modes_def, "source": str(labels_path)},
        },
        "normal_fpr": {"value": normal_fpr, "source": str(confusion_counts)},
        "comparisons": {
            "rule_compression_ratio_vs_brb_p": {"value": rule_ratio, "source": str(tab_3_2_path)},
            "speedup_vs_brb_p": {"value": speedup, "source": str(tab_3_2_path)},
        },
        "fewshot_summary": {"value": fewshot_summary, "source": str(fewshot_path)},
        "tables": {
            "tab_3_2_benchmark": {"path": str(tab_3_2_path)},
            "tab_3_3_system_perf": {"path": str(tab_3_3_path)},
            "tab_3_4_severity_perf": {"path": str(tab_3_4_path)},
            "tab_3_5_examples": {"path": str(tab_3_5_path)},
            "fewshot_results": {"path": str(fewshot_path)},
        },
        "figures": {
            "acc_compare": {"path": str(acc_fig)},
            "runtime_compare": {"path": str(runtime_fig)},
            "fewshot_curve": {"path": str(fewshot_fig)},
        },
    }

    metrics_summary_path = output_root / "metrics_summary.json"
    metrics_summary_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics summary to: {metrics_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
