#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module-level evaluation for diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from BRB.module_brb import MODULE_LABELS_V2
from pipelines.default_paths import PROJECT_ROOT, SIM_DIR


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


def _topk_indices(scores: np.ndarray, k: int) -> List[int]:
    return np.argsort(scores)[::-1][:k].tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate module-level predictions")
    parser.add_argument("--sim_dir", default=str(PROJECT_ROOT / SIM_DIR))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "Output" / "module_eval"))
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = sim_dir / "labels.json"
    module_pred_path = sim_dir / "module_predictions.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found at {labels_path}")
    if not module_pred_path.exists():
        raise FileNotFoundError(f"module_predictions.csv not found at {module_pred_path}")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    preds = _read_csv(module_pred_path)

    label_order = MODULE_LABELS_V2
    label_to_idx = {name: idx for idx, name in enumerate(label_order)}

    y_true = []
    y_pred = []
    top3_hits = 0
    total = 0

    for row in preds:
        sample_id = row.get("sample_id")
        if not sample_id or sample_id not in labels:
            continue
        label_info = labels[sample_id]
        true_label = label_info.get("module_v2") or label_info.get("module")
        if not true_label:
            continue
        if true_label not in label_to_idx:
            continue
        scores = np.array([float(row.get(name, 0.0)) for name in label_order], dtype=float)
        if scores.size == 0:
            continue
        pred_idx = int(np.argmax(scores))
        top3 = _topk_indices(scores, 3)
        total += 1
        y_true.append(label_to_idx[true_label])
        y_pred.append(pred_idx)
        if label_to_idx[true_label] in top3:
            top3_hits += 1

    if total == 0:
        raise RuntimeError("No valid module labels to evaluate.")

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc_top1 = float(np.mean(y_true_arr == y_pred_arr))
    acc_top3 = float(top3_hits / total)

    n_classes = len(label_order)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_arr, y_pred_arr):
        cm[t, p] += 1

    f1_scores = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores))

    summary = {
        "n_samples": total,
        "top1_accuracy": acc_top1,
        "top3_recall": acc_top3,
        "macro_f1": macro_f1,
        "labels_path": str(labels_path),
        "module_predictions_path": str(module_pred_path),
    }
    (output_dir / "module_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cm_csv = output_dir / "module_confusion_matrix.csv"
    with cm_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["True\\Pred"] + label_order)
        for idx, name in enumerate(label_order):
            writer.writerow([name] + cm[idx].tolist())

    # 电源模块 Top-1 泄露统计
    power_idx = label_to_idx.get("电源模块")
    power_top1 = 0
    non_power_total = 0
    if power_idx is not None:
        for t, p in zip(y_true_arr, y_pred_arr):
            if t == power_idx:
                continue
            non_power_total += 1
            if p == power_idx:
                power_top1 += 1
    leakage = {
        "non_power_samples": non_power_total,
        "power_top1_count": power_top1,
        "power_top1_ratio": float(power_top1 / non_power_total) if non_power_total else 0.0,
    }
    leak_path = output_dir / "power_top1_leakage_report.json"
    leak_path.write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        {
            "metric": "top1_accuracy",
            "value": acc_top1,
        },
        {
            "metric": "top3_recall",
            "value": acc_top3,
        },
        {
            "metric": "macro_f1",
            "value": macro_f1,
        },
    ]
    _write_csv(output_dir / "module_metrics.csv", rows, ["metric", "value"])

    print(f"Saved module metrics to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
