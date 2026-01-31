import argparse
import csv
import json
import glob
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.baseline import align_to_frequency
from baseline.config import BASELINE_ARTIFACTS, BASELINE_META, BAND_RANGES, OUTPUT_DIR
from features.extract import extract_system_features
from BRB.system_brb import system_level_infer
from BRB.module_brb import module_level_infer


def resolve(repo_root: Path, p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (repo_root / p).resolve()


def load_thresholds(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_thresholds(features, thresholds):
    """返回告警标志 dict：warn / alarm / ok，双阈值策略。"""
    flags = {}
    for k, v in thresholds.items():
        val = features.get(k, None)
        if val is None:
            continue
        low = v.get("warn", None)
        high = v.get("alarm", None)
        if high is not None and abs(val) >= high:
            flags[k] = "alarm"
        elif low is not None and abs(val) >= low:
            flags[k] = "warn"
        else:
            flags[k] = "ok"
    return flags


def _read_csv_two_columns(path: Path) -> Dict[str, np.ndarray]:
    freq_raw: List[float] = []
    amp_raw: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                freq_raw.append(float(row[0]))
                amp_raw.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return {"frequency": np.array(freq_raw, dtype=float), "amplitude": np.array(amp_raw, dtype=float)}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run detection and save results")
    parser.add_argument("--input", default="to_detect", help="待检 CSV 目录 (默认: to_detect)")
    parser.add_argument("--out_dir", default=str(Path(OUTPUT_DIR) / "runs" / "run_001"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = resolve(repo_root, args.input)
    run_dir = resolve(repo_root, args.out_dir)
    tables_dir = run_dir / "tables"
    artifacts_dir = run_dir / "artifacts"
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    baseline_artifacts = resolve(repo_root, BASELINE_ARTIFACTS)
    baseline_meta = resolve(repo_root, BASELINE_META)
    thresholds_path = resolve(repo_root, "thresholds.json")
    to_detect_glob = str(input_dir / "*.csv")

    art = np.load(baseline_artifacts)
    frequency = art["frequency"]
    rrs = art["rrs"]
    bounds = (art["upper"], art["lower"])
    with open(baseline_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    band_ranges = meta.get("band_ranges", BAND_RANGES)

    thresholds = load_thresholds(thresholds_path)

    files = glob.glob(to_detect_glob)
    if not files:
        raise FileNotFoundError(f"未找到待检 CSV：{to_detect_glob}")

    rows: List[Dict[str, object]] = []
    for fpath in files:
        curve = _read_csv_two_columns(Path(fpath))
        freq_raw = curve["frequency"]
        amp_raw = curve["amplitude"]
        if freq_raw.size == 0 or amp_raw.size == 0:
            continue
        amp = align_to_frequency(frequency, freq_raw, amp_raw)

        feats = extract_system_features(frequency, rrs, bounds, band_ranges, amp)
        sys_probs = system_level_infer(feats)
        mod_probs = module_level_infer(feats, sys_probs)
        flags = apply_thresholds(feats, thresholds)

        row = {
            "file": str(fpath),
            **feats,
            **{f"sys_{k}": v for k, v in sys_probs.items()},
            **{f"mod_{k}": v for k, v in mod_probs.items()},
            **{f"flag_{k}": v for k, v in flags.items()},
        }
        rows.append(row)

    detection_results = tables_dir / "detection_results.csv"
    _write_csv(detection_results, rows)
    features_path = artifacts_dir / "features_detect.csv"
    _write_csv(features_path, rows)
    print(f"检测结果已保存: {detection_results}")
    print(f"[INFO] features saved: {features_path}")


if __name__ == "__main__":
    main()
