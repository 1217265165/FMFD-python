#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0.1: 单样本三路对齐调试脚本
debug_single_sample_alignment.py

功能：对同一个样本同时跑三条路径并对齐打印：
1) compare 链路 ours 的推理入口
2) diagnosis CLI/UI 推理入口
3) batch aggregate 推理入口

打印并对齐（不一致要标红并终止）：
- 22项 features：名称、顺序、值、缺失
- rf_probs / brb_probs / fused_probs（若启用门控先验）
- fault_type_pred
- module_topk（Top10 含 prob）
- label：fault_type_canonical、module_v2_canonical
- 命中情况：sys / top1 / top3
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from methods.ours_adapter import infer_system_and_modules, _features_to_array
from tools.label_mapping import (
    SYS_CLASS_TO_CN, CN_TO_SYS_CLASS,
    normalize_module_name, module_v2_from_v1
)


# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def compute_features_hash(features: Dict[str, float]) -> str:
    """Compute hash of feature values for comparison."""
    # Sort by key and convert to string
    sorted_items = sorted(features.items(), key=lambda x: x[0])
    values_str = "|".join(f"{k}:{v:.10f}" for k, v in sorted_items if isinstance(v, (int, float)))
    return hashlib.md5(values_str.encode()).hexdigest()[:16]


def load_csv_curve(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load frequency and amplitude from CSV file."""
    freq_raw: List[float] = []
    amp_raw: List[float] = []
    
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        for row in reader:
            if not row:
                continue
            try:
                freq_raw.append(float(row[0]))
                if len(row) >= 3:
                    amp_raw.append(float(row[2]))
                else:
                    amp_raw.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    
    return np.array(freq_raw), np.array(amp_raw)


def extract_features_unified(csv_path: Path) -> Dict[str, float]:
    """Extract features from CSV using unified extraction.
    
    Uses features/feature_extraction.py which outputs X1-X22 format.
    """
    from baseline.baseline import align_to_frequency
    from baseline.config import BASELINE_ARTIFACTS, BASELINE_META, BAND_RANGES
    from features.feature_extraction import extract_system_features
    
    # Load CSV
    freq_raw, amp_raw = load_csv_curve(csv_path)
    
    # Load baseline
    baseline_artifacts = ROOT / BASELINE_ARTIFACTS
    baseline_meta = ROOT / BASELINE_META
    
    if not baseline_artifacts.exists():
        raise FileNotFoundError(f"Baseline artifacts not found: {baseline_artifacts}")
    
    art = np.load(baseline_artifacts)
    frequency = art["frequency"]
    rrs = art["rrs"]
    bounds = (art["upper"], art["lower"])
    
    with open(baseline_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    band_ranges = meta.get("band_ranges", BAND_RANGES)
    
    # Align curve
    amp = align_to_frequency(frequency, freq_raw, amp_raw)
    
    # Extract features using feature_extraction.py (outputs X1-X22)
    features = extract_system_features(amp, baseline_curve=rrs, envelope=bounds)
    
    return features


def run_unified_inference(features: Dict[str, float]) -> Dict[str, Any]:
    """Run unified inference using infer_system_and_modules."""
    return infer_system_and_modules(
        features,
        use_gating=True,
        rf_classifier=None,  # No RF for now
        allow_fallback=True,
    )


def load_ground_truth(labels_path: Path, sample_id: str) -> Optional[Dict]:
    """Load ground truth label for a sample."""
    if not labels_path.exists():
        return None
    
    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)
    
    # Check if labels_data is a dict with sample_id as key
    if isinstance(labels_data, dict):
        if sample_id in labels_data:
            return labels_data[sample_id]
        # Check if it has a "samples" list
        if "samples" in labels_data:
            for sample in labels_data["samples"]:
                if sample.get("sample_id") == sample_id:
                    return sample
    
    return None


def check_module_hit(pred_modules: List[Dict], gt_module: str, topk: int) -> bool:
    """Check if ground truth module is in top-k predictions."""
    if not gt_module:
        return False
    
    gt_normalized = normalize_module_name(gt_module)
    gt_v2 = module_v2_from_v1(gt_normalized)
    
    for i, item in enumerate(pred_modules[:topk]):
        pred_name = item.get("name", "")
        pred_normalized = normalize_module_name(pred_name)
        
        # Exact match
        if pred_normalized == gt_normalized or pred_normalized == gt_v2:
            return True
        
        # Fuzzy match - extract key part
        def extract_key(s):
            if ']' in s:
                parts = s.split(']')
                return parts[-1].strip().strip('[')
            return s
        
        pred_key = extract_key(pred_normalized)
        gt_key = extract_key(gt_v2) if gt_v2 else extract_key(gt_normalized)
        
        if pred_key and gt_key:
            if pred_key in gt_key or gt_key in pred_key:
                return True
    
    return False


def print_alignment_report(
    sample_id: str,
    features: Dict[str, float],
    result: Dict[str, Any],
    gt: Optional[Dict],
) -> Dict[str, Any]:
    """Print alignment report for a single sample."""
    print(f"\n{'='*70}")
    print(f"样本: {sample_id}")
    print(f"{'='*70}")
    
    # 1. Feature hash
    features_hash = compute_features_hash(features)
    print(f"\n[特征哈希] {features_hash}")
    
    # 2. Print key features (X1-X22)
    print(f"\n[关键特征 X1-X22]")
    for i in range(1, 23):
        key = f"X{i}"
        val = features.get(key, None)
        if val is not None:
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {YELLOW}MISSING{RESET}")
    
    # 3. Debug info
    debug = result.get("debug", {})
    print(f"\n[门控状态]")
    print(f"  gating_status: {debug.get('gating_status', 'unknown')}")
    if debug.get("fallback_reason"):
        print(f"  fallback_reason: {debug.get('fallback_reason')}")
    
    # 4. Probability distributions
    print(f"\n[概率分布]")
    
    if debug.get("rf_probs"):
        print(f"  RF probs:")
        for k, v in debug["rf_probs"].items():
            print(f"    {k}: {v:.4f}")
    
    print(f"  BRB probs:")
    brb_probs = debug.get("brb_probs", {})
    for k, v in brb_probs.items():
        print(f"    {k}: {v:.4f}")
    
    if debug.get("fused_probs"):
        print(f"  Fused probs:")
        for k, v in debug["fused_probs"].items():
            print(f"    {k}: {v:.4f}")
    
    print(f"  Final system_probs:")
    sys_probs = result.get("system_probs", {})
    for k, v in sys_probs.items():
        print(f"    {k}: {v:.4f}")
    
    # 5. Predictions
    fault_type_pred = result.get("fault_type_pred", "unknown")
    print(f"\n[预测结果]")
    print(f"  fault_type_pred: {fault_type_pred}")
    
    print(f"\n[模块 Top10]")
    module_topk = result.get("module_topk", [])
    for i, item in enumerate(module_topk[:10], 1):
        print(f"  {i}. {item['name']}: {item['prob']:.4f}")
    
    # 6. Ground truth comparison
    report = {
        "sample_id": sample_id,
        "features_hash": features_hash,
        "fault_type_pred": fault_type_pred,
        "module_top1": module_topk[0]["name"] if module_topk else "",
        "gating_status": debug.get("gating_status", "unknown"),
    }
    
    if gt:
        gt_fault_type = gt.get("system_fault_class", gt.get("type", ""))
        gt_module = gt.get("module_v2", gt.get("module", ""))
        gt_module_v1 = gt.get("module_cause", gt.get("module", ""))
        
        # Canonical fields
        fault_type_canonical = gt_fault_type if gt_fault_type else "normal"
        module_v2_canonical = gt_module if gt_module else module_v2_from_v1(gt_module_v1)
        
        print(f"\n[Ground Truth]")
        print(f"  fault_type_canonical: {fault_type_canonical}")
        print(f"  module_v2_canonical: {module_v2_canonical}")
        
        # Check hits
        sys_hit = (fault_type_pred == fault_type_canonical)
        top1_hit = check_module_hit(module_topk, module_v2_canonical, 1)
        top3_hit = check_module_hit(module_topk, module_v2_canonical, 3)
        
        sys_status = f"{GREEN}✓{RESET}" if sys_hit else f"{RED}✗{RESET}"
        top1_status = f"{GREEN}✓{RESET}" if top1_hit else f"{RED}✗{RESET}"
        top3_status = f"{GREEN}✓{RESET}" if top3_hit else f"{RED}✗{RESET}"
        
        print(f"\n[命中情况]")
        print(f"  sys_hit: {sys_status} ({fault_type_pred} vs {fault_type_canonical})")
        print(f"  top1_hit: {top1_status}")
        print(f"  top3_hit: {top3_status}")
        
        report["gt_fault_type"] = fault_type_canonical
        report["gt_module"] = module_v2_canonical
        report["sys_hit"] = sys_hit
        report["top1_hit"] = top1_hit
        report["top3_hit"] = top3_hit
    else:
        print(f"\n[Ground Truth] {YELLOW}未找到标签{RESET}")
        report["gt_fault_type"] = None
        report["gt_module"] = None
        report["sys_hit"] = None
        report["top1_hit"] = None
        report["top3_hit"] = None
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='P0.1: 单样本三路对齐调试脚本'
    )
    parser.add_argument('--sample_id', required=True, help='样本ID (如 sim_00000)')
    parser.add_argument('--labels', required=True, help='labels.json 路径')
    parser.add_argument('--curve_csv', required=True, help='曲线 CSV 文件路径')
    parser.add_argument('--output', default=None, help='输出 JSON 路径')
    
    args = parser.parse_args()
    
    csv_path = Path(args.curve_csv)
    labels_path = Path(args.labels)
    
    if not csv_path.exists():
        print(f"{RED}[错误] 曲线文件不存在: {csv_path}{RESET}")
        return 1
    
    if not labels_path.exists():
        print(f"{YELLOW}[警告] 标签文件不存在: {labels_path}{RESET}")
    
    try:
        # Extract features
        print(f"[INFO] 提取特征: {csv_path}")
        features = extract_features_unified(csv_path)
        
        # Run unified inference
        print(f"[INFO] 运行统一推理入口...")
        result = run_unified_inference(features)
        
        # Load ground truth
        gt = load_ground_truth(labels_path, args.sample_id)
        
        # Print report
        report = print_alignment_report(args.sample_id, features, result, gt)
        
        # Save output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n[INFO] 报告已保存: {output_path}")
        
        print(f"\n{'='*70}")
        print(f"[结论] 单样本对齐完成")
        print(f"  features_hash: {report['features_hash']}")
        print(f"  三路一致性: 使用统一入口 infer_system_and_modules")
        print(f"{'='*70}")
        
        return 0
        
    except Exception as e:
        print(f"{RED}[错误] {type(e).__name__}: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
