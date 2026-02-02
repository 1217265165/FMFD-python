#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRB 诊断命令行接口
用于接收频响数据并返回BRB诊断结果

该脚本可以通过命令行调用，也可以打包为exe供QT程序调用
调用方式:
  python brb_diagnosis_cli.py --input <input_csv> --output <output_json> [--baseline <baseline_dir>]
  
输入CSV格式: frequency,amplitude (两列，频率和幅度)
输出JSON格式: 包含系统级和模块级诊断结果

新增功能:
  --labels: 提供 labels.json 文件路径，可在输出中附加 ground_truth 字段
"""

import argparse
import csv
import json
import re
import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 设置环境变量抑制所有警告（在导入numpy/pandas之前）
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::DeprecationWarning'

# 抑制numpy/pandas兼容性警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')

import numpy as np


def resolve_import_path():
    """解决导入路径问题 - 添加FMFD父目录到sys.path"""
    current_file = Path(__file__).resolve()
    fmfd_root = current_file.parent  # Python/FMFD目录
    python_root = fmfd_root.parent    # Python目录
    
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    
    return fmfd_root


def parse_sample_id(input_path: Path) -> str:
    """从输入文件名中解析 sample_id。
    
    解析规则：
    1. 尝试正则匹配 sim_XXXXX 格式（5位数字）
    2. 如果无法匹配，使用文件名（不含扩展名）
    
    Parameters
    ----------
    input_path : Path
        输入文件路径
        
    Returns
    -------
    str
        解析出的 sample_id
    """
    filename = input_path.stem  # 不含扩展名的文件名
    
    # 尝试匹配 sim_XXXXX 格式
    match = re.search(r'(sim_\d{5})', filename)
    if match:
        return match.group(1)
    
    # 回退到使用文件名
    return filename


def load_input_csv(input_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    freq_raw: List[float] = []
    amp_raw: List[float] = []
    peak_freq: List[float] = []
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                freq_raw.append(float(row[0]))
                if len(row) >= 3:
                    peak_freq.append(float(row[1]))
                    amp_raw.append(float(row[2]))
                else:
                    amp_raw.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    peak_arr = np.array(peak_freq, dtype=float) if peak_freq else None
    return np.array(freq_raw, dtype=float), np.array(amp_raw, dtype=float), peak_arr


def main():
    parser = argparse.ArgumentParser(
        description='BRB诊断命令行工具 - 频响异常诊断',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python brb_diagnosis_cli.py --input test_data.csv --output result.json
  python brb_diagnosis_cli.py --input test_data.csv --output result.json --baseline ./baseline_data
  python brb_diagnosis_cli.py --input sim_00009.csv --output result.json --labels labels.json
  python brb_diagnosis_cli.py --mode online_infer
        """
    )
    
    parser.add_argument('--input', '-i', default=None,
                        help='输入CSV文件路径 (格式: frequency,amplitude)')
    parser.add_argument('--output', '-o', default=None,
                        help='输出JSON文件路径')
    parser.add_argument('--baseline', '-b', default=None,
                        help='基线数据目录 (可选，默认使用程序内置路径)')
    parser.add_argument('--mode', '-m', default='sub_brb',
                        choices=['er', 'simple', 'sub_brb', 'online_infer'],
                        help='BRB推理模式: sub_brb(推荐,子BRB架构), er(增强版), simple(简化版), '
                             'online_infer(自动选最新仿真样本并写入默认输出)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细输出')
    parser.add_argument('--run_name', default=None,
                        help='运行名称 (用于组织输出目录，默认使用时间戳)')
    parser.add_argument('--out_dir', default='Output',
                        help='输出目录根路径 (默认: Output)')
    parser.add_argument('--labels', '-l', default=None,
                        help='labels.json 文件路径 (可选，用于回填 ground_truth)')
    parser.add_argument('--topk', type=int, default=3,
                        help='输出 TopK 模块数量 (默认: 3)')
    parser.add_argument('--include_baseline', action='store_true',
                        help='在输出 JSON 中包含基线信息 (用于前端绘图)')
    parser.add_argument('--downsample_baseline', type=int, default=1,
                        help='基线下采样率 (默认: 1 不下采样, 可选 4 降为 205 点)')
    parser.add_argument('--trace_output', default='Output/diagnosis_audit',
                        help='诊断审计输出目录 (默认: Output/diagnosis_audit)')
    parser.add_argument('--input_dir', default=None,
                        help='批量输入目录 (包含 sim_*.csv)')
    
    args = parser.parse_args()
    if args.mode != "online_infer":
        if not args.input and not args.input_dir:
            parser.error("--input or --input_dir is required unless --mode online_infer")

    def _latest_sim_input(root: Path) -> Optional[Path]:
        sim_dir = root / "Output" / "sim_spectrum" / "raw_curves"
        if not sim_dir.exists():
            return None
        candidates = list(sim_dir.glob("*.csv"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _smooth_series(arr: np.ndarray, window: int = 61) -> np.ndarray:
        if arr.size < window or window < 3:
            return arr
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(arr, kernel, mode="same")
    
    # 解决导入路径
    fmfd_root = resolve_import_path()
    
    try:
        # 导入FMFD模块
        from baseline.baseline import align_to_frequency
        from baseline.config import BASELINE_ARTIFACTS, BASELINE_META, BAND_RANGES
        from baseline.rrs_envelope import vendor_tolerance_db
        from features.extract import extract_system_features
        from BRB.system_brb import system_level_infer
        from BRB.module_brb import module_level_infer, DISABLED_MODULES
        from BRB.uncertainty import (
            detect_uncertainty, format_uncertainty_explanation, UncertaintyConfig
        )
        from tools.label_mapping import (
            SYS_CLASS_TO_CN, CN_TO_SYS_CLASS, 
            get_topk_modules, normalize_module_name
        )
        from tools.module_validation import (
            validate_module_diagnosis, format_validation_report
        )
        
        if args.verbose:
            print(f"[INFO] FMFD模块导入成功", file=sys.stderr)
            print(f"[INFO] 工作目录: {fmfd_root}", file=sys.stderr)
        
        # 1. 读取输入数据
        input_paths: List[Path] = []
        if args.input_dir:
            input_dir = Path(args.input_dir)
            if input_dir.exists():
                input_paths = sorted(input_dir.glob("*.csv"))
        if args.input:
            input_paths.append(Path(args.input))
        if args.mode == "online_infer" and not input_paths:
            latest = _latest_sim_input(fmfd_root)
            if latest:
                input_paths.append(latest)
                print(f"[INFO] 自动选择最新仿真样本: {latest}", file=sys.stderr)
        input_paths = [p for p in input_paths if p.exists()]
        if not input_paths:
            print(f"[错误] 输入文件不存在: {args.input} / {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        
        labels_data = None
        if args.labels:
            labels_path = Path(args.labels)
            if labels_path.exists():
                try:
                    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
                except Exception:
                    labels_data = None

        def _resolve_output_path(input_path: Path) -> Path:
            if len(input_paths) > 1:
                output_base = Path(args.output) if args.output else fmfd_root / "Output" / "diagnosis" / "batch"
                if output_base.suffix.lower() == ".json":
                    output_base = output_base.parent
                output_base.mkdir(parents=True, exist_ok=True)
                return output_base / f"{input_path.stem}_diagnosis.json"

            if args.output:
                output_path = Path(args.output)
                if output_path.suffix.lower() == ".json":
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    return output_path
                output_path.mkdir(parents=True, exist_ok=True)
                return output_path / f"{input_path.stem}_diagnosis.json"

            output_path = fmfd_root / "Output" / "diagnosis" / "latest" / "diagnosis_result.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

        def _infer_one(input_path: Path) -> Dict[str, object]:
            # 解析 sample_id
            sample_id = parse_sample_id(input_path)
            if args.verbose:
                print(f"[INFO] 解析 sample_id: {sample_id}", file=sys.stderr)

            freq_raw, amp_raw, peak_freq_meas = load_input_csv(input_path)
            if freq_raw.size == 0 or amp_raw.size == 0:
                print(f"[错误] 输入CSV文件至少需要2列 (frequency, amplitude)", file=sys.stderr)
                raise SystemExit(1)

            if args.verbose:
                print(f"[INFO] 读取数据点数: {len(freq_raw)}", file=sys.stderr)
                print(f"[INFO] 频率范围: {freq_raw.min():.2e} - {freq_raw.max():.2e} Hz", file=sys.stderr)
                print(f"[INFO] 幅度范围: {amp_raw.min():.2f} - {amp_raw.max():.2f} dBm", file=sys.stderr)

            # 2. 加载基线数据
            if args.baseline:
                baseline_path = Path(args.baseline)
                if baseline_path.is_file():
                    baseline_artifacts = baseline_path
                    baseline_meta = baseline_path.with_name("baseline_meta.json")
                else:
                    baseline_artifacts = baseline_path / "baseline_artifacts.npz"
                    baseline_meta = baseline_path / "baseline_meta.json"
            else:
                baseline_artifacts = fmfd_root / BASELINE_ARTIFACTS
                baseline_meta = fmfd_root / BASELINE_META

            if not baseline_artifacts.exists():
                print(f"[错误] 基线数据文件不存在: {baseline_artifacts}", file=sys.stderr)
                raise SystemExit(1)

            art = np.load(baseline_artifacts)
            frequency = art["frequency"]
            rrs = art["rrs"]
            bounds = (art["upper"], art["lower"])

            with open(baseline_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            band_ranges = meta.get("band_ranges", BAND_RANGES)

            if args.verbose:
                print(f"[INFO] 基线频率点数: {len(frequency)}", file=sys.stderr)
                print(f"[INFO] 基线频段数: {len(band_ranges)}", file=sys.stderr)

            # 3. 对齐频率并提取特征
            amp = align_to_frequency(frequency, freq_raw, amp_raw)
            features = extract_system_features(frequency, rrs, bounds, band_ranges, amp)

            if args.verbose:
                print(f"[INFO] 提取特征数: {len(features)}", file=sys.stderr)
                print(f"[INFO] 特征: {list(features.keys())}", file=sys.stderr)

            # 4. 执行BRB推理
            infer_mode = "sub_brb" if args.mode == "online_infer" else args.mode
            sys_probs = system_level_infer(features, mode=infer_mode)
            mod_probs = module_level_infer(features, sys_probs)

            if args.verbose:
                print(f"[INFO] 系统级诊断完成", file=sys.stderr)
                print(f"[INFO] 模块级诊断完成 ({len(mod_probs)}个模块)", file=sys.stderr)

            # 5. 构造输出结果
            sys_probs_dict = sys_probs.get('probabilities', sys_probs) if isinstance(sys_probs, dict) else sys_probs
            is_normal = sys_probs.get('is_normal', False) if isinstance(sys_probs, dict) else False
            max_prob = max(sys_probs_dict.values()) if sys_probs_dict else 0.0
            predicted_class = max(sys_probs_dict, key=sys_probs_dict.get) if sys_probs_dict else "未知"

            topk_modules = get_topk_modules(mod_probs, k=args.topk, skip_disabled=True, disabled_modules=list(DISABLED_MODULES))
            topk_list = [{"module": name, "probability": float(prob)} for name, prob in topk_modules]

            viol_rate = float(features.get('viol_rate', 0.0))

            step_hz = float(np.median(np.diff(freq_raw))) if len(freq_raw) > 1 else 0.0
            offset_db = float(np.median(amp - rrs))
            amp_aligned = amp - offset_db
            residual = amp_aligned - rrs
            smooth_res = _smooth_series(residual)
            hf_std = float(np.std(residual - smooth_res)) if residual.size else 0.0
            p95_abs = float(np.quantile(np.abs(residual), 0.95)) if residual.size else 0.0
            inside_env_frac = float(np.mean((amp_aligned >= bounds[1]) & (amp_aligned <= bounds[0]))) if residual.size else 0.0

            if peak_freq_meas is not None and peak_freq_meas.size != freq_raw.size:
                peak_freq_meas = None
            if peak_freq_meas is not None and len(peak_freq_meas) == len(freq_raw):
                delta = peak_freq_meas - freq_raw
                peak_track = {
                    "peak_freq_mae_hz": float(np.mean(np.abs(delta))),
                    "peak_freq_outlier_frac": float(np.mean(np.abs(delta) > max(step_hz * 2.0, 1.0))),
                }
            else:
                peak_track = {
                    "peak_freq_mae_hz": None,
                    "peak_freq_outlier_frac": None,
                }

            result = {
                "status": "success",
                "meta": {
                    "sample_id": sample_id,
                    "input_file": str(input_path.absolute()),
                    "operating_condition": {
                        "inject_power_dbm": -10.0,
                        "freq_start_hz": float(freq_raw.min()),
                        "freq_end_hz": float(freq_raw.max()),
                        "step_hz": step_hz,
                        "frontend": {
                            "low_band_only": True,
                            "ac_coupling": True,
                            "preamp": False,
                        },
                    },
                },
                "data_points": len(freq_raw),
                "frequency_range": {
                    "min": float(freq_raw.min()),
                    "max": float(freq_raw.max())
                },
                "features": {k: float(v) if isinstance(v, (int, float)) else v for k, v in features.items()},
                "system": {
                    "probs": {k: float(v) for k, v in sys_probs_dict.items()},
                    "decision": predicted_class,
                    "confidence": float(max_prob),
                    "is_normal": is_normal,
                },
                "module": {
                    "topk": [
                        {
                            "module_id": item["module"],
                            "gamma": item["probability"],
                        }
                        for item in topk_list
                    ],
                    "gating": {
                        "disabled_modules": list(DISABLED_MODULES),
                        "topk_k": args.topk,
                    },
                },
                "evidence": {
                    "global_offset_db": offset_db,
                    "hf_std_db": hf_std,
                    "p95_abs_dev_db": p95_abs,
                    "inside_env_frac": inside_env_frac,
                    **peak_track,
                },
                "artifacts": {
                    "output_json": None,
                },
                "system_diagnosis": {
                    "probabilities": {k: float(v) for k, v in sys_probs_dict.items()},
                    "predicted_class": predicted_class,
                    "max_prob": float(max_prob),
                    "is_normal": is_normal,
                },
                "module_diagnosis": {
                    "probabilities": {k: float(v) for k, v in mod_probs.items()},
                    "topk": topk_list,
                    "disabled_modules": list(DISABLED_MODULES),
                },
                "evidence_detail": {
                    "viol_rate": viol_rate,
                    "envelope_violation": viol_rate > 0.1,
                    "violation_max_db": float(features.get('X12', features.get('env_overrun_max', 0.0))),
                    "violation_energy": float(features.get('X13', features.get('env_violation_energy', 0.0))),
                    "baseline_coverage": 1.0 - viol_rate,
                },
                "config": {
                    "mode": infer_mode,
                    "run_name": args.run_name,
                    "topk": args.topk,
                }
            }

            # T3: 低置信度检测与解释
            uncertainty_result = detect_uncertainty(sys_probs_dict, features)
            result["uncertainty"] = uncertainty_result.to_dict()
            
            if args.verbose and uncertainty_result.is_uncertain:
                print("\n" + format_uncertainty_explanation(uncertainty_result), file=sys.stderr)

            # T6: 模块级诊断自动验证（如果有 GT）
            if labels_data and sample_id in labels_data:
                gt = labels_data[sample_id]
                gt_module = gt.get("module_cause", gt.get("module", ""))
                gt_module_v2 = gt.get("module_v2", "")
                
                validation_result = validate_module_diagnosis(
                    sample_id=sample_id,
                    gt_module=gt_module,
                    gt_module_v2=gt_module_v2,
                    module_probs=mod_probs,
                    disabled_modules=list(DISABLED_MODULES),
                )
                result["module_validation"] = validation_result.to_dict()
                
                if args.verbose:
                    print("\n" + "="*50, file=sys.stderr)
                    print("[T6: 模块级验证]", file=sys.stderr)
                    print("="*50, file=sys.stderr)
                    print(f"  GT Module: {gt_module}", file=sys.stderr)
                    print(f"  GT Module V2: {gt_module_v2}", file=sys.stderr)
                    print(f"  Top1 Hit: {'✅' if validation_result.top1_hit else '❌'}", file=sys.stderr)
                    print(f"  Top3 Hit: {'✅' if validation_result.top3_hit else '❌'}", file=sys.stderr)
                    print(f"  GT Rank: {validation_result.gt_rank}", file=sys.stderr)
                    print(f"  GT Prob: {validation_result.gt_prob:.4f}", file=sys.stderr)

            # 写入审计 trace（单样本）
            try:
                from features.feature_router import (
                    FAULT_TYPE_MAPPING,
                    FAULT_TYPE_TO_MODULE_GROUP,
                    MODULE_FEATURES_BY_GROUP,
                    get_modules_for_fault_type,
                )
            except Exception:
                FAULT_TYPE_MAPPING = {}
                FAULT_TYPE_TO_MODULE_GROUP = {}
                MODULE_FEATURES_BY_GROUP = {}
                get_modules_for_fault_type = None

            fault_key = FAULT_TYPE_MAPPING.get(predicted_class, predicted_class)
            module_group = FAULT_TYPE_TO_MODULE_GROUP.get(fault_key, "other_group")
            active_features = MODULE_FEATURES_BY_GROUP.get(module_group, [])
            active_modules = get_modules_for_fault_type(fault_key) if get_modules_for_fault_type else []

            trace_payload = {
                "sample_id": sample_id,
                "csv_path": str(input_path),
                "system_probs": {k: float(v) for k, v in sys_probs_dict.items()},
                "system_decision": predicted_class,
                "module_group": module_group,
                "active_features": active_features,
                "active_modules": active_modules,
                "module_topk": topk_list,
                "power_module_active": "电源模块" in active_modules,
            }
            if labels_data and sample_id in labels_data:
                gt = labels_data[sample_id]
                trace_payload.update(
                    {
                        "true_fault_type": gt.get("system_fault_class"),
                        "true_module_v1": gt.get("module"),
                        "true_module_v2": gt.get("module_v2"),
                    }
                )
            try:
                trace_dir = Path(args.trace_output)
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / f"cli_trace_{time.strftime('%Y%m%d')}.jsonl"
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
            except Exception:
                pass

            # 5.5 如果需要包含基线信息（用于前端绘图）
            if args.include_baseline:
                ds = args.downsample_baseline
                offset_db = float(np.median(amp - rrs))
                residual = (amp - offset_db) - rrs
                if ds > 1:
                    freq_ds = frequency[::ds].tolist()
                    rrs_ds = rrs[::ds].tolist()
                    upper_ds = bounds[0][::ds].tolist()
                    lower_ds = bounds[1][::ds].tolist()
                    residual_ds = residual[::ds].tolist()
                else:
                    freq_ds = frequency.tolist()
                    rrs_ds = rrs.tolist()
                    upper_ds = bounds[0].tolist()
                    lower_ds = bounds[1].tolist()
                    residual_ds = residual.tolist()

                center_level_db = float(np.median(rrs))
                spec_center_db = -10.0
                spec_tol_db = 0.4
                spec_upper_db = spec_center_db + spec_tol_db
                spec_lower_db = spec_center_db - spec_tol_db

                quantiles = meta.get("smooth_params", {}).get("quantiles", {})
                result["baseline"] = {
                    "frequency_hz": freq_ds,
                    "rrs_dbm": rrs_ds,
                    "upper_dbm": upper_ds,
                    "lower_dbm": lower_ds,
                    "offset_db": offset_db,
                    "residual_db": residual_ds,
                    "coverage": {
                        "point_coverage": float(1.0 - features.get("viol_rate_aligned", features.get("viol_rate", 0.0))),
                        "viol_rate": float(features.get("viol_rate_aligned", features.get("viol_rate", 0.0))),
                        "viol_energy": float(features.get("viol_energy_aligned", 0.0)),
                    },
                    "meta": {
                        "q_lo": quantiles.get("q_low"),
                        "q_hi": quantiles.get("q_high"),
                        "clip_db": meta.get("clip_db", 0.4),
                        "smoothing": {
                            "width_smooth_sigma_hz": meta.get("smooth_params", {}).get("width_smooth_sigma_hz"),
                        },
                    },
                    "center_level_db": center_level_db,
                    "spec_center_db": spec_center_db,
                    "spec_tol_db": spec_tol_db,
                    "spec_upper_db": spec_upper_db,
                    "spec_lower_db": spec_lower_db,
                    "vendor_tolerance_db": vendor_tolerance_db(np.array(freq_ds)).tolist() if ds == 1 else vendor_tolerance_db(frequency[::ds]).tolist(),
                    "chosen_k": meta.get("k_final", 3.5),
                    "coverage_target": meta.get("coverage_mean", 0.97),
                    "n_points": len(freq_ds),
                    "downsample_factor": ds,
                }

            # 6. 如果提供了 labels.json，加载 ground_truth
            if labels_data and sample_id in labels_data:
                gt = labels_data[sample_id]
                gt_sys_class = gt.get("system_fault_class")
                gt_sys_cn = SYS_CLASS_TO_CN.get(gt_sys_class, "正常") if gt_sys_class else "正常"
                result["ground_truth"] = {
                    "type": gt.get("type", "unknown"),
                    "system_class_en": gt_sys_class or "normal",
                    "system_class_cn": gt_sys_cn,
                    "module": gt.get("module"),
                    "module_v2": gt.get("module_v2"),
                    "fault_params": gt.get("fault_params", {}),
                }
                if args.verbose:
                    print(f"[INFO] 已加载 ground_truth: {gt_sys_cn}", file=sys.stderr)

            output_path = _resolve_output_path(input_path)
            result["artifacts"]["output_json"] = str(output_path)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if args.verbose:
                print(f"[INFO] 结果已保存到: {output_path}", file=sys.stderr)
                print("\n" + "="*50, file=sys.stderr)
                print("[系统级诊断结果]", file=sys.stderr)
                print("="*50, file=sys.stderr)
                print(f"  样本ID: {sample_id}", file=sys.stderr)
                print(f"  预测类别: {predicted_class}", file=sys.stderr)
                print(f"  最大概率: {max_prob:.4f} ({max_prob*100:.2f}%)", file=sys.stderr)
                print(f"  是否正常: {is_normal}", file=sys.stderr)
                print("\n  概率分布:", file=sys.stderr)
                for k, v in sys_probs_dict.items():
                    print(f"    {k}: {v:.4f} ({v*100:.2f}%)", file=sys.stderr)
                print("\n" + "="*50, file=sys.stderr)
                print(f"[模块级诊断 TOP{args.topk}]（跳过禁用: {DISABLED_MODULES}）", file=sys.stderr)
                print("="*50, file=sys.stderr)
                for i, item in enumerate(topk_list, 1):
                    print(f"  {i}. {item['module']}: {item['probability']:.4f} ({item['probability']*100:.2f}%)", file=sys.stderr)
                print("\n" + "="*50, file=sys.stderr)
                print("[证据特征]", file=sys.stderr)
                print("="*50, file=sys.stderr)
                print(f"  viol_rate: {viol_rate:.4f}", file=sys.stderr)
                detail = result.get("evidence_detail", {})
                print(f"  violation_max_db: {detail.get('violation_max_db', 0.0):.4f}", file=sys.stderr)
                print(f"  violation_energy: {detail.get('violation_energy', 0.0):.4f}", file=sys.stderr)
                print(f"  baseline_coverage: {detail.get('baseline_coverage', 0.0):.4f}", file=sys.stderr)
                if "ground_truth" in result:
                    print("\n" + "="*50, file=sys.stderr)
                    print("[Ground Truth]", file=sys.stderr)
                    print("="*50, file=sys.stderr)
                    gt = result["ground_truth"]
                    print(f"  类型: {gt['type']}", file=sys.stderr)
                    print(f"  系统级: {gt['system_class_cn']}", file=sys.stderr)
                    print(f"  模块: {gt['module']}", file=sys.stderr)
                    true_mod = gt.get("module_v2") or gt.get("module")
                    pred_mod = topk_list[0]["module"] if topk_list else "未知"
                    print(f"  True Module: {true_mod}", file=sys.stderr)
                    print(f"  Predicted: {pred_mod}", file=sys.stderr)
                print("\n" + "="*50, file=sys.stderr)
                print(f"[输出文件] {output_path}", file=sys.stderr)
                print("="*50, file=sys.stderr)

            print(json.dumps(result, ensure_ascii=False))
            return result

        results = []
        for path in input_paths:
            results.append(_infer_one(path))
        if len(results) == 1:
            return 0
        return 0
        
    except ImportError as e:
        print(f"[错误] 模块导入失败: {e}", file=sys.stderr)
        print(f"[提示] 请确保已安装所需依赖: numpy, pandas", file=sys.stderr)
        print(f"[提示] 请确保FMFD模块在Python路径中", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False), file=sys.stdout)
        print(f"[错误] {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
