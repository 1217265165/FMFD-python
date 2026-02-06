#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从基线包络生成仿真样本并串联系统/模块 BRB 推理。

对应要求.md 的 4.2：基于 `run_baseline.py` 生成的产物，自动完成
“基线 → 仿真 → 特征提取 → 系统/模块 BRB 诊断” 一键流程。

使用说明（仓库根目录执行）::

    python pipelines/simulate/run_simulation_brb.py \
        --baseline_npz Output/baseline_artifacts.npz \
        --baseline_meta Output/baseline_meta.json \
        --switch_json Output/switching_features.json \
        --out_dir Output/sim_brb \
        --n_samples 200

输出::
    - Output/sim_spectrum/raw_curves/*.csv       # 每个样本一份频率-幅度 CSV，可直接给其他方法/CLI
    - Output/sim_spectrum/raw_manifest.csv       # raw_curves 下文件列表与标签
    - Output/sim_spectrum/features_brb.csv       # 对比脚本直接读取的特征+概率
    - Output/sim_spectrum/labels.json            # 与 features_brb.csv 对应的标签
    - Output/sim_spectrum/simulated_features.csv # X1~X5 + 旧版动态阈值特征 + 标签
    - Output/sim_spectrum/system_predictions.csv # 系统级概率与正常判定
    - Output/sim_spectrum/module_predictions.csv # 21 模块概率分布
    - Output/sim_spectrum/simulated_curves.csv   # 频率-幅度矩阵（可自行提取特征）
    - Output/sim_spectrum/simulated_curves.npz   # 便于复现的仿真曲线
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.baseline import compute_rrs_bounds
from baseline.config import (
    BASELINE_ARTIFACTS,
    BASELINE_META,
    BAND_RANGES,
    NORMAL_STATS_JSON,
    NORMAL_STATS_NPZ,
    OUTPUT_DIR,
    SWITCH_JSON,
)
from BRB.module_brb import DISABLED_MODULES, MODULE_LABELS_V2, module_level_infer
from BRB.system_brb import system_level_infer
from features.feature_extraction import (
    compute_dynamic_threshold_features,
    extract_module_features,
    extract_system_features,
)
from pipelines.simulate.faults import (
    inject_adc_sawtooth,
    inject_freq_miscal,
    inject_lpf_shift,
    inject_lo_path_error,
    inject_mixer1_slope,
    inject_power_noise_rrs,
    SINGLE_BAND_MODE,
)
from pipelines.simulate.check_sim_quality import run_quality_check
from pipelines.simulate.fault_models import (
    apply_template,
    module_spec_by_label,
    module_specs_by_system,
    module_templates,
    select_template,
)
# V-D.2: 导入物理核曲线生成器
from pipelines.simulate.curve_generator import (
    CurveGenerator,
    load_module_taxonomy,
    get_curve_generator,
)
from pipelines.simulate.sim_constraints import SimulationConstraints, load_baseline_stats
from pipelines.default_paths import (
    PROJECT_ROOT,
    OUTPUT_DIR,
    BASELINE_NPZ,
    BASELINE_META,
    SIM_DIR,
    SEED,
    SINGLE_BAND,
    DISABLE_PREAMP,
    DEFAULT_N_SAMPLES,
    DEFAULT_BALANCED,
    build_run_snapshot,
)


def _resolve(repo_root: Path, p: Path) -> Path:
    return p if p.is_absolute() else (repo_root / p).resolve()


def load_baseline(
    repo_root: Path, npz_path: Path, meta_path: Path, switch_path: Path
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], List[Tuple[float, float]], list]:
    npz_path = _resolve(repo_root, npz_path)
    meta_path = _resolve(repo_root, meta_path)
    switch_path = _resolve(repo_root, switch_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到基线产物 {npz_path}，请先运行 pipelines/run_baseline.py")

    data = np.load(npz_path, allow_pickle=True)
    frequency = data["frequency"]
    if "rrs" in data and "upper" in data and "lower" in data:
        rrs = data["rrs"]
        bounds = (data["upper"], data["lower"])
    else:
        traces = data["traces"]
        rrs, bounds = compute_rrs_bounds(frequency, traces)

    band_ranges = BAND_RANGES
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            band_ranges = meta.get("band_ranges", BAND_RANGES)
        except Exception:
            band_ranges = BAND_RANGES

    switch_feats = []
    if switch_path.exists():
        try:
            with open(switch_path, "r", encoding="utf-8") as f:
                switch_feats = json.load(f)
        except Exception:
            switch_feats = []

    return frequency, rrs, bounds, band_ranges, switch_feats


def _write_csv(path: Path, rows: List[Dict[str, object]], encoding: str = "utf-8") -> None:
    if not rows:
        path.write_text("", encoding=encoding)
        return

    # Collect all unique fieldnames from all rows
    all_fieldnames = set()
    for row in rows:
        all_fieldnames.update(row.keys())
    fieldnames = sorted(all_fieldnames)
    
    with path.open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_curves(path: Path, frequency: np.ndarray, curves: List[np.ndarray]) -> None:
    if not curves:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = ["frequency"] + [f"sim_{idx:05d}" for idx in range(len(curves))]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i, freq in enumerate(frequency):
            row = [freq]
            for curve in curves:
                row.append(f"{curve[i]:.4f}" if i < len(curve) else "")
            writer.writerow(row)


def _write_raw_csvs(
    base_dir: Path,
    frequency: np.ndarray,
    curves: List[np.ndarray],
    peak_freqs: List[np.ndarray],
    labels: List[str],
    modules: List[str],
    modules_v2: List[str],
) -> None:
    raw_dir = base_dir / "raw_curves"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []

    for idx, curve in enumerate(curves):
        sample_id = f"sim_{idx:05d}"
        csv_path = raw_dir / f"{sample_id}.csv"
        peak_freq_meas = peak_freqs[idx] if idx < len(peak_freqs) else frequency
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["freq_injected_hz", "peak_freq_meas_hz", "peak_amp_dbm"])
            for freq, peak_freq, amp in zip(frequency, peak_freq_meas, curve):
                writer.writerow([freq, peak_freq, f"{amp:.4f}"])

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "label": labels[idx] if idx < len(labels) else "",
                "module": modules[idx] if idx < len(modules) else "",
                "module_v2": modules_v2[idx] if idx < len(modules_v2) else "",
                "path": str(csv_path.relative_to(base_dir)),
            }
        )

    _write_csv(base_dir / "raw_manifest.csv", manifest_rows)


KIND_TO_MODULE = {
    "amp": "校准源",
    "freq": "时钟振荡器",
    "rl": "衰减器",
    "att": "衰减器",
    "lpf": "低频段前置低通滤波器",
    "mixer": "低频段第一混频器",
    "ytf": "高频段YTF滤波器",
    "clock": "时钟合成与同步网络",
    "lo": "本振混频组件",
    "adc": "ADC",
    "vbw": "数字检波器",
    "power": "电源模块",
}

# Attenuator behavior controls (AC does not auto-disable attenuator faults).
ATT_PRESENT_IN_PATH = True
ATT_FAULT_ENABLE = True
ATT_STATE_FIXED = False

PEAK_TRACK_NOISE_HZ = 5e3
PEAK_TRACK_OUTLIER_HZ = 5e6

MODULE_TEMPLATE_MAP = {}

# 中文消息常量 (用于国际化)
MSG_NO_SAMPLE_FOR_TYPE = "无 {track_type} 样本"


def _module_v2_from_fault(module_v1: str, fault_type: str) -> str:
    mapping = {
        "时钟振荡器": "[时钟板][参考域] 10MHz基准",
        "时钟合成与同步网络": "[时钟板][参考分配]",
        "本振混频组件": "[RF板][Mixer1]",
        "校准源": "[校准链路][校准源]",
        "校准信号开关": "[校准链路][校准路径开关/耦合]",
        "存储器": "[校准链路][校准表/存储]",
        "低频段前置低通滤波器": "[RF板][RF] 低频通路固定滤波/抑制网络",
        "低频段第一混频器": "[RF板][Mixer1]",
        "ADC": "[数字中频板][ADC]",
        "数字检波器": "[数字中频板][检波/对数]",
        "电源模块": "[RF板][RF] 输入连接/匹配/保护",
        "衰减器": "[RF板][RF] 步进衰减器/ATT",
    }
    if fault_type in ("clock_drift", "freq_miscal"):
        return "[LO/时钟板][LO1] 合成链"
    if fault_type == "lo_path_error":
        return "[RF板][LO1] LO1 注入链"
    if fault_type in ("peak_track_spike", "peak_track_dense", "peak_track_hole"):
        return "[数字中频板][峰值搜索]"
    return mapping.get(module_v1, module_v1)


def _select_template(module_label: str, rng: np.random.Generator) -> str:
    spec = module_spec_by_label(module_label)
    templates = module_templates(spec) if spec else ["T1"]
    if not templates:
        return "T1"
    return select_template(templates, rng)


def _choose_module_for_system(system_label: str, rng: np.random.Generator) -> str:
    specs = module_specs_by_system(system_label, DISABLED_MODULES)
    if not specs:
        return "校准源"
    return str(rng.choice([spec.module_label for spec in specs]))


def _choose_target_tier(fault_kind: str, rng: np.random.Generator) -> str:
    if fault_kind == "normal":
        return rng.choice(["in_spec_weak", "edge"], p=[0.7, 0.3])
    return rng.choice(["in_spec_weak", "edge", "mild_oos"], p=[0.45, 0.30, 0.25])


def _filter_kind_probs(kind_probs: Dict[str, float]) -> Dict[str, float]:
    if not DISABLED_MODULES:
        return kind_probs
    filtered = {}
    for kind, prob in kind_probs.items():
        module = KIND_TO_MODULE.get(kind)
        if kind in ("rl", "att") and module in DISABLED_MODULES and _has_enabled_ref_module():
            filtered[kind] = prob
            continue
        if module and module in DISABLED_MODULES:
            continue
        filtered[kind] = prob
    return filtered or kind_probs


def _choose_ref_module(rng: np.random.Generator) -> str:
    ref_modules = ["衰减器", "校准源", "存储器", "校准信号开关"]
    enabled = [module for module in ref_modules if module not in DISABLED_MODULES]
    if not enabled:
        return "校准源"
    return rng.choice(enabled)


def _track_type_cycle(rng: np.random.Generator) -> Iterator[str]:
    order = ["spike", "dense", "hole"]
    while True:
        rng.shuffle(order)
        for item in order:
            yield item


def _build_track_type_plan(n: int, rng: np.random.Generator) -> List[str]:
    if n <= 0:
        return []
    track_types = ["spike", "dense", "hole"]
    if n <= 3:
        return list(rng.choice(track_types, size=n, replace=True))
    min_count = min(20, max(1, n // 4))
    plan: List[str] = []
    for track_type in track_types:
        plan.extend([track_type] * min_count)
    remaining = n - len(plan)
    if remaining > 0:
        plan.extend(list(rng.choice(track_types, size=remaining, replace=True)))
    rng.shuffle(plan)
    return plan


def _has_enabled_ref_module() -> bool:
    ref_modules = ["衰减器", "校准源", "存储器", "校准信号开关"]
    return any(module not in DISABLED_MODULES for module in ref_modules)


def _load_normal_stats(repo_root: Path) -> dict:
    stats_path = repo_root / NORMAL_STATS_JSON
    arrays_path = repo_root / NORMAL_STATS_NPZ
    if not stats_path.exists() or not arrays_path.exists():
        return {}
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path)
    stats["frequency"] = arrays["frequency"]
    stats["rrs"] = arrays["rrs"]
    stats["sigma_smooth"] = arrays["sigma_smooth"]
    stats["sigma_i"] = arrays.get("sigma_i")
    stats["q_low"] = arrays.get("q_low")
    stats["q_high"] = arrays.get("q_high")
    stats["cap_normal_db"] = float(arrays.get("cap_normal_db", 0.4))
    return stats


def _build_active_fault_kinds(
    single_band: bool,
    disable_preamp: bool,
    coupling_mode: str,
) -> List[str]:
    kinds = [
        "amp",
        "lpf",
        "mixer",
        "adc",
        "vbw",
        "power",
        "freq",
        "clock",
        "lo",
        "rl",
        "att",
        "normal",
    ]
    if single_band:
        if "ytf" in kinds:
            kinds.remove("ytf")
    if disable_preamp and "preamp" in kinds:
        kinds.remove("preamp")
    if not ATT_PRESENT_IN_PATH or not ATT_FAULT_ENABLE:
        if "att" in kinds:
            kinds.remove("att")
    return kinds


def _constrain_by_feature_stats(
    curve: np.ndarray,
    rrs: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    normal_stats: dict,
) -> np.ndarray:
    # 基于真实正常特征分布进行轻度约束，避免特征极端离群
    feature_stats = normal_stats.get("feature_stats", {})
    if not feature_stats:
        return curve
    sys_feats = extract_system_features(curve, baseline_curve=rrs, envelope=bounds)
    delta = curve - rrs
    scale = 1.0
    for key, stats in feature_stats.items():
        if key not in sys_feats:
            continue
        val = sys_feats.get(key, 0.0)
        p95 = stats.get("p95", None)
        p05 = stats.get("p05", None)
        std = stats.get("std", 0.0)
        if p95 is not None and val > p95 + 2 * std:
            scale = min(scale, 0.85)
        if p05 is not None and val < p05 - 2 * std:
            scale = min(scale, 0.85)
    if scale < 1.0:
        curve = rrs + delta * scale
    return curve


def _smooth_series(values: np.ndarray, window: int = 61) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _compute_real_normal_stats(
    frequency: np.ndarray,
    traces: np.ndarray,
    rrs: np.ndarray,
) -> Dict[str, float]:
    residuals = traces - rrs
    global_offsets = np.median(residuals, axis=1)
    hf_std = []
    p95_abs = []
    for res in residuals:
        smooth = _smooth_series(res, window=61)
        hf_noise = res - smooth
        hf_std.append(float(np.std(hf_noise)))
        p95_abs.append(float(np.quantile(np.abs(res), 0.95)))
    hf_std = np.array(hf_std, dtype=float)
    p95_abs = np.array(p95_abs, dtype=float)
    return {
        "global_offset_abs_p10": float(np.quantile(np.abs(global_offsets), 0.10)),
        "global_offset_abs_p50": float(np.quantile(np.abs(global_offsets), 0.50)),
        "global_offset_abs_p90": float(np.quantile(np.abs(global_offsets), 0.90)),
        "global_offset_abs_p95": float(np.quantile(np.abs(global_offsets), 0.95)),
        "hf_std_p10": float(np.quantile(hf_std, 0.10)),
        "hf_std_p50": float(np.quantile(hf_std, 0.50)),
        "hf_std_p90": float(np.quantile(hf_std, 0.90)),
        "hf_std_p95": float(np.quantile(hf_std, 0.95)),
        "p95_abs_p10": float(np.quantile(p95_abs, 0.10)),
        "p95_abs_p50": float(np.quantile(p95_abs, 0.50)),
        "p95_abs_p90": float(np.quantile(p95_abs, 0.90)),
        "p95_abs_p95": float(np.quantile(p95_abs, 0.95)),
    }


def _evaluate_tier(dev_rrs: np.ndarray) -> Tuple[str, Dict[str, float]]:
    abs_dev = np.abs(dev_rrs)
    max_abs = float(np.max(abs_dev))
    edge_frac = float(np.mean((abs_dev >= 0.35) & (abs_dev <= 0.40)))
    oos_frac = float(np.mean(abs_dev > 0.40))
    if max_abs <= 0.40 and 0.03 <= edge_frac <= 0.20:
        tier = "edge"
        cap_db = 0.40
    elif max_abs <= 0.40:
        tier = "in_spec_weak"
        cap_db = 0.40
    else:
        tier = "mild_oos"
        cap_db = 0.60
    return tier, {"max_abs": max_abs, "edge_frac": edge_frac, "oos_frac": oos_frac, "cap_db": cap_db}


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def _build_peak_track_profile(
    frequency: np.ndarray,
    rng: np.random.Generator,
    track_type: str,
    severity: str,
) -> Dict[str, object]:
    """Build peak frequency tracking error profile.
    
    任务3改进: freq_error 不仅影响幅度，还影响:
    - 峰值数量 (peak_count_effect)
    - 峰值位置 (position_jitter)
    - 峰值宽度 (width_modulation)
    
    这使得 freq_error 类型的故障更容易通过峰值特征识别。
    """
    base = frequency.astype(float).copy()
    step_hz = float(base[1] - base[0]) if len(base) > 1 else 1e7
    offsets = rng.normal(0.0, PEAK_TRACK_NOISE_HZ, size=len(base))
    mask = np.zeros(len(base), dtype=float)
    bands = []
    
    # 任务3新增: 峰值特征影响参数
    peak_effects = {
        "count_multiplier": 1.0,   # 峰值数量倍增器
        "position_jitter_hz": 0.0,  # 位置抖动 (Hz)
        "width_factor": 1.0,        # 宽度调制因子
    }

    if track_type == "none":
        return {"offsets": offsets, "mask": mask, "bands": bands, "track_type": track_type, "peak_effects": peak_effects}

    if track_type == "spike":
        # W4: Random spike count with some variability
        # Minimum spike count matches light severity (4) to ensure detectability
        base_count = {"light": 4, "mid": 8, "severe": 12}[severity]
        spike_count = max(base_count - 1, base_count + rng.integers(-2, 3))
        
        # 任务3: spike 类型峰值效应
        count_multiplier = rng.uniform(0.7, 1.3)  # 峰值数量变化
        position_jitter = rng.uniform(1e6, 5e6) * ({"light": 0.5, "mid": 1.0, "severe": 1.5}[severity])
        
        # 应用数量倍增器
        spike_count = max(2, int(spike_count * count_multiplier))
        
        peak_effects["count_multiplier"] = count_multiplier
        peak_effects["position_jitter_hz"] = position_jitter
        
        # W4: Spread spikes more uniformly across frequency range
        # Divide frequency range into regions and sample from each
        n_regions = min(spike_count, 6)
        region_size = len(base) // n_regions
        spike_indices = []
        for i in range(n_regions):
            start_idx = i * region_size
            end_idx = min((i + 1) * region_size, len(base))
            if end_idx > start_idx:
                spike_indices.append(rng.integers(start_idx, end_idx))
        
        # Add remaining spikes randomly
        remaining = spike_count - len(spike_indices)
        if remaining > 0:
            available = set(range(len(base))) - set(spike_indices)
            if available:
                spike_indices.extend(rng.choice(list(available), size=min(remaining, len(available)), replace=False))
        
        idx = np.array(spike_indices[:spike_count])
        # W4: Wider amplitude range for spikes, with position jitter applied
        spike_offsets = rng.uniform(3e6, 30e6, size=len(idx)) * rng.choice([-1, 1], size=len(idx))
        spike_offsets += rng.normal(0, position_jitter, size=len(idx))  # Apply position jitter
        offsets[idx] += spike_offsets
        mask[idx] = 1.0
        return {"offsets": offsets, "mask": mask, "bands": bands, "track_type": track_type, "peak_effects": peak_effects}

    if track_type in ("dense", "hole"):
        # W4: Randomized band count
        base_count = {"light": 1, "mid": 2, "severe": 3}[severity]
        band_count = max(1, base_count + rng.integers(-1, 2))
        
        # 任务3: dense/hole 类型峰值效应
        width_factor = rng.uniform(0.8, 1.5) if track_type == "dense" else rng.uniform(0.6, 1.2)
        position_jitter = rng.uniform(0.5e6, 3e6) * ({"light": 0.5, "mid": 1.0, "severe": 1.5}[severity])
        
        peak_effects["width_factor"] = width_factor
        peak_effects["position_jitter_hz"] = position_jitter
        
        # W4: Wider ranges for amplitude and width (apply width_factor)
        amp_ranges = {"dense": (5e6, 50e6), "hole": (15e6, 70e6)}[track_type]
        base_width_ranges = {"light": (0.15e9, 0.6e9), "mid": (0.25e9, 0.9e9), "severe": (0.35e9, 1.2e9)}[severity]
        # Apply width_factor to width ranges
        width_ranges = (base_width_ranges[0] * width_factor, base_width_ranges[1] * width_factor)
        ramp_hz = rng.uniform(0.03e9, 0.12e9)
        jitter_hz = rng.uniform(0.3e6, 2.5e6) + position_jitter * 0.1  # Include position jitter
        
        # W4: Track used frequency regions to avoid overlap
        used_regions = []
        freq_min = float(base[0])
        freq_max = float(base[-1])
        
        for _ in range(band_count):
            # W4: Sample center from full allowed range (not fixed positions)
            margin_hz = 0.3e9  # Keep away from edges
            center_min = freq_min + margin_hz
            center_max = freq_max - margin_hz
            
            # Try to find non-overlapping position
            max_placement_attempts = 10
            for _ in range(max_placement_attempts):
                center_hz = float(rng.uniform(center_min, center_max))
                width_hz = float(rng.uniform(*width_ranges))
                
                start_hz = max(freq_min, center_hz - width_hz / 2)
                end_hz = min(freq_max, center_hz + width_hz / 2)
                
                # Check for overlap with existing bands
                overlap = False
                for (used_start, used_end) in used_regions:
                    if start_hz < used_end and end_hz > used_start:
                        overlap = True
                        break
                
                if not overlap:
                    used_regions.append((start_hz, end_hz))
                    break
            else:
                # If we can't find non-overlapping position, skip this band
                continue
            
            start_idx = int(np.searchsorted(base, start_hz, side="left"))
            end_idx = int(np.searchsorted(base, end_hz, side="right"))
            if end_idx <= start_idx + 4:
                continue
            ramp_pts = max(3, int(ramp_hz / max(step_hz, 1.0)))
            length = end_idx - start_idx
            # Ensure envelope length matches exactly
            if length <= 2 * ramp_pts:
                # Not enough room for ramps, use simple trapezoid
                envelope = np.linspace(0, 1, length // 2, endpoint=False)
                envelope = np.concatenate([envelope, np.linspace(1, 0, length - len(envelope))])
            else:
                left = np.linspace(0, 1, ramp_pts, endpoint=False)
                right = np.linspace(1, 0, ramp_pts, endpoint=False)
                middle_len = length - len(left) - len(right)
                middle = np.ones(middle_len)
                envelope = np.concatenate([left, middle, right])
            envelope = _smoothstep(envelope[:length])  # Ensure exact length
            
            # W4: Random amplitude with sign
            amplitude = float(rng.uniform(*amp_ranges)) * (1 if track_type == "dense" else rng.choice([-1, 1]))
            offsets[start_idx:end_idx] += amplitude * envelope
            offsets[start_idx:end_idx] += rng.normal(0.0, jitter_hz, size=length) * envelope
            mask[start_idx:end_idx] = np.maximum(mask[start_idx:end_idx], envelope)
            bands.append({
                "start_hz": start_hz, 
                "end_hz": end_hz, 
                "center_hz": center_hz,
                "width_hz": width_hz,
                "amplitude_hz": amplitude, 
                "ramp_hz": ramp_hz
            })
    return {"offsets": offsets, "mask": mask, "bands": bands, "track_type": track_type, "peak_effects": peak_effects}


def _generate_peak_freq_meas(
    frequency: np.ndarray,
    rng: np.random.Generator,
    track_type: str,
    severity: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    profile = _build_peak_track_profile(frequency, rng, track_type, severity)
    base = frequency.astype(float).copy()
    return base + profile["offsets"], profile


def _peak_freq_metrics(frequency: np.ndarray, peak_freq_meas: np.ndarray) -> Tuple[float, float]:
    diff = peak_freq_meas - frequency
    mae = float(np.mean(np.abs(diff)))
    outlier_frac = float(np.mean(np.abs(diff) > PEAK_TRACK_OUTLIER_HZ))
    return mae, outlier_frac


def _write_reject_stats(out_dir: Path, constraints: SimulationConstraints) -> None:
    stats = {"reject_counts": constraints.reject_counts, "reject_records": constraints.reject_records}
    (out_dir / "reject_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _setup_chinese_font() -> None:
    """设置中文字体支持，解决matplotlib中文乱码问题。
    
    任务4要求: 所有matplotlib输出必须正确显示中文。
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        
        # 候选字体列表 (按优先级)
        font_candidates = [
            "SimHei",           # 黑体
            "Microsoft YaHei",  # 微软雅黑
            "Noto Sans CJK SC", # Noto思源黑体
            "PingFang SC",      # 苹方字体
            "WenQuanYi Zen Hei",# 文泉驿正黑
            "Arial Unicode MS", # Arial Unicode
            "DejaVu Sans",      # 备选西文字体
        ]
        
        # 查找可用字体
        available_fonts = [entry.name for entry in fm.fontManager.ttflist]
        selected_font = None
        for font in font_candidates:
            if any(font in name for name in available_fonts):
                selected_font = font
                break
        
        # 设置字体
        if selected_font:
            matplotlib.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
        else:
            matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        
        matplotlib.rcParams["axes.unicode_minus"] = False
        matplotlib.rcParams["figure.dpi"] = 150
        
    except Exception:
        pass


def _plot_overlay_audit(
    out_dir: Path,
    frequency: np.ndarray,
    rrs: np.ndarray,
    curves: List[np.ndarray],
    labels: dict,
    traces: np.ndarray | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    # 任务4: 设置中文字体
    _setup_chinese_font()

    def quantile_band(data: np.ndarray, q_low: float = 0.1, q_high: float = 0.9):
        if data is None or data.size == 0:
            return None
        return (
            np.median(data, axis=0),
            np.quantile(data, q_low, axis=0),
            np.quantile(data, q_high, axis=0),
        )

    class_groups = {"normal": [], "amp_error": [], "freq_error": [], "ref_error": []}
    for idx in range(len(curves)):
        sample_id = f"sim_{idx:05d}"
        cls = labels.get(sample_id, {}).get("system_fault_class", "normal")
        if cls in class_groups:
            class_groups[cls].append(idx)
    x_ghz = frequency / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    real_band = quantile_band(traces, 0.1, 0.9) if traces is not None else None
    for ax, cls in zip(axes, ["normal", "amp_error", "freq_error", "ref_error"]):
        indices = class_groups.get(cls, [])
        cls_curves = np.array([curves[i] for i in indices]) if indices else np.empty((0, len(rrs)))
        sim_band = quantile_band(cls_curves, 0.1, 0.9)
        ax.plot(x_ghz, rrs, color="black", linewidth=1.5, label="RRS")
        if sim_band:
            med, low, high = sim_band
            ax.fill_between(x_ghz, low, high, color="tab:orange", alpha=0.3, label="Sim P10-P90")
            ax.plot(x_ghz, med, color="tab:orange", linewidth=1.1, label="Sim median")
        if cls == "normal" and real_band:
            med_r, low_r, high_r = real_band
            ax.fill_between(x_ghz, low_r, high_r, color="tab:blue", alpha=0.25, label="Real P10-P90")
            ax.plot(x_ghz, med_r, color="tab:blue", linewidth=1.1, label="Real median")
        ax.set_title(f"{cls} quantile band")
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Frequency (GHz)")
    axes[-1].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Amplitude (dBm)")
    axes[2].set_ylabel("Amplitude (dBm)")
    handles, labels_text = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_text, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_dir / "audit_overlay.png", dpi=150)
    plt.close(fig)


def _plot_normal_vs_real(
    out_dir: Path,
    frequency: np.ndarray,
    rrs: np.ndarray,
    traces: np.ndarray | None,
    curves: List[np.ndarray],
    labels: dict,
) -> None:
    if traces is None:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    # 任务4: 设置中文字体
    _setup_chinese_font()
    
    plt.figure(figsize=(12, 6))
    for idx in range(min(5, traces.shape[0])):
        plt.plot(frequency / 1e9, traces[idx], color="tab:gray", alpha=0.4, linewidth=0.6)
    for idx, curve in enumerate(curves[:50]):
        sample_id = f"sim_{idx:05d}"
        if labels.get(sample_id, {}).get("system_fault_class") == "normal":
            plt.plot(frequency / 1e9, curve, color="tab:blue", alpha=0.5, linewidth=0.7)
    plt.plot(frequency / 1e9, rrs, color="black", linewidth=1.5, label="RRS")
    plt.xlabel("频率 (GHz)")
    plt.ylabel("幅度 (dBm)")
    plt.title("正常样本 vs 真实数据 (统计一致性)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sample_overlay_normal_vs_real.png", dpi=150)
    plt.close()


def _plot_overlay_by_module(
    out_dir: Path,
    frequency: np.ndarray,
    curves: List[np.ndarray],
    labels: dict,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    # 任务4: 设置中文字体
    _setup_chinese_font()
    
    module_samples: Dict[str, List[int]] = {}
    for idx in range(len(curves)):
        sample_id = f"sim_{idx:05d}"
        module = labels.get(sample_id, {}).get("module")
        if not module:
            continue
        module_samples.setdefault(module, []).append(idx)
    modules = list(module_samples.keys())[:6]
    fig, axes = plt.subplots(len(modules), 1, figsize=(12, 2.2 * max(1, len(modules))), sharex=True)
    if len(modules) == 1:
        axes = [axes]
    for ax, module in zip(axes, modules):
        for idx in module_samples[module][:4]:
            ax.plot(frequency / 1e9, curves[idx], linewidth=0.7, alpha=0.7)
        ax.set_title(f"模块叠加: {module}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("频率 (GHz)")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_overlay_by_module.png", dpi=150)
    plt.close(fig)


def _plot_template_gallery(
    out_dir: Path,
    frequency: np.ndarray,
    curves: List[np.ndarray],
    labels: dict,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    # 任务4: 设置中文字体
    _setup_chinese_font()
    
    template_samples: Dict[str, List[int]] = {}
    for idx in range(len(curves)):
        sample_id = f"sim_{idx:05d}"
        template_id = labels.get(sample_id, {}).get("fault_template_id")
        if not template_id:
            continue
        template_samples.setdefault(template_id, []).append(idx)
    templates = sorted(template_samples.keys())[:6]
    fig, axes = plt.subplots(len(templates), 1, figsize=(12, 2.2 * max(1, len(templates))), sharex=True)
    if len(templates) == 1:
        axes = [axes]
    for ax, template_id in zip(axes, templates):
        for idx in template_samples[template_id][:3]:
            ax.plot(frequency / 1e9, curves[idx], linewidth=0.7, alpha=0.7)
        ax.set_title(f"模板: {template_id}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("频率 (GHz)")
    fig.tight_layout()
    fig.savefig(out_dir / "template_gallery.png", dpi=150)
    plt.close(fig)


def _plot_peak_track_audit(
    out_dir: Path,
    frequency: np.ndarray,
    peak_freqs: List[np.ndarray],
    labels: dict,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    # 任务4: 设置中文字体
    _setup_chinese_font()

    track_types = ["spike", "dense", "hole"]
    plt.figure(figsize=(12, 7))
    for idx, track_type in enumerate(track_types, start=1):
        ax = plt.subplot(len(track_types), 1, idx)
        chosen = None
        for i, _ in enumerate(peak_freqs):
            sample_id = f"sim_{i:05d}"
            if labels.get(sample_id, {}).get("peak_track_type") == track_type:
                chosen = i
                break
        if chosen is None:
            ax.text(0.5, 0.5, MSG_NO_SAMPLE_FOR_TYPE.format(track_type=track_type), ha="center", va="center")
            ax.set_axis_off()
            continue
        diff = peak_freqs[chosen] - frequency
        ax.plot(frequency / 1e9, diff / 1e6, color="tab:blue", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"峰值追踪类型: {track_type}")
        ax.set_ylabel("Δ频率 (MHz)")
        ax.grid(True, alpha=0.3)
    plt.xlabel("频率 (GHz)")
    plt.tight_layout()
    plt.savefig(out_dir / "audit_peak_track.png", dpi=150)
    plt.close()


def _plot_peakfreq_behavior(
    out_dir: Path,
    frequency: np.ndarray,
    peak_freqs: List[np.ndarray],
    labels: dict,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    # 任务4: 设置中文字体
    _setup_chinese_font()
    
    track_types = ["spike", "dense", "hole"]
    fig, axes = plt.subplots(len(track_types), 1, figsize=(12, 7), sharex=True)
    for ax, track_type in zip(axes, track_types):
        chosen = None
        for i, _ in enumerate(peak_freqs):
            sample_id = f"sim_{i:05d}"
            if labels.get(sample_id, {}).get("peak_track_type") == track_type:
                chosen = i
                break
        if chosen is None:
            ax.text(0.5, 0.5, MSG_NO_SAMPLE_FOR_TYPE.format(track_type=track_type), ha="center", va="center")
            ax.set_axis_off()
            continue
        diff = (peak_freqs[chosen] - frequency) / 1e6
        ax.plot(frequency / 1e9, diff, color="tab:purple", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"freq_error 峰值追踪: {track_type}")
        ax.set_ylabel("Δ频率 (MHz)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("频率 (GHz)")
    fig.tight_layout()
    fig.savefig(out_dir / "peakfreq_behavior_freq_error.png", dpi=150)
    plt.close(fig)


def _plot_amp_vs_ref_separability(
    out_dir: Path,
    feature_rows: List[Dict[str, object]],
    labels: dict,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    # 任务4: 设置中文字体
    _setup_chinese_font()
    
    amp_points = []
    ref_points = []
    for row in feature_rows:
        sample_id = row.get("sample_id")
        cls = labels.get(sample_id, {}).get("system_fault_class")
        x1 = float(row.get("X1", 0.0))
        x3 = float(row.get("X3", 0.0))
        if cls == "amp_error":
            amp_points.append((x1, x3))
        elif cls == "ref_error":
            ref_points.append((x1, x3))
    if not amp_points and not ref_points:
        return
    plt.figure(figsize=(6, 5))
    if amp_points:
        xs, ys = zip(*amp_points)
        plt.scatter(xs, ys, s=12, alpha=0.6, label="幅度失准")
    if ref_points:
        xs, ys = zip(*ref_points)
        plt.scatter(xs, ys, s=12, alpha=0.6, label="参考电平失准")
    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.legend()
    plt.title("幅度失准 vs 参考电平失准 可分性 (X1 vs X3)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "amp_vs_ref_separability.png", dpi=150)
    plt.close()


def _plot_grid_with_manifest(
    out_dir: Path,
    frequency: np.ndarray,
    curves: List[np.ndarray],
    labels: dict,
    max_per_batch: int = 100,
    ncols: int = 10,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    
    # 任务4: 设置中文字体
    _setup_chinese_font()
    
    sample_ids = [f"sim_{idx:05d}" for idx in range(len(curves))]
    manifest_rows: List[Dict[str, object]] = []
    batch_index = 0
    for start in range(0, len(curves), max_per_batch):
        batch_ids = sample_ids[start : start + max_per_batch]
        if not batch_ids:
            continue
        nrows = int(np.ceil(len(batch_ids) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 1.8 * nrows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(nrows, ncols)
        for idx, sample_id in enumerate(batch_ids):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            curve_idx = start + idx
            ax.plot(frequency / 1e9, curves[curve_idx], linewidth=0.6, alpha=0.8)
            info = labels.get(sample_id, {})
            title = (
                f"{sample_id}\n"
                f"{info.get('system_fault_class')} | "
                f"{info.get('template_id')} | "
                f"{info.get('module_cause')}"
            )
            ax.set_title(title, fontsize=7)
            ax.grid(True, alpha=0.2)
            manifest_rows.append(
                {
                    "batch_index": batch_index,
                    "row": row,
                    "col": col,
                    "sample_id": sample_id,
                    "system_fault_class": info.get("system_fault_class"),
                    "template_id": info.get("template_id"),
                    "module_cause": info.get("module_cause"),
                    "severity": info.get("severity"),
                    "amp_error_subtype": info.get("amp_error_subtype"),
                    "peak_track_type": info.get("peak_track_type"),
                }
            )
        for idx in range(len(batch_ids), nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis("off")
        fig.suptitle(f"Frequency Response Grid - Batch {batch_index:02d}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"all_in_grid_batch_{batch_index:02d}.png", dpi=150)
        plt.close(fig)
        batch_index += 1
    if manifest_rows:
        _write_csv(out_dir / "grid_manifest.csv", manifest_rows, encoding="utf-8-sig")


def _write_ref_error_bucket_report(
    out_dir: Path,
    feature_rows: List[Dict[str, object]],
    labels: dict,
) -> None:
    rows = []
    for row in feature_rows:
        sample_id = row.get("sample_id")
        label = labels.get(sample_id, {})
        if label.get("system_fault_class") != "ref_error":
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "template_id": label.get("template_id"),
                "tier": label.get("tier"),
                "severity": label.get("severity"),
                "x1": float(row.get("X1", 0.0)),
                "x3": float(row.get("X3", 0.0)),
            }
        )
    if not rows:
        return
    buckets: Dict[Tuple[str, str, str], List[Dict[str, float]]] = {}
    for row in rows:
        key = (str(row["template_id"]), str(row["tier"]), str(row["severity"]))
        buckets.setdefault(key, []).append(row)

    lines = ["# ref_error bucket attribution report", "", "Bucketed by (template_id, tier, severity).", ""]
    lines.append("| template_id | tier | severity | count | X1 mean | X1 std | X3 mean | X3 std |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    summary = []
    for key, entries in buckets.items():
        x1_vals = np.array([item["x1"] for item in entries])
        x3_vals = np.array([item["x3"] for item in entries])
        summary.append(
            (
                key,
                len(entries),
                float(np.mean(x1_vals)),
                float(np.std(x1_vals)),
                float(np.mean(x3_vals)),
                float(np.std(x3_vals)),
            )
        )
    summary.sort(key=lambda item: item[1], reverse=True)
    for (template_id, tier, severity), count, x1_mean, x1_std, x3_mean, x3_std in summary:
        lines.append(
            f"| {template_id} | {tier} | {severity} | {count} | "
            f"{x1_mean:.4f} | {x1_std:.4f} | {x3_mean:.4f} | {x3_std:.4f} |"
        )
    out_dir.joinpath("ref_error_bucket_report.md").write_text("\n".join(lines), encoding="utf-8")


def simulate_curve(
    frequency: np.ndarray,
    rrs: np.ndarray,
    band_ranges: List[Tuple[float, float]],
    traces: np.ndarray | None,
    normal_stats: dict,
    bounds: Tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    constraints: SimulationConstraints,
    active_kinds: List[str],
    target_class: str | None = None,
    forced_module_label: str | None = None,
    forced_peak_track_type: str | None = None,
    peak_track_cycle: Iterator[str] | None = None,
    max_attempts: int = 200,
) -> Tuple[np.ndarray, str, str, dict, np.ndarray, dict]:
    """Generate simulated curve with optional target fault class.
    
    Args:
        frequency: Frequency array
        rrs: Reference response spectrum
        band_ranges: Frequency band ranges
        traces: Optional baseline traces
        rng: Random number generator
        target_class: If specified, force generation of this class
                     Options: 'amp_error', 'freq_error', 'ref_error', 'normal'
    
    Returns:
        (curve, label_sys, label_mod)
    """
    # 核心原则：
    # - 所有仿真必须围绕 RRS 生成（仅 amp/ref 允许整体偏移且受控）
    # - 正常样本必须完全受真实统计约束（|x-RRS| <= 0.4 dB）
    # - power_noise / lpf_shift 等形态类不能出现整体漂移

    # More realistic probability distribution based on actual fault complexity
    # Amplitude faults have more module types (7) so naturally more common
    # Frequency faults have fewer modules (3) so less common
    # Reference level faults have specific modules (2)
    # NOTE: Preamp is DISABLED in single-band mode (10MHz-8.2GHz)
    if target_class is None:
        # Realistic distribution reflecting module diversity (NO PREAMP)
        kind_probs = {
            # Amplitude-related (多种模块, 概率较高) - NO PREAMP
            "amp": 0.12,      # Calibration source
            "lpf": 0.09,      # Low-pass filter
            "mixer": 0.09,    # Mixer
            "adc": 0.09,      # ADC
            "vbw": 0.08,      # Digital detector
            "power": 0.08,    # Power supply
            # Frequency-related (少量模块, 概率较低)
            "freq": 0.08,     # Frequency calibration
            "clock": 0.06,    # Clock synthesis
            "lo": 0.06,       # Local oscillator
            # Reference level (特定模块)
            "rl": 0.08,       # Reference level
            "att": 0.06,      # Attenuator
            # Normal
            "normal": 0.10,   # Normal state (increased slightly)
        }
    elif target_class == "amp_error":
        # Select from amplitude fault modules with realistic weights (NO PREAMP)
        kind_probs = {
            "amp": 0.25, "lpf": 0.18, "mixer": 0.18,
            "adc": 0.15, "vbw": 0.12, "power": 0.12
        }
    elif target_class == "freq_error":
        # Select from frequency fault modules
        kind_probs = {"freq": 0.40, "clock": 0.30, "lo": 0.30}
    elif target_class == "ref_error":
        # Select from reference level modules
        kind_probs = {"rl": 0.60, "att": 0.40}
    elif target_class == "normal":
        kind_probs = {"normal": 1.0}
    else:
        # Fallback to distribution without preamp
        kind_probs = {
            "amp": 0.15, "freq": 0.12, "rl": 0.12, "att": 0.08,
            "lpf": 0.09, "mixer": 0.09,
            "clock": 0.06, "lo": 0.06, "adc": 0.06, "vbw": 0.06,
            "power": 0.06, "normal": 0.10,
        }
    
    kind_probs = {k: v for k, v in kind_probs.items() if k in active_kinds}
    kind_probs = _filter_kind_probs(kind_probs)
    kinds = list(kind_probs.keys())
    probs = np.array(list(kind_probs.values()), dtype=float)
    probs = probs / probs.sum()
    fault_kind = rng.choice(kinds, p=probs)
    if forced_module_label:
        forced_spec = module_spec_by_label(forced_module_label)
        if forced_spec:
            target_class = forced_spec.system_label
            if target_class == "amp_error":
                fault_kind = "amp"
            elif target_class == "freq_error":
                fault_kind = "freq"
            elif target_class == "ref_error":
                fault_kind = "rl"
    
    severity = rng.choice(["light", "mid", "severe"], p=[0.55, 0.35, 0.10])
    target_tier = _choose_target_tier(fault_kind, rng) if fault_kind == "normal" else None
    last_reasons: List[str] = []
    for _ in range(max_attempts):
        label_sys = "normal"
        label_mod = "none"
        fault_params = {}  # Track injection parameters
        peak_track_type = "none"
        peak_track_profile: Dict[str, object] = {"mask": np.zeros(len(frequency)), "offsets": np.zeros(len(frequency))}
        # V-D.3: 移除 template_id = None (旧模板系统已禁用)
        fault_params["severity"] = severity
        if fault_kind != "normal":
            target_tier = _choose_target_tier(fault_kind, rng)
        fault_params["tier_target"] = target_tier

        if fault_kind == "normal":
            curve, reasons, normal_state = constraints.generate_normal(rng)
            if reasons:
                last_reasons = reasons
                constraints._record_reject("normal", fault_kind, reasons)
                continue
            fault_params["type"] = "normal"
            fault_params["normal_state"] = normal_state
            fault_params["tier_target"] = target_tier
            peak_freq_meas, _ = _generate_peak_freq_meas(frequency, rng, "none", severity)
            return curve, label_sys, label_mod, fault_params, peak_freq_meas, {"peak_track_type": "none"}

        # 将 fault_kind 映射到模块类型，用于生成模块特定的噪声
        fault_kind_to_module = {
            "amp": "校准源",
            "freq": "时钟振荡器",
            "rl": "衰减器",
            "att": "衰减器",
            "lpf": "低频段前置低通滤波器",
            "mixer": "低频段第一混频器",
            "adc": "ADC",
            "vbw": "数字检波器",
            "power": "电源模块",
            "clock": "时钟合成与同步网络",
            "lo": "本振混频组件",
        }
        module_type_for_noise = fault_kind_to_module.get(fault_kind, "校准源")
        
        # V-D.2: 将 fault_kind 映射到 CurveGenerator 的模块键 (标准键名)
        # 物理正确性说明：
        # - lpf (低通滤波器): 使用 lpf_low_band (band_insertion_loss 模拟截止频率漂移)
        # - ac_coupling (AC耦合电容): 使用高通滤波效应 (低频塌陷)
        FAULT_KIND_TO_MODULE_KEY = {
            "amp": "step_attenuator",
            "freq": "ocxo_ref",
            "rl": "cal_source",
            "att": "step_attenuator",
            "lpf": "lpf_low_band",  # ✅ 修正：低通滤波器使用 band_insertion_loss
            "mixer": "mixer1",
            "adc": "adc_module",
            "vbw": "dsp_detector",
            "power": "power_management",
            "clock": "ref_distribution",
            "lo": "lo1_synth",
            "ytf": "lpf_high_band",
        }
        
        # V-D.2: 频率 warp 参数常量
        FREQ_WARP_BIAS_MIN = -0.001
        FREQ_WARP_BIAS_MAX = 0.001
        
        # 使用模块特定的噪声模型生成故障基础
        curve, reasons = constraints.generate_fault_base(
            rng, 
            module_type=module_type_for_noise,
            severity=severity,
        )
        if reasons:
            last_reasons = reasons
            constraints._record_reject("fault_base", fault_kind, reasons)
            continue
        base_texture = curve - rrs
        if fault_kind == "amp":
            texture_scale = rng.uniform(0.4, 0.9)
        else:
            texture_scale = rng.uniform(0.2, 0.5)
        curve = rrs + base_texture * texture_scale
        
        # V-D.2: 获取物理核曲线生成器（复用同一实例以提高性能）
        module_key = FAULT_KIND_TO_MODULE_KEY.get(fault_kind, "step_attenuator")
        curve_generator = CurveGenerator(seed=int(rng.integers(0, 2**31)))
        severity_float = {"light": 0.3, "mid": 0.5, "severe": 0.8}.get(severity, 0.5)
        
        # 预创建 rrs 副本以避免多次复制
        rrs_copy = rrs.copy()

        if fault_kind == "amp":
            label_sys = "幅度失准"
            label_mod = forced_module_label or _choose_module_for_system("amp_error", rng)
            subtype = rng.choice(
                ["amp_error_offset", "amp_error_band", "amp_error_ripple"],
                p=[0.4, 0.3, 0.3],
            )
            fault_params["subtype"] = subtype
            # V-D.2/V-D.5: 使用物理核替换简单数学（始终基于 rrs_copy）
            curve = curve_generator.apply_degradation(rrs_copy, module_key, severity_float)
            fault_params["module_key"] = module_key
            # V-D.3: 禁用模板系统
        elif fault_kind == "freq":
            # V-D.2/V-D.5: 使用 peak_jitter 替代 inject_freq_miscal（基于 rrs_copy）
            curve = curve_generator.apply_degradation(rrs_copy, "ocxo_ref", severity_float)
            warp_scale = 1.0 + rng.uniform(0.00008, 0.0005) * (1 if rng.random() < 0.5 else -1)
            x_axis = np.linspace(0.0, 1.0, len(curve))
            warp_bias = rng.uniform(FREQ_WARP_BIAS_MIN, FREQ_WARP_BIAS_MAX)
            x_warp = np.clip(x_axis * warp_scale + warp_bias, 0.0, 1.0)
            curve = np.interp(x_axis, x_warp, curve)
            label_sys = "频率失准"
            label_mod = forced_module_label or _choose_module_for_system("freq_error", rng)
            fault_params["warp_scale"] = float(warp_scale)
            fault_params["warp_bias"] = float(warp_bias)
            fault_params["type"] = "freq_miscal"
            fault_params["module_key"] = "ocxo_ref"
            # V-D.3: 禁用模板系统
        elif fault_kind in ("rl", "att"):
            label_sys = "参考电平失准"
            label_mod = forced_module_label or _choose_ref_module(rng)
            # V-D.2/V-D.5: 使用物理核（基于 rrs_copy）
            curve = curve_generator.apply_degradation(rrs_copy, "cal_source", severity_float)
            fault_params.update({"type": "ref_miscal"})
            fault_params["subtype"] = "ref_error_offset"
            fault_params["module_key"] = "cal_source"
            # V-D.3: 禁用模板系统
        # NOTE: preamp case is REMOVED - it's disabled in single-band mode
        elif fault_kind == "lpf":
            label_sys = "幅度失准"
            label_mod = forced_module_label or "低频段前置低通滤波器"
            fault_params["type"] = "lpf_shift"
            fault_params["subtype"] = "amp_error_band"
            # V-D.3: 禁用模板系统
            # V-D.2: 使用 band_insertion_loss 模拟 LPF 截止频率漂移 (正确物理行为)
            curve = curve_generator.apply_degradation(rrs_copy, "lpf_low_band", severity_float)
            fault_params["module_key"] = "lpf_low_band"
        elif fault_kind == "mixer":
            label_sys = "幅度失准"
            label_mod = forced_module_label or "低频段第一混频器"
            fault_params["type"] = "mixer_ripple"
            fault_params["subtype"] = "amp_error_ripple"
            # V-D.3: 禁用模板系统
            # V-D.2: 使用 linear_slope 替代 inject_mixer1_slope
            curve = curve_generator.apply_degradation(rrs_copy, "mixer1", severity_float)
            fault_params["module_key"] = "mixer1"
        elif fault_kind == "ytf":
            label_sys = "幅度失准"
            label_mod = forced_module_label or "高频段YTF滤波器"
            fault_params["type"] = "ytf_variation"
            # V-D.3: 禁用模板系统
            # V-D.2: 使用物理核
            curve = curve_generator.apply_degradation(rrs_copy, "lpf_high_band", severity_float)
            fault_params["module_key"] = "lpf_high_band"
        elif fault_kind == "clock":
            # V-D.2/V-D.5: 使用 ref_distribution 替代 inject_freq_miscal（基于 rrs_copy）
            curve = curve_generator.apply_degradation(rrs_copy, "ref_distribution", severity_float)
            label_sys = "频率失准"
            label_mod = forced_module_label or "时钟合成与同步网络"
            fault_params["type"] = "clock_drift"
            fault_params["module_key"] = "ref_distribution"
            # V-D.3: 禁用模板系统
        elif fault_kind == "lo":
            # V-D.2/V-D.5: 使用 signal_drop 替代 inject_lo_path_error（基于 rrs_copy）
            curve = curve_generator.apply_degradation(rrs_copy, "lo1_synth", severity_float)
            label_sys = "频率失准"
            label_mod = forced_module_label or "本振混频组件"
            fault_params["type"] = "lo_path_error"
            fault_params["module_key"] = "lo1_synth"
            # V-D.3: 禁用模板系统
        elif fault_kind == "adc":
            label_sys = "幅度失准"
            label_mod = forced_module_label or "ADC"
            fault_params["type"] = "adc_bias"
            fault_params["subtype"] = "amp_error_offset"
            # V-D.3: 禁用模板系统
            # V-D.2: 使用 quantization_noise 替代 inject_adc_sawtooth
            curve = curve_generator.apply_degradation(rrs_copy, "adc_module", severity_float)
            fault_params["module_key"] = "adc_module"
        elif fault_kind == "vbw":
            label_sys = "幅度失准"
            label_mod = forced_module_label or "数字检波器"
            fault_params["type"] = "vbw_smoothing"
            fault_params["subtype"] = "amp_error_offset"
            # V-D.3: 禁用模板系统
            # V-D.2: 使用物理核
            curve = curve_generator.apply_degradation(rrs_copy, "dsp_detector", severity_float)
            fault_params["module_key"] = "dsp_detector"
        elif fault_kind == "power":
            label_sys = "幅度失准"
            label_mod = forced_module_label or "电源模块"
            fault_params["type"] = "power_noise"
            fault_params["subtype"] = "amp_error_ripple"
            # V-D.3: 禁用模板系统
            # V-D.2: 使用 high_diff_variance 替代 inject_power_noise_rrs
            curve = curve_generator.apply_degradation(rrs_copy, "power_management", severity_float)
            fault_params["module_key"] = "power_management"

        # V-D.3: 完全禁用旧模板系统
        # 旧逻辑已移除: apply_template(), template_id 分配, 模板参数更新
        # 物理特征现在完全由 CurveGenerator 控制
        
        # 频率类故障的峰值追踪逻辑（保留物理相关性）
        peak_track_type = "none"  # 默认无峰值追踪

        if peak_track_type != "none" and fault_kind in ("freq", "clock", "lo"):
            offsets = np.asarray(peak_track_profile.get("offsets", np.zeros(len(curve))))
            mask = np.asarray(peak_track_profile.get("mask", np.zeros(len(curve))))
            scale = np.clip(np.abs(offsets) / 4.0e7, 0.0, 1.0)
            bias = -rng.uniform(0.02, 0.10) * scale
            noise = rng.normal(0.0, rng.uniform(0.004, 0.02), size=len(curve))
            if peak_track_type == "spike":
                curve = curve + bias * (mask > 0)
            else:
                curve = curve + bias * mask + noise * mask
            fault_params["freq_amp_coupling"] = {
                "bias_db_min": -0.10,
                "bias_db_max": -0.02,
            }

        curve = constraints.adjust_fault_curve(curve, fault_kind, severity, rng=rng)
        if fault_kind in ("amp", "rl", "att"):
            fault_params["mean_offset_db"] = float(np.mean(curve - rrs))
        else:
            curve = _constrain_by_feature_stats(curve, rrs, bounds, normal_stats)
        result = constraints.validate_fault(curve, fault_kind)
        if result.ok:
            if peak_track_type != "none" and fault_kind in ("freq", "clock", "lo"):
                offsets = np.asarray(peak_track_profile.get("offsets", np.zeros(len(curve))))
                peak_freq_meas = frequency + offsets
            else:
                peak_freq_meas, _ = _generate_peak_freq_meas(frequency, rng, peak_track_type, severity)
            if fault_kind in ("freq", "lo") and peak_track_type != "none":
                peak_mae, peak_outlier_frac = _peak_freq_metrics(frequency, peak_freq_meas)
                if peak_outlier_frac < 0.02 and peak_mae < 2e6:
                    fault_params["peak_track_warning"] = "peak_freq mismatch too weak"
            return (
                curve,
                label_sys,
                label_mod,
                fault_params,
                peak_freq_meas,
                {"peak_track_type": peak_track_type},
            )
        last_reasons = result.reasons
        constraints._record_reject("fault", fault_kind, result.reasons)

    raise RuntimeError(
        f"Failed to generate fault curve within constraints for {fault_kind} "
        f"after retries. Last reasons: {last_reasons}"
    )


def run_simulation(args: argparse.Namespace):
    repo_root = PROJECT_ROOT
    out_dir = _resolve(repo_root, Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    build_run_snapshot(out_dir)

    print("[CRITICAL ISSUE]")
    print("1. 当前仿真数据的形态不具备物理可解释性，导致 BRB fallback 到“电源模块”")
    print("2. 标签存在逻辑互斥：Amp 类故障被标为 Power")
    print("3. 曲线未围绕 RRS 生成，而是围绕 -10 dBm 或被 clip")
    print("4. 模块标签与频谱仪真实信号链不一致（module_v1 与 物理链路脱节）")

    print("[ISSUE] 1) 仿真数据存在“阈值内但中心整体偏离 RRS”的问题。")
    print("[ISSUE] 2) 部分故障/正常过度平滑或偏离过大，compare 方法掉点。")
    print("[ISSUE] 3) 本次仅重构仿真与质量检查，不重构 RRS/包络形状。")

    print("[FACT] 数据场景：低频段单频段（10 MHz–8.2 GHz），AC 耦合，前置放大器关闭。")
    print("[FACT] 真实正常统计：相对 RRS 偏差不超过 ±0.4 dB。")
    print("[FACT] 仿真要求：normal 样本与非整体幅度类故障均围绕 RRS 生成。")

    print(f"[INFO] project_root={repo_root}")
    print(f"[INFO] single_band={SINGLE_BAND}")
    print(f"[INFO] disable_preamp={DISABLE_PREAMP}")
    print(f"[INFO] seed={args.seed}")
    print(f"[INFO] output_dir={out_dir}")

    coupling_mode = "AC"
    active_kinds = _build_active_fault_kinds(SINGLE_BAND, DISABLE_PREAMP, coupling_mode)
    active_modules = sorted(
        {
            KIND_TO_MODULE.get(kind, kind)
            for kind in active_kinds
            if kind != "normal"
        }
    )
    print(f"[INFO] coupling_mode={coupling_mode}")
    print(f"[INFO] active_fault_kinds={active_kinds}")
    print(f"[INFO] active_modules={active_modules}")
    print(f"[INFO] module_labels={MODULE_LABELS_V2}")
    print(f"[INFO] disabled_modules={DISABLED_MODULES}")

    freq, rrs, bounds, band_ranges, switch_feats = load_baseline(
        repo_root,
        Path(args.baseline_npz),
        Path(args.baseline_meta),
        Path(args.switch_json),
    )
    baseline_stats = load_baseline_stats(
        _resolve(repo_root, Path(args.baseline_npz)),
        _resolve(repo_root, Path(args.baseline_meta)),
    )
    traces = None
    npz_data = np.load(_resolve(repo_root, Path(args.baseline_npz)), allow_pickle=True)
    if "traces" in npz_data:
        traces = npz_data["traces"]
    if traces is not None:
        real_stats = _compute_real_normal_stats(freq, traces, rrs)
        real_stats_path = out_dir / "real_normal_stats.json"
        real_stats_path.write_text(json.dumps(real_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    constraints = SimulationConstraints(baseline_stats)
    rng = np.random.default_rng(args.seed)
    track_cycle = _track_type_cycle(rng)
    normal_stats = _load_normal_stats(repo_root)

    prev_dir = None
    if out_dir.exists() and (out_dir / "simulated_curves.npz").exists():
        prev_dir = out_dir.parent / f"{out_dir.name}_prev"
        if prev_dir.exists():
            for path in prev_dir.iterdir():
                if path.is_file():
                    path.unlink()
        else:
            prev_dir.mkdir(parents=True, exist_ok=True)
        for item in out_dir.iterdir():
            if item.is_file():
                item.replace(prev_dir / item.name)
        if (out_dir / "raw_curves").exists():
            for csv_file in (out_dir / "raw_curves").glob("*.csv"):
                dest_dir = prev_dir / "raw_curves"
                dest_dir.mkdir(parents=True, exist_ok=True)
                csv_file.replace(dest_dir / csv_file.name)

    curves: List[np.ndarray] = []
    peak_freqs: List[np.ndarray] = []
    feature_rows: List[Dict[str, object]] = []
    system_rows: List[Dict[str, object]] = []
    module_rows: List[Dict[str, object]] = []
    brb_rows: List[Dict[str, object]] = []
    labels: dict = {}
    sys_labels: List[str] = []
    mod_labels: List[str] = []
    mod_labels_v2: List[str] = []
    fault_params_list: List[Dict] = []  # Track fault injection parameters

    if args.module_driven:
        print("Generating module-driven dataset (Dataset-M)")
        idx = 0
        module_specs = module_specs_by_system("amp_error", DISABLED_MODULES) + \
            module_specs_by_system("freq_error", DISABLED_MODULES) + \
            module_specs_by_system("ref_error", DISABLED_MODULES)
        module_labels = [spec.module_label for spec in module_specs]
        for _ in range(args.n_samples):
            sample_id = f"sim_{idx:05d}"
            module_label = str(rng.choice(module_labels))
            spec = module_spec_by_label(module_label)
            target_class = spec.system_label if spec else None
            forced_track_type = next(track_cycle) if target_class == "freq_error" else None
            curve, label_sys, label_mod, fault_params, peak_freq_meas, peak_meta = simulate_curve(
                freq,
                rrs,
                band_ranges,
                traces,
                normal_stats,
                bounds,
                rng,
                constraints,
                active_kinds,
                target_class=target_class,
                forced_module_label=module_label,
                forced_peak_track_type=forced_track_type,
                peak_track_cycle=track_cycle,
            )
            curves.append(curve)
            peak_freqs.append(peak_freq_meas)
            sys_labels.append(label_sys)
            mod_labels.append(label_mod)
            fault_params_list.append({'sample_id': sample_id, **fault_params})

            sys_feats = extract_system_features(curve, baseline_curve=rrs, envelope=bounds)
            dyn_feats = compute_dynamic_threshold_features(curve, rrs, bounds, switch_feats)
            sys_result = system_level_infer(sys_feats)

            module_feats = extract_module_features(curve, module_id=idx)
            module_probs = module_level_infer({**module_feats, **sys_feats, **dyn_feats}, sys_result)

            sys_probs = sys_result.get("probabilities", sys_result)
            fault_class = "normal"
            if label_sys == "幅度失准":
                fault_class = "amp_error"
            elif label_sys == "频率失准":
                fault_class = "freq_error"
            elif label_sys == "参考电平失准":
                fault_class = "ref_error"

            dev_rrs = curve - rrs
            tier, tier_stats = _evaluate_tier(dev_rrs)
            peak_mae, peak_outlier_frac = _peak_freq_metrics(freq, peak_freq_meas)
            abs_range_ok = bool(-10.6 <= np.min(curve) <= -9.4 and -10.6 <= np.max(curve) <= -9.4)
            # V-D.3: 移除 template_id，使用 module_key
            module_key = fault_params.get("module_key")
            module_v2 = _module_v2_from_fault(label_mod, fault_params.get("type", ""))
            module_signature = f"{label_mod}:{module_key}" if label_mod != "none" and module_key else None
            mod_labels_v2.append(module_v2)
            hf_std_rrs = float(np.std(dev_rrs - _smooth_series(dev_rrs, window=61)))
            labels[sample_id] = {
                "type": "normal" if fault_class == "normal" else "fault",
                "system_fault_class": fault_class,
                "system_label": label_sys,
                "module_cause": None if fault_class == "normal" else label_mod,
                "module_id": None if fault_class == "normal" else label_mod,
                "module": None if fault_class == "normal" else label_mod,
                "module_v2": None if fault_class == "normal" else module_v2,
                # V-D.3: 使用 module_key 替代 template_id
                "module_key": None if fault_class == "normal" else module_key,
                "fault_template_id": None,  # V-D.3: 禁用模板系统
                "template_id": None,  # V-D.3: 禁用模板系统
                "module_signature": None if fault_class == "normal" else module_signature,
                "fault_params": fault_params,
                "tier": tier,
                "severity": fault_params.get("severity", "light"),
                "seed": int(args.seed),
                "sample_seed": int(args.seed) + idx,
                "amp_error_subtype": fault_params.get("subtype"),
                "abs_range_ok": abs_range_ok,
                "global_offset_rrs_db": float(np.median(dev_rrs)),
                "hf_std_rrs_db": hf_std_rrs,
                "p95_abs_dev_rrs_db": float(np.quantile(np.abs(dev_rrs), 0.95)),
                "inside_env_frac": float(np.mean((curve >= bounds[1]) & (curve <= bounds[0]))),
                "abs_out_of_spec_0p4": bool(tier_stats["max_abs"] > 0.40),
                "abs_cap_db": tier_stats["cap_db"],
                "peak_freq_mae_hz": peak_mae,
                "peak_freq_outlier_frac": peak_outlier_frac,
                "peak_track_type": peak_meta.get("peak_track_type", "none"),
            }

            feature_rows.append({"sample_id": sample_id, "fault_kind": label_sys, "module_label": label_mod, **sys_feats, **dyn_feats})
            system_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **sys_probs})
            module_rows.append(
                {
                    "sample_id": sample_id,
                    "fault_kind": label_sys,
                    **{label: module_probs.get(label, 0.0) for label in MODULE_LABELS_V2},
                }
            )

            brb_rows.append(
                {
                    "sample_id": sample_id,
                    **sys_feats,
                    **dyn_feats,
                    **{f"mod_{k}": v for k, v in module_probs.items()},
                }
            )
            idx += 1
    # Generate balanced samples across 4 system classes
    elif args.balanced:
        # Ensure n_samples is divisible by 4 for perfect balance
        n_per_class = args.n_samples // 4
        remaining = args.n_samples % 4
        
        class_counts = {
            'amp_error': n_per_class + (1 if remaining > 0 else 0),
            'freq_error': n_per_class + (1 if remaining > 1 else 0),
            'ref_error': n_per_class + (1 if remaining > 2 else 0),
            'normal': n_per_class,
        }
        freq_track_plan = _build_track_type_plan(class_counts["freq_error"], rng)
        
        print(f"Generating balanced dataset with {args.n_samples} samples:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
        
        # Generate samples for each class
        idx = 0
        for target_class in ['amp_error', 'freq_error', 'ref_error', 'normal']:
            for _ in range(class_counts[target_class]):
                sample_id = f"sim_{idx:05d}"
                forced_track_type = (
                    freq_track_plan.pop(0) if target_class == "freq_error" and freq_track_plan else None
                )
                curve, label_sys, label_mod, fault_params, peak_freq_meas, peak_meta = simulate_curve(
                    freq,
                    rrs,
                    band_ranges,
                    traces,
                    normal_stats,
                    bounds,
                    rng,
                    constraints,
                    active_kinds,
                    target_class=target_class,
                    forced_peak_track_type=forced_track_type,
                    peak_track_cycle=track_cycle,
                )
                curves.append(curve)
                peak_freqs.append(peak_freq_meas)
                sys_labels.append(label_sys)
                mod_labels.append(label_mod)
                fault_params_list.append({'sample_id': sample_id, **fault_params})

                # Pass baseline_curve=rrs and envelope=bounds to extract X16-X18 features properly
                sys_feats = extract_system_features(curve, baseline_curve=rrs, envelope=bounds)
                dyn_feats = compute_dynamic_threshold_features(curve, rrs, bounds, switch_feats)
                sys_result = system_level_infer(sys_feats)

                module_feats = extract_module_features(curve, module_id=idx)
                module_probs = module_level_infer({**module_feats, **sys_feats, **dyn_feats}, sys_result)

                sys_probs = sys_result.get("probabilities", sys_result)
                fault_class = "normal"
                if label_sys == "幅度失准":
                    fault_class = "amp_error"
                elif label_sys == "频率失准":
                    fault_class = "freq_error"
                elif label_sys == "参考电平失准":
                    fault_class = "ref_error"

                dev_rrs = curve - rrs
                tier, tier_stats = _evaluate_tier(dev_rrs)
                peak_mae, peak_outlier_frac = _peak_freq_metrics(freq, peak_freq_meas)
                abs_range_ok = bool(-10.6 <= np.min(curve) <= -9.4 and -10.6 <= np.max(curve) <= -9.4)
                # V-D.3: 移除 template_id，使用 module_key
                module_key = fault_params.get("module_key")
                module_v2 = _module_v2_from_fault(label_mod, fault_params.get("type", ""))
                module_signature = f"{label_mod}:{module_key}" if label_mod != "none" and module_key else None
                mod_labels_v2.append(module_v2)
                hf_std_rrs = float(np.std(dev_rrs - _smooth_series(dev_rrs, window=61)))
                labels[sample_id] = {
                    "type": "normal" if fault_class == "normal" else "fault",
                    "system_fault_class": fault_class,  # Always include, "normal" for normal samples
                    "system_label": label_sys,
                    "module_cause": None if fault_class == "normal" else label_mod,
                    "module_id": None if fault_class == "normal" else label_mod,
                    "module": None if fault_class == "normal" else label_mod,
                    "module_v2": None if fault_class == "normal" else module_v2,
                    # V-D.3: 使用 module_key 替代 template_id
                    "module_key": None if fault_class == "normal" else module_key,
                    "fault_template_id": None,  # V-D.3: 禁用模板系统
                    "template_id": None,  # V-D.3: 禁用模板系统
                    "module_signature": None if fault_class == "normal" else module_signature,
                    "fault_params": fault_params,  # Include injection parameters
                    "tier": tier,
                    "severity": fault_params.get("severity", "light"),
                    "seed": int(args.seed),
                    "sample_seed": int(args.seed) + idx,
                    "amp_error_subtype": fault_params.get("subtype"),
                    "abs_range_ok": abs_range_ok,
                    "global_offset_rrs_db": float(np.median(dev_rrs)),
                    "hf_std_rrs_db": hf_std_rrs,
                    "p95_abs_dev_rrs_db": float(np.quantile(np.abs(dev_rrs), 0.95)),
                    "inside_env_frac": float(np.mean((curve >= bounds[1]) & (curve <= bounds[0]))),
                    "abs_out_of_spec_0p4": bool(tier_stats["max_abs"] > 0.40),
                    "abs_cap_db": tier_stats["cap_db"],
                    "peak_freq_mae_hz": peak_mae,
                    "peak_freq_outlier_frac": peak_outlier_frac,
                    "peak_track_type": peak_meta.get("peak_track_type", "none"),
                }

                feature_rows.append({"sample_id": sample_id, "fault_kind": label_sys, "module_label": label_mod, **sys_feats, **dyn_feats})
                system_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **sys_probs})
                module_rows.append(
                    {
                        "sample_id": sample_id,
                        "fault_kind": label_sys,
                        **{label: module_probs.get(label, 0.0) for label in MODULE_LABELS_V2},
                    }
                )

                brb_rows.append(
                    {
                        "sample_id": sample_id,
                        **sys_feats,
                        **dyn_feats,
                        **{f"mod_{k}": v for k, v in module_probs.items()},
                    }
                )
                idx += 1
    else:
        # Realistic distribution generation (default)
        print(f"Generating realistic distribution with {args.n_samples} samples")
        print("Expected distribution (based on module diversity):")
        print("  Amplitude faults (8 modules): ~58%")
        print("  Frequency faults (3 modules): ~20%")
        print("  Reference faults (2 modules): ~14%")
        print("  Normal state: ~8%")
        print()
        
        for idx in range(args.n_samples):
            sample_id = f"sim_{idx:05d}"
            curve, label_sys, label_mod, fault_params, peak_freq_meas, peak_meta = simulate_curve(
                freq,
                rrs,
                band_ranges,
                traces,
                normal_stats,
                bounds,
                rng,
                constraints,
                active_kinds,
                peak_track_cycle=track_cycle,
            )
            curves.append(curve)
            peak_freqs.append(peak_freq_meas)
            sys_labels.append(label_sys)
            mod_labels.append(label_mod)
            fault_params_list.append({'sample_id': sample_id, **fault_params})

            # Pass baseline_curve=rrs and envelope=bounds to extract X16-X18 features properly
            sys_feats = extract_system_features(curve, baseline_curve=rrs, envelope=bounds)
            dyn_feats = compute_dynamic_threshold_features(curve, rrs, bounds, switch_feats)
            sys_result = system_level_infer(sys_feats)

            module_feats = extract_module_features(curve, module_id=idx)
            module_probs = module_level_infer({**module_feats, **sys_feats, **dyn_feats}, sys_result)

            sys_probs = sys_result.get("probabilities", sys_result)
            fault_class = "normal"
            if label_sys == "幅度失准":
                fault_class = "amp_error"
            elif label_sys == "频率失准":
                fault_class = "freq_error"
            elif label_sys == "参考电平失准":
                fault_class = "ref_error"

            dev_rrs = curve - rrs
            tier, tier_stats = _evaluate_tier(dev_rrs)
            peak_mae, peak_outlier_frac = _peak_freq_metrics(freq, peak_freq_meas)
            abs_range_ok = bool(-10.6 <= np.min(curve) <= -9.4 and -10.6 <= np.max(curve) <= -9.4)
            # V-D.3: 移除 template_id，使用 module_key
            module_key = fault_params.get("module_key")
            module_v2 = _module_v2_from_fault(label_mod, fault_params.get("type", ""))
            module_signature = f"{label_mod}:{module_key}" if label_mod != "none" and module_key else None
            mod_labels_v2.append(module_v2)
            hf_std_rrs = float(np.std(dev_rrs - _smooth_series(dev_rrs, window=61)))
            labels[sample_id] = {
                "type": "normal" if fault_class == "normal" else "fault",
                "system_fault_class": fault_class,  # Always include, "normal" for normal samples
                "system_label": label_sys,
                "module_cause": None if fault_class == "normal" else label_mod,
                "module_id": None if fault_class == "normal" else label_mod,
                "module": None if fault_class == "normal" else label_mod,
                "module_v2": None if fault_class == "normal" else module_v2,
                # V-D.3: 使用 module_key 替代 template_id
                "module_key": None if fault_class == "normal" else module_key,
                "fault_template_id": None,  # V-D.3: 禁用模板系统
                "template_id": None,  # V-D.3: 禁用模板系统
                "module_signature": None if fault_class == "normal" else module_signature,
                "fault_params": fault_params,
                "tier": tier,
                "severity": fault_params.get("severity", "light"),
                "seed": int(args.seed),
                "sample_seed": int(args.seed) + idx,
                "amp_error_subtype": fault_params.get("subtype"),
                "abs_range_ok": abs_range_ok,
                "global_offset_rrs_db": float(np.median(dev_rrs)),
                "hf_std_rrs_db": hf_std_rrs,
                "p95_abs_dev_rrs_db": float(np.quantile(np.abs(dev_rrs), 0.95)),
                "inside_env_frac": float(np.mean((curve >= bounds[1]) & (curve <= bounds[0]))),
                "abs_out_of_spec_0p4": bool(tier_stats["max_abs"] > 0.40),
                "abs_cap_db": tier_stats["cap_db"],
                "peak_freq_mae_hz": peak_mae,
                "peak_freq_outlier_frac": peak_outlier_frac,
                "peak_track_type": peak_meta.get("peak_track_type", "none"),
            }

            feature_rows.append({"sample_id": sample_id, "fault_kind": label_sys, "module_label": label_mod, **sys_feats, **dyn_feats})
            system_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **sys_probs})
            module_rows.append(
                {
                    "sample_id": sample_id,
                    "fault_kind": label_sys,
                    **{label: module_probs.get(label, 0.0) for label in MODULE_LABELS_V2},
                }
            )

            brb_rows.append(
                {
                    "sample_id": sample_id,
                    **sys_feats,
                    **dyn_feats,
                    **{f"mod_{k}": v for k, v in module_probs.items()},
                }
            )

    _write_raw_csvs(out_dir, freq, curves, peak_freqs, sys_labels, mod_labels, mod_labels_v2)
    _write_csv(out_dir / "simulated_features.csv", feature_rows)
    _write_csv(out_dir / "system_predictions.csv", system_rows)
    _write_csv(out_dir / "module_predictions.csv", module_rows)
    _write_csv(out_dir / "features_brb.csv", brb_rows, encoding="utf-8-sig")
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_curves(out_dir / "simulated_curves.csv", freq, curves)
    np.savez(out_dir / "simulated_curves.npz", frequency=freq, curves=np.array(curves))
    np.save(out_dir / "simulated_curves_float.npy", np.array(curves))
    _write_reject_stats(out_dir, constraints)
    _plot_overlay_audit(out_dir, freq, rrs, curves, labels, traces)
    _plot_peak_track_audit(out_dir, freq, peak_freqs, labels)
    _plot_normal_vs_real(out_dir, freq, rrs, traces, curves, labels)
    _plot_overlay_by_module(out_dir, freq, curves, labels)
    _plot_template_gallery(out_dir, freq, curves, labels)
    _plot_peakfreq_behavior(out_dir, freq, peak_freqs, labels)
    _plot_amp_vs_ref_separability(out_dir, feature_rows, labels)
    _plot_grid_with_manifest(out_dir, freq, curves, labels)
    _write_ref_error_bucket_report(out_dir, feature_rows, labels)

    # Validate output counts
    raw_count = len(list((out_dir / "raw_curves").glob("*.csv")))
    label_count = len(labels)
    features_path = out_dir / "features_brb.csv"
    features_count = 0
    if features_path.exists():
        with features_path.open("r", encoding="utf-8-sig") as f:
            features_count = max(0, sum(1 for _ in f) - 1)

    expected = args.n_samples
    if raw_count != expected or label_count != expected or features_count != expected:
        print(
            "[ERROR] Output counts mismatch: "
            f"raw_curves={raw_count}, labels={label_count}, features={features_count}, expected={expected}"
        )
        raise SystemExit(1)
    
    # Save fault params CSV for effect check
    if fault_params_list:
        _write_csv(out_dir / "fault_params.csv", fault_params_list, encoding="utf-8-sig")
        print(f"Saved fault_params.csv with injection parameters")
    
    # Generate freq_ref_effect_check.csv
    _generate_effect_check(out_dir, feature_rows, labels)

    # Print summary statistics
    print()
    print("=" * 60)
    print("仿真完成摘要")
    print("=" * 60)
    print(f"  raw_curves 路径: {out_dir / 'raw_curves'}")
    print(f"  labels.json 路径: {out_dir / 'labels.json'}")
    print(f"  生成样本数: {len(curves)}")
    print()
    print("  系统级分布:")
    sys_class_dist = {}
    for sample_id, lbl in labels.items():
        cls = lbl.get('system_fault_class', 'normal')
        sys_class_dist[cls] = sys_class_dist.get(cls, 0) + 1
    for cls in ['normal', 'amp_error', 'freq_error', 'ref_error']:
        count = sys_class_dist.get(cls, 0)
        pct = count / len(labels) * 100 if labels else 0
        print(f"    {cls}: {count} ({pct:.1f}%)")
    print("=" * 60)

    # Quality check (hard constraints)
    report, violations = run_quality_check(
        out_dir,
        _resolve(repo_root, Path(args.baseline_npz)),
        baseline_meta=_resolve(repo_root, Path(args.baseline_meta)),
        active_modules=active_modules,
        reject_records=constraints.reject_records,
        prev_dir=prev_dir,
    )
    if violations:
        print("[ERROR] Quality check failed: invalid samples detected.")
        for v in violations:
            print(f"  - {v.get('sample_id')}: {v.get('reason')}")
        print("Suggested adjustments: reduce beta/shift ranges or tighten mean offset correction.")
        raise SystemExit(1)

    print("[INFO] Quality check passed.")
    global_min = report.get("global_min_y")
    global_max = report.get("global_max_y")
    overall_report = report.get("overall", {})
    normal_stats_report = overall_report.get("normal", {})
    residuals = report.get("residual_comparison", {}).get("sim_normal", {})
    roughness = report.get("roughness", {})
    target_p95_low = constraints.baseline.target_p95_low
    target_p95_high = constraints.baseline.target_p95_high
    rough_low = 0.8 * constraints.baseline.rough_p50
    rough_high = 1.2 * constraints.baseline.rough_p50
    print("[INFO] Quality summary:")
    print(f"  global_min={global_min:.2f} dBm, global_max={global_max:.2f} dBm")
    print(
        f"  normal_p95_abs={residuals.get('p95_abs', 0):.4f} dB "
        f"(target {target_p95_low:.3f}–{target_p95_high:.3f})"
    )
    print(
        f"  normal_rough_p50={roughness.get('sim_normal_p50', 0):.6f} "
        f"(baseline_p50={roughness.get('baseline_p50', 0):.6f})"
    )
    print(
        f"  normal_inside_env_frac_mean={normal_stats_report.get('inside_env_frac_mean', 0):.4f}"
    )
    print(
        f"  checks: range_ok={-10.6 <= global_min <= -9.4 and -10.6 <= global_max <= -9.4}, "
        f"p95_ok={target_p95_low <= residuals.get('p95_abs', 0) <= target_p95_high}, "
        f"rough_ok={rough_low <= roughness.get('sim_normal_p50', 0) <= rough_high}, "
        f"inside_env_ok={normal_stats_report.get('inside_env_frac_mean', 0) >= 0.98}"
    )
    before_after = report.get("before_after_samples", [])
    if before_after:
        print("[INFO] Before/After sample comparison (prev vs current):")
        for row in before_after:
            before = row.get("before", {})
            after = row.get("after", {})
            print(
                f"  {row.get('sample_id')}: "
                f"before max_abs_dev={before.get('max_abs_dev'):.4f}, "
                f"mean_offset={before.get('mean_offset'):.4f}, "
                f"inside_env_frac={before.get('inside_env_frac'):.4f} -> "
                f"after max_abs_dev={after.get('max_abs_dev'):.4f}, "
                f"mean_offset={after.get('mean_offset'):.4f}, "
                f"inside_env_frac={after.get('inside_env_frac'):.4f}"
            )

    print("[INFO] Per-class quality summary:")
    for cls, stats in overall_report.items():
        print(
            f"  {cls}: min={stats.get('min_y'):.2f}, max={stats.get('max_y'):.2f}, "
            f"mean_offset_p05={stats.get('mean_offset_p05'):.4f}, "
            f"mean_offset_p50={stats.get('mean_offset_median'):.4f}, "
            f"mean_offset_p95={stats.get('mean_offset_p95'):.4f}, "
            f"p95_abs_dev_p95={stats.get('p95_abs_dev_p95'):.4f}, "
            f"rough_p50={stats.get('roughness_p50'):.6f}, "
            f"rough_p95={stats.get('roughness_p95'):.6f}"
        )

    feature_checks = _evaluate_feature_checks(feature_rows, labels)
    print("[INFO] Feature separation checks:")
    for cls in ["freq_error", "ref_error", "amp_error"]:
        check = feature_checks.get(cls, {})
        print(f"  {cls}: pass={check.get('pass')}, zscores={check.get('feature_zscores')}")

    report_path = out_dir / "sim_quality_report.json"
    if report_path.exists():
        report_json = json.loads(report_path.read_text(encoding="utf-8"))
        report_json["feature_checks"] = feature_checks
        report_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_effect_check(out_dir: Path, feature_rows: List[Dict], labels: dict):
    """Generate freq_ref_effect_check.csv to verify injection → feature correlation."""
    freq_features = ['X16', 'X17', 'X18', 'X23', 'X24', 'X25']
    ref_features = ['X3', 'X5', 'X26', 'X27', 'X28']

    stats = []
    for cls in ['normal', 'amp_error', 'freq_error', 'ref_error']:
        cls_rows = [
            row for row in feature_rows
            if labels.get(row.get('sample_id', ''), {}).get('system_fault_class', 'normal') == cls
        ]
        if not cls_rows:
            continue

        row_stats = {'class': cls, 'n': len(cls_rows)}
        for f in freq_features + ref_features:
            vals = [float(r.get(f, 0.0)) for r in cls_rows if f in r]
            if vals:
                arr = np.array(vals, dtype=float)
                row_stats[f'{f}_mean'] = float(np.mean(arr))
                row_stats[f'{f}_std'] = float(np.std(arr))
                row_stats[f'{f}_p90'] = float(np.percentile(arr, 90))
        stats.append(row_stats)

    if stats:
        output_path = out_dir / 'freq_ref_effect_check.csv'
        keys = sorted({k for row in stats for k in row.keys()})
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(stats)
        print("Saved freq_ref_effect_check.csv")
        
        stats_by_class = {row.get("class"): row for row in stats}
        print("\n=== Freq/Ref Feature Effect Check ===")
        print("Freq features (should be high for freq_error):")
        for f in ['X16', 'X23', 'X24']:
            for cls, row in stats_by_class.items():
                key = f"{f}_mean"
                if key in row:
                    print(f"  {cls:12s} {f}_mean={row.get(key, 0):.4f}")

        print("\nRef features (should be high for ref_error):")
        for f in ['X26', 'X27', 'X28']:
            for cls, row in stats_by_class.items():
                key = f"{f}_mean"
                if key in row:
                    print(f"  {cls:12s} {f}_mean={row.get(key, 0):.4f}")


def _feature_class_stats(
    feature_rows: List[Dict[str, object]],
    labels: dict,
    feature_keys: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cls in ["normal", "amp_error", "freq_error", "ref_error"]:
        cls_rows = [
            row for row in feature_rows
            if labels.get(row.get("sample_id", ""), {}).get("system_fault_class", "normal") == cls
        ]
        stats[cls] = {}
        for key in feature_keys:
            values = [float(r.get(key, 0.0)) for r in cls_rows if key in r]
            if values:
                arr = np.array(values, dtype=float)
                stats[cls][key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
            else:
                stats[cls][key] = {"mean": 0.0, "std": 0.0}
    return stats


def _evaluate_feature_checks(feature_rows: List[Dict], labels: dict) -> Dict[str, Dict[str, object]]:
    feature_keys = ["X1", "X3", "X6", "X10", "X14", "X16", "X17", "X18"]
    stats = _feature_class_stats(feature_rows, labels, feature_keys)
    checks: Dict[str, Dict[str, object]] = {}

    normal_stats = stats["normal"]
    def _zscore(cls: str, key: str) -> float:
        mean_n = normal_stats.get(key, {}).get("mean", 0.0)
        std_n = normal_stats.get(key, {}).get("std", 1e-6) or 1e-6
        mean_c = stats.get(cls, {}).get(key, {}).get("mean", 0.0)
        return (mean_c - mean_n) / std_n

    freq_keys = ["X16", "X17", "X18"]
    freq_z = {k: _zscore("freq_error", k) for k in freq_keys}
    checks["freq_error"] = {
        "feature_zscores": freq_z,
        "pass": all(abs(z) >= 3.0 for z in freq_z.values()),
    }

    ref_keys = ["X14", "X1"]
    ref_z = {k: _zscore("ref_error", k) for k in ref_keys}
    checks["ref_error"] = {
        "feature_zscores": ref_z,
        "pass": any(abs(z) >= 3.0 for z in ref_z.values()),
    }

    amp_keys = ["X6", "X3", "X10"]
    amp_z = {k: _zscore("amp_error", k) for k in amp_keys}
    checks["amp_error"] = {
        "feature_zscores": amp_z,
        "pass": any(abs(z) >= 3.0 for z in amp_z.values()),
    }

    checks["stats"] = stats
    return checks


def build_argparser():
    parser = argparse.ArgumentParser(description="仿真频响并执行 BRB 诊断")
    parser.add_argument("--baseline_npz", default=BASELINE_NPZ)
    parser.add_argument("--baseline_meta", default=BASELINE_META)
    parser.add_argument("--switch_json", default=SWITCH_JSON)
    parser.add_argument("--out_dir", default=SIM_DIR)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="总样本数（默认400，建议4的倍数以便完美平衡）",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=DEFAULT_BALANCED,
        help="生成系统级均衡的样本（每类相同数量，默认开启）",
    )
    parser.add_argument(
        "--module_driven",
        action="store_true",
        default=False,
        help="生成模块机理驱动的 Dataset-M（模块采样优先，系统类不强制均衡）",
    )
    parser.add_argument("--realistic", dest="balanced", action="store_false",
                       help="使用真实概率分布（反映模块多样性：幅度58%%,频率20%%,参考14%%,正常8%%）")
    return parser


def main() -> int:
    import os
    import sys

    script_dir = Path(__file__).resolve().parent
    repo_root = PROJECT_ROOT
    os.chdir(repo_root)
    parser = build_argparser()
    args = parser.parse_args()
    print("=" * 60)
    print("FMFD Simulation Pipeline (系统级均衡仿真)")
    print("=" * 60)
    print(f"  Samples: {args.n_samples} (balanced={args.balanced})")
    print(f"  Output:  {args.out_dir}")
    print("=" * 60)
    print()
    run_simulation(args)
    if sys.platform == "win32":
        try:
            if sys.stdin.isatty():
                print()
                print("=" * 60)
                print("Simulation complete! Press Enter to exit...")
                input()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
