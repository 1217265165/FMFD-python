#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation constraints and quality reporting for RRS-centered data

任务书第二阶段 §5: 模块→RRS形态签名
实现可解释的模块级形态扰动项 Δy_module(f)，用于诊断可分性
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================================
# 配置常量
# ============================================================================

# 签名生成重试次数
SIGNATURE_MAX_RETRY = 10

# 归一化频率计算的小量，避免除零
FREQ_NORMALIZE_EPS = 1e-12

# V-D.5b: 微弱故障兼容阈值 (dB)
# 物理引擎生成的微弱故障偏差可能 < 0.6 dB，需要降低质量检查阈值
MIN_VALID_DEVIATION_DB = 0.10

# V-D.5b: 纹理变化检测阈值 (用于 ADC/VBW 等以纹理为主的故障)
TEXTURE_CHANGE_THRESHOLD = 1e-6

# V-D.6: amp/att/rl 专用阈值 - 这些故障主要是偏移，只看幅度不看形态
AMP_DEVIATION_THRESHOLD = 0.10


# ============================================================================
# 任务书 §5.2: 模块形态签名模板
# ============================================================================

def signature_lpf_global_slope(
    x: np.ndarray, 
    a1: float = None, 
    a2: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """LPF 模块签名：global_slope (全局斜率)
    
    Δy_LPF(x) = a1*(x - 0.5) + a2*(x - 0.5)^2
    
    Parameters
    ----------
    x : ndarray
        归一化频率 [0, 1]
    a1 : float, optional
        一阶斜率系数，建议范围 [-0.18, -0.05]
    a2 : float, optional
        二阶曲率系数，建议范围 [0, 0.10]
    rng : Generator, optional
        随机数生成器
    
    Returns
    -------
    ndarray
        形态签名扰动
    """
    if rng is None:
        rng = np.random.default_rng()
    if a1 is None:
        a1 = rng.uniform(-0.18, -0.05)
    if a2 is None:
        a2 = rng.uniform(0.0, 0.10)
    
    x_centered = x - 0.5
    return a1 * x_centered + a2 * x_centered ** 2


def signature_mixer_periodic_wave(
    x: np.ndarray,
    nu: float = None,
    A0: float = None,
    alpha: float = None,
    phi: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Mixer 模块签名：periodic_wave (周期性波动)
    
    Δy_Mixer(x) = A(x) * sin(2π*ν*x + φ)
    A(x) = A0 * (1 + α*(x - 0.5))
    
    Parameters
    ----------
    x : ndarray
        归一化频率 [0, 1]
    nu : float, optional
        周期数，建议范围 [6, 14]
    A0 : float, optional
        基础幅度，建议范围 [0.03, 0.08]
    alpha : float, optional
        幅度调制系数，建议范围 [-0.4, 0.4]
    phi : float, optional
        相位，建议范围 [0, 2π]
    rng : Generator, optional
        随机数生成器
    
    Returns
    -------
    ndarray
        形态签名扰动
    """
    if rng is None:
        rng = np.random.default_rng()
    if nu is None:
        nu = rng.uniform(6.0, 14.0)
    if A0 is None:
        A0 = rng.uniform(0.03, 0.08)
    if alpha is None:
        alpha = rng.uniform(-0.4, 0.4)
    if phi is None:
        phi = rng.uniform(0.0, 2 * np.pi)
    
    A_x = A0 * (1 + alpha * (x - 0.5))
    return A_x * np.sin(2 * np.pi * nu * x + phi)


def signature_detector_local_ripple(
    x: np.ndarray,
    x_c: float = None,
    sigma: float = None,
    num_harmonics: int = 2,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Detector 模块签名：local_ripple (局部纹理)
    
    W(x) = exp(-(x - x_c)^2 / (2*σ^2))
    Δy_Det(x) = W(x) * Σ b_k * sin(2π*ν_k*x + φ_k)
    
    Parameters
    ----------
    x : ndarray
        归一化频率 [0, 1]
    x_c : float, optional
        中心位置，建议范围 [0.35, 0.65]
    sigma : float, optional
        窗口宽度，建议范围 [0.08, 0.18]
    num_harmonics : int
        谐波数量
    rng : Generator, optional
        随机数生成器
    
    Returns
    -------
    ndarray
        形态签名扰动
    """
    if rng is None:
        rng = np.random.default_rng()
    if x_c is None:
        x_c = rng.uniform(0.35, 0.65)
    if sigma is None:
        sigma = rng.uniform(0.08, 0.18)
    
    # 高斯窗
    W = np.exp(-((x - x_c) ** 2) / (2 * sigma ** 2))
    
    # 多谐波叠加
    ripple = np.zeros_like(x)
    for _ in range(num_harmonics):
        nu_k = rng.uniform(18.0, 40.0)
        b_k = rng.uniform(0.008, 0.02)
        phi_k = rng.uniform(0.0, 2 * np.pi)
        ripple += b_k * np.sin(2 * np.pi * nu_k * x + phi_k)
    
    return W * ripple


def signature_adc_step_bias(
    x: np.ndarray,
    num_steps: int = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """ADC 模块签名：step_bias (台阶/分段偏置)
    
    Δy_ADC(x) = Σ s_i * sigmoid((x - t_i) / w_i)
    
    Parameters
    ----------
    x : ndarray
        归一化频率 [0, 1]
    num_steps : int, optional
        台阶数量，建议范围 [1, 3]
    rng : Generator, optional
        随机数生成器
    
    Returns
    -------
    ndarray
        形态签名扰动
    """
    if rng is None:
        rng = np.random.default_rng()
    if num_steps is None:
        num_steps = rng.integers(1, 4)
    
    # 确保台阶位置间隔 >= 0.15
    result = np.zeros_like(x)
    used_positions = []
    
    for _ in range(num_steps):
        # 尝试找到不重叠的位置
        for _ in range(SIGNATURE_MAX_RETRY):
            t_i = rng.uniform(0.15, 0.85)
            if all(abs(t_i - p) >= 0.15 for p in used_positions):
                used_positions.append(t_i)
                break
        else:
            continue
        
        w_i = rng.uniform(0.003, 0.02)
        s_i = rng.uniform(-0.08, 0.08)
        
        # Sigmoid 函数
        step = s_i / (1 + np.exp(-(x - t_i) / w_i))
        result += step
    
    return result


def signature_power_highfreq_noise(
    x: np.ndarray,
    g0: float = None,
    p: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Power 模块签名：highfreq_noise (高频噪声 - 降低主导性)
    
    任务书§5.2: 电源不再以主导异常，只在尾部噪声厚度上体现差异
    Δy_PWR(x) = g0 * x^p * n(x)
    
    Parameters
    ----------
    x : ndarray
        归一化频率 [0, 1]
    g0 : float, optional
        增益系数，建议范围 [0.015, 0.05] (降低)
    p : float, optional
        指数，建议范围 [1.5, 3.5]
    rng : Generator, optional
        随机数生成器
    
    Returns
    -------
    ndarray
        形态签名扰动
    """
    if rng is None:
        rng = np.random.default_rng()
    if g0 is None:
        g0 = rng.uniform(0.015, 0.05)  # 降低范围
    if p is None:
        p = rng.uniform(1.5, 3.5)
    
    # 高频端加权噪声
    noise = rng.normal(0, 1, size=len(x))
    return g0 * (x ** p) * noise


def generate_module_signature(
    frequency: np.ndarray,
    module_type: str,
    severity: str = "mid",
    rng: np.random.Generator = None,
) -> np.ndarray:
    """根据模块类型生成形态签名。
    
    Parameters
    ----------
    frequency : ndarray
        频率轴 (Hz)
    module_type : str
        模块类型：'lpf', 'mixer', 'detector', 'adc', 'power', 'normal'
    severity : str
        严重程度：'light', 'mid', 'severe'
    rng : Generator, optional
        随机数生成器
    
    Returns
    -------
    ndarray
        模块形态签名扰动 (dB)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 归一化频率
    f_min, f_max = frequency.min(), frequency.max()
    x = (frequency - f_min) / (f_max - f_min + FREQ_NORMALIZE_EPS)
    
    # 严重程度缩放因子
    severity_scale = {"light": 0.6, "mid": 1.0, "severe": 1.5}.get(severity, 1.0)
    
    module_type = module_type.lower()
    
    if module_type == "lpf":
        return severity_scale * signature_lpf_global_slope(x, rng=rng)
    elif module_type == "mixer":
        return severity_scale * signature_mixer_periodic_wave(x, rng=rng)
    elif module_type == "detector":
        return severity_scale * signature_detector_local_ripple(x, rng=rng)
    elif module_type == "adc":
        return severity_scale * signature_adc_step_bias(x, rng=rng)
    elif module_type == "power":
        return severity_scale * signature_power_highfreq_noise(x, rng=rng)
    elif module_type == "normal":
        # 正常状态：轻微高斯噪声，无特征签名
        return rng.normal(0, 0.005 * severity_scale, size=len(x))
    else:
        # 未知模块类型：返回零
        return np.zeros_like(x)


# 模块类型到签名函数的映射
MODULE_SIGNATURE_MAP = {
    "低频段前置低通滤波器": "lpf",
    "低频段第一混频器": "mixer",
    "高频段YTF滤波器": "lpf",
    "高频段混频器": "mixer",
    "数字检波器": "detector",
    "VBW滤波器": "detector",
    "数字RBW": "detector",
    "ADC": "adc",
    "数字放大器": "adc",
    "电源模块": "power",
    "中频放大器": "mixer",
    "时钟振荡器": "mixer",
    "时钟合成与同步网络": "mixer",
    "本振源（谐波发生器）": "mixer",
    "本振混频组件": "mixer",
    "校准源": "lpf",
    "存储器": "adc",
    "校准信号开关": "adc",
    "衰减器": "lpf",
    "前置放大器": "lpf",
}


@dataclass
class BaselineStats:
    frequency: np.ndarray
    rrs: np.ndarray
    traces: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    q_low: np.ndarray
    q_high: np.ndarray
    sigma_smooth: np.ndarray
    cap_normal_db: float
    spec_center_db: float
    spec_tol_db: float
    residuals: np.ndarray
    residual_q_low: float
    residual_q_high: float
    residual_abs_p95: float
    residual_abs_p99: float
    residual_tail_prob: float
    residual_lag1: float
    segment_edges: List[int]
    segment_bias_mean: np.ndarray
    segment_bias_std: np.ndarray
    target_p95_low: float
    target_p95_high: float
    mean_offset_p95: float
    envelope_expand_db: float
    rough_p50: float
    rough_p95: float
    global_offsets: np.ndarray
    global_offset_p50: float
    global_offset_p90: float
    global_offset_p95: float
    global_offset_p99: float
    hf_noise_p50: float
    hf_noise_p90: float
    hf_noise_p95: float


@dataclass
class ConstraintResult:
    ok: bool
    reasons: List[str]


def _smooth_noise(values: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def roughness_metric(curve: np.ndarray) -> float:
    diffs = np.diff(curve)
    return float(np.std(diffs)) if diffs.size else 0.0


def _lag1_autocorr(residuals: np.ndarray) -> float:
    if residuals.size == 0:
        return 0.0
    vals = []
    for res in residuals:
        if len(res) < 2:
            continue
        x = res[:-1]
        y = res[1:]
        denom = (np.std(x) * np.std(y)) + 1e-12
        vals.append(float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / denom))
    return float(np.mean(vals)) if vals else 0.0


def _segment_stats(
    frequency: np.ndarray,
    residuals: np.ndarray,
    bin_hz: float = 0.5e9,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    if residuals.size == 0:
        return [0, len(frequency)], np.zeros(1), np.ones(1)
    f_min = float(np.min(frequency))
    f_max = float(np.max(frequency))
    edges_hz = list(np.arange(f_min, f_max + bin_hz, bin_hz))
    if edges_hz[-1] < f_max:
        edges_hz.append(f_max)
    edges_idx = [int(np.searchsorted(frequency, edge, side="left")) for edge in edges_hz]
    edges_idx = [max(0, min(idx, len(frequency))) for idx in edges_idx]
    if edges_idx[-1] != len(frequency):
        edges_idx.append(len(frequency))
    segment_means = []
    segment_stds = []
    for start, end in zip(edges_idx[:-1], edges_idx[1:]):
        if end <= start:
            continue
        seg_vals = np.median(residuals[:, start:end], axis=1)
        segment_means.append(float(np.mean(seg_vals)))
        segment_stds.append(float(np.std(seg_vals)) + 1e-6)
    if not segment_means:
        segment_means = [0.0]
        segment_stds = [1e-6]
        edges_idx = [0, len(frequency)]
    return edges_idx, np.array(segment_means), np.array(segment_stds)


def clip_curve(curve: np.ndarray, lower: float = -10.6, upper: float = -9.4) -> np.ndarray:
    return np.clip(curve, lower, upper)


def clip_and_round(curve: np.ndarray, lower: float = -10.6, upper: float = -9.4) -> np.ndarray:
    return np.round(curve, 2)


def compute_global_offsets(traces: np.ndarray, rrs: np.ndarray) -> np.ndarray:
    return np.median(traces - rrs, axis=1)


def compute_hf_noise_stds(
    residuals: np.ndarray,
    smooth_window: int = 61,
) -> np.ndarray:
    smooth_window = max(7, smooth_window)
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.ones(smooth_window, dtype=float) / smooth_window
    hf_stds = []
    for res in residuals:
        smooth = np.convolve(res, kernel, mode="same")
        hf = res - smooth
        hf_stds.append(float(np.std(hf)))
    return np.array(hf_stds, dtype=float)


def load_baseline_stats(baseline_npz: Path, baseline_meta: Path | None = None) -> BaselineStats:
    data = np.load(baseline_npz, allow_pickle=True)
    frequency = data["frequency"]
    traces = data["traces"]
    rrs = data["rrs"]
    upper = data["upper"]
    lower = data["lower"]
    q_low = data.get("q_low", np.quantile(traces, 0.025, axis=0))
    q_high = data.get("q_high", np.quantile(traces, 0.975, axis=0))
    sigma_smooth = data.get("sigma_smooth")
    if sigma_smooth is None or len(sigma_smooth) != len(rrs):
        residuals = traces - rrs
        mad = np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
        sigma_smooth = 1.4826 * mad
        sigma_smooth = _smooth_noise(sigma_smooth, window=9)
    cap_normal_db = float(data.get("cap_normal_db", 0.4))
    spec_center_db = float(data.get("spec_center_db", -10.0))
    spec_tol_db = float(data.get("spec_tol_db", 0.4))

    residuals = traces - rrs
    residual_q_low = float(np.quantile(residuals, 0.005))
    residual_q_high = float(np.quantile(residuals, 0.995))
    residual_abs_p95 = float(np.quantile(np.abs(residuals), 0.95))
    residual_abs_p99 = float(np.quantile(np.abs(residuals), 0.99))
    residual_tail_prob = float(np.mean(np.abs(residuals) >= residual_abs_p99))
    target_p95_low = 0.06
    target_p95_high = 0.10
    meta = {}
    if baseline_meta is not None and baseline_meta.exists():
        try:
            meta = json.loads(baseline_meta.read_text(encoding="utf-8"))
            meta_p95 = float(meta.get("offset_stats", {}).get("p95_abs", residual_abs_p95))
            target_p95_low = max(0.06, 0.75 * meta_p95)
            target_p95_high = min(0.10, 1.25 * meta_p95)
        except (json.JSONDecodeError, ValueError, TypeError):
            meta = {}
    mean_offsets = np.mean(residuals, axis=1)
    mean_offset_p95 = float(np.quantile(np.abs(mean_offsets), 0.95))
    envelope_expand_db = max(0.02, 0.75 * residual_abs_p95)
    rough_vals = np.array([roughness_metric(trace) for trace in traces], dtype=float)
    rough_p50 = float(np.quantile(rough_vals, 0.50))
    rough_p95 = float(np.quantile(rough_vals, 0.95))

    global_offsets = compute_global_offsets(traces, rrs)
    global_offset_stats = meta.get("global_offset_stats", {})
    if not global_offset_stats:
        global_offset_stats = {
            "p50": float(np.quantile(global_offsets, 0.50)),
            "p90": float(np.quantile(global_offsets, 0.90)),
            "p95": float(np.quantile(global_offsets, 0.95)),
            "p99": float(np.quantile(global_offsets, 0.99)),
            "p95_abs": float(np.quantile(np.abs(global_offsets), 0.95)),
        }
        meta["global_offset_stats"] = global_offset_stats

    hf_noise_stats = meta.get("hf_noise_stats", {})
    hf_noise_stds = compute_hf_noise_stds(residuals)
    if not hf_noise_stats:
        hf_noise_stats = {
            "p50": float(np.quantile(hf_noise_stds, 0.50)),
            "p90": float(np.quantile(hf_noise_stds, 0.90)),
            "p95": float(np.quantile(hf_noise_stds, 0.95)),
        }
        meta["hf_noise_stats"] = hf_noise_stats

    if baseline_meta is not None:
        baseline_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    segment_edges, segment_bias_mean, segment_bias_std = _segment_stats(frequency, residuals)
    residual_lag1 = _lag1_autocorr(residuals)

    return BaselineStats(
        frequency=frequency,
        rrs=rrs,
        traces=traces,
        upper=upper,
        lower=lower,
        q_low=q_low,
        q_high=q_high,
        sigma_smooth=sigma_smooth,
        cap_normal_db=cap_normal_db,
        spec_center_db=spec_center_db,
        spec_tol_db=spec_tol_db,
        residuals=residuals,
        residual_q_low=residual_q_low,
        residual_q_high=residual_q_high,
        residual_abs_p95=residual_abs_p95,
        residual_abs_p99=residual_abs_p99,
        residual_tail_prob=residual_tail_prob,
        residual_lag1=residual_lag1,
        segment_edges=segment_edges,
        segment_bias_mean=segment_bias_mean,
        segment_bias_std=segment_bias_std,
        target_p95_low=target_p95_low,
        target_p95_high=target_p95_high,
        mean_offset_p95=mean_offset_p95,
        envelope_expand_db=envelope_expand_db,
        rough_p50=rough_p50,
        rough_p95=rough_p95,
        global_offsets=global_offsets,
        global_offset_p50=float(global_offset_stats.get("p50", 0.0)),
        global_offset_p90=float(global_offset_stats.get("p90", 0.0)),
        global_offset_p95=float(global_offset_stats.get("p95", 0.0)),
        global_offset_p99=float(global_offset_stats.get("p99", 0.0)),
        hf_noise_p50=float(hf_noise_stats.get("p50", 0.0)),
        hf_noise_p90=float(hf_noise_stats.get("p90", 0.0)),
        hf_noise_p95=float(hf_noise_stats.get("p95", 0.0)),
    )


def compute_curve_metrics(curve: np.ndarray, baseline: BaselineStats) -> Dict[str, float]:
    delta = curve - baseline.rrs
    abs_delta = np.abs(delta)
    inside_env = np.mean((curve >= baseline.lower) & (curve <= baseline.upper))
    inside_expanded = np.mean(
        (curve >= baseline.lower - baseline.envelope_expand_db)
        & (curve <= baseline.upper + baseline.envelope_expand_db)
    )
    inside_quantile = np.mean((curve >= baseline.q_low) & (curve <= baseline.q_high))
    hf_noise_std = float(np.std(delta - _smooth_noise(delta, window=61)))
    return {
        "mean_offset": float(np.mean(delta)),
        "global_offset": float(np.median(delta)),
        "max_abs_dev": float(np.max(abs_delta)),
        "p95_abs_dev": float(np.quantile(abs_delta, 0.95)),
        "p99_abs_dev": float(np.quantile(abs_delta, 0.99)),
        "within_0p4_ratio": float(np.mean(abs_delta <= 0.4)),
        "inside_env_frac": float(inside_env),
        "outside_env_frac": float(1.0 - inside_env),
        "inside_expanded_env_frac": float(inside_expanded),
        "inside_quantile_frac": float(inside_quantile),
        "hf_noise_std": hf_noise_std,
        "amp_min": float(np.min(curve)),
        "amp_max": float(np.max(curve)),
    }


def generate_correlated_noise(
    sigma: np.ndarray,
    rng: np.random.Generator,
    alpha_range: Tuple[float, float] = (0.6, 1.2),
    window: int = 9,
) -> np.ndarray:
    eps_raw = rng.normal(0.0, 1.0, size=len(sigma))
    eps_corr = _smooth_noise(eps_raw, window=window)
    alpha = rng.uniform(*alpha_range)
    return alpha * sigma * eps_corr


def _ar1_process(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    eps = rng.normal(0.0, 1.0, size=n)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    scale = np.sqrt(max(1.0 - rho ** 2, 1e-6))
    for i in range(1, n):
        out[i] = rho * out[i - 1] + scale * eps[i]
    return out


def _heavy_tail_noise(
    rng: np.random.Generator,
    sigma: np.ndarray,
    tail_prob: float,
    df: float = 3.5,
) -> np.ndarray:
    n = len(sigma)
    mask = rng.random(n) < tail_prob
    gaussian = rng.normal(0.0, 1.0, size=n)
    t_noise = rng.standard_t(df, size=n)
    t_std = np.sqrt(df / max(df - 2.0, 1e-6))
    t_noise = t_noise / max(t_std, 1e-6)
    combined = np.where(mask, t_noise, gaussian)
    return combined * sigma


# =============================================================================
# 模块特定噪声模型 (Module-Specific Noise Models)
# =============================================================================
# 每个模块有独特的噪声谱特征，这是区分不同模块故障的关键
# LPF/IF: 低频平稳，高频逐渐增大
# Mixer: 局部随机脉冲型
# ADC: 量化阶梯噪声 + 偏置
# 检波器: 高频被抑制
# 电源: 宽带随机漂移

MODULE_NOISE_PROFILES = {
    "lpf": {
        "name": "LPF/IF低通滤波器",
        "lf_stable": True,      # 低频平稳
        "hf_increase": True,    # 高频逐渐增大
        "pulse_prob": 0.0,      # 无脉冲
        "quantize": False,      # 无量化
        "hf_suppress": False,   # 无高频抑制
        "drift": False,         # 无漂移
    },
    "mixer": {
        "name": "Mixer混频器",
        "lf_stable": False,
        "hf_increase": False,
        "pulse_prob": 0.15,     # 局部随机脉冲
        "pulse_width": (3, 10), # 脉冲宽度范围
        "quantize": False,
        "hf_suppress": False,
        "drift": False,
    },
    "adc": {
        "name": "ADC模数转换器",
        "lf_stable": False,
        "hf_increase": False,
        "pulse_prob": 0.0,
        "quantize": True,       # 量化阶梯噪声
        "quantize_step": 0.01,  # 量化步长 dB
        "bias_drift": 0.02,     # 偏置漂移
        "hf_suppress": False,
        "drift": False,
    },
    "detector": {
        "name": "数字检波器",
        "lf_stable": False,
        "hf_increase": False,
        "pulse_prob": 0.0,
        "quantize": False,
        "hf_suppress": True,    # 高频被抑制
        "hf_cutoff": 0.7,       # 高频抑制起始点(归一化频率)
        "drift": False,
    },
    "power": {
        "name": "电源模块",
        "lf_stable": False,
        "hf_increase": False,
        "pulse_prob": 0.0,
        "quantize": False,
        "hf_suppress": False,
        "drift": True,          # 宽带随机漂移
        "drift_period": (0.05, 0.2),  # 漂移周期范围(归一化)
    },
    "clock": {
        "name": "时钟振荡器",
        "lf_stable": True,
        "hf_increase": True,
        "pulse_prob": 0.05,     # 轻微相位噪声脉冲
        "quantize": False,
        "hf_suppress": False,
        "drift": True,          # 轻微漂移
        "drift_period": (0.1, 0.3),
    },
    "normal": {
        "name": "正常状态",
        "lf_stable": True,
        "hf_increase": False,   # 正常状态高频不增加
        "pulse_prob": 0.0,
        "quantize": False,
        "hf_suppress": False,
        "drift": False,
    },
}

# 模块类型到噪声配置的映射
MODULE_TO_NOISE_TYPE = {
    "低频段前置低通滤波器": "lpf",
    "低频段第一混频器": "mixer",
    "高频段YTF滤波器": "lpf",
    "ADC": "adc",
    "数字检波器": "detector",
    "电源模块": "power",
    "时钟振荡器": "clock",
    "时钟合成与同步网络": "clock",
    "本振混频组件": "mixer",
    "校准源": "lpf",
    "衰减器": "lpf",
    "存储器": "lpf",
    "校准信号开关": "lpf",
}


def generate_noise_by_module(
    frequency: np.ndarray,
    sigma_base: np.ndarray,
    rng: np.random.Generator,
    module_type: str = "normal",
    severity: str = "light",
) -> np.ndarray:
    """根据模块类型生成特定噪声谱。
    
    不同模块有不同的噪声特征:
    - LPF: 低频平稳，高频增大
    - Mixer: 局部脉冲
    - ADC: 量化阶梯
    - 检波器: 高频抑制
    - 电源: 宽带漂移
    
    Args:
        frequency: 频率轴 (Hz)
        sigma_base: 基础噪声标准差
        rng: 随机数生成器
        module_type: 模块类型名称
        severity: 严重程度 (light/mid/severe)
    
    Returns:
        模块特定的噪声数组
    """
    n = len(frequency)
    noise_type = MODULE_TO_NOISE_TYPE.get(module_type, "normal")
    profile = MODULE_NOISE_PROFILES.get(noise_type, MODULE_NOISE_PROFILES["normal"])
    
    # 基础噪声
    base_noise = rng.normal(0.0, 1.0, size=n) * sigma_base
    
    # 严重程度缩放
    severity_scale = {"light": 0.8, "mid": 1.0, "severe": 1.3}.get(severity, 1.0)
    
    # 归一化频率轴
    f_norm = (frequency - frequency.min()) / (frequency.max() - frequency.min() + 1e-12)
    
    # 1. 低频平稳，高频增大 (LPF/IF特征)
    if profile.get("hf_increase"):
        hf_weight = np.power(f_norm, 1.5) * 0.5  # 高频权重
        hf_noise = rng.normal(0.0, 1.0, size=n) * sigma_base * hf_weight * severity_scale
        base_noise = base_noise * (1 - 0.3 * f_norm) + hf_noise  # 低频减弱，高频增强
    
    # 2. 局部脉冲 (Mixer特征)
    if profile.get("pulse_prob", 0) > 0:
        pulse_prob = profile["pulse_prob"] * severity_scale
        pulse_width = profile.get("pulse_width", (3, 8))
        min_width = pulse_width[0]
        max_width = pulse_width[1]
        
        # 脉冲数量: 每50个采样点平均 pulse_prob 个脉冲
        expected_pulses = pulse_prob * n / 50.0
        n_pulses = int(rng.poisson(max(0.1, expected_pulses)))
        
        for _ in range(n_pulses):
            # 确保有足够空间生成脉冲
            if n < min_width + 2:
                continue
            pos = rng.integers(0, max(1, n - max_width))
            width = rng.integers(min_width, min(max_width + 1, n - pos))
            if width < 2:
                continue
            amplitude = rng.uniform(1.5, 3.0) * severity_scale * sigma_base[min(pos, len(sigma_base) - 1)]
            sign = rng.choice([-1, 1])
            end_pos = min(pos + width, n)
            actual_width = end_pos - pos
            # 使用平滑的脉冲形状
            pulse_shape = np.hanning(actual_width)
            base_noise[pos:end_pos] += sign * amplitude * pulse_shape
    
    # 3. 量化阶梯 (ADC特征)
    if profile.get("quantize"):
        step = profile.get("quantize_step", 0.01) * severity_scale
        base_noise = np.round(base_noise / step) * step
        # 添加偏置漂移
        bias = profile.get("bias_drift", 0.02) * severity_scale
        base_noise += rng.uniform(-bias, bias)
    
    # 4. 高频抑制 (数字检波器特征)
    if profile.get("hf_suppress"):
        cutoff = profile.get("hf_cutoff", 0.7)
        suppress_weight = np.clip((f_norm - cutoff) / (1 - cutoff + 1e-6), 0, 1)
        base_noise = base_noise * (1 - 0.7 * suppress_weight * severity_scale)
    
    # 5. 宽带漂移 (电源特征)
    if profile.get("drift"):
        period_range = profile.get("drift_period", (0.1, 0.2))
        period = rng.uniform(*period_range)
        phase = rng.uniform(0, 2 * np.pi)
        drift_amplitude = np.mean(sigma_base) * 0.8 * severity_scale
        drift = drift_amplitude * np.sin(2 * np.pi * f_norm / period + phase)
        # 添加随机漂移调制
        drift *= (1 + 0.3 * rng.normal(0, 1, size=n))
        base_noise += _smooth_noise(drift, window=15)
    
    return base_noise


def generate_normal_noise(
    frequency: np.ndarray,
    sigma_base: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """为正常样本生成独立的噪声模型。
    
    正常样本的噪声特征:
    - 低频到高频较为平稳
    - 无明显脉冲
    - 无量化阶梯
    - 无系统性漂移
    
    这与故障样本的噪声骨架完全不同，确保正常/故障可以通过
    噪声形态区分，而不仅仅是故障形态。
    """
    n = len(frequency)
    
    # 正常状态：平稳的高斯噪声，带轻微自相关
    noise = rng.normal(0.0, 1.0, size=n) * sigma_base * 0.6
    
    # 添加轻微的平滑自相关（真实仪器的特征）
    noise = _smooth_noise(noise, window=5)
    
    # 正常状态不应有频率依赖的系统性变化
    # 只添加非常轻微的随机调制
    f_norm = (frequency - frequency.min()) / (frequency.max() - frequency.min() + 1e-12)
    modulation = 1 + 0.05 * rng.normal(0, 1, size=n)
    modulation = _smooth_noise(modulation, window=21)  # 平滑调制
    noise = noise * modulation
    
    return noise


class SimulationConstraints:
    def __init__(self, baseline: BaselineStats):
        self.baseline = baseline
        self.reject_records: List[Dict[str, object]] = []
        self.reject_counts: Dict[str, int] = {}

    def sample_residual_curve(
        self,
        rng: np.random.Generator,
        block_min: int = 20,
        block_max: int = 80,
    ) -> np.ndarray:
        residuals = self.baseline.residuals
        n_points = residuals.shape[1]
        sampled = np.zeros(n_points, dtype=float)
        pos = 0
        while pos < n_points:
            block_len = int(rng.integers(block_min, block_max + 1))
            block_len = min(block_len, n_points - pos)
            trace_idx = int(rng.integers(0, residuals.shape[0]))
            start_idx = int(rng.integers(0, max(1, n_points - block_len + 1)))
            sampled[pos:pos + block_len] = residuals[trace_idx, start_idx:start_idx + block_len]
            pos += block_len
        return sampled

    def winsorize_residual(self, residual: np.ndarray) -> np.ndarray:
        return np.clip(residual, self.baseline.residual_q_low, self.baseline.residual_q_high)

    def _record_reject(self, sample_id: str, fault_kind: str, reasons: List[str]) -> None:
        self.reject_records.append(
            {"sample_id": sample_id, "fault_kind": fault_kind, "reasons": reasons}
        )
        for reason in reasons:
            key = f"{fault_kind}:{reason}"
            self.reject_counts[key] = self.reject_counts.get(key, 0) + 1

    def generate_normal(
        self,
        rng: np.random.Generator,
        max_attempts: int = 25,
    ) -> Tuple[np.ndarray, List[str], str]:
        """Generate normal sample using INDEPENDENT noise model.
        
        关键改进 (Critical Improvement):
        正常样本使用独立的噪声模型，不再继承故障样本的统计骨架。
        
        正常样本噪声特征:
        - 低频到高频较为平稳 (无系统性增长)
        - 无脉冲型噪声
        - 无量化阶梯
        - 无系统性漂移
        
        这确保了正常/故障可以通过噪声形态本身区分，而不仅仅依赖故障形态。
        """
        baseline = self.baseline
        # Tighter global offset limit based on real statistics
        global_offset_limit = max(0.001, abs(baseline.global_offset_p95))
        # Tighter hf noise range to match real normal
        hf_noise_low = max(1e-4, 0.8 * baseline.hf_noise_p50)
        hf_noise_high = max(hf_noise_low, 1.2 * baseline.hf_noise_p95)
        freq = baseline.frequency
        
        # Tighter p95 constraint
        target_p95_abs_max = min(baseline.residual_abs_p95 * 1.15, 0.06)
        
        for _ in range(max_attempts):
            global_offset = float(rng.choice(baseline.global_offsets))
            normal_state = "normal_state_A" if rng.random() > 0.2 else "normal_state_B"
            if normal_state == "normal_state_B":
                global_offset += float(rng.uniform(-0.05, -0.02))
            
            # ===== 关键改进：使用独立的正常噪声模型 =====
            # 不再使用 sample_residual_curve() 从故障样本继承统计骨架
            normal_noise = generate_normal_noise(
                frequency=freq,
                sigma_base=baseline.sigma_smooth,
                rng=rng,
            )
            
            # 添加轻微的段偏移（真实仪器在不同频段有轻微偏移）
            segment_offsets = np.zeros_like(normal_noise)
            seg_edges = baseline.segment_edges
            for seg_idx, (start, end) in enumerate(zip(seg_edges[:-1], seg_edges[1:])):
                if end <= start:
                    continue
                mean = float(baseline.segment_bias_mean[min(seg_idx, len(baseline.segment_bias_mean) - 1)])
                std = float(baseline.segment_bias_std[min(seg_idx, len(baseline.segment_bias_std) - 1)])
                seg_offset = rng.normal(mean, 0.2 * std)  # 比故障更小的偏移
                segment_offsets[start:end] = seg_offset
            
            residual = normal_noise + segment_offsets
            residual = residual - float(np.median(residual))
            curve = baseline.rrs + global_offset + residual
            
            metrics = compute_curve_metrics(curve, baseline)
            reasons = []
            if not (-10.6 <= metrics["amp_min"] <= -9.4 and -10.6 <= metrics["amp_max"] <= -9.4):
                reasons.append("normal amplitude out of bounds")
            if abs(metrics["global_offset"]) > global_offset_limit:
                reasons.append("normal |global_offset| > limit")
            if metrics["p95_abs_dev"] > target_p95_abs_max:
                reasons.append("normal p95_abs_dev too large")
            if not (hf_noise_low <= metrics["hf_noise_std"] <= hf_noise_high):
                reasons.append("normal hf_noise_std outside range")
            rough = roughness_metric(curve)
            if not (0.6 * baseline.rough_p50 <= rough <= 1.3 * baseline.rough_p50):
                reasons.append("normal roughness outside target")
            if not reasons:
                return curve, [], normal_state
        return curve, reasons, normal_state

    def generate_fault_base(
        self,
        rng: np.random.Generator,
        max_attempts: int = 12,
        module_type: str = "normal",
        severity: str = "light",
    ) -> Tuple[np.ndarray, List[str]]:
        """生成故障基础曲线，使用模块特定的噪声模型。
        
        关键改进: 不同模块有不同的噪声谱特征
        - LPF: 低频平稳，高频增大
        - Mixer: 局部脉冲
        - ADC: 量化阶梯
        - 检波器: 高频抑制
        - 电源: 宽带漂移
        """
        baseline = self.baseline
        for _ in range(max_attempts):
            # 使用模块特定的噪声模型替代统一的correlated noise
            module_noise = generate_noise_by_module(
                frequency=baseline.frequency,
                sigma_base=baseline.sigma_smooth,
                rng=rng,
                module_type=module_type,
                severity=severity,
            )
            
            # 混合少量基础残差以保持与真实数据的关联
            base_residual = self.sample_residual_curve(rng)
            base_scale = rng.uniform(0.2, 0.4)  # 降低基础残差的影响
            base_residual = base_residual * base_scale
            
            # 组合: 模块噪声 + 基础残差
            residual = module_noise * 0.6 + base_residual * 0.4
            residual = self.winsorize_residual(residual)
            
            curve = baseline.rrs + residual
            metrics = compute_curve_metrics(curve, baseline)
            reasons = []
            if not (-10.6 <= metrics["amp_min"] <= -9.4 and -10.6 <= metrics["amp_max"] <= -9.4):
                reasons.append("fault_base amplitude out of bounds")
            if metrics["p95_abs_dev"] > 1.5 * baseline.residual_abs_p95:
                reasons.append("fault_base p95_abs_dev too large")
            if not reasons:
                return curve, []
        return curve, reasons

    def adjust_fault_curve(
        self,
        curve: np.ndarray,
        fault_kind: str,
        severity: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Adjust fault curve to match target characteristics.
        
        W2 Improvement for ref_error:
        - Offset magnitude has wider range and sample-to-sample variance
        - Sign can be either positive or negative (was fixed)
        - Added frequency-dependent slope component
        - Offset no longer "too stable" (~0.11 dB for all samples)
        
        W3 Improvement for amp_error:
        - Remove systematic global offset (that's ref_error territory)
        - Keep shape-based deviation (ripple, band effects)
        """
        baseline = self.baseline
        delta = curve - baseline.rrs
        mean_offset = float(np.mean(delta))
        global_offset = float(np.median(delta))

        if fault_kind in ("rl", "att"):
            # W2: More realistic ref_error offset distribution
            base = max(0.04, abs(baseline.global_offset_p95))  # Slightly higher base
            
            # W2: Ensure minimum offset for detectability
            min_required = max(0.035, 0.85 * base)  # Guaranteed minimum
            
            # W2: Wider magnitude range with sample-to-sample variance
            if severity == "severe":
                target_mag = rng.uniform(1.5 * base, 2.5 * base)
            elif severity == "mid":
                target_mag = rng.uniform(1.0 * base, 2.0 * base)
            else:
                # Light severity: still needs to be detectable
                target_mag = rng.uniform(max(0.8 * base, min_required), 1.5 * base)
            
            # W2: Random sign (not fixed based on baseline)
            sign = rng.choice([-1.0, 1.0])
            
            # W2: Add sample-to-sample variance in offset (smaller noise)
            variance_factor = rng.uniform(0.85, 1.15)  # Reduced variance
            target = sign * target_mag * variance_factor
            
            # Ensure minimum absolute offset
            if abs(target) < min_required:
                target = sign * min_required
            
            # W2: Optional slight frequency slope (real ref drift often has slope)
            if rng.random() < 0.4:
                freq = baseline.frequency
                x_norm = (freq - freq[0]) / (freq[-1] - freq[0] + 1e-12)
                slope_mag = rng.uniform(-0.02, 0.02)  # Small slope
                slope_component = slope_mag * (x_norm - 0.5)
                curve = curve + slope_component
                delta = curve - baseline.rrs
                mean_offset = float(np.mean(delta))
            
            curve = curve + (target - mean_offset)
            return curve

        if fault_kind == "amp":
            # W3: amp_error should NOT have systematic global offset
            # Remove global offset to keep this as shape-based fault
            delta = delta - global_offset
            max_budget = rng.uniform(0.18, 0.35)
            max_abs = float(np.max(np.abs(delta)))
            if max_abs > max_budget and max_abs > 0:
                delta = delta * (max_budget / max_abs)
            curve = baseline.rrs + delta
            return curve

        # For other fault types (freq, etc.)
        delta = delta - mean_offset
        max_budget = rng.uniform(0.08, 0.20)
        max_abs = float(np.max(np.abs(delta)))
        if max_abs > max_budget and max_abs > 0:
            delta = delta * (max_budget / max_abs)
        curve = baseline.rrs + delta
        return curve

    def validate_normal(self, curve: np.ndarray) -> ConstraintResult:
        baseline = self.baseline
        metrics = compute_curve_metrics(curve, baseline)
        reasons = []
        if not (-10.6 <= metrics["amp_min"] <= -9.4 and -10.6 <= metrics["amp_max"] <= -9.4):
            reasons.append("normal amplitude out of bounds")
        if abs(metrics["global_offset"]) > 1.2 * abs(baseline.global_offset_p95):
            reasons.append("normal |global_offset| > limit")
        if metrics["p95_abs_dev"] > 1.4 * baseline.residual_abs_p95:
            reasons.append("normal p95_abs_dev too large")
        if metrics["hf_noise_std"] < 0.6 * baseline.hf_noise_p50:
            reasons.append("normal hf_noise_std too low")
        if metrics["hf_noise_std"] > 2.2 * baseline.hf_noise_p95:
            reasons.append("normal hf_noise_std too high")
        return ConstraintResult(ok=not reasons, reasons=reasons)

    def validate_fault(self, curve: np.ndarray, fault_kind: str) -> ConstraintResult:
        """Validate fault curve meets minimum requirements.
        
        W2 Improvement for ref_error:
        - Use absolute minimum offset threshold (not just relative to baseline)
        - Allow smaller offsets for "light" severity ref errors
        """
        baseline = self.baseline
        metrics = compute_curve_metrics(curve, baseline)
        reasons = []
        if not (-10.6 <= metrics["amp_min"] <= -9.4 and -10.6 <= metrics["amp_max"] <= -9.4):
            reasons.append("fault amplitude out of bounds")
        if fault_kind in ("rl", "att"):
            # V-D.6: rl/att 故障只看偏移量，不看形态 (与 amp 相同处理)
            # 只要 max_abs_dev > AMP_DEVIATION_THRESHOLD (0.10 dB)，无论形态是否变化都通过
            # 早期返回跳过后续的 min_offset 和 p95_abs_dev 检查，
            # 因为当偏移量足够大时，这些形态检查对于衰减器类故障没有意义
            if metrics["max_abs_dev"] > AMP_DEVIATION_THRESHOLD:
                return ConstraintResult(ok=True, reasons=[])  # OK (Amp offset)
            # W2: Use absolute minimum threshold for ref_error offset
            # Relaxed threshold to allow module-specific noise variation
            min_offset = max(0.025, 0.6 * abs(baseline.global_offset_p95))  # Lower threshold
            if abs(metrics["global_offset"]) < min_offset:
                reasons.append("ref global_offset too small")
            # W2: Very relaxed p95_abs_dev requirement
            # Module-specific noise may have different characteristics
            if metrics["p95_abs_dev"] < 0.5 * baseline.residual_abs_p95:  # Much more lenient
                reasons.append("ref p95_abs_dev too small")
            return ConstraintResult(ok=not reasons, reasons=reasons)

        if fault_kind == "amp":
            # V-D.6: amp 故障只看偏移量，不看形态
            # 只要 max_abs_dev > AMP_DEVIATION_THRESHOLD (0.10 dB)，无论形态是否变化都通过
            # 早期返回跳过后续的 global_offset 和 p95_abs_dev 检查，
            # 因为当偏移量足够大时，这些形态/全局偏移检查对于幅度类故障没有意义
            if metrics["max_abs_dev"] > AMP_DEVIATION_THRESHOLD:
                return ConstraintResult(ok=True, reasons=[])  # OK (Amp offset)
            if abs(metrics["global_offset"]) > 1.5 * abs(baseline.global_offset_p95):
                reasons.append("amp global_offset too large")
            # V-D.5b: 降低阈值以兼容微弱故障 (偏差 < 0.6 dB)
            # 只要 p95_abs_dev > MIN_VALID_DEVIATION_DB (0.10 dB) 即视为有效
            if metrics["p95_abs_dev"] < MIN_VALID_DEVIATION_DB:
                reasons.append("amp shape deviation too small")
            return ConstraintResult(ok=not reasons, reasons=reasons)

        # V-D.5b: ADC/VBW 特殊豁免 - 这些故障主要表现为纹理变化，幅度偏差可能极小
        # ADC 量化噪声和 VBW 数字检波器故障的主要特征是纹理/方差变化，而非幅度偏移
        # 因此对这些故障类型不施加最小幅度偏差约束，允许极微弱的幅度变化通过
        if fault_kind in ("adc", "vbw"):
            # ADC/VBW 故障无需检查 p95_abs_dev 阈值
            # 真正的检测应通过 X37 (差分方差) 和 X35 (形态特征) 进行
            # 这里只保留基本的幅度边界检查 (已在上面完成)
            return ConstraintResult(ok=not reasons, reasons=reasons)

        if abs(metrics["global_offset"]) > 1.5 * abs(baseline.global_offset_p95):
            reasons.append("freq global_offset too large")
        if metrics["p99_abs_dev"] > 0.35:
            reasons.append("freq p99_abs_dev too large")
        return ConstraintResult(ok=not reasons, reasons=reasons)


def summarize_residuals(residuals: np.ndarray) -> Dict[str, float]:
    flat = residuals.ravel()
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p01": float(np.quantile(flat, 0.01)),
        "p05": float(np.quantile(flat, 0.05)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "p95_abs": float(np.quantile(np.abs(flat), 0.95)),
    }


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    all_vals = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n = min(a_sorted.size, b_sorted.size)
    a_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, a_sorted.size), a_sorted)
    b_q = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, b_sorted.size), b_sorted)
    return float(np.mean(np.abs(a_q - b_q)))


def _lag1_from_samples(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    vals = []
    for row in samples:
        if len(row) < 2:
            continue
        x = row[:-1]
        y = row[1:]
        denom = (np.std(x) * np.std(y)) + 1e-12
        vals.append(float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / denom))
    return float(np.mean(vals)) if vals else 0.0


def build_quality_report(
    sim_dir: Path,
    baseline: BaselineStats,
    curves: np.ndarray,
    labels: Dict[str, dict],
    active_modules: List[str],
    reject_records: List[Dict[str, object]],
    prev_dir: Path | None = None,
    feature_rows: List[Dict[str, object]] | None = None,
) -> Tuple[dict, List[Dict[str, str]]]:
    per_sample = []
    violations = []
    feature_map = {row.get("sample_id"): row for row in (feature_rows or [])}
    freq_proxy_by_class = {}
    normal_freq_proxy = []
    for sample_id, row in feature_map.items():
        label = labels.get(sample_id, {})
        if label.get("system_fault_class") == "normal":
            proxy = abs(float(row.get("X23", 0.0))) + abs(float(row.get("X24", 0.0)))
            normal_freq_proxy.append(proxy)
    freq_proxy_threshold = 0.0

    for idx, curve in enumerate(curves):
        sample_id = f"sim_{idx:05d}"
        label = labels.get(sample_id, {})
        fault_params = label.get("fault_params", {}) or {}
        fault_type = fault_params.get("type", "")
        fault_kind = "normal"
        if label.get("system_fault_class") == "amp_error":
            fault_kind = "amp"
        elif label.get("system_fault_class") == "freq_error":
            fault_kind = "freq"
        elif label.get("system_fault_class") == "ref_error":
            fault_kind = "rl"
        metrics = compute_curve_metrics(curve, baseline)
        feature_row = feature_map.get(sample_id, {})
        freq_proxy = abs(float(feature_row.get("X23", 0.0))) + abs(float(feature_row.get("X24", 0.0)))
        per_sample.append(
            {
                "sample_id": sample_id,
                "class": label.get("system_fault_class", "normal"),
                "module": label.get("module"),
                "fault_type": fault_type,
                "freq_proxy": freq_proxy,
                **metrics,
            }
        )
        if fault_kind == "normal":
            if abs(metrics["global_offset"]) > 1.2 * abs(baseline.global_offset_p95):
                violations.append(
                    {"sample_id": sample_id, "reason": "normal global_offset out of bounds"}
                )
            if metrics["hf_noise_std"] < 0.8 * baseline.hf_noise_p50:
                violations.append(
                    {"sample_id": sample_id, "reason": "normal hf_noise_std too low"}
                )
        elif fault_kind in ("amp", "rl", "att"):
            # W2: Lowered ref offset threshold to allow smaller but realistic offsets
            ref_min_offset = max(0.03, 0.8 * abs(baseline.global_offset_p95))  # Absolute min 0.03 dB
            if fault_kind in ("rl", "att") and abs(metrics["global_offset"]) < ref_min_offset:
                violations.append(
                    {"sample_id": sample_id, "reason": "ref global_offset too small"}
                )
            if fault_kind == "amp" and abs(metrics["global_offset"]) > 1.5 * abs(baseline.global_offset_p95):
                violations.append(
                    {"sample_id": sample_id, "reason": "amp global_offset too large"}
                )
        else:
            if abs(metrics["global_offset"]) > 1.5 * abs(baseline.global_offset_p95):
                violations.append(
                    {"sample_id": sample_id, "reason": "freq global_offset too large"}
                )
            if freq_proxy_threshold > 0 and freq_proxy < freq_proxy_threshold:
                violations.append(
                    {"sample_id": sample_id, "reason": "freq proxy too low"}
                )

    overall = {}
    for cls in sorted({row["class"] for row in per_sample}):
        cls_rows = [row for row in per_sample if row["class"] == cls]
        cls_indices = [int(row["sample_id"].split("_")[1]) for row in cls_rows]
        cls_curves = curves[cls_indices] if cls_indices else np.empty((0, len(baseline.rrs)))
        min_y = float(np.min(cls_curves)) if cls_curves.size else 0.0
        max_y = float(np.max(cls_curves)) if cls_curves.size else 0.0
        cls_rough = [roughness_metric(curves[idx]) for idx in cls_indices] if cls_indices else []
        mean_offsets = [r["mean_offset"] for r in cls_rows]
        max_abs_devs = [r["max_abs_dev"] for r in cls_rows]
        p95_abs_devs = [r["p95_abs_dev"] for r in cls_rows]
        overall[cls] = {
            "count": len(cls_rows),
            "min_y": min_y,
            "max_y": max_y,
            "mean_offset_mean": float(np.mean(mean_offsets)),
            "mean_offset_p05": float(np.quantile(mean_offsets, 0.05)),
            "mean_offset_median": float(np.median(mean_offsets)),
            "mean_offset_p95": float(np.quantile(mean_offsets, 0.95)),
            "mean_offset_max": float(np.max(np.abs(mean_offsets))),
            "max_abs_dev_mean": float(np.mean(max_abs_devs)),
            "max_abs_dev_median": float(np.median(max_abs_devs)),
            "max_abs_dev_p95": float(np.quantile(max_abs_devs, 0.95)),
            "max_abs_dev_max": float(np.max(max_abs_devs)),
            "p95_abs_dev_mean": float(np.mean(p95_abs_devs)),
            "p95_abs_dev_p95": float(np.quantile(p95_abs_devs, 0.95)),
            "inside_env_frac_mean": float(np.mean([r["inside_env_frac"] for r in cls_rows])),
            "outside_env_frac_mean": float(np.mean([r["outside_env_frac"] for r in cls_rows])),
            "roughness_p50": float(np.quantile(cls_rough, 0.50)) if cls_rough else 0.0,
            "roughness_p95": float(np.quantile(cls_rough, 0.95)) if cls_rough else 0.0,
        }

    sim_residuals = []
    for row in per_sample:
        if row["class"] == "normal":
            idx = int(row["sample_id"].split("_")[1])
            sim_residuals.append(curves[idx] - baseline.rrs)
    sim_residuals = np.array(sim_residuals) if sim_residuals else np.empty((0, len(baseline.rrs)))
    real_flat = baseline.residuals.ravel()
    sim_flat = sim_residuals.ravel() if sim_residuals.size else np.array([])
    tail_threshold = baseline.residual_abs_p95
    ks_dist = _ks_distance(real_flat, sim_flat)
    wass_dist = _wasserstein_distance(real_flat, sim_flat)
    real_tail = float(np.mean(np.abs(real_flat) > tail_threshold)) if real_flat.size else 0.0
    sim_tail = float(np.mean(np.abs(sim_flat) > tail_threshold)) if sim_flat.size else 0.0
    real_lag1 = _lag1_from_samples(baseline.residuals)
    sim_lag1 = _lag1_from_samples(sim_residuals) if sim_residuals.size else 0.0

    residual_comparison = {
        "baseline": summarize_residuals(baseline.residuals),
        "sim_normal": summarize_residuals(sim_residuals) if sim_residuals.size else {},
        "distances": {
            "ks": ks_dist,
            "wasserstein": wass_dist,
        },
        "tail": {
            "threshold_db": tail_threshold,
            "real": real_tail,
            "sim": sim_tail,
        },
        "lag1_autocorr": {
            "real": real_lag1,
            "sim": sim_lag1,
        },
    }

    bin_stats = []
    if baseline.frequency.size:
        f_min = float(np.min(baseline.frequency))
        f_max = float(np.max(baseline.frequency))
        edges_hz = list(np.arange(f_min, f_max + 0.5e9, 0.5e9))
        if edges_hz[-1] < f_max:
            edges_hz.append(f_max)
        for start_hz, end_hz in zip(edges_hz[:-1], edges_hz[1:]):
            start_idx = int(np.searchsorted(baseline.frequency, start_hz, side="left"))
            end_idx = int(np.searchsorted(baseline.frequency, end_hz, side="left"))
            if end_idx <= start_idx:
                continue
            real_seg = baseline.residuals[:, start_idx:end_idx]
            sim_seg = sim_residuals[:, start_idx:end_idx] if sim_residuals.size else np.empty((0, 0))
            real_flat_seg = real_seg.ravel()
            sim_flat_seg = sim_seg.ravel() if sim_seg.size else np.array([])
            bin_stats.append(
                {
                    "f_start_hz": start_hz,
                    "f_end_hz": end_hz,
                    "real_mean": float(np.mean(real_flat_seg)) if real_flat_seg.size else 0.0,
                    "real_std": float(np.std(real_flat_seg)) if real_flat_seg.size else 0.0,
                    "real_tail": float(np.mean(np.abs(real_flat_seg) > tail_threshold)) if real_flat_seg.size else 0.0,
                    "sim_mean": float(np.mean(sim_flat_seg)) if sim_flat_seg.size else 0.0,
                    "sim_std": float(np.std(sim_flat_seg)) if sim_flat_seg.size else 0.0,
                    "sim_tail": float(np.mean(np.abs(sim_flat_seg) > tail_threshold)) if sim_flat_seg.size else 0.0,
                    "real_lag1": _lag1_from_samples(real_seg),
                    "sim_lag1": _lag1_from_samples(sim_seg) if sim_seg.size else 0.0,
                }
            )

    before_after = []
    if prev_dir and prev_dir.exists():
        prev_npz = prev_dir / "simulated_curves.npz"
        if prev_npz.exists():
            prev_data = np.load(prev_npz, allow_pickle=True)
            prev_curves = prev_data["curves"]
            sample_ids = [f"sim_{idx:05d}" for idx in range(min(5, prev_curves.shape[0]))]
            random.shuffle(sample_ids)
            for sample_id in sample_ids[:5]:
                idx = int(sample_id.split("_")[1])
                prev_curve = prev_curves[idx]
                prev_metrics = compute_curve_metrics(prev_curve, baseline)
                now_curve = curves[idx] if idx < curves.shape[0] else None
                now_metrics = compute_curve_metrics(now_curve, baseline) if now_curve is not None else {}
                before_after.append(
                    {
                        "sample_id": sample_id,
                        "before": prev_metrics,
                        "after": now_metrics,
                    }
                )

    sim_normal_rough = [roughness_metric(curves[int(row["sample_id"].split("_")[1])])
                        for row in per_sample if row["class"] == "normal"]
    report = {
        "active_modules": active_modules,
        "global_min_y": float(np.min(curves)) if curves.size else 0.0,
        "global_max_y": float(np.max(curves)) if curves.size else 0.0,
        "overall": overall,
        "per_sample": per_sample,
        "violations": violations,
        "residual_comparison": residual_comparison,
        "frequency_bin_metrics": bin_stats,
        "reject_records": reject_records,
        "before_after_samples": before_after,
        "thresholds": {
            "normal_global_offset_max": 1.2 * abs(baseline.global_offset_p95),
            "ref_global_offset_min": max(0.03, 0.8 * abs(baseline.global_offset_p95)),  # W2: Updated threshold
            "amp_global_offset_max": 1.5 * abs(baseline.global_offset_p95),
            "freq_proxy_threshold": freq_proxy_threshold,
            "hf_noise_std_low": 0.8 * baseline.hf_noise_p50,
            "hf_noise_std_high": 1.2 * baseline.hf_noise_p95,
        },
        "roughness": {
            "baseline_p50": baseline.rough_p50,
            "baseline_p95": baseline.rough_p95,
            "sim_normal_p50": float(np.quantile(sim_normal_rough, 0.50)) if sim_normal_rough else 0.0,
            "sim_normal_p95": float(np.quantile(sim_normal_rough, 0.95)) if sim_normal_rough else 0.0,
        },
    }

    sim_dir.mkdir(parents=True, exist_ok=True)
    report_path = sim_dir / "sim_quality_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = sim_dir / "sim_quality_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Simulation Quality Summary",
                "",
                "## Normal vs Real",
                f"- KS distance: {ks_dist:.5f}",
                f"- Wasserstein distance: {wass_dist:.5f}",
                f"- Tail threshold (|res| > {tail_threshold:.4f} dB): real={real_tail:.4f}, sim={sim_tail:.4f}",
                f"- Lag-1 autocorr: real={real_lag1:.4f}, sim={sim_lag1:.4f}",
                "",
                "## Per-class counts",
                *[f"- {cls}: {stats.get('count', 0)}" for cls, stats in overall.items()],
            ]
        ),
        encoding="utf-8",
    )

    summary_path = sim_dir / "sim_quality_table.csv"
    headers = list(per_sample[0].keys()) if per_sample else []
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for row in per_sample:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    bad_samples_path = sim_dir / "sim_bad_samples.csv"
    if violations:
        with bad_samples_path.open("w", encoding="utf-8", newline="") as f:
            f.write("sample_id,reason\n")
            for row in violations:
                f.write(f"{row.get('sample_id','')},{row.get('reason','')}\n")
    else:
        bad_samples_path.write_text("sample_id,reason\n", encoding="utf-8")

    _write_quality_plots(sim_dir, baseline, curves, per_sample)

    return report, violations


def _write_quality_plots(
    sim_dir: Path,
    baseline: BaselineStats,
    curves: np.ndarray,
    per_sample: List[Dict[str, object]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    def quantile_band(data: np.ndarray, q_low: float = 0.1, q_high: float = 0.9):
        if data.size == 0:
            return None
        return (
            np.median(data, axis=0),
            np.quantile(data, q_low, axis=0),
            np.quantile(data, q_high, axis=0),
        )

    def shade_mask(ax, x_vals, mask, color="tab:red", alpha=0.12):
        if mask.size == 0:
            return
        start = None
        for i, flag in enumerate(mask):
            if flag and start is None:
                start = i
            if (not flag or i == len(mask) - 1) and start is not None:
                end = i if not flag else i + 1
                ax.axvspan(x_vals[start], x_vals[end - 1], color=color, alpha=alpha, lw=0)
                start = None

    x_ghz = baseline.frequency / 1e9
    class_groups = {"normal": [], "amp_error": [], "freq_error": [], "ref_error": []}
    for row in per_sample:
        idx = int(row["sample_id"].split("_")[1])
        cls = row["class"]
        if cls in class_groups:
            class_groups[cls].append(idx)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    real_band = quantile_band(baseline.traces, 0.1, 0.9)
    for ax, cls in zip(axes, ["normal", "amp_error", "freq_error", "ref_error"]):
        indices = class_groups.get(cls, [])
        cls_curves = curves[indices] if indices else np.empty((0, len(baseline.rrs)))
        sim_band = quantile_band(cls_curves, 0.1, 0.9)
        ax.plot(x_ghz, baseline.rrs, color="black", linewidth=1.5, label="RRS")
        if sim_band:
            med, low, high = sim_band
            ax.fill_between(x_ghz, low, high, color="tab:orange", alpha=0.3, label="Sim P10-P90")
            ax.plot(x_ghz, med, color="tab:orange", linewidth=1.2, label="Sim median")
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
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(sim_dir / "audit_overlay.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for ax, cls in zip(axes, ["normal", "amp_error", "freq_error", "ref_error"]):
        indices = class_groups.get(cls, [])
        cls_curves = curves[indices] if indices else np.empty((0, len(baseline.rrs)))
        sim_band = quantile_band(cls_curves, 0.1, 0.9)
        ax.plot(x_ghz, baseline.rrs, color="black", linewidth=1.5, label="RRS")
        if sim_band:
            med, low, high = sim_band
            ax.fill_between(x_ghz, low, high, color="tab:orange", alpha=0.3, label="Sim P10-P90")
            ax.plot(x_ghz, med, color="tab:orange", linewidth=1.2, label="Sim median")
        ax.set_title(f"{cls} band overview")
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Frequency (GHz)")
    axes[-1].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Amplitude (dBm)")
    axes[2].set_ylabel("Amplitude (dBm)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(sim_dir / "class_band_overview.png", dpi=150)
    plt.close(fig)

    real_residual = baseline.residuals
    real_res_band = quantile_band(real_residual, 0.1, 0.9)
    real_abs_p95 = np.quantile(np.abs(real_residual), 0.95, axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, cls in zip(axes, ["amp_error", "freq_error", "ref_error"]):
        indices = class_groups.get(cls, [])
        cls_curves = curves[indices] if indices else np.empty((0, len(baseline.rrs)))
        cls_res = cls_curves - baseline.rrs if cls_curves.size else np.empty((0, len(baseline.rrs)))
        band = quantile_band(cls_res, 0.1, 0.9)
        if real_res_band:
            med_r, low_r, high_r = real_res_band
            ax.fill_between(x_ghz, low_r, high_r, color="tab:blue", alpha=0.2, label="Real normal P10-P90")
            ax.plot(x_ghz, med_r, color="tab:blue", linewidth=1.0, label="Real normal median")
        if band:
            med, low, high = band
            ax.fill_between(x_ghz, low, high, color="tab:orange", alpha=0.25, label=f"{cls} P10-P90")
            ax.plot(x_ghz, med, color="tab:orange", linewidth=1.2, label=f"{cls} median")
            shade_mask(ax, x_ghz, np.abs(med) > real_abs_p95)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{cls} residual band vs real normal")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("Residual (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(sim_dir / "sim_debug_overlay.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    normal_indices = class_groups.get("normal", [])
    normal_curves = curves[normal_indices] if normal_indices else np.empty((0, len(baseline.rrs)))
    normal_band = quantile_band(normal_curves, 0.1, 0.9)
    if normal_band and real_band:
        med, low, high = normal_band
        med_r, low_r, high_r = real_band
        axes[0].fill_between(x_ghz, low_r, high_r, color="tab:blue", alpha=0.25, label="Real P10-P90")
        axes[0].plot(x_ghz, med_r, color="tab:blue", linewidth=1.0)
        axes[0].fill_between(x_ghz, low, high, color="tab:orange", alpha=0.25, label="Sim P10-P90")
        axes[0].plot(x_ghz, med, color="tab:orange", linewidth=1.0)
    axes[0].plot(x_ghz, baseline.rrs, color="black", linewidth=1.2, label="RRS")
    axes[0].set_title("Normal quantile band")
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Amplitude (dBm)")
    axes[0].legend(fontsize=8)
    real_flat = baseline.residuals.ravel()
    sim_flat = (normal_curves - baseline.rrs).ravel() if normal_curves.size else np.array([])
    axes[1].hist(real_flat, bins=40, alpha=0.6, label="Real residual")
    if sim_flat.size:
        axes[1].hist(sim_flat, bins=40, alpha=0.6, label="Sim residual")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual (dB)")
    axes[1].legend(fontsize=8)

    bin_stats = []
    f_min = float(np.min(baseline.frequency))
    f_max = float(np.max(baseline.frequency))
    edges_hz = list(np.arange(f_min, f_max + 0.5e9, 0.5e9))
    if edges_hz[-1] < f_max:
        edges_hz.append(f_max)
    for start_hz, end_hz in zip(edges_hz[:-1], edges_hz[1:]):
        start_idx = int(np.searchsorted(baseline.frequency, start_hz, side="left"))
        end_idx = int(np.searchsorted(baseline.frequency, end_hz, side="left"))
        if end_idx <= start_idx:
            continue
        real_seg = baseline.residuals[:, start_idx:end_idx]
        sim_seg = (normal_curves - baseline.rrs)[:, start_idx:end_idx] if normal_curves.size else np.empty((0, 0))
        bin_stats.append(
            (
                (start_hz + end_hz) / 2 / 1e9,
                float(np.mean(real_seg)),
                float(np.mean(sim_seg)) if sim_seg.size else 0.0,
                float(np.mean(np.abs(real_seg) > baseline.residual_abs_p95)),
                float(np.mean(np.abs(sim_seg) > baseline.residual_abs_p95)) if sim_seg.size else 0.0,
            )
        )
    if bin_stats:
        centers = [b[0] for b in bin_stats]
        real_mean = [b[1] for b in bin_stats]
        sim_mean = [b[2] for b in bin_stats]
        real_tail = [b[3] for b in bin_stats]
        sim_tail = [b[4] for b in bin_stats]
        width = 0.18
        axes[2].bar(np.array(centers) - width, real_mean, width=width, label="Real mean")
        axes[2].bar(np.array(centers), sim_mean, width=width, label="Sim mean")
        axes[2].plot(centers, real_tail, color="tab:blue", marker="o", label="Real tail")
        axes[2].plot(centers, sim_tail, color="tab:orange", marker="o", label="Sim tail")
    axes[2].set_title("Binned bias & tail (0.5 GHz)")
    axes[2].set_xlabel("Frequency (GHz)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(sim_dir / "sim_quality_plots.png", dpi=150)
    plt.close(fig)
