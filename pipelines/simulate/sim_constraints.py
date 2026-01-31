#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation constraints and quality reporting for RRS-centered data.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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
        max_attempts: int = 12,
    ) -> Tuple[np.ndarray, List[str]]:
        baseline = self.baseline
        global_offset_limit = max(0.001, 1.2 * abs(baseline.global_offset_p95))
        hf_noise_low = max(1e-4, 0.8 * baseline.hf_noise_p50)
        hf_noise_high = max(hf_noise_low, 1.2 * baseline.hf_noise_p95)
        for _ in range(max_attempts):
            global_offset = float(rng.choice(baseline.global_offsets))
            slow_amp = min(0.25, 0.8 * baseline.residual_abs_p95)
            x = np.linspace(0.0, 1.0, len(baseline.rrs))
            slow_drift = (
                rng.uniform(-1.0, 1.0) * (x - 0.5)
                + rng.uniform(-0.5, 0.5) * (x - 0.5) ** 2
                + rng.uniform(0.6, 1.0) * np.sin(2 * np.pi * (rng.uniform(0.4, 1.2) * x + rng.random()))
            )
            slow_drift = slow_drift / (np.max(np.abs(slow_drift)) + 1e-6)
            slow_drift = slow_drift * rng.uniform(0.2, 0.6) * slow_amp
            hf_std = float(rng.uniform(hf_noise_low, hf_noise_high))
            hf_noise = rng.normal(0.0, hf_std, size=len(baseline.rrs))
            hf_noise = _smooth_noise(hf_noise, window=5)
            curve = baseline.rrs + global_offset + slow_drift + hf_noise
            metrics = compute_curve_metrics(curve, baseline)
            reasons = []
            if not (-10.6 <= metrics["amp_min"] <= -9.4 and -10.6 <= metrics["amp_max"] <= -9.4):
                reasons.append("normal amplitude out of bounds")
            if abs(metrics["global_offset"]) > global_offset_limit:
                reasons.append("normal |global_offset| > limit")
            if metrics["p95_abs_dev"] > 1.3 * baseline.residual_abs_p95:
                reasons.append("normal p95_abs_dev too large")
            if not (hf_noise_low <= metrics["hf_noise_std"] <= hf_noise_high):
                reasons.append("normal hf_noise_std outside range")
            rough = roughness_metric(curve)
            if not (0.7 * baseline.rough_p50 <= rough <= 1.3 * baseline.rough_p50):
                reasons.append("normal roughness outside target")
            if not reasons:
                return curve, []
        return curve, reasons

    def generate_fault_base(
        self,
        rng: np.random.Generator,
        max_attempts: int = 12,
    ) -> Tuple[np.ndarray, List[str]]:
        baseline = self.baseline
        for _ in range(max_attempts):
            base_residual = self.sample_residual_curve(rng)
            scale = rng.uniform(0.85, 1.15)
            residual = self.winsorize_residual(base_residual * scale)
            curve = baseline.rrs + residual
            jitter = generate_correlated_noise(
                baseline.sigma_smooth, rng, alpha_range=(0.04, 0.12)
            )
            curve = curve + jitter
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
        baseline = self.baseline
        delta = curve - baseline.rrs
        mean_offset = float(np.mean(delta))
        global_offset = float(np.median(delta))

        if fault_kind in ("rl", "att"):
            base = max(0.04, abs(baseline.global_offset_p95))
            if severity == "severe":
                target = rng.uniform(3.2 * base, 4.2 * base)
            elif severity == "mid":
                target = rng.uniform(2.4 * base, 3.4 * base)
            else:
                target = rng.uniform(2.0 * base, 2.8 * base)
            target = target * (-1 if rng.random() < 0.5 else 1)
            curve = curve + (target - mean_offset)
            return curve

        if fault_kind == "amp":
            delta = delta - global_offset
            max_budget = rng.uniform(0.18, 0.35)
            max_abs = float(np.max(np.abs(delta)))
            if max_abs > max_budget and max_abs > 0:
                delta = delta * (max_budget / max_abs)
            curve = baseline.rrs + delta
            return curve

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
        if metrics["p95_abs_dev"] > 1.3 * baseline.residual_abs_p95:
            reasons.append("normal p95_abs_dev too large")
        if metrics["hf_noise_std"] < 0.8 * baseline.hf_noise_p50:
            reasons.append("normal hf_noise_std too low")
        if metrics["hf_noise_std"] > 1.2 * baseline.hf_noise_p95:
            reasons.append("normal hf_noise_std too high")
        return ConstraintResult(ok=not reasons, reasons=reasons)

    def validate_fault(self, curve: np.ndarray, fault_kind: str) -> ConstraintResult:
        baseline = self.baseline
        metrics = compute_curve_metrics(curve, baseline)
        reasons = []
        if not (-10.6 <= metrics["amp_min"] <= -9.4 and -10.6 <= metrics["amp_max"] <= -9.4):
            reasons.append("fault amplitude out of bounds")
        if fault_kind in ("rl", "att"):
            if abs(metrics["global_offset"]) < 2.0 * abs(baseline.global_offset_p95):
                reasons.append("ref global_offset too small")
            if metrics["p95_abs_dev"] < baseline.residual_abs_p95:
                reasons.append("ref p95_abs_dev too small")
            return ConstraintResult(ok=not reasons, reasons=reasons)

        if fault_kind == "amp":
            if abs(metrics["global_offset"]) > 1.5 * abs(baseline.global_offset_p95):
                reasons.append("amp global_offset too large")
            if metrics["p95_abs_dev"] < baseline.residual_abs_p95:
                reasons.append("amp shape deviation too small")
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
            if fault_kind in ("rl", "att") and abs(metrics["global_offset"]) < 2.0 * abs(baseline.global_offset_p95):
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

    residual_comparison = {
        "baseline": summarize_residuals(baseline.residuals),
        "sim_normal": summarize_residuals(sim_residuals) if sim_residuals.size else {},
    }

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
        "reject_records": reject_records,
        "before_after_samples": before_after,
        "thresholds": {
            "normal_global_offset_max": 1.2 * abs(baseline.global_offset_p95),
            "ref_global_offset_min": 2.0 * abs(baseline.global_offset_p95),
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

    normal_indices = [int(row["sample_id"].split("_")[1]) for row in per_sample if row["class"] == "normal"]
    rng = np.random.default_rng(2025)
    rng.shuffle(normal_indices)
    sample_ids = normal_indices[:10]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_ghz = baseline.frequency / 1e9
    real_indices = list(range(min(10, baseline.traces.shape[0])))
    rng.shuffle(real_indices)
    for idx in real_indices[:10]:
        axes[0].plot(x_ghz, baseline.traces[idx], color="tab:blue", alpha=0.25)
    for idx in sample_ids:
        axes[0].plot(x_ghz, curves[idx], color="tab:orange", alpha=0.6)
    axes[0].plot(x_ghz, baseline.rrs, color="black", linewidth=1.5, label="RRS")
    axes[0].set_title("Real Normal (blue) vs Sim Normal (orange)")
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Amplitude (dBm)")
    axes[0].legend()

    real_res = baseline.residuals.ravel()
    sim_res = []
    for idx in normal_indices:
        sim_res.append(curves[idx] - baseline.rrs)
    sim_res = np.array(sim_res).ravel() if sim_res else np.array([])
    axes[1].hist(real_res, bins=40, alpha=0.6, label="Real residual")
    if sim_res.size:
        axes[1].hist(sim_res, bins=40, alpha=0.6, label="Sim residual")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual (dB)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(sim_dir / "sim_quality_plots.png", dpi=150)
    plt.close(fig)

    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        x_ghz = baseline.frequency / 1e9
        ax.plot(x_ghz, baseline.rrs, color="black", linewidth=1.5, label="RRS")
        ax.plot(x_ghz, baseline.upper, color="tab:green", linewidth=1.0, alpha=0.6, label="Upper")
        ax.plot(x_ghz, baseline.lower, color="tab:green", linewidth=1.0, alpha=0.6, label="Lower")

        normal_indices = [int(row["sample_id"].split("_")[1]) for row in per_sample if row["class"] == "normal"]
        fault_indices = [int(row["sample_id"].split("_")[1]) for row in per_sample if row["class"] != "normal"]
        rng = np.random.default_rng(2025)
        rng.shuffle(normal_indices)
        rng.shuffle(fault_indices)
        for idx in normal_indices[:20]:
            ax.plot(x_ghz, curves[idx], color="tab:blue", alpha=0.3)
        for idx in fault_indices[:20]:
            ax.plot(x_ghz, curves[idx], color="tab:red", alpha=0.3)

        normal_offsets = [row["global_offset"] for row in per_sample if row["class"] == "normal"]
        if normal_offsets:
            p50 = float(np.quantile(normal_offsets, 0.50))
            p95 = float(np.quantile(np.abs(normal_offsets), 0.95))
            ax.text(
                0.02,
                0.95,
                f"normal global_offset P50={p50:.4f}, P95_abs={p95:.4f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )
        ax.set_title("Sim Debug Overlay (Normal vs Fault)")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Amplitude (dBm)")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(sim_dir / "sim_debug_overlay.png", dpi=150)
        plt.close(fig)
    except Exception:
        return
