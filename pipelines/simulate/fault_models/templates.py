from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TemplateResult:
    curve: np.ndarray
    template_id: str
    params: Dict[str, float]
    peak_track_type: str


def _severity_range(severity: str, light: Tuple[float, float], mid: Tuple[float, float], severe: Tuple[float, float]) -> Tuple[float, float]:
    if severity == "severe":
        return severe
    if severity == "mid":
        return mid
    return light


def _norm_freq(frequency: np.ndarray) -> np.ndarray:
    f_min = float(np.min(frequency))
    f_max = float(np.max(frequency))
    denom = max(f_max - f_min, 1.0)
    return (frequency - f_min) / denom


def _template_smooth_shift(
    base: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    offset_range = _severity_range(severity, (0.04, 0.10), (0.10, 0.20), (0.20, 0.34))
    offset = float(rng.uniform(*offset_range)) * (-1 if rng.random() < 0.5 else 1)
    curve = base + offset
    return TemplateResult(curve=curve, template_id="T1", params={"offset_db": offset}, peak_track_type="none")


def _template_tilt_rolloff(
    base: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    slope_range = _severity_range(severity, (-0.08, -0.03), (-0.14, -0.06), (-0.20, -0.10))
    slope = float(rng.uniform(*slope_range))
    f_norm = _norm_freq(frequency) - 0.5
    curve = base + slope * f_norm
    return TemplateResult(curve=curve, template_id="T2", params={"slope_db": slope}, peak_track_type="none")


def _template_stable_ripple(
    base: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    amp_range = _severity_range(severity, (0.03, 0.07), (0.06, 0.12), (0.10, 0.18))
    period_range = _severity_range(severity, (0.08, 0.16), (0.06, 0.12), (0.04, 0.10))
    amplitude = float(rng.uniform(*amp_range))
    period = float(rng.uniform(*period_range))
    f_norm = _norm_freq(frequency)
    phase = float(rng.uniform(0, 2 * np.pi))
    curve = base + amplitude * np.sin(2 * np.pi * f_norm / period + phase)
    return TemplateResult(
        curve=curve,
        template_id="T3",
        params={"ripple_amp_db": amplitude, "period_norm": period},
        peak_track_type="none",
    )


def _template_step_fixed(
    base: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    step_range = _severity_range(severity, (0.05, 0.12), (0.10, 0.20), (0.18, 0.30))
    step = float(rng.uniform(*step_range)) * (-1 if rng.random() < 0.5 else 1)
    idx = int(rng.integers(low=int(0.25 * len(frequency)), high=int(0.75 * len(frequency))))
    curve = base.copy()
    curve[idx:] = curve[idx:] + step
    return TemplateResult(
        curve=curve,
        template_id="T4",
        params={"step_db": step, "step_index": float(idx)},
        peak_track_type="none",
    )


def _template_spike_sparse(
    base: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    spike_count_range = _severity_range(severity, (4, 8), (8, 14), (14, 22))
    spike_amp_range = _severity_range(severity, (0.08, 0.15), (0.12, 0.22), (0.18, 0.30))
    spike_count = int(rng.integers(spike_count_range[0], spike_count_range[1] + 1))
    spike_amp = float(rng.uniform(*spike_amp_range))
    indices = rng.choice(len(base), size=spike_count, replace=False)
    curve = base.copy()
    curve[indices] = curve[indices] + spike_amp * rng.choice([-1, 1], size=spike_count)
    return TemplateResult(
        curve=curve,
        template_id="T5",
        params={"spike_count": float(spike_count), "spike_amp_db": spike_amp},
        peak_track_type="none",
    )


def _template_scatter_thick(
    base: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    noise_range = _severity_range(severity, (0.02, 0.05), (0.04, 0.08), (0.06, 0.12))
    noise_std = float(rng.uniform(*noise_range))
    noise = rng.normal(0.0, noise_std, size=len(base))
    curve = base + noise
    return TemplateResult(
        curve=curve,
        template_id="T6",
        params={"noise_std": noise_std},
        peak_track_type="none",
    )


def _template_quant_grain(
    base: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    step_range = _severity_range(severity, (0.008, 0.015), (0.012, 0.022), (0.018, 0.030))
    step = float(rng.uniform(*step_range))
    residual = base - rrs
    quantized = np.round(residual / step) * step
    curve = rrs + quantized
    return TemplateResult(
        curve=curve,
        template_id="T7",
        params={"quant_step_db": step},
        peak_track_type="none",
    )


def _template_peak_track(
    base: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    track_type = rng.choice(["spike", "dense", "hole"], p=[0.35, 0.45, 0.20])
    return TemplateResult(
        curve=base,
        template_id="T8",
        params={"track_type": track_type},
        peak_track_type=track_type,
    )


def _template_band_offset(
    base: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    offset_range = _severity_range(severity, (0.06, 0.12), (0.10, 0.18), (0.14, 0.24))
    width_range = _severity_range(severity, (0.2, 0.5), (0.3, 0.7), (0.4, 0.9))
    f_norm = _norm_freq(frequency)
    center = float(rng.uniform(0.2, 0.8))
    width = float(rng.uniform(*width_range))
    start = max(0.0, center - width / 2)
    end = min(1.0, center + width / 2)
    mask = (f_norm >= start) & (f_norm <= end)
    offset = float(rng.uniform(*offset_range)) * (-1 if rng.random() < 0.5 else 1)
    curve = base.copy()
    curve[mask] = curve[mask] + offset
    return TemplateResult(
        curve=curve,
        template_id="T9",
        params={"band_offset_db": offset, "band_start_norm": start, "band_end_norm": end},
        peak_track_type="none",
    )


TEMPLATE_TABLE = {
    "T1": _template_smooth_shift,
    "T2": _template_tilt_rolloff,
    "T3": _template_stable_ripple,
    "T4": _template_step_fixed,
    "T5": _template_spike_sparse,
    "T6": _template_scatter_thick,
    "T7": _template_quant_grain,
    "T8": _template_peak_track,
    "T9": _template_band_offset,
}


def select_template(templates: List[str], rng: np.random.Generator) -> str:
    if not templates:
        raise ValueError("templates list cannot be empty")
    return str(rng.choice(templates))


def apply_template(
    template_id: str,
    base_curve: np.ndarray,
    frequency: np.ndarray,
    rrs: np.ndarray,
    rng: np.random.Generator,
    severity: str,
) -> TemplateResult:
    if template_id not in TEMPLATE_TABLE:
        raise ValueError(f"Unknown template_id: {template_id}")
    return TEMPLATE_TABLE[template_id](base_curve, frequency, rrs, rng, severity)
