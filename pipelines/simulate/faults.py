import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import sawtooth

# Single-band mode flag: When True, preamp is disabled and switch-point step injection is disabled
SINGLE_BAND_MODE = True

# ============ 故障严重度配置 (2024-01 新增) ============
# 每种故障注入支持三档严重度: light, mid, severe

SEVERITY_LEVELS = ['light', 'mid', 'severe']

# 幅度失准参数（按严重度分档）
AMP_MISCAL_PARAMS = {
    'light':  {'gain_sigma': 0.005, 'bias_sigma': 0.15, 'comp_mean': 0.005, 'comp_std': 0.002},
    'mid':    {'gain_sigma': 0.010, 'bias_sigma': 0.25, 'comp_mean': 0.010, 'comp_std': 0.004},
    'severe': {'gain_sigma': 0.020, 'bias_sigma': 0.40, 'comp_mean': 0.015, 'comp_std': 0.006},
}

# 参考电平失准参数（按严重度分档）
REFLEVEL_PARAMS = {
    'light':  {'offset_sigma': 0.2, 'offset_clip': 0.6, 'comp_mean': 0.06, 'comp_std': 0.03},
    'mid':    {'offset_sigma': 0.4, 'offset_clip': 0.8, 'comp_mean': 0.10, 'comp_std': 0.05},
    'severe': {'offset_sigma': 0.7, 'offset_clip': 1.0, 'comp_mean': 0.15, 'comp_std': 0.07},
}


# 辅助：估计局部/全局 sigma，用于自适应幅度/噪声
def _estimate_sigma(amp, window_frac=0.02, min_window=21):
    x = np.asarray(amp, dtype=float)
    n = len(x)
    w = max(min_window, int(round(n * window_frac)))
    if w % 2 == 0:
        w += 1
    half = w // 2
    pad = np.pad(x, (half, half), mode="edge")
    sig = np.zeros(n)
    for i in range(n):
        seg = pad[i:i + w]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        sig[i] = max(1e-6, 1.4826 * mad)
    return sig, float(np.median(sig))

# -------------------------
# 系统级故障/畸变注入（自适应幅度）
# -------------------------
def inject_amplitude_miscal(amp, gain=None, bias=None, comp=None, rng=None, 
                            severity='mid', return_params=False):
    """
    幅度失准：A' = gain*A + bias + comp*A^2
    若 gain/bias/comp 未给出，则基于严重度和当前曲线的 σ 自适应随机生成。
    
    优化版（2024-01）：
    - 支持 severity 参数：light/mid/severe 三档
    - 避免与参考电平失准重叠太多
    
    Parameters
    ----------
    severity : str
        故障严重度: 'light', 'mid', 'severe'
    return_params : bool
        If True, return (curve, params_dict) instead of just curve
    """
    rng = rng or np.random.default_rng()
    _, sig_med = _estimate_sigma(amp)
    
    # 获取对应严重度的参数
    params_cfg = AMP_MISCAL_PARAMS.get(severity, AMP_MISCAL_PARAMS['mid'])
    
    if gain is None:
        gain = 1.0 + rng.normal(0, params_cfg['gain_sigma'] * max(1.0, sig_med))
    if bias is None:
        bias = rng.normal(0, params_cfg['bias_sigma'] * max(0.2, sig_med))
    if comp is None:
        comp = rng.normal(params_cfg['comp_mean'], params_cfg['comp_std'])
        # 限制 comp 范围，避免二次项把低频段放大成类似 ref 的整体偏移
        comp = np.clip(comp, -0.03, 0.03)
    
    result = gain * amp + bias + comp * (amp ** 2)
    
    if return_params:
        return result, {
            'severity': severity,
            'gain': float(gain),
            'bias': float(bias),
            'comp': float(comp),
        }
    return result

def inject_freq_miscal(
    frequency,
    amp,
    delta_f=None,
    rng=None,
    return_params=False,
    severity="mid",
):
    """
    频率失准：频率轴整体平移后重采样；delta_f 未给出时按带宽 ppm 生成。
    
    优化版（2024-01）：
    - 使用 cubic 插值替代 linear，实现更平滑的频率响应过渡
    - 添加多次平滑操作减小插值误差
    
    Parameters
    ----------
    frequency : array
        Frequency axis
    amp : array
        Amplitude data
    delta_f : float, optional
        Frequency shift in Hz. If None, generated from ppm.
    rng : Generator, optional
        Random number generator
    return_params : bool
        If True, return (curve, params_dict) instead of just curve
        
    Returns
    -------
    array or (array, dict)
        Modified amplitude, optionally with injection parameters
    """
    rng = rng or np.random.default_rng()
    bw = frequency[-1] - frequency[0]
    step_hz = frequency[1] - frequency[0] if len(frequency) > 1 else 1e7
    
    if delta_f is None:
        # Constrained ppm range for single-band low-frequency path
        # light: ±80 ppm, mid: ±200 ppm, severe: ±500 ppm
        ppm_ranges = {
            "light": 80e-6,
            "mid": 200e-6,
            "severe": 500e-6,
        }
        ppm_limit = ppm_ranges.get(severity, ppm_ranges["mid"])
        ppm = rng.uniform(-ppm_limit, ppm_limit)
        min_ppm = 0.1 * ppm_limit
        if abs(ppm) < min_ppm:
            ppm = np.sign(ppm) * min_ppm if ppm != 0 else rng.choice([-1, 1]) * min_ppm
        delta_f = ppm * bw
    else:
        ppm = delta_f / bw if bw > 0 else 0
    
    f_shift = frequency + delta_f
    # Optimized: use cubic interpolation for smoother frequency response
    interp = interp1d(
        f_shift,
        amp,
        kind="cubic",
        bounds_error=False,
        fill_value=(float(amp[0]), float(amp[-1])),
        assume_sorted=True,
    )
    result = interp(frequency)
    
    # Calculate effective shift in bins
    shift_bins = delta_f / step_hz if step_hz > 0 else 0
    
    params = {
        'delta_f_hz': float(delta_f),
        'ppm': float(ppm * 1e6),  # Convert to actual ppm
        'shift_bins': float(shift_bins),
        'bandwidth_hz': float(bw),
        'severity': severity,
    }
    
    if return_params:
        return result, params
    return result

def inject_reflevel_miscal(frequency, amp, band_ranges, step_biases=None, compression_coef=None,
                           compression_start_percent=0.8, rng=None, single_band_mode=None,
                           return_params=False, severity='mid'):
    """
    参考电平失准：在切换点施加错误步进；高幅度区压缩。
    step_biases/compression_coef 未给出时按严重度和 σ 自适应随机生成。
    
    优化版（2024-01）：
    - 支持 severity 参数：light/mid/severe 三档
    - Type-A 全局偏移限制在 [-1.0, +1.0] dB
    - Type-B 压缩可限制在特定频段（高频段）
    
    In single-band mode (single_band_mode=True or SINGLE_BAND_MODE global):
    - Switch-point step injection is DISABLED
    - Type-A: Global offset (reference level shift)
    - Type-B: High-amplitude compression/saturation
    
    Parameters
    ----------
    severity : str
        故障严重度: 'light', 'mid', 'severe'
    return_params : bool
        If True, return (curve, params_dict) instead of just curve
    """
    rng = rng or np.random.default_rng()
    out = amp.copy()
    _, sig_med = _estimate_sigma(amp)
    
    # 获取对应严重度的参数
    params_cfg = REFLEVEL_PARAMS.get(severity, REFLEVEL_PARAMS['mid'])
    
    # Determine if single-band mode is active
    if single_band_mode is None:
        single_band_mode = SINGLE_BAND_MODE
    
    params = {
        'single_band_mode': single_band_mode,
        'severity': severity,
        'ref_type': 'none',
        'global_offset_db': 0.0,
        'compression_coef': 0.0,
        'compression_threshold_db': 0.0,
    }
    
    # Step injection at switch points (DISABLED in single-band mode)
    if not single_band_mode and len(band_ranges) > 1:
        if step_biases is None:
            step_biases = [rng.normal(0.6 * sig_med, 0.2 * max(0.2, sig_med))
                           for _ in range(len(band_ranges) - 1)]
        for i in range(len(band_ranges) - 1):
            end_f = band_ranges[i][1]
            m_end = np.argmin(np.abs(frequency - end_f))
            out[m_end:] += step_biases[i]
        params['ref_type'] = 'step'
        params['step_biases'] = [float(b) for b in step_biases]
    else:
        # Type-A global offset with severity-based parameters
        global_offset = rng.normal(0, params_cfg['offset_sigma'] * max(0.5, sig_med))
        # Clip to physical limits
        global_offset = np.clip(global_offset, -params_cfg['offset_clip'], params_cfg['offset_clip'])
        # Ensure minimum offset for detectability
        if abs(global_offset) < 0.10:
            global_offset = 0.10 * np.sign(global_offset) if global_offset != 0 else rng.choice([-1, 1]) * 0.10
        out = out + global_offset
        params['ref_type'] = 'global_offset'
        params['global_offset_db'] = float(global_offset)
    
    # Type-B compression with severity-based parameters
    if compression_coef is None:
        compression_coef = abs(rng.normal(params_cfg['comp_mean'], params_cfg['comp_std']))
        # Ensure minimum compression for detectability
        compression_coef = max(0.03, compression_coef)
    
    thr = np.percentile(out, 100 * compression_start_percent)
    mask = out >= thr
    out[mask] = out[mask] - compression_coef * (out[mask] - thr)
    
    params['compression_coef'] = float(compression_coef)
    params['compression_threshold_db'] = float(thr)
    params['compression_start_percent'] = float(compression_start_percent)
    
    if return_params:
        return out, params
    return out

# -------------------------
# 模块级示例畸变（自适应幅度）
# -------------------------

# PREAMP DISABLED: inject_preamp_degradation is kept for backward compatibility
# but will raise an error in single-band mode and should not be called
def inject_preamp_degradation(frequency, amp, hf_drop_db=None, rng=None):
    """
    前置放大器衰减：随频率线性下滑，高频端下降 hf_drop_db。
    
    **DISABLED IN SINGLE-BAND MODE**
    
    In single-band mode (10MHz-8.2GHz with preamp OFF), this function
    should NOT be called. It is preserved for backward compatibility only.
    """
    if SINGLE_BAND_MODE:
        raise ValueError(
            "inject_preamp_degradation is DISABLED in single-band mode. "
            "Preamp is OFF for 10MHz-8.2GHz frequency range."
        )
    
    rng = rng or np.random.default_rng()
    _, sig_med = _estimate_sigma(amp)
    if hf_drop_db is None:
        hf_drop_db = rng.uniform(0.5, 2.0) * max(1.0, sig_med)
    slope = hf_drop_db / (frequency[-1] - frequency[0])
    return amp - slope * (frequency - frequency[0])

def inject_lpf_shift(frequency, amp, cutoff_shift=None, rng=None, severity="mid"):
    """低频 LPF 拐点漂移：sigmoid 模拟滚降过渡。"""
    rng = rng or np.random.default_rng()
    bw = frequency[-1] - frequency[0]
    shift_limits = {
        "light": 0.005 * bw,
        "mid": 0.010 * bw,
        "severe": 0.020 * bw,
    }
    if cutoff_shift is None:
        cutoff_shift = rng.uniform(-shift_limits.get(severity, 0.010 * bw),
                                    shift_limits.get(severity, 0.010 * bw))
    f = frequency
    center = f[0] + 0.85 * bw + cutoff_shift
    center = float(np.clip(center, f[0] + 0.70 * bw, f[0] + 0.95 * bw))
    width = 0.05 * bw
    trans = 1 / (1 + np.exp((f - center) / width))
    drop_ranges = {
        "light": (0.04, 0.08),
        "mid": (0.08, 0.14),
        "severe": (0.14, 0.18),
    }
    drop_low, drop_high = drop_ranges.get(severity, (0.3, 0.6))
    drop_db = rng.uniform(drop_low, drop_high)
    attenuation = drop_db * (1.0 - trans)
    return amp - attenuation

def inject_mixer_ripple(frequency, amp, ripple_db=None, period=None, rng=None, severity="mid"):
    """混频器带内纹波：正弦微纹波叠加。"""
    rng = rng or np.random.default_rng()
    ripple_ranges = {
        "light": (0.01, 0.03),
        "mid": (0.03, 0.06),
        "severe": (0.06, 0.09),
    }
    if ripple_db is None:
        low, high = ripple_ranges.get(severity, (0.03, 0.06))
        ripple_db = rng.uniform(low, high)
    if period is None:
        bw = frequency[-1] - frequency[0]
        period = rng.uniform(0.05 * bw, 0.2 * bw)
    ripple = ripple_db * np.sin(2 * np.pi * frequency / period)
    return amp + ripple


def inject_rf_match_ripple(frequency, rrs, ripple_db=None, period=None, rng=None, severity="mid"):
    """输入匹配驻波：在 RRS 上叠加正弦周期波纹。"""
    rng = rng or np.random.default_rng()
    ripple_ranges = {
        "light": (0.02, 0.05),
        "mid": (0.05, 0.10),
        "severe": (0.10, 0.18),
    }
    if ripple_db is None:
        low, high = ripple_ranges.get(severity, (0.05, 0.10))
        ripple_db = rng.uniform(low, high)
    if period is None:
        bw = frequency[-1] - frequency[0]
        period = rng.uniform(0.03 * bw, 0.12 * bw)
    ripple = ripple_db * np.sin(2 * np.pi * frequency / period)
    return rrs + ripple


def inject_mixer1_slope(frequency, rrs, slope_db=None, rng=None, severity="mid"):
    """Mixer1 转换损耗滚降：线性斜率叠加。"""
    rng = rng or np.random.default_rng()
    slope_ranges = {
        "light": (-0.4, -0.2),
        "mid": (-0.8, -0.4),
        "severe": (-1.2, -0.8),
    }
    if slope_db is None:
        low, high = slope_ranges.get(severity, (-0.8, -0.4))
        slope_db = rng.uniform(low, high)
    f_norm = (frequency - frequency[0]) / (frequency[-1] - frequency[0] + 1e-12)
    return rrs + slope_db * f_norm


def inject_adc_sawtooth(frequency, rrs, amplitude_db=None, cycles=None, rng=None, severity="mid"):
    """ADC 量化非线性：在 RRS 上叠加锯齿纹理。"""
    rng = rng or np.random.default_rng()
    amp_ranges = {
        "light": (0.02, 0.05),
        "mid": (0.05, 0.12),
        "severe": (0.12, 0.20),
    }
    if amplitude_db is None:
        low, high = amp_ranges.get(severity, (0.05, 0.12))
        amplitude_db = rng.uniform(low, high)
    if cycles is None:
        cycles = rng.uniform(10, 25)
    phase = np.linspace(0.0, 2 * np.pi * cycles, len(frequency))
    texture = amplitude_db * sawtooth(phase, width=0.5)
    return rrs + texture


def inject_power_noise_rrs(rrs, sigma, rng=None, severity="mid"):
    """电源宽带噪声：基于 RRS 的高频锯齿噪声。"""
    rng = rng or np.random.default_rng()
    beta_ranges = {
        "light": (1.5, 2.0),
        "mid": (2.0, 3.0),
        "severe": (3.0, 4.0),
    }
    beta_low, beta_high = beta_ranges.get(severity, (2.0, 3.0))
    beta = rng.uniform(beta_low, beta_high)
    noise_std = beta * np.asarray(sigma, dtype=float)
    noise = rng.normal(0, noise_std, size=len(rrs))
    noise = noise - float(np.mean(noise))
    return rrs + noise

def inject_ytf_variation(frequency, amp, notch_depth_db=None, notch_center=None, rng=None):
    """YTF 滤波器：高频端陷波/带宽变化。"""
    rng = rng or np.random.default_rng()
    f = frequency
    if notch_depth_db is None:
        _, sig_med = _estimate_sigma(amp)
        notch_depth_db = rng.uniform(0.5, 2.0) * max(1.0, sig_med)
    if notch_center is None:
        notch_center = f[0] + 0.8 * (f[-1] - f[0])
    width = 0.01 * (f[-1] - f[0])
    notch = -notch_depth_db * np.exp(-0.5 * ((f - notch_center) / width) ** 2)
    return amp + notch

def inject_clock_drift(frequency, amp, delta_f=None, rng=None):
    """时钟系统：全局 Δf。"""
    return inject_freq_miscal(frequency, amp, delta_f=delta_f, rng=rng)

def inject_lo_path_error(
    frequency, amp, band_ranges, band_shifts=None, rng=None, severity="mid"
):
    """本振/路径相关：分段 Δf，分频段重采样。"""
    rng = rng or np.random.default_rng()
    out = amp.copy()
    if band_shifts is None:
        bw = frequency[-1] - frequency[0]
        ppm_ranges = {
            "light": 80e-6,
            "mid": 200e-6,
            "severe": 500e-6,
        }
        ppm_limit = ppm_ranges.get(severity, ppm_ranges["mid"])
        band_shifts = [rng.uniform(-ppm_limit, ppm_limit) * bw for _ in band_ranges]
    for (start, end), df in zip(band_ranges, band_shifts):
        mask = (frequency >= start) & (frequency <= end)
        if np.any(mask):
            f_seg = frequency[mask]
            a_seg = out[mask]
            out[mask] = inject_freq_miscal(f_seg, a_seg, delta_f=df, rng=rng, severity=severity)
    return out

def inject_adc_bias(amp, gain=None, bias=None, comp=None, rng=None):
    """数字 IF/ADC 偏置或非线性。"""
    rng = rng or np.random.default_rng()
    _, sig_med = _estimate_sigma(amp)
    if gain is None:
        gain = 1.0 + rng.normal(0, 0.05 * max(1.0, sig_med))
    if bias is None:
        bias = rng.normal(0, 0.2 * max(0.2, sig_med))
    if comp is None:
        comp = rng.normal(0.05, 0.02)
    return inject_amplitude_miscal(amp, gain, bias, comp, rng=rng)

def inject_vbw_smoothing(amp, window=None, rng=None):
    """数字 IF/VBW：滑动平均模拟平滑。"""
    rng = rng or np.random.default_rng()
    if window is None:
        window = int(max(50, min(len(amp) // 10, rng.integers(200, 800))))
    if window < 3:
        return amp
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(amp, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")[:len(amp)]

def inject_power_noise(
    amp,
    noise_std=None,
    rng=None,
    sigma=None,
    severity="mid",
    return_params=False,
):
    """电源噪声：全频随机噪声提升。"""
    rng = rng or np.random.default_rng()
    beta_ranges = {
        "light": (1.0, 1.5),
        "mid": (1.5, 2.2),
        "severe": (2.2, 3.0),
    }
    if sigma is not None:
        beta_low, beta_high = beta_ranges.get(severity, (1.5, 2.2))
        beta = rng.uniform(beta_low, beta_high)
        noise_std = beta * np.asarray(sigma, dtype=float)
    elif noise_std is None:
        _, sig_med = _estimate_sigma(amp)
        noise_std = rng.uniform(0.1, 0.3) * max(1.0, sig_med)
        beta = None
    else:
        beta = None
    noise = rng.normal(0, noise_std, size=len(amp))
    noise = noise - float(np.mean(noise))
    result = amp + noise
    if return_params:
        return result, {
            "severity": severity,
            "beta": float(beta) if beta is not None else None,
            "noise_std_mean": float(np.mean(noise_std)) if np.ndim(noise_std) else float(noise_std),
        }
    return result


# ============ 数字中频板故障模型 (新增) ============
# 严重度权重体系与文档 simulation_methodology.md 保持一致
_SEVERITY_WEIGHTS = {"light": 0.3, "mid": 0.5, "severe": 0.8}


def inject_rbw_ripple(
    amp,
    max_ripple_db=3.0,
    severity="mid",
    rng=None,
    return_params=False,
):
    """
    [数字中频板][数字IF域] RBW滤波器故障：周期性正弦纹波叠加。

    机理：数字 RBW 滤波器系数异常，导致曲线出现规则的"非物理"Sinc
    旁瓣效应或过补偿起伏。

    建模：叠加带高斯包络约束的正弦波 (Sine Ripple)。
        actual_ripple_db = max_ripple_db × severity_weight

    Parameters
    ----------
    amp : array_like
        原始幅度曲线 (dBm)。
    max_ripple_db : float
        基准最大起伏幅度 (dB)。 **禁止写死**，通过此参数暴露。
    severity : str
        严重度等级 ('light', 'mid', 'severe')。
    rng : numpy.random.Generator, optional
    return_params : bool
        是否返回注入参数字典。
    """
    rng = rng or np.random.default_rng()
    amp = np.asarray(amp, dtype=float)
    n = len(amp)
    if n < 4:
        return (amp.copy(), {}) if return_params else amp.copy()

    severity_weight = _SEVERITY_WEIGHTS.get(severity, 0.5)
    actual_ripple_db = max_ripple_db * severity_weight

    # 随机纹波周期数与相位
    ripple_periods = rng.uniform(3.0, 12.0)
    phase = rng.uniform(0.0, 2 * np.pi)
    x = np.linspace(0, 2 * np.pi * ripple_periods, n)

    # 高斯包络约束，避免边缘突变
    envelope = np.exp(-0.5 * (np.linspace(-2.0, 2.0, n) ** 2))

    ripple = actual_ripple_db * np.sin(x + phase) * envelope
    result = amp + ripple

    if return_params:
        return result, {
            "severity": severity,
            "max_ripple_db": float(max_ripple_db),
            "actual_ripple_db": float(actual_ripple_db),
            "ripple_periods": float(ripple_periods),
        }
    return result


def inject_vbw_ema_lag(
    amp,
    max_alpha=0.85,
    severity="mid",
    rng=None,
    return_params=False,
):
    """
    [数字中频板][数字IF域] VBW滤波器故障：一阶指数平滑 (EMA) 迟滞/拖尾。

    机理：VBW 包络平滑常数异常，细小纹波被过度抹平，陡峭过渡带出现
    向扫描方向的迟滞拖尾形变。

    建模：一阶 EMA 单向滤波 y[n] = α·y[n-1] + (1-α)·x[n]。
        actual_alpha = max_alpha × severity_weight

    Parameters
    ----------
    amp : array_like
        原始幅度曲线 (dBm)。
    max_alpha : float
        基准最大平滑/迟滞系数 (0 < max_alpha < 1)。 **禁止写死**。
    severity : str
        严重度等级 ('light', 'mid', 'severe')。
    rng : numpy.random.Generator, optional
    return_params : bool
        是否返回注入参数字典。
    """
    rng = rng or np.random.default_rng()
    amp = np.asarray(amp, dtype=float)
    n = len(amp)
    if n < 2:
        return (amp.copy(), {}) if return_params else amp.copy()

    severity_weight = _SEVERITY_WEIGHTS.get(severity, 0.5)
    actual_alpha = max_alpha * severity_weight

    # 一阶 EMA 向右迟滞
    result = np.empty_like(amp)
    result[0] = amp[0]
    for i in range(1, n):
        result[i] = actual_alpha * result[i - 1] + (1.0 - actual_alpha) * amp[i]

    if return_params:
        return result, {
            "severity": severity,
            "max_alpha": float(max_alpha),
            "actual_alpha": float(actual_alpha),
        }
    return result
