#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
曲线生成器 (Curve Generator) - 基于映射.md的物理退化函数
============================================================
实现物理意义明确的频响曲线退化模型，替代旧的随机加噪逻辑。

每个退化函数对应映射.md中定义的"曲线表征"：
- AC耦合入口: 低频塌陷 (高通滤波器效应)
- ADC量化: 梳齿纹理/颗粒感
- LO失锁: 黑洞频段
- 增益压缩: 平顶效应
- 等等...

使用方法:
    from pipelines.simulate.curve_generator import CurveGenerator
    
    generator = CurveGenerator()
    degraded_curve = generator.apply_degradation(
        curve=baseline_curve,
        module_key="ac_coupling",
        severity=0.5
    )
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from scipy.signal import butter, filtfilt, sawtooth

# 获取配置文件路径
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def load_module_taxonomy() -> Dict:
    """加载模块分类体系配置。"""
    config_path = CONFIG_DIR / "module_taxonomy_v2.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


class CurveGenerator:
    """
    曲线生成器：基于物理模型的频响曲线退化。
    
    每个模块故障对应一个或多个物理退化函数。
    退化强度由 severity 参数控制 (0-1)。
    
    Attributes
    ----------
    module_config : Dict
        模块分类体系配置
    rng : np.random.Generator
        随机数生成器
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化曲线生成器。
        
        Parameters
        ----------
        seed : int, optional
            随机数种子，用于可复现性
        """
        self.module_config = load_module_taxonomy()
        self.rng = np.random.default_rng(seed)
        
        # 模块别名
        self.module_aliases = self.module_config.get("MODULE_ALIASES", {})
        
        # 曲线特征签名
        self.curve_signatures = self.module_config.get("CURVE_SIGNATURES", {})
        
        # 模块到退化函数的映射
        self._degradation_map = {
            "ac_coupling": self.apply_high_pass_filter_effect,
            "input_connector": self.apply_periodic_ripple,
            "step_attenuator": self.apply_global_shift,
            "lpf_low_band": self.apply_band_insertion_loss,
            "lpf_high_band": self.apply_band_insertion_loss,
            "mixer1": self.apply_linear_slope,
            "lo1_inject": self.apply_linear_slope,
            "if_amp_chain": self.apply_smooth_shift,
            "rbw_filter": self.apply_band_insertion_loss,
            "adc_module": self.apply_quantization_noise,
            "adc_clock": self.apply_jitter_noise,
            "dsp_gain_cal": self.apply_global_shift,
            "dsp_window": self.apply_band_insertion_loss,
            "dsp_detector": self.apply_smooth_shift,
            "peak_search": self.apply_isolated_spike,
            "lo1_synth": self.apply_signal_drop,
            "lo2_synth": self.apply_signal_drop,
            "ocxo_ref": self.apply_peak_jitter,
            "ref_distribution": self.apply_peak_jitter,
            "cal_source": self.apply_smooth_shift,
            "cal_switch": self.apply_step_discontinuity,
            "cal_storage": self.apply_global_shift,
            "power_management": self.apply_high_diff_variance,
        }
    
    # =========================================================================
    # 物理退化函数 (基于映射.md)
    # =========================================================================
    
    def apply_high_pass_filter_effect(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        cutoff_ratio: float = 0.1
    ) -> np.ndarray:
        """
        AC耦合入口退化：低频塌陷（高通滤波器效应）。
        
        物理机理：隔直电容老化导致低频截止点右移，低频响应下降。
        
        V-D.4 校准：最大衰减控制在 0.3-0.5 dB，保持"圆弧状滚降"形态。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        cutoff_ratio : float
            截止频率比例（相对于带宽）
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        if n < 10:
            return curve.copy()
        
        # V-D.4: 校准后的截止点范围（影响 5%-15% 带宽）
        cutoff_idx = int(n * cutoff_ratio * (0.5 + 1.0 * severity))
        cutoff_idx = min(cutoff_idx, n // 6)  # 最多影响 1/6 带宽
        
        result = curve.copy()
        if cutoff_idx > 0:
            # V-D.4: 使用圆弧滚降（1 - cos）形态，最大衰减 0.3-0.5 dB
            # 形态特征：左端最低，向右平滑过渡到零
            max_attenuation = 0.3 + 0.2 * severity  # 0.3-0.5 dB
            freq_axis = np.linspace(0, np.pi / 2, cutoff_idx)
            # 使用 (1 - cos) 生成圆弧状滚降
            rolloff = max_attenuation * (1 - np.cos(freq_axis[::-1]))
            result[:cutoff_idx] = curve[:cutoff_idx] - rolloff
        
        return result
    
    def apply_periodic_ripple(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        ripple_period: int = 50
    ) -> np.ndarray:
        """
        输入连接/匹配退化：周期性纹波（阻抗失配）。
        
        物理机理：阻抗失配导致驻波，表现为周期性纹波。
        
        V-D.4 校准：纹波 Vpp 控制在 0.2-0.4 dB，保持清晰周期性。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        ripple_period : int
            纹波周期（点数）
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        
        # V-D.4: Vpp = 0.2-0.4 dB，所以 amplitude (半幅) = 0.1-0.2 dB
        amplitude = 0.10 + 0.10 * severity  # 0.10-0.20 dB 半幅
        
        # 随机化周期（保持在合理范围）
        period = ripple_period * (0.7 + 0.6 * self.rng.random())
        
        # 生成清晰正弦纹波（关键形态特征）
        x = np.arange(n)
        phase = self.rng.random() * 2 * np.pi
        ripple = amplitude * np.sin(2 * np.pi * x / period + phase)
        
        # V-D.4: 添加微小二次谐波增强形态识别
        if severity > 0.4:
            harmonic2 = (amplitude * 0.15) * np.sin(4 * np.pi * x / period + phase)
            ripple += harmonic2
        
        return curve + ripple
    
    def apply_global_shift(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        shift_range: float = 0.3
    ) -> np.ndarray:
        """
        衰减器/增益退化：整体偏移。
        
        物理机理：增益或衰减值偏离标称值，曲线整体上移或下移。
        
        V-D.4 校准：最大偏移量控制在 0.3 dB。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        shift_range : float
            最大偏移量 (dB)
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        # V-D.4: 偏移量控制在 ±0.15-0.30 dB
        direction = 1 if self.rng.random() > 0.5 else -1
        shift = direction * shift_range * severity * (0.5 + 0.5 * self.rng.random())
        
        return curve + shift
    
    def apply_band_insertion_loss(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        band_type: str = "low"
    ) -> np.ndarray:
        """
        滤波器退化：频段插损增大。
        
        物理机理：滤波器老化导致通带平坦度下降，插入损耗增加。
        
        V-D.4 校准：最大插损控制在 0.4-0.6 dB。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        band_type : str
            频段类型 ("low" 或 "high")
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        result = curve.copy()
        
        # 确定影响区域（缩小范围）
        if band_type == "low":
            start_idx = 0
            end_idx = n // 5  # V-D.4: 只影响 1/5 带宽
        else:
            start_idx = 4 * n // 5
            end_idx = n
        
        # V-D.4: 最大衰减 0.4-0.6 dB
        affected_len = end_idx - start_idx
        if affected_len > 0:
            # 使用半余弦窗生成平滑衰减
            window = np.cos(np.linspace(0, np.pi / 2, affected_len))
            max_attenuation = 0.4 + 0.2 * severity  # 0.4-0.6 dB
            attenuation = max_attenuation * (1 - window)
            result[start_idx:end_idx] = curve[start_idx:end_idx] - attenuation
        
        return result
    
    def apply_linear_slope(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        max_slope: float = 0.001
    ) -> np.ndarray:
        """
        混频器退化：线性斜率变化。
        
        物理机理：混频器增益随频率线性变化。
        
        V-D.4 校准：最大端点偏差控制在 ±0.3 dB。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        max_slope : float
            最大斜率系数
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        
        # V-D.4: 斜率使端点偏差控制在 ±0.15-0.30 dB
        direction = 1 if self.rng.random() > 0.5 else -1
        max_deviation = 0.15 + 0.15 * severity  # 0.15-0.30 dB at endpoints
        
        # 生成线性斜坡，中点为零
        ramp = np.linspace(-max_deviation, max_deviation, n) * direction
        
        return curve + ramp
    
    def apply_quantization_noise(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        bit_depth: int = 12
    ) -> np.ndarray:
        """
        ADC量化退化：非高斯量化纹理 + DNL/INL 误差。
        
        物理机理：ADC 位数限制导致量化台阶（取整），DNL 误差导致周期性纹理。
        
        V-D.4 校准：纹理幅度控制在 0.1-0.2 dB，重点在"非高斯性"形态特征。
        形态特征：周期性 DNL 梳齿效应，可被 X37 (差分方差) 和 X35 (形态特征) 识别。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线 (dB)
        severity : float
            退化严重度 (0-1)
        bit_depth : int
            有效位数
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        result = curve.copy()
        
        # V-D.4: 幅度限制在 0.1-0.2 dB
        max_noise_amp = 0.10 + 0.10 * severity  # 0.10-0.20 dB
        
        # ADC 量化误差分配比例 (基于 ADC 物理特性):
        # - DNL (60%): 主要形态特征，周期性梳齿效应，可被 X37 差分方差检测
        # - INL (30%): 低频包络漂移，表现为非线性累积误差
        # - Jitter (10%): 采样抖动，使用均匀分布确保非高斯性
        DNL_RATIO = 0.6
        INL_RATIO = 0.3
        JITTER_RATIO = 0.2  # 略大于 10% 以增强可检测性
        
        # 核心特征 1：周期性 DNL 误差（非高斯性的关键）
        # DNL 周期与 ADC 码字相关，产生梳齿效应
        dnl_period = max(4, 8 + int(8 * (1 - severity)))  # 8-16 点周期
        dnl = max_noise_amp * DNL_RATIO * sawtooth(2 * np.pi * np.arange(n) / dnl_period)
        result = result + dnl
        
        # 核心特征 2：INL 积累误差（低频包络）
        # INL 表现为缓慢的非线性漂移
        inl_period = n / (2 + self.rng.random() * 2)
        inl = max_noise_amp * INL_RATIO * np.sin(2 * np.pi * np.arange(n) / inl_period)
        result = result + inl
        
        # 核心特征 3：微量量化抖动（确保非高斯分布）
        # 使用均匀分布而非高斯分布
        jitter = self.rng.uniform(-max_noise_amp * JITTER_RATIO, max_noise_amp * JITTER_RATIO, n)
        result = result + jitter
        
        return result
    
    def apply_signal_drop(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        drop_width_ratio: float = 0.05
    ) -> np.ndarray:
        """
        LO临界失锁退化：相位噪声恶化/底噪抬升。
        
        物理机理：LO 处于临界失锁状态，导致特定频段相位噪声恶化，
        表现为底噪抬升和密集毛刺，而非完全信号丢失（黑洞）。
        
        V-D.4 校准：底噪抬升 0.3-0.5 dB，加入密集毛刺纹理。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        drop_width_ratio : float
            受影响频段宽度比例
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        result = curve.copy()
        
        # V-D.4: 受影响区域宽度
        affected_width = int(n * drop_width_ratio * (0.5 + 0.5 * severity))
        affected_width = max(20, min(affected_width, n // 5))
        
        # 随机选择受影响区域位置
        max_start = n - affected_width
        if max_start > 0:
            start_idx = self.rng.integers(0, max_start)
            end_idx = start_idx + affected_width
            
            # V-D.4: 底噪抬升 0.3-0.5 dB（使用负值因为是信号衰减）
            noise_floor_rise = 0.3 + 0.2 * severity  # 0.3-0.5 dB
            
            # 使用平滑窗口避免硬边界
            window = np.hanning(affected_width)
            
            # 应用底噪抬升
            result[start_idx:end_idx] = curve[start_idx:end_idx] - noise_floor_rise * window
            
            # V-D.4: 添加密集毛刺（相位噪声特征）
            spike_density = 0.1 + 0.1 * severity  # 10-20% 的点有毛刺
            num_spikes = int(affected_width * spike_density)
            if num_spikes > 0:
                spike_positions = self.rng.choice(
                    affected_width, size=num_spikes, replace=False
                )
                spike_amp = 0.1 + 0.1 * severity  # 0.1-0.2 dB 毛刺
                for pos in spike_positions:
                    direction = 1 if self.rng.random() > 0.5 else -1
                    result[start_idx + pos] += direction * spike_amp * self.rng.random()
        
        return result
    
    def apply_jitter_noise(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        jitter_rms: float = 0.08
    ) -> np.ndarray:
        """
        采样时钟抖动退化：抖动噪声。
        
        物理机理：采样时钟抖动导致频域展宽和噪声增加。
        
        V-D.4 校准：噪声幅度控制在 0.1-0.15 dB，使用裁剪防止极端值。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        jitter_rms : float
            基准抖动噪声 (dB rms)
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        
        # V-D.4: 抖动噪声幅度控制在 0.08-0.15 dB RMS
        noise_amp = jitter_rms * (1 + 0.8 * severity)  # 0.08-0.15 dB
        
        # 高斯白噪声 + 裁剪防止极端值
        noise = self.rng.normal(0, noise_amp, n)
        max_noise = 0.25  # 硬限制最大噪声
        noise = np.clip(noise, -max_noise, max_noise)
        
        # 添加微弱高频分量（时钟抖动特征）
        if severity > 0.4:
            high_freq_period = 5 + self.rng.random() * 10
            high_freq = noise_amp * 0.3 * np.sin(
                2 * np.pi * np.arange(n) / high_freq_period
            )
            noise += high_freq
        
        return curve + noise
    
    def apply_peak_jitter(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        jitter_points: int = 2
    ) -> np.ndarray:
        """
        参考时钟退化：频率轴微抖。
        
        物理机理：参考时钟不稳定导致频率读数轻微抖动。
        
        V-D.4 校准：抖动范围控制在 1-2 点，幅度变化 < 0.3 dB。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        jitter_points : int
            基准抖动点数
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        result = curve.copy()
        
        # V-D.4: 频率轴抖动控制在 1-2 点
        jitter = max(1, int(jitter_points * severity))
        
        # 随机位移每个点（模拟频率抖动）
        shifts = self.rng.integers(-jitter, jitter + 1, n)
        indices = np.clip(np.arange(n) + shifts, 0, n - 1)
        result = curve[indices]
        
        return result
    
    def apply_smooth_shift(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        max_shift: float = 0.25
    ) -> np.ndarray:
        """
        中频链路退化：平滑偏移。
        
        物理机理：增益/偏置漂移导致曲线平滑偏移。
        
        V-D.4 校准：最大偏移控制在 0.25-0.5 dB。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        max_shift : float
            最大偏移量 (dB)
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        
        # V-D.4: 平滑偏移控制在 0.25-0.5 dB
        actual_max_shift = max_shift * (1 + severity)  # 0.25-0.50 dB
        
        # 生成平滑的偏移曲线（低频正弦）
        period = n * (2 + 3 * self.rng.random())
        phase = self.rng.random() * 2 * np.pi
        
        shift = actual_max_shift * np.sin(
            2 * np.pi * np.arange(n) / period + phase
        )
        
        return curve + shift
    
    def apply_step_discontinuity(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        step_size: float = 0.2
    ) -> np.ndarray:
        """
        开关/切换退化：阶跃不连续。
        
        物理机理：信号路径切换时的阻抗不匹配或接触问题。
        
        V-D.4 校准：阶跃大小控制在 0.2-0.4 dB。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        step_size : float
            阶跃大小 (dB)
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        result = curve.copy()
        
        # 短曲线保护
        if n < 20:
            return result
        
        # V-D.4: 阶跃数量控制
        num_steps = 1 + int(severity)  # 1-2 个阶跃
        max_steps = max(1, n // 10)
        step_positions = self.rng.choice(n, size=min(num_steps, max_steps), replace=False)
        step_positions.sort()
        
        # V-D.4: 应用阶跃，每个阶跃 0.1-0.4 dB
        cumulative_step = 0
        for pos in step_positions:
            direction = 1 if self.rng.random() > 0.5 else -1
            step = direction * step_size * (1 + severity) * (0.5 + 0.5 * self.rng.random())
            cumulative_step += step
            result[pos:] += step
        
        return result
    
    def apply_isolated_spike(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        num_spikes: int = 2
    ) -> np.ndarray:
        """
        峰值搜索退化：孤立尖峰。
        
        物理机理：峰值检测逻辑错误导致虚假峰值。
        
        V-D.4 校准：尖峰高度控制在 0.2-0.4 dB。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        num_spikes : int
            尖峰数量
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        if n < 1:
            return curve.copy()
        
        result = curve.copy()
        
        # V-D.4: 尖峰数量控制
        actual_spikes = max(1, int(num_spikes * severity))
        actual_spikes = min(actual_spikes, n)
        
        # 随机选择尖峰位置
        spike_positions = self.rng.choice(n, size=actual_spikes, replace=False)
        
        # V-D.4: 尖峰高度控制在 0.2-0.4 dB
        spike_height = 0.2 + 0.2 * severity  # 0.2-0.4 dB
        for pos in spike_positions:
            direction = 1 if self.rng.random() > 0.3 else -1
            result[pos] += direction * spike_height * (0.5 + 0.5 * self.rng.random())
        
        return result
    
    def apply_high_diff_variance(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        noise_scale: float = 0.1
    ) -> np.ndarray:
        """
        电源退化：高差分方差（噪声增加）。
        
        物理机理：电源纹波或稳定性问题导致噪声增加。
        
        V-D.4 校准：噪声幅度控制在 0.08-0.12 dB，使用裁剪防止极端值。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        noise_scale : float
            噪声比例
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        
        # V-D.4: 高斯噪声幅度控制在 0.08-0.12 dB RMS
        noise_amp = noise_scale * (0.8 + 0.4 * severity)  # 0.08-0.12 dB
        noise = self.rng.normal(0, noise_amp, n)
        # 裁剪防止极端值
        noise = np.clip(noise, -0.25, 0.25)
        
        # 添加低频调制（电源纹波特征）
        ripple_freq = 50 + self.rng.random() * 50  # 50-100 周期数
        period = max(10, n / ripple_freq)
        ripple = noise_amp * 0.3 * np.sin(2 * np.pi * np.arange(n) / period)
        
        return curve + noise + ripple
    
    def apply_compression(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        threshold: float = -1
    ) -> np.ndarray:
        """
        增益压缩退化：微平顶效应。
        
        物理机理：放大器轻微饱和导致高电平信号被微压缩。
        
        V-D.4 校准：压缩量控制在 0.2-0.4 dB。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        threshold : float
            压缩阈值 (dB from max)
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        result = curve.copy()
        
        # V-D.4: 计算压缩阈值，使压缩量控制在 0.2-0.4 dB
        max_val = np.max(curve)
        compression_threshold = max_val + threshold
        
        # V-D.4: 最大压缩量控制在 0.2-0.4 dB
        max_compression = 0.2 + 0.2 * severity
        
        # 应用软压缩：使用 tanh 压缩，限制输出在 max_compression 范围内
        above_threshold = curve > compression_threshold
        if np.any(above_threshold):
            excess = curve[above_threshold] - compression_threshold
            # 压缩后的增量 = tanh(excess / scale) * max_compression
            # 这确保无论 excess 多大，压缩量都不会超过 max_compression
            scale = max_compression  # 控制压缩曲线的形状
            compressed_excess = np.tanh(excess / max(scale, 0.1)) * max_compression
            result[above_threshold] = compression_threshold + compressed_excess
        
        return result
    
    # =========================================================================
    # 公共接口
    # =========================================================================
    
    def apply_degradation(
        self,
        curve: np.ndarray,
        module_key: str,
        severity: float = 0.5
    ) -> np.ndarray:
        """
        应用指定模块的退化效应。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        module_key : str
            模块键名（如 "ac_coupling", "adc_module" 等）
        severity : float
            退化严重度 (0-1)
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        if module_key in self._degradation_map:
            return self._degradation_map[module_key](curve, severity)
        else:
            # 未知模块，返回原曲线
            return curve.copy()
    
    def apply_multiple_degradations(
        self,
        curve: np.ndarray,
        module_keys: List[str],
        severities: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        应用多个模块的退化效应（串联）。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        module_keys : List[str]
            模块键名列表
        severities : List[float], optional
            各模块的退化严重度，默认均为 0.5
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        if severities is None:
            severities = [0.5] * len(module_keys)
        
        result = curve.copy()
        for module_key, severity in zip(module_keys, severities):
            result = self.apply_degradation(result, module_key, severity)
        
        return result
    
    def get_module_full_name(self, module_key: str) -> str:
        """获取模块的完整中文名称。"""
        return self.module_aliases.get(module_key, module_key)
    
    def get_fault_type_for_module(self, module_key: str) -> str:
        """获取模块对应的系统级故障类型。"""
        fault_to_modules = self.module_config.get("FAULT_TYPE_TO_MODULES", {})
        
        for fault_type, modules_dict in fault_to_modules.items():
            primary = modules_dict.get("primary", [])
            secondary = modules_dict.get("secondary", [])
            if module_key in primary or module_key in secondary:
                return fault_type
        
        return "amp_error"  # 默认返回幅度失准
    
    def list_available_modules(self) -> List[str]:
        """列出所有可用的模块键名。"""
        return list(self._degradation_map.keys())


# 便捷接口
_default_generator: Optional[CurveGenerator] = None


def get_curve_generator(seed: Optional[int] = None) -> CurveGenerator:
    """获取曲线生成器实例。"""
    global _default_generator
    if _default_generator is None or seed is not None:
        _default_generator = CurveGenerator(seed=seed)
    return _default_generator


def apply_degradation(
    curve: np.ndarray,
    module_key: str,
    severity: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    应用退化效应（便捷接口）。
    
    Parameters
    ----------
    curve : np.ndarray
        原始频响曲线
    module_key : str
        模块键名
    severity : float
        退化严重度 (0-1)
    seed : int, optional
        随机种子
        
    Returns
    -------
    np.ndarray
        退化后的曲线
    """
    generator = get_curve_generator(seed)
    return generator.apply_degradation(curve, module_key, severity)
