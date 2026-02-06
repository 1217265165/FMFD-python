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
        
        # 计算截止点（受严重度影响）
        cutoff_idx = int(n * cutoff_ratio * (1 + 2 * severity))
        cutoff_idx = min(cutoff_idx, n // 3)  # 最多影响 1/3 带宽
        
        # 创建高通滤波器响应
        result = curve.copy()
        if cutoff_idx > 0:
            # 低频塌陷：使用指数衰减
            decay_factor = 3 + 5 * severity  # dB per decade
            freq_axis = np.linspace(0.01, 1, cutoff_idx)
            attenuation = decay_factor * (1 - freq_axis) * severity
            result[:cutoff_idx] = curve[:cutoff_idx] - attenuation
        
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
        
        # 纹波幅度与严重度成正比
        amplitude = 0.5 + 2.0 * severity  # 0.5-2.5 dB
        
        # 随机化周期
        period = ripple_period * (0.5 + self.rng.random())
        
        # 生成正弦纹波
        x = np.arange(n)
        ripple = amplitude * np.sin(2 * np.pi * x / period)
        
        # 添加小幅度高频纹波
        if severity > 0.3:
            high_freq_ripple = (amplitude * 0.3) * np.sin(2 * np.pi * x / (period / 3))
            ripple += high_freq_ripple
        
        return curve + ripple
    
    def apply_global_shift(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        shift_range: float = 2.0
    ) -> np.ndarray:
        """
        衰减器/增益退化：整体偏移。
        
        物理机理：增益或衰减值偏离标称值，曲线整体上移或下移。
        
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
        # 偏移量随机化（有方向）
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
        
        # 确定影响区域
        if band_type == "low":
            start_idx = 0
            end_idx = n // 3
        else:
            start_idx = 2 * n // 3
            end_idx = n
        
        # 生成平滑的衰减曲线
        affected_len = end_idx - start_idx
        if affected_len > 0:
            # 使用半余弦窗生成平滑衰减
            window = np.cos(np.linspace(0, np.pi / 2, affected_len))
            attenuation = (1 + 3 * severity) * (1 - window)  # 1-4 dB
            result[start_idx:end_idx] = curve[start_idx:end_idx] - attenuation
        
        return result
    
    def apply_linear_slope(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        max_slope: float = 0.01
    ) -> np.ndarray:
        """
        混频器退化：线性斜率变化。
        
        物理机理：混频器增益随频率线性变化。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        max_slope : float
            最大斜率 (dB/点)
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        
        # 斜率随机化（有方向）
        direction = 1 if self.rng.random() > 0.5 else -1
        slope = direction * max_slope * severity * n
        
        # 生成线性斜坡
        ramp = np.linspace(-slope / 2, slope / 2, n)
        
        return curve + ramp
    
    def apply_quantization_noise(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        bit_depth: int = 12
    ) -> np.ndarray:
        """
        ADC量化退化：梳齿纹理/颗粒感。
        
        物理机理：ADC 位数不足或 DNL 问题导致量化噪声。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        bit_depth : int
            有效位数
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        # 计算量化步长（LSB）
        curve_range = np.ptp(curve)
        if curve_range < 1e-6:
            curve_range = 1.0
        
        # 有效位数随严重度降低
        effective_bits = bit_depth - int(4 * severity)  # 最多损失 4 位
        effective_bits = max(4, effective_bits)
        
        lsb = curve_range / (2 ** effective_bits)
        
        # 添加量化噪声（均匀分布，约 0.5 LSB rms）
        quant_noise = self.rng.uniform(-lsb / 2, lsb / 2, len(curve))
        
        # 添加 DNL 误差（周期性）
        if severity > 0.3:
            dnl_period = 2 ** (effective_bits // 2)
            dnl_amp = lsb * severity
            dnl = dnl_amp * sawtooth(2 * np.pi * np.arange(len(curve)) / dnl_period)
            quant_noise += dnl
        
        return curve + quant_noise
    
    def apply_signal_drop(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        drop_width_ratio: float = 0.1
    ) -> np.ndarray:
        """
        LO失锁退化：黑洞频段（信号置零）。
        
        物理机理：LO 锁定失败导致特定频段无输出。
        
        Parameters
        ----------
        curve : np.ndarray
            原始频响曲线
        severity : float
            退化严重度 (0-1)
        drop_width_ratio : float
            黑洞宽度比例
            
        Returns
        -------
        np.ndarray
            退化后的曲线
        """
        n = len(curve)
        result = curve.copy()
        
        # 黑洞宽度与严重度成正比
        drop_width = int(n * drop_width_ratio * (0.5 + severity))
        drop_width = max(10, min(drop_width, n // 4))
        
        # 随机选择黑洞位置
        max_start = n - drop_width
        if max_start > 0:
            start_idx = self.rng.integers(0, max_start)
            
            # 计算噪底（取曲线最低值附近）
            noise_floor = np.percentile(curve, 5) - 10 * severity
            
            # 使用平滑过渡而非硬切换
            transition_len = min(20, drop_width // 4)
            for i in range(drop_width):
                if i < transition_len:
                    factor = i / transition_len
                elif i > drop_width - transition_len:
                    factor = (drop_width - i) / transition_len
                else:
                    factor = 1.0
                
                idx = start_idx + i
                if idx < n:
                    result[idx] = curve[idx] * (1 - factor) + noise_floor * factor
        
        return result
    
    def apply_jitter_noise(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        jitter_rms: float = 0.5
    ) -> np.ndarray:
        """
        采样时钟抖动退化：抖动噪声。
        
        物理机理：采样时钟抖动导致频域展宽和噪声增加。
        
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
        
        # 抖动噪声幅度
        noise_amp = jitter_rms * (0.5 + 2 * severity)
        
        # 高斯白噪声
        noise = self.rng.normal(0, noise_amp, n)
        
        # 高频分量（时钟抖动表现为高频噪声）
        if severity > 0.3:
            high_freq = noise_amp * 0.5 * np.sin(
                2 * np.pi * np.arange(n) / (5 + self.rng.random() * 10)
            )
            noise += high_freq
        
        return curve + noise
    
    def apply_peak_jitter(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        jitter_points: int = 5
    ) -> np.ndarray:
        """
        参考时钟退化：峰值频率抖动。
        
        物理机理：参考时钟不稳定导致频率读数抖动。
        
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
        
        # 频率轴抖动（点数）
        jitter = int(jitter_points * severity * (0.5 + self.rng.random()))
        if jitter < 1:
            return result
        
        # 随机位移每个点（模拟频率抖动）- 向量化实现
        shifts = self.rng.integers(-jitter, jitter + 1, n)
        indices = np.clip(np.arange(n) + shifts, 0, n - 1)
        result = curve[indices]
        
        return result
    
    def apply_smooth_shift(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        max_shift: float = 1.5
    ) -> np.ndarray:
        """
        中频链路退化：平滑偏移。
        
        物理机理：增益/偏置漂移导致曲线平滑偏移。
        
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
        
        # 生成平滑的偏移曲线（低频正弦）
        period = n * (2 + 3 * self.rng.random())
        phase = self.rng.random() * 2 * np.pi
        
        shift = max_shift * severity * np.sin(
            2 * np.pi * np.arange(n) / period + phase
        )
        
        return curve + shift
    
    def apply_step_discontinuity(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        step_size: float = 1.0
    ) -> np.ndarray:
        """
        开关/切换退化：阶跃不连续。
        
        物理机理：信号路径切换时的阻抗不匹配或接触问题。
        
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
        
        # 随机选择阶跃位置
        num_steps = 1 + int(2 * severity)
        max_steps = max(1, n // 10)  # 确保至少可选 1 个位置
        step_positions = self.rng.choice(n, size=min(num_steps, max_steps), replace=False)
        step_positions.sort()
        
        # 应用阶跃
        cumulative_step = 0
        for pos in step_positions:
            direction = 1 if self.rng.random() > 0.5 else -1
            step = direction * step_size * severity * (0.5 + 0.5 * self.rng.random())
            cumulative_step += step
            result[pos:] += step
        
        return result
    
    def apply_isolated_spike(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        num_spikes: int = 3
    ) -> np.ndarray:
        """
        峰值搜索退化：孤立尖峰。
        
        物理机理：峰值检测逻辑错误导致虚假峰值。
        
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
        
        # 尖峰数量与严重度成正比
        actual_spikes = max(1, int(num_spikes * severity))
        actual_spikes = min(actual_spikes, n)  # 确保不超过曲线长度
        
        # 随机选择尖峰位置
        spike_positions = self.rng.choice(n, size=actual_spikes, replace=False)
        
        # 添加尖峰
        curve_range = np.ptp(curve)
        if curve_range < 1e-6:
            curve_range = 1.0
        for pos in spike_positions:
            spike_height = curve_range * 0.3 * (0.5 + severity)
            direction = 1 if self.rng.random() > 0.3 else -1
            result[pos] += direction * spike_height
        
        return result
    
    def apply_high_diff_variance(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        noise_scale: float = 0.5
    ) -> np.ndarray:
        """
        电源退化：高差分方差（噪声增加）。
        
        物理机理：电源纹波或稳定性问题导致噪声增加。
        
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
        
        # 高斯噪声
        noise_amp = noise_scale * (0.5 + 2 * severity)
        noise = self.rng.normal(0, noise_amp, n)
        
        # 添加低频调制（电源纹波）
        # ripple_freq 是纹波周期数，period = n / ripple_freq
        ripple_freq = 50 + self.rng.random() * 50  # 50-100 周期数
        period = max(10, n / ripple_freq)  # 确保周期至少为 10 点
        ripple = noise_amp * 0.3 * np.sin(2 * np.pi * np.arange(n) / period)
        
        return curve + noise + ripple
    
    def apply_compression(
        self, 
        curve: np.ndarray, 
        severity: float = 0.5,
        threshold: float = -10
    ) -> np.ndarray:
        """
        增益压缩退化：平顶效应。
        
        物理机理：放大器饱和导致高电平信号被压缩。
        
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
        
        # 计算压缩阈值
        max_val = np.max(curve)
        compression_threshold = max_val + threshold + (1 - severity) * 5
        
        # 压缩比例因子
        compression_scale = 3 + 5 * (1 - severity)
        
        # 应用软压缩（tanh）
        above_threshold = curve > compression_threshold
        if np.any(above_threshold):
            excess = curve[above_threshold] - compression_threshold
            compressed = compression_threshold + np.tanh(excess / compression_scale) * compression_scale
            result[above_threshold] = compressed
        
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
