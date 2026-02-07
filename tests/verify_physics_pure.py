#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-D.3 物理特征验证脚本
======================
验证 CurveGenerator 生成的物理故障特征是否正确。

生成 4 条物理特征曲线对比图:
1. AC Coupling: 低频端（左侧）大幅下陷
2. ADC Module: 量化阶梯/锯齿纹理
3. Input Connector: 全频段驻波纹波
4. LO Unlock: 某一段频率直接掉到底噪

使用方法:
    python tests/verify_physics_pure.py

输出:
    tests/output/物理特征对比图.png
"""
from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches

from pipelines.simulate.curve_generator import CurveGenerator


def setup_chinese_font():
    """设置中文字体支持。"""
    try:
        # 尝试使用系统中文字体
        for font_name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']:
            if font_manager.findfont(font_name, fallback_to_default=False):
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                return
    except Exception:
        pass
    # 回退到默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


def generate_baseline_curve(n_points: int = 1000, seed: int = 42) -> np.ndarray:
    """生成基线频响曲线。"""
    rng = np.random.default_rng(seed)
    
    # 模拟典型频响: 基线 + 小幅度随机波动
    baseline = -10.0 + np.sin(np.linspace(0, 4 * np.pi, n_points)) * 0.1
    baseline += rng.normal(0, 0.02, n_points)
    
    return baseline


def verify_physics_features():
    """验证物理特征并生成对比图。"""
    setup_chinese_font()
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 生成基线
    n_points = 1000
    baseline = generate_baseline_curve(n_points, seed=42)
    freq_axis = np.linspace(0, n_points - 1, n_points)
    
    # 创建 CurveGenerator
    generator = CurveGenerator(seed=42)
    
    # 定义要验证的物理特征
    test_cases = [
        {
            "module_key": "ac_coupling",
            "title": "AC Coupling (AC耦合)",
            "expected": "Low freq rolloff (低频塌陷)",
            "severity": 0.7,
            "color": "red",
        },
        {
            "module_key": "adc_module", 
            "title": "ADC Module (ADC量化)",
            "expected": "Quantization steps (量化阶梯)",
            "severity": 0.7,
            "color": "orange",
        },
        {
            "module_key": "input_connector",
            "title": "Input Connector (阻抗失配)",
            "expected": "Periodic ripple (驻波纹波)",
            "severity": 0.7,
            "color": "green",
        },
        {
            "module_key": "lo1_synth",
            "title": "LO Unlock (LO失锁)",
            "expected": "Signal drop (黑洞频段)",
            "severity": 0.7,
            "color": "blue",
        },
    ]
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    results = []
    
    for idx, case in enumerate(test_cases):
        ax = axes[idx]
        module_key = case["module_key"]
        severity = case["severity"]
        
        # 应用退化
        degraded = generator.apply_degradation(baseline.copy(), module_key, severity)
        
        # 计算差异
        diff = degraded - baseline
        
        # 绘制曲线
        ax.plot(freq_axis, baseline, 'k-', alpha=0.3, linewidth=1, label='Baseline')
        ax.plot(freq_axis, degraded, color=case["color"], alpha=0.8, linewidth=1.2, label='Degraded')
        
        # 突出显示差异区域
        ax.fill_between(freq_axis, baseline, degraded, alpha=0.2, color=case["color"])
        
        # 设置标题和标签
        ax.set_title(f"{case['title']}\nExpected: {case['expected']}", fontsize=11)
        ax.set_xlabel('Frequency Index')
        ax.set_ylabel('Amplitude (dB)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 计算物理特征指标
        metrics = {
            "module_key": module_key,
            "diff_max": float(np.max(np.abs(diff))),
            "diff_std": float(np.std(diff)),
            "low_freq_drop": float(np.mean(baseline[:100]) - np.mean(degraded[:100])),
            "high_freq_drop": float(np.mean(baseline[-100:]) - np.mean(degraded[-100:])),
        }
        results.append(metrics)
        
        # 添加物理特征注释
        if module_key == "ac_coupling":
            # 验证低频塌陷
            if metrics["low_freq_drop"] > 0.5:
                status = "✅ PASS"
            else:
                status = "⚠️ WEAK"
            ax.annotate(f"Low freq drop: {metrics['low_freq_drop']:.2f} dB\n{status}",
                       xy=(50, degraded[50]), fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        elif module_key == "adc_module":
            # 验证量化阶梯
            # 计算差分的标准差（量化阶梯会产生高频跳变）
            diff_of_diff = np.diff(diff)
            step_signature = float(np.std(diff_of_diff))
            if step_signature > 0.001:
                status = "✅ PASS"
            else:
                status = "⚠️ WEAK"
            ax.annotate(f"Step signature: {step_signature:.4f}\n{status}",
                       xy=(500, degraded[500]), fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        elif module_key == "input_connector":
            # 验证纹波
            # 使用 FFT 检测周期性
            fft = np.abs(np.fft.fft(diff))
            fft_peak = float(np.max(fft[5:100]))  # 排除 DC
            if fft_peak > 1.0:
                status = "✅ PASS"
            else:
                status = "⚠️ WEAK"
            ax.annotate(f"Ripple FFT peak: {fft_peak:.2f}\n{status}",
                       xy=(500, degraded[500]), fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        elif module_key == "lo1_synth":
            # 验证信号丢失
            min_value = float(np.min(degraded))
            drop_depth = float(np.mean(baseline)) - min_value
            if drop_depth > 3.0:
                status = "✅ PASS"
            else:
                status = "⚠️ WEAK"
            ax.annotate(f"Drop depth: {drop_depth:.2f} dB\n{status}",
                       xy=(np.argmin(degraded), min_value), fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 添加总标题
    fig.suptitle('V-D.3 Physics Feature Verification (物理特征验证)\nCurveGenerator Output - Template System DISABLED',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图片
    output_path = output_dir / "物理特征对比图.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 物理特征对比图已保存: {output_path}")
    
    # 打印验证结果
    print("\n" + "=" * 60)
    print("V-D.3 物理特征验证结果")
    print("=" * 60)
    for r in results:
        print(f"\n{r['module_key']}:")
        print(f"  - 最大差异: {r['diff_max']:.4f} dB")
        print(f"  - 差异标准差: {r['diff_std']:.4f}")
        print(f"  - 低频下降: {r['low_freq_drop']:.4f} dB")
        print(f"  - 高频下降: {r['high_freq_drop']:.4f} dB")
    
    return results


if __name__ == "__main__":
    verify_physics_features()
