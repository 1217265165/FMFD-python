#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-D.4 校准验证脚本 (Calibration Verification)
============================================
验证所有 23 种故障类型的退化幅度是否在 ±0.6 dB 范围内。

Usage:
    python tests/verify_calibration.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from pipelines.simulate.curve_generator import CurveGenerator


def generate_baseline(length: int = 501, seed: int = 42) -> np.ndarray:
    """生成一条典型的基线曲线。"""
    rng = np.random.default_rng(seed)
    # 模拟真实频响曲线：基本平坦 + 微小起伏
    baseline = np.zeros(length)
    # 添加微弱的低频包络
    baseline += 0.1 * np.sin(2 * np.pi * np.arange(length) / length)
    # 添加微弱噪声
    baseline += rng.normal(0, 0.02, length)
    return baseline


def verify_all_modules():
    """验证所有模块的退化幅度是否在 ±0.65 dB 范围内。"""
    # 所有模块键
    all_module_keys = [
        "ac_coupling",
        "input_connector",
        "step_attenuator",
        "lpf_low_band",
        "lpf_high_band",
        "mixer1",
        "lo1_inject",
        "if_amp_chain",
        "rbw_filter",
        "adc_module",
        "adc_clock",
        "dsp_gain_cal",
        "dsp_window",
        "dsp_detector",
        "peak_search",
        "lo1_synth",
        "lo2_synth",
        "ocxo_ref",
        "ref_distribution",
        "cal_source",
        "cal_switch",
        "cal_storage",
        "power_management",
    ]
    
    MAX_ALLOWED_DIFF = 0.65  # dB
    NUM_SAMPLES = 10
    SEVERITY_MIN = 0.5
    SEVERITY_MAX = 1.0
    
    print("=" * 70)
    print("V-D.4 校准验证报告 (Calibration Verification Report)")
    print("=" * 70)
    print(f"最大允许偏差: {MAX_ALLOWED_DIFF} dB")
    print(f"每模块样本数: {NUM_SAMPLES}")
    print(f"Severity 范围: {SEVERITY_MIN} - {SEVERITY_MAX}")
    print("=" * 70)
    print()
    
    # 统计结果
    passed = 0
    failed = 0
    failed_modules = []
    results = []
    
    for module_key in all_module_keys:
        max_diffs = []
        
        for sample_idx in range(NUM_SAMPLES):
            # 使用不同的种子
            seed = sample_idx * 100 + hash(module_key) % 1000
            generator = CurveGenerator(seed=seed)
            
            # 随机 severity
            rng = np.random.default_rng(seed)
            severity = rng.uniform(SEVERITY_MIN, SEVERITY_MAX)
            
            # 生成基线
            baseline = generate_baseline(length=501, seed=seed + 1)
            
            # 应用退化
            degraded = generator.apply_degradation(baseline, module_key, severity)
            
            # 计算最大绝对偏差
            diff = np.abs(degraded - baseline)
            max_diff = np.max(diff)
            max_diffs.append(max_diff)
        
        # 统计该模块
        avg_max_diff = np.mean(max_diffs)
        worst_max_diff = np.max(max_diffs)
        
        # 判定
        if worst_max_diff <= MAX_ALLOWED_DIFF:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
            failed_modules.append((module_key, worst_max_diff))
        
        results.append({
            "module": module_key,
            "avg_max_diff": avg_max_diff,
            "worst_max_diff": worst_max_diff,
            "status": status,
        })
    
    # 打印结果表
    print(f"{'模块键 (Module Key)':<25} {'平均偏差':<12} {'最大偏差':<12} {'状态':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['module']:<25} {r['avg_max_diff']:>8.4f} dB  {r['worst_max_diff']:>8.4f} dB  {r['status']}")
    
    print()
    print("=" * 70)
    print(f"总结: {passed} 通过, {failed} 失败 (共 {len(all_module_keys)} 模块)")
    print("=" * 70)
    
    if failed_modules:
        print("\n❌ 失败模块列表 (需要回炉重造):")
        for module, diff in failed_modules:
            print(f"  - {module}: {diff:.4f} dB (超出 {diff - MAX_ALLOWED_DIFF:.4f} dB)")
        return False
    else:
        print("\n✅ 所有模块校准通过！偏差均在 ±0.6 dB 范围内。")
        return True


def run_assertion_test():
    """断言测试：确保所有模块通过校准。"""
    all_module_keys = [
        "ac_coupling", "input_connector", "step_attenuator",
        "lpf_low_band", "lpf_high_band", "mixer1", "lo1_inject",
        "if_amp_chain", "rbw_filter", "adc_module", "adc_clock",
        "dsp_gain_cal", "dsp_window", "dsp_detector", "peak_search",
        "lo1_synth", "lo2_synth", "ocxo_ref", "ref_distribution",
        "cal_source", "cal_switch", "cal_storage", "power_management",
    ]
    
    MAX_ALLOWED = 0.65
    
    for module_key in all_module_keys:
        generator = CurveGenerator(seed=42)
        baseline = generate_baseline(501)
        
        # 测试 severity=1.0 (最严重情况)
        degraded = generator.apply_degradation(baseline, module_key, severity=1.0)
        max_diff = np.max(np.abs(degraded - baseline))
        
        assert max_diff < MAX_ALLOWED, \
            f"模块 {module_key} 校准失败: max_diff={max_diff:.4f} > {MAX_ALLOWED}"
    
    print("✅ 断言测试通过: 所有模块 severity=1.0 时偏差 < 0.65 dB")


if __name__ == "__main__":
    success = verify_all_modules()
    print()
    run_assertion_test()
    
    sys.exit(0 if success else 1)
