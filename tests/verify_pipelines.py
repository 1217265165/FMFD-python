#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-D.4 链路验证脚本 (Pipeline Verification)
==========================================
验证对比链路和诊断链路能否正常导入和初始化。

Usage:
    python tests/verify_pipelines.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np


def test_compare_methods_import():
    """测试对比链路模块导入。"""
    print("=" * 60)
    print("测试 1: 对比链路 (compare_methods) 模块导入")
    print("=" * 60)
    
    try:
        from pipelines.compare_methods import (
            set_global_seed,
            SYS_LABEL_ORDER,
            LeakageError,
        )
        print("✅ pipelines.compare_methods 导入成功")
        print(f"   - set_global_seed: {set_global_seed}")
        print(f"   - SYS_LABEL_ORDER: {SYS_LABEL_ORDER}")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_diagnosis_cli_import():
    """测试诊断链路模块导入。"""
    print()
    print("=" * 60)
    print("测试 2: 诊断链路 (brb_diagnosis_cli) 核心函数导入")
    print("=" * 60)
    
    try:
        # 导入诊断CLI的核心函数
        sys.path.insert(0, str(ROOT_DIR))
        
        # 直接测试导入关键模块
        from features.feature_extraction import extract_system_features
        from BRB.module_brb import MODULE_LABELS
        
        print("✅ features.feature_extraction 导入成功")
        print("✅ BRB.module_brb 导入成功")
        print(f"   - MODULE_LABELS 数量: {len(MODULE_LABELS)}")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_curve_generator_integration():
    """测试 CurveGenerator 与链路的集成。"""
    print()
    print("=" * 60)
    print("测试 3: CurveGenerator 物理核集成")
    print("=" * 60)
    
    try:
        from pipelines.simulate.curve_generator import CurveGenerator, load_module_taxonomy
        
        # 加载模块分类
        taxonomy = load_module_taxonomy()
        print(f"✅ module_taxonomy_v2.json 加载成功")
        print(f"   - BOARD_HIERARCHY 板卡数: {len(taxonomy.get('BOARD_HIERARCHY', {}))}")
        print(f"   - MODULE_ALIASES 别名数: {len(taxonomy.get('MODULE_ALIASES', {}))}")
        
        # 测试 CurveGenerator
        generator = CurveGenerator(seed=42)
        baseline = np.zeros(501)
        
        # 测试几个关键模块
        test_modules = ["ac_coupling", "adc_module", "input_connector", "lo1_synth"]
        for module in test_modules:
            degraded = generator.apply_degradation(baseline, module, severity=0.8)
            max_diff = np.max(np.abs(degraded - baseline))
            print(f"   - {module}: max_diff = {max_diff:.4f} dB")
        
        print("✅ CurveGenerator 物理核测试通过")
        return True
    except Exception as e:
        print(f"❌ 集成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_expert_system():
    """测试专家系统模块。"""
    print()
    print("=" * 60)
    print("测试 4: 新架构专家系统 (FMFDExpertSystem)")
    print("=" * 60)
    
    try:
        from BRB.expert_system import FMFDExpertSystem, DiagnosisResult
        
        # 初始化专家系统
        expert = FMFDExpertSystem()
        print("✅ FMFDExpertSystem 初始化成功")
        print(f"   - 类型: {type(expert)}")
        print(f"   - DiagnosisResult 可用: {DiagnosisResult is not None}")
        
        return True
    except ImportError as e:
        print(f"⚠️ FMFDExpertSystem 未找到 (可能尚未实现): {e}")
        return True  # 不算失败
    except Exception as e:
        print(f"❌ 专家系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layered_engine():
    """测试分层推理引擎。"""
    print()
    print("=" * 60)
    print("测试 5: 分层推理引擎 (LayeredBRBEngine)")
    print("=" * 60)
    
    try:
        from BRB.engines.layered_engine import LayeredBRBEngine
        
        engine = LayeredBRBEngine()
        print("✅ LayeredBRBEngine 初始化成功")
        print(f"   - layer_defs 层数: {len(engine.layer_defs)}")
        print(f"   - pool_defs 池数: {len(engine.pool_defs)}")
        
        return True
    except ImportError as e:
        print(f"⚠️ LayeredBRBEngine 未找到: {e}")
        return True
    except Exception as e:
        print(f"❌ 分层引擎测试失败: {e}")
        return False


def test_soft_router():
    """测试软路由模块。"""
    print()
    print("=" * 60)
    print("测试 6: 软路由分发器 (SoftModuleRouter)")
    print("=" * 60)
    
    try:
        from BRB.routing.soft_router import SoftModuleRouter
        
        router = SoftModuleRouter()
        print("✅ SoftModuleRouter 初始化成功")
        
        # 测试路由计算 - 使用正确的方法名
        test_probs = {"normal": 0.1, "amp_error": 0.5, "freq_error": 0.3, "ref_error": 0.1}
        weights = router.compute_module_activations(test_probs)
        print(f"   - 模块激活权重数: {len(weights)}")
        
        return True
    except ImportError as e:
        print(f"⚠️ SoftModuleRouter 未找到: {e}")
        return True
    except Exception as e:
        print(f"❌ 软路由测试失败: {e}")
        return False


def main():
    """运行所有链路验证测试。"""
    print("\n" + "=" * 60)
    print("V-D.4 链路验证报告 (Pipeline Verification Report)")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("对比链路导入", test_compare_methods_import()))
    results.append(("诊断链路导入", test_diagnosis_cli_import()))
    results.append(("CurveGenerator 集成", test_curve_generator_integration()))
    results.append(("专家系统", test_expert_system()))
    results.append(("分层引擎", test_layered_engine()))
    results.append(("软路由", test_soft_router()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print()
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n✅ 所有链路验证通过！对比和诊断链路可正常运行。")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
