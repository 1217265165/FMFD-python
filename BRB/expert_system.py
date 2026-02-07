#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一专家系统 (FMFD Expert System)
==================================
封装完整的故障诊断流程，提供统一接口。

诊断流程:
1. FeatureExtract: 从频响曲线提取特征 (X1-X37)
2. SystemInfer: 分层 BRB 推理系统级故障类型
3. SoftRouter: 软路由计算模块激活权重
4. ModuleInfer: 模块级推理定位故障模块

统一接口:
- diagnose(curve_data) -> DiagnosisResult
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 导入子模块
from .engines.layered_engine import LayeredBRBEngine, get_layered_engine
from .routing.soft_router import SoftModuleRouter, get_soft_router

# 获取配置文件路径
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class DiagnosisResult:
    """诊断结果数据类。
    
    Attributes
    ----------
    system_fault_type : str
        系统级故障类型 (normal/amp_error/freq_error/ref_error)
    system_probs : Dict[str, float]
        系统级故障概率分布
    system_confidence : float
        系统级诊断置信度
    top_modules : List[Tuple[str, float]]
        Top-K 模块及概率
    module_probs : Dict[str, float]
        完整模块概率分布
    layer_trace : Dict
        分层推理追踪信息
    routing_trace : Dict
        软路由追踪信息
    features : Dict[str, float]
        提取的特征值
    """
    system_fault_type: str = "normal"
    system_probs: Dict[str, float] = field(default_factory=dict)
    system_confidence: float = 0.0
    top_modules: List[Tuple[str, float]] = field(default_factory=list)
    module_probs: Dict[str, float] = field(default_factory=dict)
    layer_trace: Dict = field(default_factory=dict)
    routing_trace: Dict = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式。"""
        return {
            "system_diagnosis": {
                "predicted_class": self.system_fault_type,
                "probabilities": self.system_probs,
                "confidence": self.system_confidence,
            },
            "module_diagnosis": {
                "top_modules": [
                    {"module": name, "probability": prob}
                    for name, prob in self.top_modules
                ],
                "all_probs": self.module_probs,
            },
            "trace": {
                "layer_trace": self.layer_trace,
                "routing_trace": self.routing_trace,
            },
            "features": self.features,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class FMFDExpertSystem:
    """
    FMFD 统一专家系统。
    
    封装完整的诊断流程，提供统一的 diagnose() 接口。
    
    Attributes
    ----------
    layered_engine : LayeredBRBEngine
        分层推理引擎
    soft_router : SoftModuleRouter
        软路由器
    top_k_modules : int
        返回的 Top-K 模块数量
    """
    
    def __init__(
        self,
        alpha: float = 2.0,
        top_k_modules: int = 5,
        feature_extractor: Optional[Any] = None
    ):
        """
        初始化专家系统。
        
        Parameters
        ----------
        alpha : float
            BRB Softmax 温度参数
        top_k_modules : int
            返回的 Top-K 模块数量
        feature_extractor : callable, optional
            自定义特征提取器，接受 curve_data 返回 features dict
        """
        self.layered_engine = LayeredBRBEngine(alpha=alpha)
        self.soft_router = SoftModuleRouter()
        self.top_k_modules = top_k_modules
        self.feature_extractor = feature_extractor
        
    def _extract_features(
        self, 
        curve_data: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, float]:
        """从曲线数据提取特征。
        
        Parameters
        ----------
        curve_data : np.ndarray
            频响曲线数据
        baseline : np.ndarray, optional
            基线曲线（用于计算残差特征）
        envelope : Tuple, optional
            上下包络 (upper, lower)
            
        Returns
        -------
        Dict
            特征字典 {X1: value, X2: value, ...}
        """
        if self.feature_extractor is not None:
            return self.feature_extractor(curve_data, baseline, envelope)
        
        # 使用内置特征提取
        try:
            from features.feature_extraction import extract_system_features
            return extract_system_features(curve_data, baseline, envelope)
        except ImportError:
            # 回退到简化特征提取
            return self._simple_feature_extract(curve_data, baseline)
    
    def _simple_feature_extract(
        self, 
        curve_data: np.ndarray,
        baseline: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """简化特征提取（回退方案）。"""
        arr = np.asarray(curve_data, dtype=float)
        features = {}
        
        # X1: 整体幅度偏移
        features["X1"] = float(np.mean(arr))
        
        # X2: 带内平坦度
        inband = arr[:int(len(arr) * 0.6)] if len(arr) > 5 else arr
        features["X2"] = float(np.var(inband))
        
        # X3: 高频段衰减斜率
        tail = arr[int(len(arr) * 0.8):] if len(arr) > 5 else arr
        if len(tail) >= 2:
            idx = np.arange(len(tail))
            coef = np.polyfit(idx, tail, 1)[0]
            features["X3"] = float(coef)
        else:
            features["X3"] = 0.0
        
        # X4: 频率标度非线性度
        x_axis = np.linspace(0, 1, len(arr))
        try:
            coef = np.polyfit(x_axis, arr, 1)
            fit = np.polyval(coef, x_axis)
            residual = arr - fit
            features["X4"] = float(np.std(residual))
        except Exception:
            features["X4"] = 0.0
        
        # X5: 幅度缩放一致性
        centered = arr - np.mean(arr)
        denom = np.max(np.abs(centered)) + 1e-12
        normalized = centered / denom
        features["X5"] = float(np.std(normalized))
        
        # 其他特征设置默认值
        for i in range(6, 38):
            features[f"X{i}"] = 0.0
        
        return features
    
    def diagnose(
        self,
        curve_data: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        return_trace: bool = True
    ) -> DiagnosisResult:
        """执行完整诊断。
        
        Parameters
        ----------
        curve_data : np.ndarray
            频响曲线数据
        baseline : np.ndarray, optional
            基线曲线
        envelope : Tuple, optional
            上下包络
        return_trace : bool
            是否返回详细追踪信息
            
        Returns
        -------
        DiagnosisResult
            诊断结果
        """
        # Step 1: 特征提取
        features = self._extract_features(curve_data, baseline, envelope)
        
        # Step 2: 分层系统级推理
        layer_trace = self.layered_engine.infer_with_trace(features)
        system_probs = layer_trace["final_beliefs"]
        system_fault_type = layer_trace["predicted_fault_type"]
        system_confidence = layer_trace["max_probability"]
        
        # Step 3: 软路由
        routing_trace = self.soft_router.route_with_trace(system_probs)
        module_probs = routing_trace["module_probs_v2"]
        top_modules = routing_trace["top_5_modules"][:self.top_k_modules]
        
        # 构建结果
        result = DiagnosisResult(
            system_fault_type=system_fault_type,
            system_probs=system_probs,
            system_confidence=system_confidence,
            top_modules=top_modules,
            module_probs=module_probs,
            layer_trace=layer_trace if return_trace else {},
            routing_trace=routing_trace if return_trace else {},
            features=features,
        )
        
        return result
    
    def diagnose_from_features(
        self,
        features: Dict[str, float],
        return_trace: bool = True
    ) -> DiagnosisResult:
        """从已提取的特征进行诊断。
        
        Parameters
        ----------
        features : Dict
            特征字典
        return_trace : bool
            是否返回详细追踪信息
            
        Returns
        -------
        DiagnosisResult
            诊断结果
        """
        # Step 2: 分层系统级推理
        layer_trace = self.layered_engine.infer_with_trace(features)
        system_probs = layer_trace["final_beliefs"]
        system_fault_type = layer_trace["predicted_fault_type"]
        system_confidence = layer_trace["max_probability"]
        
        # Step 3: 软路由
        routing_trace = self.soft_router.route_with_trace(system_probs)
        module_probs = routing_trace["module_probs_v2"]
        top_modules = routing_trace["top_5_modules"][:self.top_k_modules]
        
        # 构建结果
        result = DiagnosisResult(
            system_fault_type=system_fault_type,
            system_probs=system_probs,
            system_confidence=system_confidence,
            top_modules=top_modules,
            module_probs=module_probs,
            layer_trace=layer_trace if return_trace else {},
            routing_trace=routing_trace if return_trace else {},
            features=features,
        )
        
        return result


# 创建默认专家系统实例
_default_expert_system = None


def get_expert_system() -> FMFDExpertSystem:
    """获取专家系统实例（单例模式）。"""
    global _default_expert_system
    if _default_expert_system is None:
        _default_expert_system = FMFDExpertSystem()
    return _default_expert_system


def diagnose(
    curve_data: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    envelope: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> DiagnosisResult:
    """执行诊断（便捷接口）。
    
    Parameters
    ----------
    curve_data : np.ndarray
        频响曲线数据
    baseline : np.ndarray, optional
        基线曲线
    envelope : Tuple, optional
        上下包络
        
    Returns
    -------
    DiagnosisResult
        诊断结果
    """
    expert = get_expert_system()
    return expert.diagnose(curve_data, baseline, envelope)


def diagnose_from_features(features: Dict[str, float]) -> DiagnosisResult:
    """从特征进行诊断（便捷接口）。
    
    Parameters
    ----------
    features : Dict
        特征字典
        
    Returns
    -------
    DiagnosisResult
        诊断结果
    """
    expert = get_expert_system()
    return expert.diagnose_from_features(features)
