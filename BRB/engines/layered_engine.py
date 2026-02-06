#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层推理引擎 (Layered BRB Engine)
==================================
实现分层特征注入的 BRB 推理引擎。

对应架构手稿 Diagram 2:
- Layer 1: 基础特征 → 初始 Belief
- Layer 2: 派生特征 + Layer 1 Belief → 更新 Belief
- Layer 3: 高级特征 + Layer 2 Belief → 最终 Belief

特征从 config/feature_definitions.json 动态加载。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 获取配置文件路径
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def load_feature_definitions() -> Dict:
    """加载特征定义配置。"""
    config_path = CONFIG_DIR / "feature_definitions.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_module_taxonomy() -> Dict:
    """加载模块分类体系配置。"""
    config_path = CONFIG_DIR / "module_taxonomy_v2.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


class LayeredBRBEngine:
    """
    分层 BRB 推理引擎。
    
    实现从 Layer 1 到 Layer 3 的流式推理：
    - 每一层接收当前层特征和上一层的 Belief
    - 输出更新后的 Belief 概率分布
    
    Attributes
    ----------
    feature_config : Dict
        从 feature_definitions.json 加载的特征配置
    module_config : Dict
        从 module_taxonomy_v2.json 加载的模块配置
    alpha : float
        Softmax 温度参数
    """
    
    def __init__(self, alpha: float = 2.0, config_override: Optional[Dict] = None):
        """
        初始化分层推理引擎。
        
        Parameters
        ----------
        alpha : float
            Softmax 温度参数，控制输出分布的锐度
        config_override : Dict, optional
            配置覆盖，用于测试
        """
        self.alpha = alpha
        self.feature_config = config_override or load_feature_definitions()
        self.module_config = load_module_taxonomy()
        
        # 预加载层定义
        self.layer_defs = self.feature_config.get("LAYER_DEFINITIONS", {})
        self.pool_defs = self.feature_config.get("FEATURE_POOLS", {})
        self.norm_params = self.feature_config.get("NORMALIZATION_PARAMS", {})
        self.feature_schema = self.feature_config.get("FEATURE_SCHEMA", {})
        
    def _get_layer_features(self, layer: int) -> List[str]:
        """获取指定层的特征列表。"""
        layer_key = f"layer_{layer}"
        return self.layer_defs.get(layer_key, {}).get("features", [])
    
    def _get_pool_features(self, pool_name: str) -> List[str]:
        """获取指定池的特征列表。"""
        return self.pool_defs.get(pool_name, {}).get("features", [])
    
    def _normalize_feature(self, feature_key: str, value: float) -> float:
        """归一化单个特征值到 [0, 1] 范围。"""
        params = self.norm_params.get(feature_key, {})
        lower = params.get("lower", 0.0)
        upper = params.get("upper", 1.0)
        method = params.get("method", "linear")
        
        if method == "log":
            # 对数归一化
            if value <= 0:
                value = 1e-12
            if lower <= 0:
                lower = 1e-12
            if upper <= 0:
                upper = 1e-12
            value = math.log(value)
            lower = math.log(lower)
            upper = math.log(upper)
        
        # 线性归一化
        if upper == lower:
            return 0.5
        normalized = (value - lower) / (upper - lower)
        return max(0.0, min(1.0, normalized))
    
    def _triangular_membership(
        self, value: float, low: float = 0.15, center: float = 0.35, high: float = 0.7
    ) -> Tuple[float, float, float]:
        """三角隶属度函数，返回 (Low, Normal, High) 隶属度。"""
        if value <= low:
            return 1.0, 0.0, 0.0
        if low < value < center:
            low_mem = (center - value) / (center - low)
            normal_mem = 1.0 - low_mem
            return low_mem, normal_mem, 0.0
        if center <= value < high:
            high_mem = (value - center) / (high - center)
            normal_mem = 1.0 - high_mem
            return 0.0, normal_mem, high_mem
        return 0.0, 0.0, 1.0
    
    def _compute_layer_activation(
        self,
        features: Dict[str, float],
        layer_features: List[str],
        fault_pool: str
    ) -> float:
        """计算单层的激活度。
        
        Parameters
        ----------
        features : Dict
            完整特征字典
        layer_features : List[str]
            当前层使用的特征列表
        fault_pool : str
            故障类型对应的特征池名称
            
        Returns
        -------
        float
            激活度 [0, 1]
        """
        pool_features = set(self._get_pool_features(fault_pool))
        relevant_features = [f for f in layer_features if f in pool_features]
        
        if not relevant_features:
            return 0.0
        
        total_activation = 0.0
        total_weight = 0.0
        
        for feat_key in relevant_features:
            raw_value = abs(float(features.get(feat_key, 0.0)))
            normalized = self._normalize_feature(feat_key, raw_value)
            _, _, high_degree = self._triangular_membership(normalized)
            
            # 权重可以根据特征重要性调整
            weight = 1.0
            total_activation += weight * high_degree
            total_weight += weight
        
        if total_weight > 0:
            return total_activation / total_weight
        return 0.0
    
    def _apply_prior_belief(
        self, 
        current_activation: float, 
        prior_belief: float,
        blend_weight: float = 0.3
    ) -> float:
        """融合当前层激活度和上一层 Belief。
        
        Parameters
        ----------
        current_activation : float
            当前层计算的激活度
        prior_belief : float
            上一层输出的 Belief
        blend_weight : float
            上一层 Belief 的融合权重
            
        Returns
        -------
        float
            融合后的激活度
        """
        return (1.0 - blend_weight) * current_activation + blend_weight * prior_belief
    
    def _softmax_normalize(self, activations: Dict[str, float]) -> Dict[str, float]:
        """对激活度进行 Softmax 归一化。"""
        if not activations:
            return {}
        
        max_val = max(activations.values())
        exp_values = {}
        for key, val in activations.items():
            exp_values[key] = math.exp(self.alpha * (val - max_val))
        
        total = sum(exp_values.values())
        if total <= 0:
            uniform = 1.0 / len(activations)
            return {k: uniform for k in activations}
        
        return {k: v / total for k, v in exp_values.items()}
    
    def infer_layer(
        self,
        features: Dict[str, float],
        layer: int,
        prior_beliefs: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """执行单层推理。
        
        Parameters
        ----------
        features : Dict
            完整特征字典
        layer : int
            当前层编号 (1, 2, 3)
        prior_beliefs : Dict, optional
            上一层输出的 Belief 分布
            
        Returns
        -------
        Dict
            当前层输出的 Belief 分布
        """
        layer_features = self._get_layer_features(layer)
        
        # 计算各故障类型的激活度
        activations = {}
        for fault_type, pool_name in [
            ("amp_error", "amp_pool"),
            ("freq_error", "freq_pool"),
            ("ref_error", "ref_pool"),
        ]:
            activation = self._compute_layer_activation(features, layer_features, pool_name)
            
            # 融合上一层 Belief
            if prior_beliefs and fault_type in prior_beliefs:
                prior = prior_beliefs[fault_type]
                # 层越深，融合权重越大
                blend_weight = 0.2 + 0.1 * (layer - 1)
                activation = self._apply_prior_belief(activation, prior, blend_weight)
            
            activations[fault_type] = activation
        
        # 计算 normal 的激活度（1 - max(其他激活度)）
        max_fault_activation = max(activations.values()) if activations else 0.0
        activations["normal"] = max(0.0, 1.0 - max_fault_activation)
        
        # Softmax 归一化
        return self._softmax_normalize(activations)
    
    def infer_full(self, features: Dict[str, float]) -> Dict[str, float]:
        """执行完整的分层推理（3 层）。
        
        Parameters
        ----------
        features : Dict
            完整特征字典
            
        Returns
        -------
        Dict
            最终的故障类型概率分布
        """
        # Layer 1
        layer1_beliefs = self.infer_layer(features, layer=1, prior_beliefs=None)
        
        # Layer 2
        layer2_beliefs = self.infer_layer(features, layer=2, prior_beliefs=layer1_beliefs)
        
        # Layer 3
        layer3_beliefs = self.infer_layer(features, layer=3, prior_beliefs=layer2_beliefs)
        
        return layer3_beliefs
    
    def infer_with_trace(self, features: Dict[str, float]) -> Dict:
        """执行推理并返回详细追踪信息。
        
        Parameters
        ----------
        features : Dict
            完整特征字典
            
        Returns
        -------
        Dict
            包含各层输出和最终结果的详细信息
        """
        # Layer 1
        layer1_beliefs = self.infer_layer(features, layer=1, prior_beliefs=None)
        
        # Layer 2
        layer2_beliefs = self.infer_layer(features, layer=2, prior_beliefs=layer1_beliefs)
        
        # Layer 3
        layer3_beliefs = self.infer_layer(features, layer=3, prior_beliefs=layer2_beliefs)
        
        # 确定最终预测
        final_pred = max(layer3_beliefs, key=layer3_beliefs.get)
        
        return {
            "layer_1_beliefs": layer1_beliefs,
            "layer_2_beliefs": layer2_beliefs,
            "layer_3_beliefs": layer3_beliefs,
            "final_beliefs": layer3_beliefs,
            "predicted_fault_type": final_pred,
            "max_probability": layer3_beliefs[final_pred],
        }


# 创建默认引擎实例
_default_engine = None


def get_layered_engine(alpha: float = 2.0) -> LayeredBRBEngine:
    """获取分层推理引擎实例（单例模式）。"""
    global _default_engine
    if _default_engine is None:
        _default_engine = LayeredBRBEngine(alpha=alpha)
    return _default_engine


def layered_system_infer(features: Dict[str, float]) -> Dict[str, float]:
    """执行分层系统级推理（便捷接口）。
    
    Parameters
    ----------
    features : Dict
        特征字典
        
    Returns
    -------
    Dict
        故障类型概率分布
    """
    engine = get_layered_engine()
    return engine.infer_full(features)
