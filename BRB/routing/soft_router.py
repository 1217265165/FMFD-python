#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
软路由模块 (Soft Module Router)
================================
实现系统级故障类型到模块级激活权重的软路由。

对应架构手稿 Diagram 1:
- 输入: 系统级 4 类故障概率分布
- 读取: coupling_matrix.json 中的激活权重
- 输出: 所有模块的激活权重

软激活策略:
- 不做硬切换，而是根据故障概率加权计算模块激活权重
- 支持跨故障类型的耦合效应
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 获取配置文件路径
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def load_coupling_matrix() -> Dict:
    """加载耦合矩阵配置。"""
    config_path = CONFIG_DIR / "coupling_matrix.json"
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


class SoftModuleRouter:
    """
    软路由器：根据系统级诊断结果计算模块激活权重。
    
    实现软激活策略：
    1. 读取系统级故障概率分布
    2. 应用故障类型间的耦合矩阵
    3. 计算每个模块的激活权重
    4. 返回模块激活权重分布
    
    Attributes
    ----------
    coupling_config : Dict
        耦合矩阵配置
    module_config : Dict
        模块分类体系配置
    top_k : int
        考虑的故障类型数量
    min_threshold : float
        最小激活阈值
    """
    
    def __init__(
        self, 
        config_override: Optional[Dict] = None,
        taxonomy_override: Optional[Dict] = None,
        top_k: int = 2,
        min_threshold: float = 0.1
    ):
        """
        初始化软路由器。
        
        Parameters
        ----------
        config_override : Dict, optional
            耦合矩阵配置覆盖
        taxonomy_override : Dict, optional
            模块分类体系配置覆盖
        top_k : int
            考虑的故障类型数量
        min_threshold : float
            最小激活阈值
        """
        self.coupling_config = config_override or load_coupling_matrix()
        self.module_config = taxonomy_override or load_module_taxonomy()
        self.top_k = top_k
        self.min_threshold = min_threshold
        
        # 预加载配置
        self.fault_coupling = self.coupling_config.get("FAULT_TYPE_COUPLING", {})
        self.module_weights = self.coupling_config.get("MODULE_ACTIVATION_WEIGHTS", {})
        self.board_weights = self.coupling_config.get("BOARD_ACTIVATION_WEIGHTS", {})
        self.router_config = self.coupling_config.get("SOFT_ROUTER_CONFIG", {})
        
        # 模块别名到全名的映射
        self.module_aliases = self.module_config.get("MODULE_ALIASES", {})
        
    def _apply_fault_coupling(
        self, 
        system_probs: Dict[str, float]
    ) -> Dict[str, float]:
        """应用故障类型间的耦合效应。
        
        Parameters
        ----------
        system_probs : Dict
            系统级故障概率分布
            
        Returns
        -------
        Dict
            应用耦合后的调整概率
        """
        # 获取主导故障类型
        fault_types = ["amp_error", "freq_error", "ref_error"]
        fault_probs = {ft: system_probs.get(ft, 0.0) for ft in fault_types}
        
        if not any(fault_probs.values()):
            return system_probs.copy()
        
        # 获取 Top-K 故障类型
        sorted_faults = sorted(fault_probs.items(), key=lambda x: x[1], reverse=True)
        top_faults = sorted_faults[:self.top_k]
        primary_fault = top_faults[0][0] if top_faults else "normal"
        
        # 应用耦合矩阵
        adjusted = {}
        coupling_row = self.fault_coupling.get(primary_fault, {})
        
        for fault_type in fault_types + ["normal"]:
            original_prob = system_probs.get(fault_type, 0.0)
            coupling_factor = coupling_row.get(fault_type, 1.0)
            adjusted[fault_type] = original_prob * coupling_factor
        
        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def _get_module_base_weight(
        self, 
        module_alias: str, 
        fault_type: str
    ) -> float:
        """获取模块在指定故障类型下的基础权重。"""
        fault_weights = self.module_weights.get(fault_type, {})
        
        # 检查是否是 normal
        if fault_type == "normal":
            default_weight = fault_weights.get("_default_weight", 0.05)
            return default_weight
        
        return fault_weights.get(module_alias, 0.1)
    
    def _softmax(self, values: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
        """Softmax 归一化。"""
        if not values:
            return {}
        
        max_val = max(values.values())
        exp_values = {}
        for key, val in values.items():
            exp_values[key] = math.exp((val - max_val) / temperature)
        
        total = sum(exp_values.values())
        if total <= 0:
            uniform = 1.0 / len(values)
            return {k: uniform for k in values}
        
        return {k: v / total for k, v in exp_values.items()}
    
    def compute_module_activations(
        self, 
        system_probs: Dict[str, float],
        apply_coupling: bool = True
    ) -> Dict[str, float]:
        """计算所有模块的激活权重。
        
        Parameters
        ----------
        system_probs : Dict
            系统级故障概率分布 {fault_type: probability}
        apply_coupling : bool
            是否应用故障类型耦合
            
        Returns
        -------
        Dict
            模块激活权重 {module_alias: weight}
        """
        # 应用耦合（如果启用）
        if apply_coupling:
            adjusted_probs = self._apply_fault_coupling(system_probs)
        else:
            adjusted_probs = system_probs.copy()
        
        # 计算每个模块的激活权重
        module_activations = {}
        
        for module_alias in self.module_aliases.keys():
            # 加权平均各故障类型下的模块权重
            weighted_activation = 0.0
            
            for fault_type in ["amp_error", "freq_error", "ref_error", "normal"]:
                fault_prob = adjusted_probs.get(fault_type, 0.0)
                module_weight = self._get_module_base_weight(module_alias, fault_type)
                weighted_activation += fault_prob * module_weight
            
            module_activations[module_alias] = weighted_activation
        
        # 过滤低于阈值的模块
        filtered = {
            k: v for k, v in module_activations.items() 
            if v >= self.min_threshold
        }
        
        # 如果过滤后为空，返回所有模块
        if not filtered:
            filtered = module_activations
        
        return filtered
    
    def compute_module_probs(
        self, 
        system_probs: Dict[str, float],
        normalize: bool = True,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """计算模块概率分布。
        
        Parameters
        ----------
        system_probs : Dict
            系统级故障概率分布
        normalize : bool
            是否进行 Softmax 归一化
        temperature : float
            Softmax 温度参数
            
        Returns
        -------
        Dict
            模块概率分布（别名格式）
        """
        activations = self.compute_module_activations(system_probs)
        
        if normalize:
            return self._softmax(activations, temperature)
        
        return activations
    
    def compute_module_probs_v2(
        self, 
        system_probs: Dict[str, float],
        normalize: bool = True
    ) -> Dict[str, float]:
        """计算模块概率分布（使用 V2 完整名称）。
        
        Parameters
        ----------
        system_probs : Dict
            系统级故障概率分布
        normalize : bool
            是否归一化
            
        Returns
        -------
        Dict
            模块概率分布（V2 完整名称格式）
        """
        alias_probs = self.compute_module_probs(system_probs, normalize)
        
        # 转换为 V2 名称
        v2_probs = {}
        for alias, prob in alias_probs.items():
            v2_name = self.module_aliases.get(alias, alias)
            v2_probs[v2_name] = prob
        
        return v2_probs
    
    def get_top_k_modules(
        self, 
        system_probs: Dict[str, float], 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """获取激活度最高的 K 个模块。
        
        Parameters
        ----------
        system_probs : Dict
            系统级故障概率分布
        k : int
            返回的模块数量
            
        Returns
        -------
        List[Tuple[str, float]]
            [(模块V2名称, 概率), ...] 按概率降序排列
        """
        v2_probs = self.compute_module_probs_v2(system_probs)
        sorted_modules = sorted(v2_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_modules[:k]
    
    def route_with_trace(
        self, 
        system_probs: Dict[str, float]
    ) -> Dict:
        """执行路由并返回详细追踪信息。
        
        Parameters
        ----------
        system_probs : Dict
            系统级故障概率分布
            
        Returns
        -------
        Dict
            包含路由过程详细信息的字典
        """
        # 应用耦合
        adjusted_probs = self._apply_fault_coupling(system_probs)
        
        # 计算激活权重
        alias_activations = self.compute_module_activations(system_probs)
        alias_probs = self.compute_module_probs(system_probs)
        v2_probs = self.compute_module_probs_v2(system_probs)
        
        # 获取 Top-5
        top_5 = self.get_top_k_modules(system_probs, k=5)
        
        return {
            "input_probs": system_probs,
            "adjusted_probs": adjusted_probs,
            "module_activations": alias_activations,
            "module_probs_alias": alias_probs,
            "module_probs_v2": v2_probs,
            "top_5_modules": top_5,
        }


# 创建默认路由器实例
_default_router = None


def get_soft_router() -> SoftModuleRouter:
    """获取软路由器实例（单例模式）。"""
    global _default_router
    if _default_router is None:
        _default_router = SoftModuleRouter()
    return _default_router


def soft_route_modules(system_probs: Dict[str, float]) -> Dict[str, float]:
    """执行软路由（便捷接口）。
    
    Parameters
    ----------
    system_probs : Dict
        系统级故障概率分布
        
    Returns
    -------
    Dict
        模块概率分布（V2 名称）
    """
    router = get_soft_router()
    return router.compute_module_probs_v2(system_probs)
