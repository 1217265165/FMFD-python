#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3: 低置信度机制与系统级解释

定义低置信度门限和 UNCERTAIN 状态输出策略
当 max_prob < 55% 或 max_prob - second_prob < 10% 时标记 UNCERTAIN
并输出冲突解释字段
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# 特征贡献度计算常量
FEATURE_PREFERENCE_THRESHOLD = 1.5  # 特征倾向判定阈值（top_contrib > runnerup_contrib * 1.5 则倾向 top）
CONFLICT_THRESHOLD = 0.3  # 冲突特征判定阈值（贡献差值 < max_contrib * 0.3 则为冲突）


@dataclass
class UncertaintyConfig:
    """低置信度机制配置
    
    Attributes
    ----------
    max_prob_threshold : float
        最大概率低于此值时标记 UNCERTAIN (默认 0.55)
    gap_threshold : float
        max_prob - second_prob 小于此值时标记 UNCERTAIN (默认 0.10)
    top_support_features : int
        输出支持 top_class 的特征数量 (默认 5)
    top_conflict_features : int
        输出冲突特征数量 (默认 3)
    """
    max_prob_threshold: float = 0.55
    gap_threshold: float = 0.10
    top_support_features: int = 5
    top_conflict_features: int = 3


@dataclass
class UncertaintyResult:
    """低置信度检测结果
    
    Attributes
    ----------
    is_uncertain : bool
        是否为低置信度状态
    reason : str
        低置信度原因
    top_class : str
        最高概率类别
    runner_up : str
        次高概率类别
    max_prob : float
        最高概率
    second_prob : float
        次高概率
    prob_gap : float
        概率差值
    support_top_features : List[Tuple[str, float]]
        支持 top_class 的特征列表
    support_runnerup_features : List[Tuple[str, float]]
        支持 runner_up 的特征列表
    conflict_features : List[Tuple[str, float, float]]
        冲突特征列表 (feature, top_contrib, runnerup_contrib)
    suggested_actions : List[str]
        建议验证动作
    """
    is_uncertain: bool = False
    reason: str = ""
    top_class: str = ""
    runner_up: str = ""
    max_prob: float = 0.0
    second_prob: float = 0.0
    prob_gap: float = 0.0
    support_top_features: List[Tuple[str, float]] = field(default_factory=list)
    support_runnerup_features: List[Tuple[str, float]] = field(default_factory=list)
    conflict_features: List[Tuple[str, float, float]] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "is_uncertain": self.is_uncertain,
            "reason": self.reason,
            "top_class": self.top_class,
            "runner_up": self.runner_up,
            "max_prob": self.max_prob,
            "second_prob": self.second_prob,
            "prob_gap": self.prob_gap,
            "support_top_features": [
                {"feature": f, "contribution": c}
                for f, c in self.support_top_features
            ],
            "support_runnerup_features": [
                {"feature": f, "contribution": c}
                for f, c in self.support_runnerup_features
            ],
            "conflict_features": [
                {"feature": f, "top_contrib": t, "runnerup_contrib": r}
                for f, t, r in self.conflict_features
            ],
            "suggested_actions": self.suggested_actions,
        }


# 特征对类别的贡献映射
FEATURE_CLASS_AFFINITY = {
    # 幅度失准相关特征
    "X1": {"幅度失准": 0.8, "参考电平失准": 0.5, "频率失准": 0.1, "正常": 0.1},
    "X2": {"幅度失准": 0.7, "参考电平失准": 0.3, "频率失准": 0.2, "正常": 0.1},
    "X3": {"幅度失准": 0.5, "参考电平失准": 0.6, "频率失准": 0.2, "正常": 0.1},
    "X5": {"幅度失准": 0.7, "参考电平失准": 0.5, "频率失准": 0.1, "正常": 0.1},
    "X6": {"幅度失准": 0.9, "参考电平失准": 0.2, "频率失准": 0.1, "正常": 0.1},
    "X10": {"幅度失准": 0.8, "参考电平失准": 0.3, "频率失准": 0.1, "正常": 0.1},
    "X11": {"幅度失准": 0.6, "参考电平失准": 0.5, "频率失准": 0.2, "正常": 0.1},
    "X12": {"幅度失准": 0.7, "参考电平失准": 0.4, "频率失准": 0.1, "正常": 0.1},
    "X19": {"幅度失准": 0.8, "参考电平失准": 0.3, "频率失准": 0.1, "正常": 0.1},
    "X20": {"幅度失准": 0.7, "参考电平失准": 0.2, "频率失准": 0.2, "正常": 0.1},
    
    # 频率失准相关特征
    "X4": {"频率失准": 0.9, "幅度失准": 0.1, "参考电平失准": 0.1, "正常": 0.1},
    "X14": {"频率失准": 0.6, "幅度失准": 0.3, "参考电平失准": 0.2, "正常": 0.1},
    "X15": {"频率失准": 0.7, "幅度失准": 0.2, "参考电平失准": 0.2, "正常": 0.1},
    "X16": {"频率失准": 0.9, "幅度失准": 0.1, "参考电平失准": 0.1, "正常": 0.1},
    "X17": {"频率失准": 0.9, "幅度失准": 0.1, "参考电平失准": 0.1, "正常": 0.1},
    "X18": {"频率失准": 0.8, "幅度失准": 0.1, "参考电平失准": 0.2, "正常": 0.1},
    
    # 参考电平相关特征
    "X13": {"参考电平失准": 0.7, "幅度失准": 0.4, "频率失准": 0.1, "正常": 0.1},
    "X7": {"参考电平失准": 0.6, "幅度失准": 0.5, "频率失准": 0.2, "正常": 0.1},
    "X8": {"参考电平失准": 0.5, "幅度失准": 0.3, "频率失准": 0.3, "正常": 0.1},
    "X9": {"参考电平失准": 0.4, "幅度失准": 0.3, "频率失准": 0.4, "正常": 0.1},
    
    # 幅度细粒度特征
    "X21": {"幅度失准": 0.6, "参考电平失准": 0.3, "频率失准": 0.2, "正常": 0.1},
    "X22": {"幅度失准": 0.7, "参考电平失准": 0.2, "频率失准": 0.2, "正常": 0.1},
}

# 建议验证动作映射
SUGGESTED_ACTIONS = {
    "幅度失准": [
        "复测全频段幅度响应",
        "检查前端衰减器/放大器链路",
        "对比多个信号源的幅度一致性",
        "检查内部校准源输出",
    ],
    "频率失准": [
        "复测频率标度线性度",
        "检查本振锁定状态",
        "对比外部频率标准",
        "检查峰值频率搜索精度",
    ],
    "参考电平失准": [
        "复测参考电平校准点",
        "检查 ADC 量化状态",
        "对比不同 RBW 下的幅度一致性",
        "检查内部校准表完整性",
    ],
    "正常": [
        "当前诊断为正常，但置信度较低",
        "建议增加测试样本或更换测试信号",
    ],
}


def compute_feature_contributions(
    features: Dict[str, float],
    top_class: str,
    runner_up: str,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float, float]]]:
    """计算特征对不同类别的贡献
    
    Parameters
    ----------
    features : Dict[str, float]
        输入特征字典
    top_class : str
        最高概率类别
    runner_up : str
        次高概率类别
        
    Returns
    -------
    Tuple
        (support_top, support_runnerup, conflict)
    """
    support_top: List[Tuple[str, float]] = []
    support_runnerup: List[Tuple[str, float]] = []
    conflict: List[Tuple[str, float, float]] = []
    
    for feat_name, feat_value in features.items():
        if feat_name not in FEATURE_CLASS_AFFINITY:
            continue
            
        affinity = FEATURE_CLASS_AFFINITY[feat_name]
        top_affinity = affinity.get(top_class, 0.1)
        runnerup_affinity = affinity.get(runner_up, 0.1)
        
        # 计算贡献值 = 特征值 * 类别亲和度
        top_contrib = abs(feat_value) * top_affinity
        runnerup_contrib = abs(feat_value) * runnerup_affinity
        
        # 判断特征倾向
        if top_contrib > runnerup_contrib * FEATURE_PREFERENCE_THRESHOLD:
            support_top.append((feat_name, top_contrib))
        elif runnerup_contrib > top_contrib * FEATURE_PREFERENCE_THRESHOLD:
            support_runnerup.append((feat_name, runnerup_contrib))
        elif abs(top_contrib - runnerup_contrib) < max(top_contrib, runnerup_contrib) * CONFLICT_THRESHOLD:
            # 两边贡献接近，认为是冲突特征
            conflict.append((feat_name, top_contrib, runnerup_contrib))
    
    # 排序并返回
    support_top.sort(key=lambda x: x[1], reverse=True)
    support_runnerup.sort(key=lambda x: x[1], reverse=True)
    conflict.sort(key=lambda x: x[1] + x[2], reverse=True)
    
    return support_top, support_runnerup, conflict


def detect_uncertainty(
    probabilities: Dict[str, float],
    features: Dict[str, float],
    config: Optional[UncertaintyConfig] = None,
) -> UncertaintyResult:
    """检测系统级输出的低置信度状态
    
    Parameters
    ----------
    probabilities : Dict[str, float]
        系统级概率分布
    features : Dict[str, float]
        输入特征
    config : UncertaintyConfig, optional
        配置参数
        
    Returns
    -------
    UncertaintyResult
        低置信度检测结果
    """
    cfg = config or UncertaintyConfig()
    result = UncertaintyResult()
    
    # 排序概率
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_probs) < 2:
        return result
    
    result.top_class = sorted_probs[0][0]
    result.max_prob = sorted_probs[0][1]
    result.runner_up = sorted_probs[1][0]
    result.second_prob = sorted_probs[1][1]
    result.prob_gap = result.max_prob - result.second_prob
    
    # 检测 UNCERTAIN 条件
    reasons = []
    if result.max_prob < cfg.max_prob_threshold:
        reasons.append(f"max_prob({result.max_prob:.2%}) < {cfg.max_prob_threshold:.0%}")
    if result.prob_gap < cfg.gap_threshold:
        reasons.append(f"prob_gap({result.prob_gap:.2%}) < {cfg.gap_threshold:.0%}")
    
    if reasons:
        result.is_uncertain = True
        result.reason = " && ".join(reasons)
        
        # 计算特征贡献
        support_top, support_runnerup, conflict = compute_feature_contributions(
            features, result.top_class, result.runner_up
        )
        
        result.support_top_features = support_top[:cfg.top_support_features]
        result.support_runnerup_features = support_runnerup[:cfg.top_support_features]
        result.conflict_features = conflict[:cfg.top_conflict_features]
        
        # 生成建议动作
        actions = []
        if result.top_class in SUGGESTED_ACTIONS:
            actions.extend(SUGGESTED_ACTIONS[result.top_class][:2])
        if result.runner_up in SUGGESTED_ACTIONS:
            actions.extend(SUGGESTED_ACTIONS[result.runner_up][:1])
        actions.append("建议重复测试或更换测试点以提高置信度")
        result.suggested_actions = actions
    
    return result


def format_uncertainty_explanation(result: UncertaintyResult) -> str:
    """格式化低置信度解释为可读文本
    
    Parameters
    ----------
    result : UncertaintyResult
        低置信度检测结果
        
    Returns
    -------
    str
        可读文本
    """
    if not result.is_uncertain:
        return "置信度正常，无需额外解释"
    
    lines = [
        "=" * 60,
        "⚠️ 低置信度警告 (UNCERTAIN)",
        "=" * 60,
        f"原因: {result.reason}",
        f"Top1: {result.top_class} ({result.max_prob:.2%})",
        f"Top2: {result.runner_up} ({result.second_prob:.2%})",
        f"概率差: {result.prob_gap:.2%}",
        "",
        f"支持 [{result.top_class}] 的特征:",
    ]
    
    for feat, contrib in result.support_top_features:
        lines.append(f"  - {feat}: {contrib:.4f}")
    
    lines.append("")
    lines.append(f"支持 [{result.runner_up}] 的特征:")
    for feat, contrib in result.support_runnerup_features:
        lines.append(f"  - {feat}: {contrib:.4f}")
    
    if result.conflict_features:
        lines.append("")
        lines.append("冲突特征 (同时拉扯两类):")
        for feat, top_c, ru_c in result.conflict_features:
            lines.append(f"  - {feat}: {result.top_class}={top_c:.4f}, {result.runner_up}={ru_c:.4f}")
    
    lines.append("")
    lines.append("建议验证动作:")
    for i, action in enumerate(result.suggested_actions, 1):
        lines.append(f"  {i}. {action}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
