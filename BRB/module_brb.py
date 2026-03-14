"""
Module-level BRB reasoning (对应小论文 3.2 分层推理与规则压缩章节)。

本模块实现 20 个模块的压缩式 BRB 推理：利用系统级诊断结果
作为虚拟先验属性 V（式 (3)），仅激活与异常类型相关的规则组，
避免全组合爆炸。规则数≈45、参数≈38，显著少于传统设计。

优化特性（对应准确率提升需求）：
1. 特征分流：根据系统级异常类型，仅使用相关特征进行推理
2. 模块组激活：仅激活与异常类型相关的模块组进行推理
3. 规则压缩：通过物理链路知识压缩规则组合
"""
from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Optional

from pipelines.default_paths import DISABLE_PREAMP, SINGLE_BAND
from .utils import BRBRule, SimpleBRB, normalize_feature
from tools.label_mapping import module_v2_from_v1


# 软门控阈值配置
SOFT_GATING_CONFIG = {
    "TH_NORMAL_STRONG": 0.85,  # Normal 概率高于此值时标记为低置信度
    "TH_P_ABN_FORCE": 0.20,    # 异常概率之和高于此值时强制执行模块诊断
    "MIN_MODULE_SCORE": 0.01,  # 模块分数最小阈值，避免 BRB 输出全零
    "PROB_SUM_TOLERANCE": 0.01,  # 概率和验证的容差
}


MODULE_LABELS: List[str] = [
    "衰减器",
    "前置放大器",  # DISABLED in single-band mode
    "低频段前置低通滤波器",
    "低频段第一混频器",
    "高频段YTF滤波器",
    "高频段混频器",
    "时钟振荡器",
    "时钟合成与同步网络",
    "本振源（谐波发生器）",
    "本振混频组件",
    "校准源",
    "存储器",
    "校准信号开关",
    "中频放大器",
    "ADC",
    "数字RBW",
    "数字放大器",
    "数字检波器",
    "VBW滤波器",
    "电源模块",
]

MODULE_LABELS_V2: List[str] = []
_seen_labels = set()
for label in MODULE_LABELS:
    mapped = module_v2_from_v1(label)
    if mapped not in _seen_labels:
        MODULE_LABELS_V2.append(mapped)
        _seen_labels.add(mapped)

# Single-band mode flag: When True, high-band modules are disabled
SINGLE_BAND_MODE = SINGLE_BAND
AC_COUPLED = True

DISABLED_MODULES = []
if SINGLE_BAND_MODE:
    DISABLED_MODULES.extend(["高频段YTF滤波器", "高频段混频器"])
if DISABLE_PREAMP:
    DISABLED_MODULES.append("前置放大器")
if AC_COUPLED:
    # AC 耦合下隔直功能无意义，先禁用“隔直衰减器”（模块名沿用“衰减器”）
    DISABLED_MODULES.append("衰减器")
DISABLED_MODULES = list(dict.fromkeys(DISABLED_MODULES))

# Default BRB rule weights; can be overridden via set_module_rule_weights()
_DEFAULT_RULE_WEIGHTS = [0.8, 0.6, 0.7, 0.5, 0.5, 0.4, 0.15]
_module_rule_weights = list(_DEFAULT_RULE_WEIGHTS)

# Tunable prior scales for hierarchical_module_infer (optimized by CMA-ES)
# Structure: 18 parameters
#   [0:8]  = amp_error module prior scales (ADC, Mixer1, Filter, Power, IF, DSP, RBW, VBW)
#   [8:12] = freq_error module prior scales (RefDist, Mixer1, LO1, OCXO)
#   [12:15] = ref_error module prior scales (CalSrc, CalStore, CalSwitch)
#   [15:18] = feature sensitivity per fault type (amp, freq, ref)
_DEFAULT_HIERARCHICAL_PARAMS = [
    # amp_error priors (8)
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    # freq_error priors (4)
    1.0, 1.0, 1.0, 1.0,
    # ref_error priors (3)
    1.0, 1.0, 1.0,
    # feature sensitivity (3)
    1.0, 1.0, 1.0,
]
_hierarchical_params = list(_DEFAULT_HIERARCHICAL_PARAMS)

# Module name ordering for each fault type (matches FAULT_MODULE_PRIORS)
_AMP_MODULES = [
    "[数字中频板][ADC] 数字检波与平均",
    "[RF板][Mixer1]",
    "[RF板][RF] 低频通路固定滤波/抑制网络",
    "[电源板] 电源管理模块",
    "[数字中频板][IF] 中频放大/衰减链",
    "[数字中频板][DSP] 数字增益/偏置校准",
    "[数字中频板][数字IF域] RBW滤波器",
    "[数字中频板][数字IF域] VBW滤波器",
]
_FREQ_MODULES = [
    "[时钟板][参考分配]",
    "[RF板][Mixer1]",
    "[LO/时钟板][LO1] 合成链",
    "[时钟板][参考域] 10MHz 基准 OCXO",
]
_REF_MODULES = [
    "[校准链路][校准源]",
    "[校准链路][校准表/存储]",
    "[校准链路][校准路径开关/耦合]",
]


def set_module_rule_weights(weights):
    """Override BRB module rule weights (from CMA-ES optimization).
    
    Parameters
    ----------
    weights : list of 7 floats
        Rule weights for: [ref_rule, amp_rule, freq_rule, 
        hf_filter_rule, lf_filter_rule, digital_rule, power_rule]
    """
    global _module_rule_weights
    if len(weights) != 7:
        raise ValueError(f"Expected 7 rule weights, got {len(weights)}")
    _module_rule_weights = list(weights)


def get_module_rule_weights():
    """Return current BRB module rule weights."""
    return list(_module_rule_weights)


def set_hierarchical_params(params):
    """Override hierarchical module inference parameters (from CMA-ES).
    
    Parameters
    ----------
    params : list of 18 floats
        [0:8] amp prior scales, [8:12] freq prior scales,
        [12:15] ref prior scales, [15:18] feature sensitivity
    """
    global _hierarchical_params
    if len(params) != 18:
        raise ValueError(f"Expected 18 hierarchical params, got {len(params)}")
    _hierarchical_params = list(params)


def get_hierarchical_params():
    """Return current hierarchical module inference parameters."""
    return list(_hierarchical_params)


def _filter_belief(belief: Dict[str, float]) -> Dict[str, float]:
    if not DISABLED_MODULES:
        return belief
    return {k: v for k, v in belief.items() if k not in DISABLED_MODULES}


def _sanitize_rules(rules: List[BRBRule]) -> List[BRBRule]:
    sanitized: List[BRBRule] = []
    for rule in rules:
        belief = _filter_belief(rule.belief)
        if not belief:
            continue
        if belief == rule.belief:
            sanitized.append(rule)
        else:
            sanitized.append(BRBRule(weight=rule.weight, belief=belief))
    return sanitized


def _apply_disabled_modules(module_probs: Dict[str, float]) -> Dict[str, float]:
    if not DISABLED_MODULES:
        return module_probs
    updated = dict(module_probs)
    for name in DISABLED_MODULES:
        if name in updated:
            updated[name] = 0.0
    total = sum(updated.values())
    if total > 0:
        updated = {k: v / total for k, v in updated.items()}
    return updated


def _map_module_probs_to_v2(module_probs: Dict[str, float]) -> Dict[str, float]:
    mapped: Dict[str, float] = {}
    for key, val in module_probs.items():
        target = module_v2_from_v1(key)
        mapped[target] = mapped.get(target, 0.0) + val
    total = sum(mapped.values())
    if total > 0:
        mapped = {k: v / total for k, v in mapped.items()}
    return mapped

# 模块分组 - 按物理链路和功能相关性
# NOTE: 前置放大器 is DISABLED in single-band mode
MODULE_GROUPS = {
    # 幅度链路模块组 (前置放大器 excluded in single-band mode)
    'amp_group': [
        '衰减器', '中频放大器', '数字放大器', 'ADC',
        '数字RBW', '数字检波器', 'VBW滤波器'
    ],
    # 频率链路模块组
    'freq_group': [
        '时钟振荡器', '时钟合成与同步网络', '本振源（谐波发生器）', '本振混频组件',
        '高频段YTF滤波器', '高频段混频器', '低频段前置低通滤波器', '低频段第一混频器'
    ],
    # 参考电平链路模块组
    'ref_group': [
        '校准源', '存储器', '校准信号开关', '衰减器'
    ],
    # 通用模块
    'other_group': [
        '电源模块'
    ],
}

# 异常类型到模块组的映射
FAULT_TYPE_TO_MODULE_GROUP = {
    '幅度失准': 'amp_group',
    '频率失准': 'freq_group',
    '参考电平失准': 'ref_group',
}


def _mean(values: Iterable[float]) -> float:
    arr = list(values)
    return float(statistics.mean(arr)) if arr else 0.0


def _aggregate_module_score(features: Dict[str, float], anomaly_type: str = None) -> float:
    """Aggregate module-level symptom scores with feature streaming.

    对应文中模块层虚拟属性 V 构建思路：根据系统层异常类型，选择性使用相关特征。
    
    Parameters
    ----------
    features : dict
        模块层特征字典，包含传统字段和X6-X22扩展字段。
    anomaly_type : str, optional
        系统层检测到的异常类型（"幅度失准", "频率失准", "参考电平失准"），
        用于特征分流。如果为None，使用全部特征。
    
    Returns
    -------
    float
        聚合后的模块层异常分数(0-1)。
        注意：返回值至少为 0.01 以避免 BRB 输出全零。
    """

    # 传统特征
    md_step_raw = max(
        features.get("step_score", 0.0),
        features.get("switch_step_err_max", 0.0),
        features.get("nonswitch_step_max", 0.0),
        features.get("X7", 0.0),  # 增益非线性
    )
    md_step = normalize_feature(md_step_raw, 0.2, 1.5)
    md_slope = normalize_feature(abs(features.get("res_slope", 0.0)), 1e-12, 1e-10)
    md_ripple = normalize_feature(features.get("ripple_var", features.get("X6", 0.0)), 0.001, 0.02)
    md_df = normalize_feature(abs(features.get("df", 0.0)), 1e6, 5e7)
    md_viol = normalize_feature(features.get("viol_rate", features.get("X11", 0.0)), 0.02, 0.2)
    md_gain_bias = max(
        normalize_feature(abs(features.get("bias", 0.0)), 0.1, 1.0),
        normalize_feature(abs(features.get("gain", 1.0) - 1.0), 0.02, 0.2),
    )

    # 特征分流：根据异常类型选择相关特征
    if anomaly_type == "幅度失准":
        # 幅度模块：使用幅度相关特征X1,X2,X5,X11-X13,X19-X22
        amp_features = [
            md_step,
            md_ripple,
            md_gain_bias,
            normalize_feature(features.get("X11", 0.0), 0.01, 0.3),  # 包络超出率
            normalize_feature(features.get("X12", 0.0), 0.5, 5.0),  # 最大违规
            normalize_feature(features.get("X13", 0.0), 0.1, 10.0),  # 违规能量
            normalize_feature(abs(features.get("X19", 0.0)), 1e-12, 1e-10),  # 低频斜率
            normalize_feature(abs(features.get("X20", 0.0)), 0.5, 5.0),  # 峰度
            normalize_feature(features.get("X21", 0.0), 1, 20),  # 峰值数
            normalize_feature(features.get("X22", 0.0), 0.1, 0.8),  # 主频占比
            normalize_feature(features.get("X36", 0.0), 0.1, 0.9),  # 周期性
            normalize_feature(features.get("X37", 0.0), 0.001, 0.2),  # 线性拟合残差
        ]
        result = _mean(amp_features)
    
    elif anomaly_type == "频率失准":
        # 频率模块：使用频率相关特征X4,X14-X15,X16-X18
        freq_features = [
            md_df,
            normalize_feature(features.get("X14", 0.0), 0.01, 1.0),  # 低频残差
            normalize_feature(features.get("X15", 0.0), 0.01, 0.5),  # 高频残差
            normalize_feature(abs(features.get("X16", 0.0)), 0.001, 0.1),  # 频移
            normalize_feature(abs(features.get("X17", 0.0)), 0.001, 0.05),  # 缩放
            normalize_feature(abs(features.get("X18", 0.0)), 0.001, 0.05),  # 平移
        ]
        result = _mean(freq_features)
    
    elif anomaly_type == "参考电平失准":
        # 参考电平模块：使用参考相关特征X1,X3,X5,X11-X13
        ref_features = [
            md_slope,
            md_gain_bias,
            normalize_feature(features.get("X11", 0.0), 0.01, 0.3),  # 包络超出率
            normalize_feature(features.get("X12", 0.0), 0.5, 5.0),  # 最大违规
            normalize_feature(features.get("X13", 0.0), 0.1, 10.0),  # 违规能量
            normalize_feature(features.get("X35", 0.0), 0.001, 0.02),  # 差分方差
        ]
        result = _mean(ref_features)
    
    else:
        # 默认：使用所有传统特征
        result = _mean([md_step, md_slope, md_ripple, md_df, md_viol, md_gain_bias])
    
    # 任务书§2.2: 确保返回值非零，避免 BRB 输出全零
    return max(SOFT_GATING_CONFIG["MIN_MODULE_SCORE"], result)


def _validate_features(features: Dict[str, float]) -> Dict[str, object]:
    """验证特征向量，检查 NaN/Inf，返回诊断信息。
    
    任务2.1: 特征向量检查 - 每个 feature 的 min/max/mean，是否 NaN/Inf
    """
    diagnostics = {
        "nan_features": [],
        "inf_features": [],
        "valid": True,
        "feature_stats": {},
    }
    
    for key, val in features.items():
        if not isinstance(val, (int, float)):
            continue
        if val != val:  # NaN check
            diagnostics["nan_features"].append(key)
            diagnostics["valid"] = False
        elif abs(val) == float('inf'):
            diagnostics["inf_features"].append(key)
            diagnostics["valid"] = False
        else:
            diagnostics["feature_stats"][key] = float(val)
    
    return diagnostics


def _validate_module_probs(probs: Dict[str, float]) -> Dict[str, object]:
    """验证模块概率分布，确保 sum ≈ 1 且无全零。
    
    任务2.4: 概率输出检查 - sum(probs) 必须≈1，Top1 概率 > 0
    """
    total = sum(probs.values())
    max_prob = max(probs.values()) if probs else 0.0
    diagnostics = {
        "sum": total,
        "sum_valid": abs(total - 1.0) < SOFT_GATING_CONFIG["PROB_SUM_TOLERANCE"],
        "max_prob": max_prob,
        "top1_nonzero": max_prob > 0,
        "all_zero": all(v == 0 for v in probs.values()),
    }
    return diagnostics


def module_level_infer(
    features: Dict[str, float], 
    sys_probs: Dict[str, float],
    enable_soft_gating: bool = True,
) -> Dict[str, float]:
    """Perform compressed module-level BRB inference with feature streaming.

    改进说明（对应任务书 §3 软门控）：
    1. 即使系统级判断为 Normal，只要异常概率 > 0.2，仍执行模块推理
    2. Normal 概率 >= 0.85 时，输出模块 TopK 但标记为低置信度
    3. 添加特征验证和概率验证的诊断信息

    Parameters
    ----------
    features : dict
        模块层特征，至少包含 step_score、res_slope、ripple_var、df、
        viol_rate、gain、bias，可额外提供 X6-X22扩展特征用于特征分流。
    sys_probs : dict
        系统级诊断概率分布，作为虚拟先验属性 V 对规则进行加权。
        既支持直接传入概率字典，也支持 system_level_infer 的完整
        返回值（会自动提取其中的 ``probabilities`` 字段）。
    enable_soft_gating : bool
        是否启用软门控。启用时，即使系统判断为 Normal，
        也会输出模块级候选（可能标记低置信度）。

    Returns
    -------
    dict
        20 个模块的概率分布，顺序与 ``MODULE_LABELS`` 对齐。
        NOTE: 前置放大器 is set to 0 in single-band mode.
    """
    # 任务2.1: 特征验证
    feature_diagnostics = _validate_features(features)
    if not feature_diagnostics["valid"]:
        import warnings
        warnings.warn(
            f"[BRB Module] 特征验证失败: NaN={feature_diagnostics['nan_features']}, "
            f"Inf={feature_diagnostics['inf_features']}"
        )

    probs = sys_probs.get("probabilities", sys_probs)
    normal_prob = probs.get("正常", 0.0)
    amp_prior = probs.get("幅度失准", 0.3)
    freq_prior = probs.get("频率失准", 0.3)
    ref_prior = probs.get("参考电平失准", 0.3)
    
    # 任务书 §3.2: 软门控策略
    # 计算异常概率之和 P_abn = 1 - P(Normal)
    p_abn = 1.0 - normal_prob
    
    # 软门控条件判断
    if enable_soft_gating:
        # 条件1: 如果 Normal 概率很高 (>= 0.85)，仍执行推理但结果可能标记为低置信度
        # 条件2: 如果 P_abn > 0.2，强制执行模块推理
        force_module_infer = (p_abn > SOFT_GATING_CONFIG["TH_P_ABN_FORCE"])
        
        # 即使 is_normal=True，只要 p_abn > 阈值，也执行模块推理
        # 这样避免了"Normal 就直接返回全 0"的情况
    
    # 确定主导异常类型，用于特征分流
    max_prob_val = max(amp_prior, freq_prior, ref_prior)
    if amp_prior == max_prob_val:
        anomaly_type = "幅度失准"
    elif freq_prior == max_prob_val:
        anomaly_type = "频率失准"
    else:
        anomaly_type = "参考电平失准"

    # 使用特征分流计算模块层分数
    md = _aggregate_module_score(features, anomaly_type)

    # Per-rule feature-based activations for module discrimination
    # Different features activate different module groups
    # _nf normalizes feature x to [0, 1] range using (lo, hi) thresholds
    def _nf(x, lo, hi):
        return normalize_feature(x, lo, hi)
    
    # Ref-specific activation (校准链路 modules)
    ref_act = max(md, _nf(features.get("X35", 0.0), 0.001, 0.02),
                  _nf(abs(features.get("res_slope", 0.0)), 1e-12, 1e-10))
    
    # Amp-specific activation (放大器/滤波器 modules)
    amp_act = max(md, _nf(features.get("X11", 0.0), 0.01, 0.3),
                  _nf(features.get("X12", 0.0), 0.5, 5.0),
                  _nf(abs(features.get("bias", 0.0)), 0.1, 1.0))
    
    # Freq-specific activation (时钟/振荡器 modules)
    freq_act = max(md, _nf(abs(features.get("df", 0.0)), 1e6, 5e7),
                   _nf(abs(features.get("X16", 0.0)), 0.001, 0.1),
                   _nf(abs(features.get("X17", 0.0)), 0.001, 0.05))

    # Rules with tunable weights from _module_rule_weights (optimizable via CMA-ES)
    # Beliefs updated to match simulation module distribution
    rw = _module_rule_weights
    
    # Build rules with per-rule activations instead of global md
    rule_specs = [
        (rw[0] * ref_prior, ref_act,
         {"校准源": 0.38, "存储器": 0.32, "校准信号开关": 0.30}),
        (rw[1] * amp_prior, amp_act,
         {"低频段第一混频器": 0.25, "低频段前置低通滤波器": 0.21,
          "数字检波器": 0.17, "ADC": 0.13, "电源模块": 0.10,
          "中频放大器": 0.05, "本振混频组件": 0.05, "数字放大器": 0.04}),
        (rw[2] * freq_prior, freq_act,
         {"时钟合成与同步网络": 0.37, "本振混频组件": 0.33,
          "本振源（谐波发生器）": 0.17, "时钟振荡器": 0.13}),
        (rw[3] * freq_prior, freq_act,
         {"高频段YTF滤波器": 0.60, "高频段混频器": 0.40}),
        (rw[4] * freq_prior, freq_act,
         {"低频段前置低通滤波器": 0.60, "低频段第一混频器": 0.40}),
        (rw[5] * amp_prior, amp_act,
         {"数字RBW": 0.30, "数字检波器": 0.35, "VBW滤波器": 0.25, "ADC": 0.10}),
        (rw[6], md, {"电源模块": 1.0}),
    ]
    
    # Filter disabled modules from beliefs
    filtered_specs = []
    for w, act, bel in rule_specs:
        filtered_bel = _filter_belief(bel)
        if filtered_bel:
            filtered_specs.append((w, act, filtered_bel))
    
    # Compute weighted combination with per-rule activations
    total_act = sum(w * act for w, act, _ in filtered_specs) + 1e-9
    out = {lab: 0.0 for lab in MODULE_LABELS}
    for w, act, bel in filtered_specs:
        contribution = (w * act) / total_act
        for lab in MODULE_LABELS:
            out[lab] += contribution * bel.get(lab, 0.0)
    
    # Normalize
    s = sum(out.values()) + 1e-9
    for lab in MODULE_LABELS:
        out[lab] = out[lab] / s
    
    # 任务2.4: 概率验证
    prob_diagnostics = _validate_module_probs(out)
    if prob_diagnostics["all_zero"]:
        import warnings
        warnings.warn("[BRB Module] 模块概率全为 0，检查特征映射和规则激活")

    return _map_module_probs_to_v2(_apply_disabled_modules(out))


def module_level_infer_with_activation(
    features: Dict[str, float], 
    sys_probs: Dict[str, float],
    only_activate_relevant: bool = True,
    uncertain_max_prob_threshold: float = 0.45,
    uncertain_top2_diff_threshold: float = 0.15,
    uncertain_reliability_threshold: float = 0.6,
    expand_top_m: int = 10,
    contract_top_k: int = 5
) -> Dict[str, float]:
    """优化版模块级推理：仅激活与异常类型相关的模块组 + 候选路由兜底。
    
    对应小论文规则压缩策略：根据系统级诊断结果，
    仅对可能受影响的模块子集执行推理，减少冗余计算。
    
    Enhanced with candidate routing fallback (v5: reliability-based):
    - If system-level is uncertain (max_prob < threshold or reliability < T_rel):
      Expand candidates to Top-M (8~12)
    - If high certainty and reliability, contract to Top-K (3~6)
    
    Parameters
    ----------
    features : dict
        模块层特征字典。
    sys_probs : dict
        系统级诊断概率分布 (may include 'reliability' dict from v5).
    only_activate_relevant : bool
        如果为True，仅激活与检测到的异常类型相关的模块组。
        如果为False，行为与 module_level_infer 相同。
    uncertain_max_prob_threshold : float
        If max_prob below this, expand candidates.
    uncertain_top2_diff_threshold : float
        If top1-top2 diff below this, expand candidates.
    uncertain_reliability_threshold : float
        If reliability below this, expand candidates (v5 NEW).
    expand_top_m : int
        Number of candidates when uncertain.
    contract_top_k : int
        Number of candidates when certain.
        
    Returns
    -------
    dict
        20个模块的概率分布。
    """
    probs = sys_probs.get("probabilities", sys_probs)
    
    # v5: Get reliability from system result (if available)
    reliability_info = sys_probs.get("reliability", {})
    reliability = reliability_info.get("overall", 1.0) if isinstance(reliability_info, dict) else 1.0
    
    # 检查是否为正常状态
    normal_prob = probs.get("正常", 0.0)
    
    # 任务书 §3.2 软门控改进：
    # 即使 Normal 概率 > 0.5，只要 P_abn > 0.2 也要执行模块推理
    # 不再简单返回均匀分布
    p_abn = 1.0 - normal_prob
    force_inference = (p_abn > SOFT_GATING_CONFIG["TH_P_ABN_FORCE"])
    
    # 注意：即使 Normal 概率很高 (>= 0.85) 且 P_abn <= 0.2，
    # 我们仍然执行推理（可以标记为低置信度），而不是返回均匀分布
    # 这样保证模块级诊断始终输出有意义的结果
    _ = (normal_prob >= SOFT_GATING_CONFIG["TH_NORMAL_STRONG"] and not force_inference)  # 可用于标记低置信度
    
    amp_prior = probs.get("幅度失准", 0.3)
    freq_prior = probs.get("频率失准", 0.3)
    ref_prior = probs.get("参考电平失准", 0.3)
    
    # v5: Check for uncertainty - include reliability in decision
    max_prob = sys_probs.get("max_prob", max(amp_prior, freq_prior, ref_prior, normal_prob))
    is_uncertain = (
        max_prob < uncertain_max_prob_threshold or
        reliability < uncertain_reliability_threshold
    )
    
    # 确定主导异常类型
    max_prob_val = max(amp_prior, freq_prior, ref_prior)
    if amp_prior == max_prob_val:
        anomaly_type = "幅度失准"
        active_group = "amp_group"
    elif freq_prior == max_prob_val:
        anomaly_type = "频率失准"
        active_group = "freq_group"
    else:
        anomaly_type = "参考电平失准"
        active_group = "ref_group"
    
    # 使用特征分流计算模块层分数
    md = _aggregate_module_score(features, anomaly_type)
    
    if only_activate_relevant:
        # 仅激活相关模块组的规则
        active_modules = set(
            module for module in MODULE_GROUPS.get(active_group, [])
            if module not in DISABLED_MODULES
        )
        active_modules.update(
            module for module in MODULE_GROUPS.get('other_group', [])
            if module not in DISABLED_MODULES
        )
        
        # 构建针对性的规则
        rules = _build_targeted_rules(anomaly_type, amp_prior, freq_prior, ref_prior)
    else:
        # 使用完整规则集（降低电源模块权重）
        rules = [
            BRBRule(
                weight=0.8 * ref_prior,
                belief={"衰减器": 0.60, "校准源": 0.08, "存储器": 0.06, "校准信号开关": 0.16},
            ),
            BRBRule(
                weight=0.6 * amp_prior,
                belief={"中频放大器": 0.35, "数字放大器": 0.30, "衰减器": 0.20, "ADC": 0.15},
            ),
            BRBRule(
                weight=0.7 * freq_prior,
                belief={"时钟振荡器": 0.35, "时钟合成与同步网络": 0.35, "本振源（谐波发生器）": 0.15, "本振混频组件": 0.15},
            ),
            BRBRule(weight=0.5 * freq_prior, belief={"高频段YTF滤波器": 0.60, "高频段混频器": 0.40}),
            BRBRule(weight=0.5 * freq_prior, belief={"低频段前置低通滤波器": 0.60, "低频段第一混频器": 0.40}),
            BRBRule(weight=0.4 * amp_prior, belief={"数字RBW": 0.30, "数字检波器": 0.35, "VBW滤波器": 0.25, "ADC": 0.10}),
            # 降低电源模块权重（从 0.3 降到 0.15）
            BRBRule(weight=0.15, belief={"电源模块": 1.0}),
        ]

    rules = _sanitize_rules(rules)
    brb = SimpleBRB(MODULE_LABELS, rules)
    result = brb.infer([md])

    return _map_module_probs_to_v2(_apply_disabled_modules(result))


def _build_targeted_rules(anomaly_type: str, amp_prior: float, freq_prior: float, ref_prior: float) -> List[BRBRule]:
    """根据异常类型构建针对性的规则集。
    
    实现规则压缩：仅保留与检测到的异常类型相关的规则，
    显著减少规则数量。
    
    NOTE: 前置放大器 is EXCLUDED from all rules in single-band mode.
    """
    rules = []
    
    if anomaly_type == "幅度失准":
        # Beliefs aligned with training data distribution
        rules.extend([
            BRBRule(
                weight=0.8 * amp_prior,
                belief={"低频段第一混频器": 0.25, "低频段前置低通滤波器": 0.21,
                        "数字检波器": 0.17, "ADC": 0.13, "电源模块": 0.10,
                        "中频放大器": 0.05, "本振混频组件": 0.05, "数字放大器": 0.04},
            ),
        ])
        
    elif anomaly_type == "频率失准":
        # Beliefs aligned with training data distribution
        rules.extend([
            BRBRule(
                weight=0.8 * freq_prior,
                belief={"时钟合成与同步网络": 0.37, "本振混频组件": 0.33,
                        "本振源（谐波发生器）": 0.17, "时钟振荡器": 0.13},
            ),
        ])
        
    elif anomaly_type == "参考电平失准":
        # Beliefs aligned with training data distribution
        rules.extend([
            BRBRule(
                weight=0.8 * ref_prior,
                belief={"校准源": 0.38, "存储器": 0.32, "校准信号开关": 0.30},
            ),
        ])
    
    # 通用规则（电源模块 - 低权重）
    rules.extend([
        BRBRule(weight=0.1, belief={"电源模块": 1.0}),
    ])
    
    return rules


def get_top_k_modules(module_probs: Dict[str, float], k: int = 3) -> List[tuple]:
    """获取概率最高的前K个模块。
    
    Parameters
    ----------
    module_probs : dict
        模块概率分布。
    k : int
        返回的模块数量。
        
    Returns
    -------
    list
        (模块名称, 概率) 元组列表，按概率降序排列。
    """
    sorted_modules = sorted(module_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_modules[:k]


# =============================================================================
# P3: 条件化分层 BRB 改造
# =============================================================================

# 故障类型 → 子图激活映射
FAULT_TO_SUBGRAPH = {
    "freq_error": "LO_Clock_Network",
    "amp_error": "RF_IF_ADC_Network",
    "ref_error": "Calibration_Network",
    "normal": None
}

# 板级 → 模块映射
BOARD_MODULES = {
    "RF板": [
        "[RF板][RF] 输入衰减器组",
        "[RF板][RF] 低频通路固定滤波/抑制网络",
        "[RF板][Mixer1]"
    ],
    "数字中频板": [
        "[数字中频板][IF] 中频放大/衰减链",
        "[数字中频板][ADC] 数字检波与平均",
        "[数字中频板][DSP] 数字增益/偏置校准",
        "[数字中频板][数字IF域] RBW滤波器",
        "[数字中频板][数字IF域] VBW滤波器"
    ],
    "LO/时钟板": [
        "[LO/时钟板][LO1] 合成链",
        "[时钟板][参考域] 10MHz 基准 OCXO",
        "[时钟板][参考分配]"
    ],
    "校准链路": [
        "[校准链路][校准源]",
        "[校准链路][校准路径开关/耦合]",
        "[校准链路][校准表/存储]"
    ],
    "电源板": [
        "[电源板] 电源管理模块"
    ]
}

# 子图 → 板级映射
SUBGRAPH_TO_BOARDS = {
    "LO_Clock_Network": ["LO/时钟板"],
    "RF_IF_ADC_Network": ["RF板", "数字中频板", "电源板"],  # 电源板: ~10% of amp_error in training data
    "Calibration_Network": ["校准链路"]
}


def hierarchical_module_infer(
    fault_type: str,
    features: Dict[str, float],
    use_board_prior: bool = True
) -> Dict[str, float]:
    """
    P3: 条件化分层 BRB 推理
    
    根据 ML 预测的故障类型，激活对应的子图进行模块定位。
    
    Parameters
    ----------
    fault_type : str
        ML 预测的故障类型 (freq_error, amp_error, ref_error, normal)
    features : dict
        特征字典
    use_board_prior : bool
        是否使用板级先验
        
    Returns
    -------
    dict
        模块概率分布（使用 V2 命名）
    """
    # 获取子图
    subgraph = FAULT_TO_SUBGRAPH.get(fault_type, None)
    
    if subgraph is None or fault_type == "normal":
        # Normal 样本，返回均匀分布
        all_modules = []
        for board_modules in BOARD_MODULES.values():
            all_modules.extend(board_modules)
        uniform_prob = 1.0 / len(all_modules) if all_modules else 0.0
        return {m: uniform_prob for m in all_modules}
    
    # 获取激活的板级
    active_boards = SUBGRAPH_TO_BOARDS.get(subgraph, list(BOARD_MODULES.keys()))
    
    # 获取候选模块（V2 名称）
    candidate_modules = []
    for board in active_boards:
        candidate_modules.extend(BOARD_MODULES.get(board, []))
    
    if not candidate_modules:
        # 回退到所有模块
        for board_modules in BOARD_MODULES.values():
            candidate_modules.extend(board_modules)
    
    # 构造系统级概率（基于故障类型）
    sys_probs = {
        "normal": 0.0,
        "amp_error": 0.0,
        "freq_error": 0.0,
        "ref_error": 0.0
    }
    sys_probs[fault_type] = 0.9  # 高置信度
    
    # Data-aligned module distribution priors per fault type
    # hp[0:15] are scale factors (optimized by P-CMA-ES with box projection).
    # Always: scaled_prior = base_prior * scale_factor, then normalize to Σ=1.
    # This guarantees valid probability output while allowing full optimization freedom.
    hp = _hierarchical_params
    _BASE_AMP_PRIORS = [0.19, 0.14, 0.17, 0.10, 0.04, 0.06, 0.16, 0.14]
    _BASE_FREQ_PRIORS = [0.30, 0.30, 0.22, 0.18]
    _BASE_REF_PRIORS = [0.35, 0.33, 0.32]

    def _scale_and_normalize(base, scales, modules):
        scaled = [b * max(sc, 0.01) for b, sc in zip(base, scales)]
        total = sum(scaled) or 1.0
        return {m: v / total for m, v in zip(modules, scaled)}

    if fault_type == "amp_error":
        filtered_probs = _scale_and_normalize(_BASE_AMP_PRIORS, hp[0:8], _AMP_MODULES)
    elif fault_type == "freq_error":
        filtered_probs = _scale_and_normalize(_BASE_FREQ_PRIORS, hp[8:12], _FREQ_MODULES)
    elif fault_type == "ref_error":
        filtered_probs = _scale_and_normalize(_BASE_REF_PRIORS, hp[12:15], _REF_MODULES)
    else:
        filtered_probs = {}
    
    # Feature sensitivity multiplier for this fault type
    feat_sens = hp[15] if fault_type == "amp_error" else hp[16] if fault_type == "freq_error" else hp[17]

    # Feature-based adjustment: use discriminative features to shift priors
    # Thresholds data-driven from training feature statistics per V2 module:
    #   Module      | X7    | X13   | X22   | X33   | X35      | X36   | X37   | X38      | X39
    #   Power       | 0.184 | 0.436 | 0.022 | 1.028 | 0.003902 | 0.408 | 0.049 |          |
    #   Filter(LPF) | 0.133 | 2.157 | 0.052 | 0.490 | 0.000334 | 0.873 | 0.037 |          |
    #   ADC         | 0.164 | 0.177 | 0.044 | 0.727 | 0.001399 | 0.722 | 0.049 |          |
    #   Mixer1      | 0.080 | 0.000 | 0.041 | 0.302 | 0.000312 | 0.812 | 0.029 |          |
    #   IF/DSP      | 0.080 | 0.000 | 0.041 | 0.573 | 0.000312 | 0.812 | 0.029 |          |
    #   数字检波器   | 0.080 | 1.065 | 0.057 | 0.357 | 0.000312 | 0.834 | 0.032 |          |
    #   RBW         | 0.078 | 0.126 | 0.114 | 0.306 | 0.000322 | 0.943 | 0.063 |  (high)  | (normal)
    #   VBW         | 0.057 | 0.120 | 0.052 | 0.419 | 0.000158 | 0.894 | 0.027 |  (low)   | (low)
    if fault_type == "amp_error":
        x7 = features.get("X7", features.get("step_score", 0.08))
        x13 = features.get("X13", 0.0)
        x22 = features.get("X22", 0.04)
        x33 = features.get("X33", 0.57)
        x35 = features.get("X35", 0.0003)
        x36 = features.get("X36", 0.81)
        x37 = features.get("X37", 0.029)
        x38 = features.get("X38", 0.0)  # 2nd-order gradient variance
        x39 = features.get("X39", 0.0)  # high-freq residual energy ratio
        rmse = features.get("shape_rmse", 0.0)

        adj = {}

        # Power: X36<0.61 (midpoint 0.408↔0.722), X35>0.002 (midpoint 0.004↔0.001), X7>0.12
        power_score = 1.0
        if x36 < 0.61:
            power_score += 4.0
        if x35 > 0.002:
            power_score += 3.0
        if x7 > 0.12:
            power_score += 1.0
        adj["[电源板] 电源管理模块"] = power_score

        # Filter(LPF): X13>0.5 (midpoint 2.157↔0.436), X36>0.83
        # When X13 is near zero (no LPF signature), penalize to free top-3
        # slots for default-feature modules.
        filter_score = 1.0
        if x13 < 0.1:
            filter_score *= 0.5  # penalize: no LPF signature present
        elif x13 > 0.8:
            filter_score += 5.0
        elif x13 > 0.3:
            filter_score += 2.5
        if x36 > 0.83:
            filter_score += 1.0
        adj["[RF板][RF] 低频通路固定滤波/抑制网络"] = filter_score

        # ADC: X7>0.09, X36 in [0.65, 0.80] (midpoint 0.722↔0.812), X35 in [0.0005, 0.002]
        # When features are at default (no ADC signature), penalize to free
        # top-3 slots for modules that cannot be distinguished by features.
        adc_score = 1.0
        if rmse < 0.005 and abs(x7 - 0.08) < 0.005:
            adc_score *= 0.3  # penalize default-feature ADC
        else:
            if x7 > 0.09 and x36 > 0.65:
                adc_score += 2.0
            if rmse > 0.04:
                adc_score += 2.0
            if x35 > 0.0005 and x35 < 0.002:
                adc_score += 1.0
        adj["[数字中频板][ADC] 数字检波与平均"] = adc_score

        # Mixer1: X6 high (mean=0.006), features at default otherwise
        x6 = features.get("X6", 0.001)
        mixer_score = 1.0
        if x7 < 0.085 and x13 < 0.1 and rmse > 0.0:
            mixer_score += 3.0
        if x6 > 0.003 and x35 < 0.0004:
            mixer_score += 1.5
        adj["[RF板][Mixer1]"] = mixer_score

        # IF/DSP: these modules have all features at default when they are the
        # fault source. HOWEVER, light-severity faults of OTHER modules also
        # produce default features.  A large bonus here would crowd out the
        # true module from Top-3.  Reduce the boost to avoid this.
        if_score = 1.0
        if rmse < 0.001 and x13 < 0.001 and abs(x7 - 0.08) < 0.002:
            if_score += 0.5  # mild boost only; priors handle the rest
        adj["[数字中频板][IF] 中频放大/衰减链"] = if_score
        adj["[数字中频板][DSP] 数字增益/偏置校准"] = if_score * 0.8

        # RBW: X22>0.079 (midpoint 0.114↔0.045), X36>0.87 (midpoint 0.943↔0.812)
        # X37>0.050 (midpoint 0.063↔0.036), X39 low (0.019, midpoint with default 0.063 → <0.041)
        # Data: RBW X22 mean=0.114 (best discriminator ratio=2.86)
        rbw_score = 1.0
        if x22 > 0.079:
            rbw_score += 5.0
        elif x22 > 0.055:
            rbw_score += 2.0
        if x36 > 0.87:
            rbw_score += 2.0
        if x37 > 0.050:
            rbw_score += 2.0
        if x39 < 0.041 and x39 > 0:
            rbw_score += 1.0  # RBW has low hf_energy (sinc adds low-freq FFT energy)
        adj["[数字中频板][数字IF域] RBW滤波器"] = rbw_score

        # VBW: X35<0.000235 (midpoint 0.000158↔0.000312), X7<0.069 (midpoint 0.057↔0.080)
        # X33<0.484 (midpoint 0.419↔0.549), X39<0.046 (midpoint 0.029↔0.063)
        # X38<0.000448 (midpoint 0.000278↔0.000617) — EMA smoothing reduces 2nd-order grad var
        # Data: VBW X35 mean=0.000158 std=0.000069
        vbw_score = 1.0
        if x35 < 0.000235:
            vbw_score += 5.0
        elif x35 < 0.000312:
            vbw_score += 2.0
        if x7 < 0.069:
            vbw_score += 3.0
        elif x7 < 0.078:
            vbw_score += 1.0
        if x33 < 0.484:
            vbw_score += 1.5
        if x39 < 0.046 and x39 > 0:
            vbw_score += 1.5  # VBW EMA kills high-freq residual energy
        if x38 < 0.000448 and x38 > 0:
            vbw_score += 1.0  # VBW EMA reduces 2nd-order gradient variance
        adj["[数字中频板][数字IF域] VBW滤波器"] = vbw_score

        for mod in filtered_probs:
            filtered_probs[mod] *= adj.get(mod, 1.0) ** feat_sens

    elif fault_type == "freq_error":
        # Discriminating features (from training data V2 analysis):
        # Mixer1: X13=0.97 (very high), X14=0.002 (very low), X24=0.012 (very low)
        # 参考分配: X24=0.527 (highest), X36=0.842, X35=0.00025
        # LO1: X36=0.900 (high), X35=0.00015, X24=0.446
        # OCXO: X36=0.885, X23=0.000086 (lowest), X24=0.454
        x13 = features.get("X13", 0.45)
        x14 = features.get("X14", 0.007)
        x7 = features.get("X7", 0.07)
        x36 = features.get("X36", 0.85)
        x35 = features.get("X35", 0.0002)
        x24 = features.get("X24", 0.45)
        x23 = features.get("X23", 0.0001)

        adj = {}
        # Mixer1: very high X13 (>0.6) and low X14 (<0.003) and low X24
        mixer_s = 1.0
        if x13 > 0.7:
            mixer_s += 5.0
        elif x13 > 0.5:
            mixer_s += 2.0
        if x14 < 0.003:
            mixer_s += 2.0
        if x24 < 0.05:
            mixer_s += 2.0
        adj["[RF板][Mixer1]"] = mixer_s

        # 参考分配: highest X24 (0.527), moderate X36 (0.842), higher X35 (0.00025)
        # Use softer thresholds with gradient scoring for overlap zone
        ref_dist_s = 1.0
        if x24 > 0.50:
            ref_dist_s += 3.0
        elif x24 > 0.40:
            ref_dist_s += 1.0 + 2.0 * (x24 - 0.40) / 0.10  # linear ramp [0.40, 0.50]
        elif x24 > 0.30:
            ref_dist_s += 0.5  # partial credit for being in range
        if x35 > 0.00020:
            ref_dist_s += 1.5
        elif x35 > 0.00012:
            ref_dist_s += 0.5  # partial credit
        if x36 < 0.86:
            ref_dist_s += 1.0
        adj["[时钟板][参考分配]"] = ref_dist_s

        # LO1: high X36 (0.900), lowest X35 (0.00015), lower X24
        # Use softer thresholds for overlap with 参考分配/OCXO
        lo1_s = 1.0
        if x36 > 0.88:
            lo1_s += 2.0
        elif x36 > 0.84:
            lo1_s += 1.0  # partial credit (was missing before)
        if x35 < 0.00016:
            lo1_s += 2.0
        elif x35 < 0.00022:
            lo1_s += 1.0  # softened from 0.00020
        if x24 < 0.46 and x24 > 0.30:
            lo1_s += 1.0
        elif x24 > 0.46 and x24 < 0.50:
            lo1_s += 0.5  # partial credit for borderline X24
        adj["[LO/时钟板][LO1] 合成链"] = lo1_s

        # OCXO: high X36 (0.885), lower X23 (0.000086), moderate X24 (0.454)
        ocxo_s = 1.0
        if x36 > 0.84:
            ocxo_s += 2.0
        if x23 < 0.00012:
            ocxo_s += 2.0
        elif x23 < 0.00015:
            ocxo_s += 1.0
        if x24 < 0.55 and x24 > 0.30:
            ocxo_s += 1.0
        adj["[时钟板][参考域] 10MHz 基准 OCXO"] = ocxo_s

        for mod in filtered_probs:
            filtered_probs[mod] *= adj.get(mod, 1.0) ** feat_sens

    elif fault_type == "ref_error":
        # Ref modules weakly distinguishable; use multiple correlated features
        # Primary: X29 (Cal Source: +3.98, Storage: -0.77, Switch: +0.40) but high variance
        # Secondary: offset_slope (Cal+0.006 vs Stor-0.009), band_offset_db_1 (Cal-0.041 vs Stor+0.023)
        # Tertiary: X28 (Switch: +0.089 vs others ~-0.04), X14
        slope = features.get("offset_slope", features.get("res_slope", 0.0))
        band1 = features.get("band_offset_db_1", 0.0)
        x14 = features.get("X14", 0.1)
        x29 = features.get("X29", 0.0)
        x28 = features.get("X28", 0.0)

        adj = {}
        # 校准源: strongly positive X29 (with non-Switch X28), OR slope+band1 combined
        cal_s = 1.0
        if x29 > 2.0 and x28 < 0.2:
            cal_s += 3.0  # strong X29 + non-Switch X28
        elif x29 > 2.0:
            cal_s += 1.0  # X29 positive but X28 suggests Switch
        elif x29 > 0.5 and x28 < 0.2:
            cal_s += 1.5
        if slope > 0.03 and band1 < -0.15:
            cal_s += 2.5  # very strong slope+band1 = Cal Source even if X29 negative
        elif slope > 0.01 and band1 < -0.05:
            cal_s += 1.5
        adj["[校准链路][校准源]"] = cal_s

        # 校准表/存储: negative X29 confirmed by negative slope, higher X14
        stor_s = 1.0
        if x29 < -2.0 and slope < 0:
            stor_s += 3.0  # both X29 and slope agree
        elif x29 < -2.0:
            stor_s += 1.5  # only X29 (might be noisy)
        elif x29 < -0.5 and slope < 0:
            stor_s += 2.0  # moderate negative X29 + negative slope
        elif x29 < 0:
            stor_s += 0.5
        if slope < -0.01:
            stor_s += 0.5
        if x14 > 0.14:
            stor_s += 0.5
        adj["[校准链路][校准表/存储]"] = stor_s

        # 校准路径开关/耦合: positive X28 is strongest indicator, moderate X29
        sw_s = 1.0
        if x28 > 0.3:
            sw_s += 2.5  # strongly positive X28
        elif x28 > 0.1:
            sw_s += 1.0
        if x29 > -0.5 and x29 < 2.0 and x28 > 0:
            sw_s += 1.0
        if band1 > 0.1:
            sw_s += 0.5
        adj["[校准链路][校准路径开关/耦合]"] = sw_s

        for mod in filtered_probs:
            filtered_probs[mod] *= adj.get(mod, 1.0) ** feat_sens

    # Ignorance floor: ensure every candidate retains a minimum probability
    # so that low-but-not-zero alternatives stay in Top-3 contention.
    _IGNORANCE_FLOOR = 0.005
    for mod in filtered_probs:
        if filtered_probs[mod] < _IGNORANCE_FLOOR:
            filtered_probs[mod] = _IGNORANCE_FLOOR

    # Normalize
    total = sum(filtered_probs.values())
    if total > 0:
        filtered_probs = {m: p / total for m, p in filtered_probs.items()}
    else:
        uniform_prob = 1.0 / len(filtered_probs) if filtered_probs else 0.0
        filtered_probs = {m: uniform_prob for m in filtered_probs}
    
    return filtered_probs


def hierarchical_module_infer_soft_gating(
    final_probs: Dict[str, float],
    features: Dict[str, float],
    delta: float = 0.1,
    use_board_prior: bool = True,
) -> Dict:
    """
    Top-2 truncated soft-gating multi-hypothesis module inference.

    Strategy:
      1. Keep only the Top-2 fault-type hypotheses (zero out the rest).
      2. Activate the 2nd hypothesis only when the system-level is genuinely
         uncertain (gap < delta AND 2nd prob > floor).
      3. Renormalize Top-2 probabilities to sum = 1 with a small secondary
         weight floor.
      4. Run BRB module inference for each active hypothesis.
      5. Fuse module probabilities using the renormalized hypothesis weights.

    Parameters
    ----------
    final_probs : dict
        System-level fault type probabilities {fault_type: prob}
    features : dict
        Feature dictionary
    delta : float
        Maximum gap between top-1 and top-2 probabilities to activate
        multi-hypothesis routing.
    use_board_prior : bool
        Whether to use board-level priors

    Returns
    -------
    dict
        {
            "fused_topk": List[{"name": str, "prob": float}],
            "used_fault_hypotheses": List[Tuple[str, float]],
            "per_hypothesis_topk": Dict[str, List],
            "single_hypothesis": bool
        }
    """
    # ── Configuration ──────────────────────────────────────────────────
    # Minimum probability to even consider a fault hypothesis
    _MIN_FAULT_PROB = 0.01
    # Minimum probability for the 2nd hypothesis to be considered
    _TOP2_MIN_PROB = 0.02
    # Minimum fraction of total weight reserved for the 2nd hypothesis
    # when multi-hypothesis is activated (raised from 0.15 to prevent
    # cascade errors when RF system-level prediction is slightly wrong)
    _SECONDARY_WEIGHT_FLOOR = 0.25

    # ── Step 1: Sort fault types, filter out "normal" ──────────────────
    sorted_faults = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
    fault_hypotheses = [
        (ft, p) for ft, p in sorted_faults
        if ft != "normal" and p > _MIN_FAULT_PROB
    ]

    if not fault_hypotheses:
        all_modules = []
        for board_modules in BOARD_MODULES.values():
            all_modules.extend(board_modules)
        uniform_prob = 1.0 / len(all_modules) if all_modules else 0.0
        topk = [{"name": m, "prob": uniform_prob} for m in all_modules[:5]]
        return {
            "fused_topk": topk,
            "used_fault_hypotheses": [],
            "per_hypothesis_topk": {},
            "single_hypothesis": True,
        }

    # ── Step 2: Top-2 truncation with delta-gated activation ──────────
    top1_ft, top1_prob = fault_hypotheses[0]

    use_top2 = False
    top2_ft, top2_prob = None, 0.0
    if len(fault_hypotheses) >= 2:
        top2_ft, top2_prob = fault_hypotheses[1]
        # Top-2 truncation: always activate the 2nd hypothesis when its
        # probability exceeds the minimum floor.  This prevents a single
        # high-confidence (but possibly wrong) system prediction from
        # completely silencing the alternative fault-type subgraph.
        if top2_prob > _TOP2_MIN_PROB:
            use_top2 = True

    # ── Step 3: Run BRB module inference per hypothesis ────────────────
    per_hypothesis_topk = {}
    per_hypothesis_probs = {}

    probs_1 = hierarchical_module_infer(top1_ft, features, use_board_prior)
    per_hypothesis_probs[top1_ft] = probs_1
    sorted_1 = sorted(probs_1.items(), key=lambda x: x[1], reverse=True)
    per_hypothesis_topk[top1_ft] = [{"name": m, "prob": p} for m, p in sorted_1[:5]]

    if use_top2 and top2_ft:
        probs_2 = hierarchical_module_infer(top2_ft, features, use_board_prior)
        per_hypothesis_probs[top2_ft] = probs_2
        sorted_2 = sorted(probs_2.items(), key=lambda x: x[1], reverse=True)
        per_hypothesis_topk[top2_ft] = [{"name": m, "prob": p} for m, p in sorted_2[:5]]

        # ── Step 4: Top-2 renormalized weighted fusion ─────────────────
        raw_total = top1_prob + top2_prob
        if raw_total > 0:
            raw_w2_ratio = top2_prob / raw_total
        else:
            raw_w2_ratio = 0.5

        # Apply a small floor so the 2nd hypothesis is not negligible
        effective_w2_ratio = max(raw_w2_ratio, _SECONDARY_WEIGHT_FLOOR)
        effective_w1_ratio = 1.0 - effective_w2_ratio

        all_modules = set(probs_1.keys()) | set(probs_2.keys())
        fused_probs = {}
        for m in all_modules:
            p1 = probs_1.get(m, 0.0)
            p2 = probs_2.get(m, 0.0)
            fused_probs[m] = effective_w1_ratio * p1 + effective_w2_ratio * p2

        used_hypotheses = [(top1_ft, top1_prob), (top2_ft, top2_prob)]

        # ── Step 5: Diversity guarantee ───────────────────────────────
        # When multi-hypothesis is active, ensure the best module from
        # the SECONDARY hypothesis has a minimum presence in the fused
        # top-3.  Only apply when the primary hypothesis has more than
        # 3 modules (otherwise all primary modules already fit in
        # top-3 and adding a secondary module would displace one).
        h1_module_count = sum(1 for p in probs_1.values() if p > 0)
        if h1_module_count > 3:
            top_h2_mod = max(probs_2.items(), key=lambda x: x[1])[0]
            fused_sorted_vals = sorted(fused_probs.values(), reverse=True)
            if len(fused_sorted_vals) >= 3:
                third_val = fused_sorted_vals[2]
                if fused_probs[top_h2_mod] < third_val:
                    fused_probs[top_h2_mod] = third_val + 1e-9
    else:
        fused_probs = probs_1
        used_hypotheses = [(top1_ft, top1_prob)]
    
    # Sort and get top-K
    sorted_fused = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)
    fused_topk = [{"name": m, "prob": p} for m, p in sorted_fused[:10]]
    
    return {
        "fused_topk": fused_topk,
        "used_fault_hypotheses": used_hypotheses,
        "per_hypothesis_topk": per_hypothesis_topk,
        "single_hypothesis": not use_top2,
    }
