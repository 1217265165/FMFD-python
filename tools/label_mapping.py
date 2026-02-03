#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一标签映射模块 - 系统级与模块级标签定义

该模块定义了系统级故障类别的中英文映射关系，确保：
1. 所有诊断输出、标签读取、评估脚本使用统一的映射
2. 输出顺序固定，不依赖 sorted() 推断
3. 前端展示、评估脚本、softmax 层都使用相同顺序

使用方式：
    from tools.label_mapping import (
        SYS_CLASS_TO_CN, CN_TO_SYS_CLASS, 
        SYS_LABEL_ORDER_CN, SYS_LABEL_ORDER_EN,
        normalize_module_name
    )
"""

import re

# ============== 系统级类别映射（固定，不允许排序推断）==============

# 系统级故障类别：英文 -> 中文
SYS_CLASS_TO_CN = {
    "normal": "正常",
    "amp_error": "幅度失准",
    "freq_error": "频率失准",
    "ref_error": "参考电平失准",
}

# 系统级故障类别：中文 -> 英文
CN_TO_SYS_CLASS = {v: k for k, v in SYS_CLASS_TO_CN.items()}

# 统一系统输出顺序（前端、评估、softmax都用这一套）
SYS_LABEL_ORDER_CN = ["正常", "幅度失准", "频率失准", "参考电平失准"]
SYS_LABEL_ORDER_EN = ["normal", "amp_error", "freq_error", "ref_error"]

# 系统级类别数量
NUM_SYSTEM_CLASSES = len(SYS_LABEL_ORDER_CN)


# ============== 模块机理映射（HIERARCHY_MAP）==============
# 映射来源：映射.md（物理模块 -> 系统异常 -> 曲线形态 -> 标签）
# NOTE: 请勿擅自更改模块语义，所有模块级标签必须经该映射表校验。
HIERARCHY_MAP = {
    "RF_AC": {
        "system_class": "amp_error",
        "label_group": "Amp",
        "curve_signature": "low_freq_rolloff",
    },
    "RF_Match": {
        "system_class": "amp_error",
        "label_group": "Amp",
        "curve_signature": "periodic_ripple",
    },
    "ATT": {
        "system_class": "amp_error",
        "label_group": "Amp",
        "curve_signature": "global_shift_or_spikes",
    },
    "BPF": {
        "system_class": "amp_error",
        "label_group": "Amp",
        "curve_signature": "band_insertion_loss",
    },
    "Mixer1": {
        "system_class": "amp_error",
        "label_group": "Amp",
        "curve_signature": "linear_slope",
    },
    "LO1": {
        "system_class": "freq_error",
        "label_group": "Freq",
        "curve_signature": "peak_jitter",
    },
    "IF1_BPF": {
        "system_class": "amp_error",
        "label_group": "Amp",
        "curve_signature": "low_with_texture",
    },
    "IF1_AMP": {
        "system_class": "amp_error",
        "label_group": "Amp",
        "curve_signature": "smooth_shift",
    },
    "Mixer2": {
        "system_class": "amp_error",
        "label_group": "Amp",
        "curve_signature": "upper_band_drop",
    },
    "LO2": {
        "system_class": "freq_error",
        "label_group": "Freq",
        "curve_signature": "blackout_band",
    },
    "IF2_BPF": {
        "system_class": "amp_error",
        "label_group": "Amp",
        "curve_signature": "stable_ripple",
    },
    "ADC": {
        "system_class": "ref_error",
        "label_group": "Ref",
        "curve_signature": "sawtooth_quant",
    },
    "DDC": {
        "system_class": "freq_error",
        "label_group": "Freq",
        "curve_signature": "fixed_freq_offset",
    },
    "SCALE": {
        "system_class": "ref_error",
        "label_group": "Ref",
        "curve_signature": "smooth_shift",
    },
    "RBW": {
        "system_class": "ref_error",
        "label_group": "Ref",
        "curve_signature": "regular_nonphysical_ripple",
    },
    "PEAK1": {
        "system_class": "freq_error",
        "label_group": "Freq",
        "curve_signature": "isolated_spike",
    },
    "PEAK2": {
        "system_class": "freq_error",
        "label_group": "Freq",
        "curve_signature": "dense_spikes",
    },
    "PEAK3": {
        "system_class": "freq_error",
        "label_group": "Freq",
        "curve_signature": "blackout_gaps",
    },
    "Power": {
        "system_class": "other",
        "label_group": "Other",
        "curve_signature": "high_diff_variance",
    },
}


def canonical_module_key(module_name: str) -> str:
    """Normalize module names into HIERARCHY_MAP keys when possible."""
    if not module_name:
        return ""
    name = normalize_module_name(module_name)
    key_map = {
        "AC耦合": "RF_AC",
        "匹配": "RF_Match",
        "ATT": "ATT",
        "衰减": "ATT",
        "固定滤波": "BPF",
        "滤波": "BPF",
        "Mixer1": "Mixer1",
        "混频器": "Mixer1",
        "LO1": "LO1",
        "IF1滤波": "IF1_BPF",
        "IF1放大": "IF1_AMP",
        "Mixer2": "Mixer2",
        "LO2": "LO2",
        "IF2滤波": "IF2_BPF",
        "ADC": "ADC",
        "DDC": "DDC",
        "标度": "SCALE",
        "RBW": "RBW",
        "峰值搜索1": "PEAK1",
        "峰值搜索2": "PEAK2",
        "峰值搜索3": "PEAK3",
        "电源": "Power",
        "Power": "Power",
    }
    if name in HIERARCHY_MAP:
        return name
    for token, key in key_map.items():
        if token in name:
            return key
    return ""


def expected_system_class_for_module(module_name: str) -> str:
    """Return expected system class for a module based on HIERARCHY_MAP."""
    key = canonical_module_key(module_name)
    if not key:
        return ""
    return HIERARCHY_MAP.get(key, {}).get("system_class", "")


MODULE_V1_TO_V2 = {
    "时钟振荡器": "[时钟板][参考域] 10MHz 基准 OCXO",
    "时钟合成与同步网络": "[时钟板][参考分配]",
    "本振源（谐波发生器）": "[LO/时钟板][LO1] 合成链",
    "本振源(谐波发生器)": "[LO/时钟板][LO1] 合成链",
    "本振混频组件": "[RF板][Mixer1]",
    "校准源": "[校准链路][校准源]",
    "校准信号开关": "[校准链路][校准路径开关/耦合]",
    "存储器": "[校准链路][校准表/存储]",
    "低频段前置低通滤波器": "[RF板][RF] 低频通路固定滤波/抑制网络",
    "低频段第一混频器": "[RF板][Mixer1]",
    "高频段YTF滤波器": "[RF板][YTF]",
    "高频段混频器": "[RF板][Mixer2]",
    "中频放大器": "[数字中频板][IF] 中频放大/衰减链",
    "ADC": "[数字中频板][ADC] 数字检波与平均",
    "数字RBW": "[数字中频板][IF] RBW数字滤波器",
    "数字放大器": "[数字中频板][DSP] 数字增益/偏置校准",
    "数字检波器": "[数字中频板][ADC] 数字检波与平均",
    "VBW滤波器": "[数字中频板][VBW]",
    "电源模块": "[电源板] 电源管理模块",
    "衰减器": "[RF板][RF] 输入衰减器组",
    "前置放大器": "前置放大器",
    "参考电平衰减器组": "[RF板][RF] 输入衰减器组",
    "数字增益/偏置校准": "[数字中频板][DSP] 数字增益/偏置校准",
    "中频窗函数与加权": "[数字中频板][DSP] 中频窗函数与加权",
    "输入衰减器组": "[RF板][RF] 输入衰减器组",
    "输入连接与匹配网络": "[RF板][RF] 输入连接/匹配/保护",
    "RBW带宽滤波器": "[数字中频板][IF] RBW数字滤波器",
    "ADC采样时钟": "[数字中频板][ADC] 采样时钟",
}


def module_v2_from_v1(module_name: str) -> str:
    """Map module_v1 names to module_v2 labels."""
    if not module_name:
        return ""
    name = normalize_module_name(module_name)
    return MODULE_V1_TO_V2.get(name, name)


# ============== 禁用模块列表 ==============
# NOTE: DISABLED_MODULES is imported from BRB.module_brb to maintain single source of truth.
# Do NOT define DISABLED_MODULES here. Use:
#   from BRB.module_brb import DISABLED_MODULES


# ============== 模块名规范化 ==============

def normalize_module_name(name: str) -> str:
    """规范化模块名字符串，防止"看起来一样但不相等"的问题。
    
    处理规则：
    1. 去掉首尾空格
    2. 全角半角括号统一（中文括号 → 英文括号）
    3. 连续空格压缩为单个空格
    4. 保留括号内描述（如"本振源（谐波发生器）"）
    
    Parameters
    ----------
    name : str
        原始模块名字符串
        
    Returns
    -------
    str
        规范化后的模块名
    """
    if not name or not isinstance(name, str):
        return ""
    
    # 1. 去掉首尾空格
    result = name.strip()
    
    # 2. 全角半角括号统一（中文括号 → 英文括号）
    result = result.replace("（", "(").replace("）", ")")
    
    # 3. 连续空格压缩为单个空格
    result = re.sub(r'\s+', ' ', result)
    
    return result


def is_module_disabled(module_name: str, disabled_modules: list = None) -> bool:
    """检查模块是否被禁用。
    
    Parameters
    ----------
    module_name : str
        模块名称
    disabled_modules : list, optional
        禁用模块列表。如果为 None，返回 False（无模块被禁用）。
        建议使用 BRB.module_brb.DISABLED_MODULES。
        
    Returns
    -------
    bool
        True 表示模块被禁用，不参与 TopK 和命中统计
    """
    if disabled_modules is None:
        return False
    
    normalized = normalize_module_name(module_name)
    return any(
        normalize_module_name(disabled) == normalized 
        for disabled in disabled_modules
    )


def get_topk_modules(module_probs: dict, k: int = 3, skip_disabled: bool = True, disabled_modules: list = None) -> list:
    """获取概率最高的前K个模块，可选跳过禁用模块。
    
    Parameters
    ----------
    module_probs : dict
        模块概率分布字典，key=模块名，value=概率
    k : int
        返回的模块数量
    skip_disabled : bool
        是否跳过禁用模块，默认 True
    disabled_modules : list, optional
        禁用模块列表。如果为 None 且 skip_disabled=True，则不跳过任何模块。
        建议使用 BRB.module_brb.DISABLED_MODULES。
        
    Returns
    -------
    list
        [(模块名, 概率), ...] 元组列表，按概率降序排列
    """
    items = list(module_probs.items())
    
    if skip_disabled and disabled_modules:
        items = [(name, prob) for name, prob in items if not is_module_disabled(name, disabled_modules)]
    
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    return sorted_items[:k]


# ============== 标签验证工具 ==============

def validate_system_class(sys_class: str, allow_none: bool = True) -> bool:
    """验证系统级类别是否有效。
    
    Parameters
    ----------
    sys_class : str
        系统级类别（英文或中文）
    allow_none : bool
        是否允许 None/空值（normal 类型可能无 system_fault_class）
        
    Returns
    -------
    bool
        True 表示有效
    """
    if sys_class is None or sys_class == "":
        return allow_none
    
    return sys_class in SYS_CLASS_TO_CN or sys_class in CN_TO_SYS_CLASS


def get_system_class_cn(sys_class: str) -> str:
    """获取系统级类别的中文名称。
    
    Parameters
    ----------
    sys_class : str
        系统级类别（英文或中文）
        
    Returns
    -------
    str
        中文类别名称，如果无法识别则返回原值
    """
    if sys_class is None:
        return "正常"  # None 默认为正常
    
    if sys_class in SYS_CLASS_TO_CN:
        return SYS_CLASS_TO_CN[sys_class]
    
    if sys_class in CN_TO_SYS_CLASS:
        return sys_class  # 已经是中文
    
    return sys_class  # 无法识别，返回原值


def get_system_class_en(sys_class: str) -> str:
    """获取系统级类别的英文名称。
    
    Parameters
    ----------
    sys_class : str
        系统级类别（英文或中文）
        
    Returns
    -------
    str
        英文类别名称，如果无法识别则返回原值
    """
    if sys_class is None:
        return "normal"  # None 默认为正常
    
    if sys_class in CN_TO_SYS_CLASS:
        return CN_TO_SYS_CLASS[sys_class]
    
    if sys_class in SYS_CLASS_TO_CN:
        return sys_class  # 已经是英文
    
    return sys_class  # 无法识别，返回原值
