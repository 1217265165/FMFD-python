#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3.2: Canonical name normalization utilities.

Provides unified canonicalization for fault types and module names
to ensure consistent comparison across all evaluation paths.
"""

from typing import Optional


# Canonical fault types (English)
CANONICAL_FAULT_TYPES = {"normal", "amp_error", "freq_error", "ref_error"}

# Chinese to English fault type mapping
FAULT_TYPE_CN_TO_EN = {
    "正常": "normal",
    "幅度失准": "amp_error", 
    "频率失准": "freq_error",
    "参考电平失准": "ref_error",
}

FAULT_TYPE_EN_TO_CN = {v: k for k, v in FAULT_TYPE_CN_TO_EN.items()}


def canonical_fault_type(name: str) -> str:
    """
    Normalize fault type to canonical English form.
    
    Parameters
    ----------
    name : str
        Fault type in any format (Chinese or English, with variations)
        
    Returns
    -------
    str
        Canonical fault type: one of {"normal", "amp_error", "freq_error", "ref_error"}
    """
    if not name:
        return "normal"
    
    name = str(name).strip().lower()
    
    # Already canonical
    if name in CANONICAL_FAULT_TYPES:
        return name
    
    # Chinese mapping
    for cn, en in FAULT_TYPE_CN_TO_EN.items():
        if cn in name or name in cn:
            return en
    
    # Common variations
    if "amp" in name or "幅度" in name:
        return "amp_error"
    if "freq" in name or "频率" in name:
        return "freq_error"
    if "ref" in name or "参考" in name:
        return "ref_error"
    if "normal" in name or "正常" in name:
        return "normal"
    
    return "normal"


def canonical_module_v2(name: str) -> str:
    """
    Normalize module name to canonical V2 format.
    
    The canonical format is: [板级][功能块] 描述
    
    Parameters
    ----------
    name : str
        Module name in V1 or V2 format
        
    Returns
    -------
    str
        Canonical V2 module name
    """
    if not name:
        return ""
    
    name = str(name).strip()
    
    # V1 to V2 mapping
    V1_TO_V2 = {
        # RF板
        "衰减器": "[RF板][RF] 输入衰减器组",
        "输入衰减器": "[RF板][RF] 输入衰减器组",
        "前置放大器": "前置放大器",  # Disabled
        "低频段前置低通滤波器": "[RF板][RF] 低频通路固定滤波/抑制网络",
        "低频段第一混频器": "[RF板][Mixer1]",
        "高频段YTF滤波器": "[RF板][YTF]",
        "高频段混频器": "[RF板][Mixer2]",
        
        # 时钟板
        "时钟振荡器": "[时钟板][参考域] 10MHz 基准 OCXO",
        "时钟合成与同步网络": "[时钟板][参考分配]",
        "本振源（谐波发生器）": "[LO/时钟板][LO1] 合成链",
        "本振混频组件": "[LO/时钟板][LO1] 合成链",
        
        # 校准链路
        "校准源": "[校准链路][校准源]",
        "存储器": "[校准链路][校准表/存储]",
        "校准信号开关": "[校准链路][校准路径开关/耦合]",
        
        # 数字中频板
        "中频放大器": "[数字中频板][IF] 中频放大/衰减链",
        "ADC": "[数字中频板][ADC] 数字检波与平均",
        "数字RBW": "[数字中频板][IF] RBW数字滤波器",
        "数字放大器": "[数字中频板][DSP] 数字增益/偏置校准",
        "数字检波器": "[数字中频板][ADC] 数字检波与平均",
        "VBW滤波器": "[数字中频板][VBW]",
        
        # 电源板
        "电源模块": "[电源板] 电源管理模块",
    }
    
    # Check if already V2 format
    if name.startswith("["):
        return name
    
    # Direct V1 to V2 mapping
    if name in V1_TO_V2:
        return V1_TO_V2[name]
    
    # Fuzzy matching for partial names
    name_lower = name.lower()
    for v1, v2 in V1_TO_V2.items():
        v1_lower = v1.lower()
        if v1_lower in name_lower or name_lower in v1_lower:
            return v2
    
    # Return as-is if no match
    return name


def modules_match(pred: str, gt: str) -> bool:
    """
    Check if predicted module matches ground truth.
    
    Uses canonical names for comparison with fuzzy matching.
    
    Parameters
    ----------
    pred : str
        Predicted module name
    gt : str
        Ground truth module name
        
    Returns
    -------
    bool
        True if modules match
    """
    if not pred or not gt:
        return False
    
    # Canonicalize both
    pred_canonical = canonical_module_v2(pred)
    gt_canonical = canonical_module_v2(gt)
    
    # Exact match
    if pred_canonical == gt_canonical:
        return True
    
    # Extract key parts for fuzzy matching
    def extract_key(s: str) -> str:
        if ']' in s:
            parts = s.split(']')
            return parts[-1].strip().strip('[')
        return s
    
    pred_key = extract_key(pred_canonical)
    gt_key = extract_key(gt_canonical)
    
    if pred_key and gt_key:
        if pred_key in gt_key or gt_key in pred_key:
            return True
    
    # Specific substring matching
    keywords = [
        ("中频放大", "中频放大"),
        ("检波", "检波"),
        ("ADC", "ADC"),
        ("低频通路", "低频通路"),
        ("Mixer1", "Mixer1"),
        ("混频", "混频"),
        ("RBW", "RBW"),
        ("数字增益", "数字放大"),
        ("校准源", "校准源"),
        ("时钟", "时钟"),
        ("本振", "本振"),
        ("电源", "电源"),
    ]
    
    for kw1, kw2 in keywords:
        if kw1 in pred and kw2 in gt:
            return True
        if kw2 in pred and kw1 in gt:
            return True
    
    return False


def fault_types_match(pred: str, gt: str) -> bool:
    """
    Check if predicted fault type matches ground truth.
    
    Parameters
    ----------
    pred : str
        Predicted fault type
    gt : str
        Ground truth fault type
        
    Returns
    -------
    bool
        True if fault types match
    """
    return canonical_fault_type(pred) == canonical_fault_type(gt)
