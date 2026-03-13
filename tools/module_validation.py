#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T6: 模块级诊断自动验证机制

对 sim_* 仿真数据运行模块诊断后，自动与 labels.json 比对
计算 Top1/Top3/Top5 命中率、GT 排名和 GT 概率
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tools.label_mapping import normalize_module_name, module_v2_from_v1


@dataclass
class ModuleValidationResult:
    """单样本模块验证结果
    
    Attributes
    ----------
    sample_id : str
        样本 ID
    gt_module : str
        Ground Truth 模块 (来自 labels.json)
    gt_module_v2 : str
        GT 模块 V2 版本
    top1 : str
        BRB 输出 Top1 模块
    top3 : List[str]
        BRB 输出 Top3 模块
    top5 : List[str]
        BRB 输出 Top5 模块
    gt_rank : int
        GT 在排序中的位置 (1-based, -1 表示未找到)
    gt_prob : float
        GT 的后验概率
    top1_hit : bool
        Top1 是否命中 GT
    top3_hit : bool
        GT 是否在 Top3 中
    top5_hit : bool
        GT 是否在 Top5 中
    """
    sample_id: str = ""
    gt_module: str = ""
    gt_module_v2: str = ""
    top1: str = ""
    top3: List[str] = field(default_factory=list)
    top5: List[str] = field(default_factory=list)
    gt_rank: int = -1
    gt_prob: float = 0.0
    top1_hit: bool = False
    top3_hit: bool = False
    top5_hit: bool = False
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "sample_id": self.sample_id,
            "gt_module": self.gt_module,
            "gt_module_v2": self.gt_module_v2,
            "top1": self.top1,
            "top3": self.top3,
            "top5": self.top5,
            "gt_rank": self.gt_rank,
            "gt_prob": self.gt_prob,
            "top1_hit": self.top1_hit,
            "top3_hit": self.top3_hit,
            "top5_hit": self.top5_hit,
        }


@dataclass
class ModuleValidationSummary:
    """模块验证汇总统计
    
    Attributes
    ----------
    total_samples : int
        总样本数
    valid_samples : int
        有效样本数 (有 GT 的样本)
    top1_hit_count : int
        Top1 命中数
    top3_hit_count : int
        Top3 命中数
    top5_hit_count : int
        Top5 命中数
    top1_hit_rate : float
        Top1 命中率
    top3_hit_rate : float
        Top3 命中率
    top5_hit_rate : float
        Top5 命中率
    mean_gt_rank : float
        GT 平均排名
    mean_gt_prob : float
        GT 平均概率
    """
    total_samples: int = 0
    valid_samples: int = 0
    top1_hit_count: int = 0
    top3_hit_count: int = 0
    top5_hit_count: int = 0
    top1_hit_rate: float = 0.0
    top3_hit_rate: float = 0.0
    top5_hit_rate: float = 0.0
    mean_gt_rank: float = 0.0
    mean_gt_prob: float = 0.0
    per_class_stats: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "top1_hit_count": self.top1_hit_count,
            "top3_hit_count": self.top3_hit_count,
            "top5_hit_count": self.top5_hit_count,
            "top1_hit_rate": self.top1_hit_rate,
            "top3_hit_rate": self.top3_hit_rate,
            "top5_hit_rate": self.top5_hit_rate,
            "mean_gt_rank": self.mean_gt_rank,
            "mean_gt_prob": self.mean_gt_prob,
            "per_class_stats": self.per_class_stats,
        }


def _find_gt_in_topk(
    gt_module: str,
    gt_module_v2: str,
    module_probs: Dict[str, float],
    disabled_modules: Optional[List[str]] = None,
) -> Tuple[int, float, List[str]]:
    """在模块概率分布中查找 GT 的位置和概率
    
    Parameters
    ----------
    gt_module : str
        GT 模块名 (v1)
    gt_module_v2 : str
        GT 模块名 (v2)
    module_probs : Dict[str, float]
        BRB 输出的模块概率分布
    disabled_modules : List[str], optional
        禁用模块列表
        
    Returns
    -------
    Tuple[int, float, List[str]]
        (gt_rank, gt_prob, sorted_modules)
    """
    # 排除禁用模块
    if disabled_modules:
        module_probs = {
            k: v for k, v in module_probs.items()
            if normalize_module_name(k) not in [
                normalize_module_name(d) for d in disabled_modules
            ]
        }
    
    # 按概率排序
    sorted_items = sorted(module_probs.items(), key=lambda x: x[1], reverse=True)
    sorted_modules = [m for m, _ in sorted_items]
    
    # 规范化 GT 名称
    gt_v1_norm = normalize_module_name(gt_module)
    gt_v2_norm = normalize_module_name(gt_module_v2)
    
    # 查找 GT
    gt_rank = -1
    gt_prob = 0.0
    
    for rank, (mod_name, prob) in enumerate(sorted_items, 1):
        mod_norm = normalize_module_name(mod_name)
        mod_v2 = normalize_module_name(module_v2_from_v1(mod_name))
        
        # 匹配 v1 或 v2
        if mod_norm == gt_v1_norm or mod_norm == gt_v2_norm or mod_v2 == gt_v2_norm:
            gt_rank = rank
            gt_prob = prob
            break
    
    return gt_rank, gt_prob, sorted_modules


def validate_module_diagnosis(
    sample_id: str,
    gt_module: str,
    gt_module_v2: str,
    module_probs: Dict[str, float],
    disabled_modules: Optional[List[str]] = None,
) -> ModuleValidationResult:
    """验证单样本的模块级诊断
    
    Parameters
    ----------
    sample_id : str
        样本 ID
    gt_module : str
        GT 模块名 (v1)
    gt_module_v2 : str
        GT 模块名 (v2)
    module_probs : Dict[str, float]
        BRB 输出的模块概率分布
    disabled_modules : List[str], optional
        禁用模块列表
        
    Returns
    -------
    ModuleValidationResult
        验证结果
    """
    result = ModuleValidationResult(
        sample_id=sample_id,
        gt_module=gt_module,
        gt_module_v2=gt_module_v2,
    )
    
    gt_rank, gt_prob, sorted_modules = _find_gt_in_topk(
        gt_module, gt_module_v2, module_probs, disabled_modules
    )
    
    result.gt_rank = gt_rank
    result.gt_prob = gt_prob
    
    # TopK 模块
    result.top1 = sorted_modules[0] if sorted_modules else ""
    result.top3 = sorted_modules[:3]
    result.top5 = sorted_modules[:5]
    
    # 命中判断
    if gt_rank > 0:
        result.top1_hit = (gt_rank == 1)
        result.top3_hit = (gt_rank <= 3)
        result.top5_hit = (gt_rank <= 5)
    
    return result


def compute_validation_summary(
    results: List[ModuleValidationResult],
    labels_data: Optional[Dict] = None,
) -> ModuleValidationSummary:
    """计算验证汇总统计
    
    Parameters
    ----------
    results : List[ModuleValidationResult]
        验证结果列表
    labels_data : Dict, optional
        labels.json 数据 (用于按类统计)
        
    Returns
    -------
    ModuleValidationSummary
        汇总统计
    """
    summary = ModuleValidationSummary()
    summary.total_samples = len(results)
    
    # 过滤有效样本 (gt_rank > 0 表示找到了 GT)
    valid_results = [r for r in results if r.gt_rank > 0]
    summary.valid_samples = len(valid_results)
    
    if not valid_results:
        return summary
    
    # 计算命中统计
    summary.top1_hit_count = sum(1 for r in valid_results if r.top1_hit)
    summary.top3_hit_count = sum(1 for r in valid_results if r.top3_hit)
    summary.top5_hit_count = sum(1 for r in valid_results if r.top5_hit)
    
    summary.top1_hit_rate = summary.top1_hit_count / summary.valid_samples
    summary.top3_hit_rate = summary.top3_hit_count / summary.valid_samples
    summary.top5_hit_rate = summary.top5_hit_count / summary.valid_samples
    
    # 计算平均排名和概率
    ranks = [r.gt_rank for r in valid_results]
    probs = [r.gt_prob for r in valid_results]
    summary.mean_gt_rank = sum(ranks) / len(ranks)
    summary.mean_gt_prob = sum(probs) / len(probs)
    
    # 按系统类别统计
    if labels_data:
        per_class: Dict[str, Dict] = {}
        for r in valid_results:
            sample_data = labels_data.get(r.sample_id, {})
            sys_class = sample_data.get("system_fault_class", "unknown")
            
            if sys_class not in per_class:
                per_class[sys_class] = {
                    "count": 0,
                    "top1_hit": 0,
                    "top3_hit": 0,
                    "ranks": [],
                }
            
            per_class[sys_class]["count"] += 1
            if r.top1_hit:
                per_class[sys_class]["top1_hit"] += 1
            if r.top3_hit:
                per_class[sys_class]["top3_hit"] += 1
            per_class[sys_class]["ranks"].append(r.gt_rank)
        
        # 计算每类统计
        for cls, stats in per_class.items():
            count = stats["count"]
            stats["top1_rate"] = stats["top1_hit"] / count if count > 0 else 0
            stats["top3_rate"] = stats["top3_hit"] / count if count > 0 else 0
            stats["mean_rank"] = sum(stats["ranks"]) / len(stats["ranks"]) if stats["ranks"] else 0
            del stats["ranks"]  # 移除原始数据
        
        summary.per_class_stats = per_class
    
    return summary


def run_batch_validation(
    labels_path: Path,
    diagnosis_results: Dict[str, Dict],
    output_path: Optional[Path] = None,
    disabled_modules: Optional[List[str]] = None,
) -> Tuple[List[ModuleValidationResult], ModuleValidationSummary]:
    """批量运行模块验证
    
    Parameters
    ----------
    labels_path : Path
        labels.json 文件路径
    diagnosis_results : Dict[str, Dict]
        诊断结果字典 {sample_id: {"module_probs": {...}, ...}}
    output_path : Path, optional
        输出 JSON 文件路径
    disabled_modules : List[str], optional
        禁用模块列表
        
    Returns
    -------
    Tuple[List[ModuleValidationResult], ModuleValidationSummary]
        (验证结果列表, 汇总统计)
    """
    # 加载标签
    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)
    
    results: List[ModuleValidationResult] = []
    
    for sample_id, diag_result in diagnosis_results.items():
        # 从标签获取 GT
        label = labels_data.get(sample_id, {})
        gt_module = label.get("module_cause", "")
        gt_module_v2 = label.get("module_v2", "")
        
        if not gt_module:
            continue
        
        # 获取模块概率
        module_probs = diag_result.get("module_probs", {})
        if not module_probs:
            continue
        
        # 验证
        result = validate_module_diagnosis(
            sample_id=sample_id,
            gt_module=gt_module,
            gt_module_v2=gt_module_v2,
            module_probs=module_probs,
            disabled_modules=disabled_modules,
        )
        results.append(result)
    
    # 计算汇总
    summary = compute_validation_summary(results, labels_data)
    
    # 输出 JSON
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "summary": summary.to_dict(),
            "results": [r.to_dict() for r in results],
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return results, summary


def format_validation_report(summary: ModuleValidationSummary) -> str:
    """格式化验证报告为可读文本
    
    Parameters
    ----------
    summary : ModuleValidationSummary
        汇总统计
        
    Returns
    -------
    str
        可读文本
    """
    lines = [
        "=" * 60,
        "T6: 模块级诊断自动验证报告",
        "=" * 60,
        f"总样本数: {summary.total_samples}",
        f"有效样本数: {summary.valid_samples}",
        "",
        "命中统计:",
        f"  Top1 命中率: {summary.top1_hit_rate:.2%} ({summary.top1_hit_count}/{summary.valid_samples})",
        f"  Top3 命中率: {summary.top3_hit_rate:.2%} ({summary.top3_hit_count}/{summary.valid_samples})",
        f"  Top5 命中率: {summary.top5_hit_rate:.2%} ({summary.top5_hit_count}/{summary.valid_samples})",
        "",
        f"GT 平均排名: {summary.mean_gt_rank:.2f}",
        f"GT 平均概率: {summary.mean_gt_prob:.4f}",
    ]
    
    if summary.per_class_stats:
        lines.append("")
        lines.append("按系统类别统计:")
        for cls, stats in summary.per_class_stats.items():
            lines.append(f"  [{cls}]")
            lines.append(f"    样本数: {stats['count']}")
            lines.append(f"    Top1: {stats['top1_rate']:.2%}, Top3: {stats['top3_rate']:.2%}")
            lines.append(f"    平均排名: {stats['mean_rank']:.2f}")
    
    # 验收标准检查
    lines.append("")
    lines.append("验收标准检查 (AC5):")
    top1_pass = summary.top1_hit_rate >= 0.60
    top3_pass = summary.top3_hit_rate >= 0.85
    rank_pass = summary.mean_gt_rank <= 3
    
    lines.append(f"  Top1 >= 60%: {'✅ PASS' if top1_pass else '❌ FAIL'} ({summary.top1_hit_rate:.2%})")
    lines.append(f"  Top3 >= 85%: {'✅ PASS' if top3_pass else '❌ FAIL'} ({summary.top3_hit_rate:.2%})")
    lines.append(f"  平均排名 <= 3: {'✅ PASS' if rank_pass else '❌ FAIL'} ({summary.mean_gt_rank:.2f})")
    
    overall_pass = top1_pass and top3_pass and rank_pass
    lines.append("")
    lines.append(f"整体验收: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    lines.append("=" * 60)
    
    return "\n".join(lines)
