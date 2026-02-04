#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量诊断结果汇总工具
aggregate_batch_diagnosis.py

功能:
- 读取 Output/batch_diagnosis/ 下的所有 *_diagnosis.json 文件
- 汇总系统级和模块级准确率
- 输出 module_localization_report.json

使用方法:
    python tools/aggregate_batch_diagnosis.py --input_dir Output/batch_diagnosis --output Output/batch_diagnosis/module_localization_report.json
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional


def load_diagnosis_results(input_dir: str) -> List[Dict]:
    """加载所有诊断结果文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[错误] 目录不存在: {input_dir}")
        return []
    
    results = []
    for json_file in sorted(input_path.glob("*_diagnosis.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"[警告] 读取失败 {json_file}: {e}")
    
    return results


def evaluate_module_localization(results: List[Dict]) -> Dict:
    """
    评估模块级定位性能
    
    返回:
        - sys_acc: 系统级准确率
        - mod_top1: Top1 准确率
        - mod_top3: Top3 准确率
        - by_fault_type: 按故障类型分组的指标
    """
    stats = {
        "total": 0,
        "sys_correct": 0,
        "top1_correct": 0,
        "top3_correct": 0,
        "by_fault_type": defaultdict(lambda: {"total": 0, "sys": 0, "top1": 0, "top3": 0}),
        "error_cases": [],
        "samples_detail": []
    }
    
    for result in results:
        # 检查是否有 ground_truth
        gt = result.get("ground_truth", {})
        if not gt:
            continue
        
        # 跳过 normal 样本（无模块标签）
        gt_fault_type = gt.get("system_class_en", "normal")
        if gt_fault_type == "normal":
            continue
        
        gt_module = gt.get("module_v2") or gt.get("module", "")
        if not gt_module:
            continue
        
        stats["total"] += 1
        stats["by_fault_type"][gt_fault_type]["total"] += 1
        
        # 获取预测结果
        sys_pred = result.get("system_diagnosis", {}).get("predicted_class", "")
        module_topk = result.get("module_diagnosis", {}).get("topk", [])
        
        # 系统级准确率
        sys_correct = (sys_pred == gt_fault_type) or (
            # 处理中英文映射
            (sys_pred == "幅度失准" and gt_fault_type == "amp_error") or
            (sys_pred == "频率失准" and gt_fault_type == "freq_error") or
            (sys_pred == "参考电平失准" and gt_fault_type == "ref_error")
        )
        if sys_correct:
            stats["sys_correct"] += 1
            stats["by_fault_type"][gt_fault_type]["sys"] += 1
        
        # 模块级准确率
        pred_top1 = module_topk[0]["module"] if module_topk else ""
        pred_top3 = [m["module"] for m in module_topk[:3]]
        
        # 模块匹配函数
        def modules_match(pred: str, gt: str) -> bool:
            """检查两个模块名是否匹配"""
            if not pred or not gt:
                return False
            if pred == gt:
                return True
            
            # 关键词匹配
            keywords = [
                ("中频放大", "中频放大"),
                ("检波", "检波"),
                ("ADC", "ADC"),
                ("低频通路", "低频通路"),
                ("Mixer1", "Mixer1"),
                ("混频", "混频"),
                ("输入连接", "输入连接"),
                ("匹配", "匹配"),
                ("RBW", "RBW"),
                ("数字放大", "数字放大"),
                ("数字增益", "数字"),
                ("LO1", "LO1"),
                ("时钟", "时钟"),
                ("振荡器", "振荡器"),
            ]
            for kw1, kw2 in keywords:
                if kw1 in pred and kw2 in gt:
                    return True
            
            return False
        
        top1_hit = modules_match(pred_top1, gt_module)
        top3_hit = any(modules_match(m, gt_module) for m in pred_top3)
        
        # 也检查 module_validation 字段（如果有）
        validation = result.get("module_validation", {})
        if validation:
            top1_hit = top1_hit or validation.get("top1_hit", False)
            top3_hit = top3_hit or validation.get("top3_hit", False)
        
        if top1_hit:
            stats["top1_correct"] += 1
            stats["by_fault_type"][gt_fault_type]["top1"] += 1
        
        if top3_hit:
            stats["top3_correct"] += 1
            stats["by_fault_type"][gt_fault_type]["top3"] += 1
        
        # 记录样本详情
        sample_id = result.get("meta", {}).get("sample_id", "")
        stats["samples_detail"].append({
            "sample_id": sample_id,
            "fault_type": gt_fault_type,
            "gt_module": gt_module,
            "pred_top1": pred_top1,
            "sys_correct": sys_correct,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
        })
        
        # 记录错误案例
        if not top1_hit:
            stats["error_cases"].append({
                "sample_id": sample_id,
                "fault_type": gt_fault_type,
                "gt_module": gt_module,
                "pred_top1": pred_top1,
                "pred_top3": pred_top3,
            })
    
    # 计算准确率
    total = stats["total"]
    if total > 0:
        stats["sys_acc"] = stats["sys_correct"] / total
        stats["mod_top1"] = stats["top1_correct"] / total
        stats["mod_top3"] = stats["top3_correct"] / total
    else:
        stats["sys_acc"] = 0.0
        stats["mod_top1"] = 0.0
        stats["mod_top3"] = 0.0
    
    # 按故障类型计算
    for ft, ft_stats in stats["by_fault_type"].items():
        ft_total = ft_stats["total"]
        if ft_total > 0:
            ft_stats["sys_acc"] = ft_stats["sys"] / ft_total
            ft_stats["top1_acc"] = ft_stats["top1"] / ft_total
            ft_stats["top3_acc"] = ft_stats["top3"] / ft_total
        else:
            ft_stats["sys_acc"] = 0.0
            ft_stats["top1_acc"] = 0.0
            ft_stats["top3_acc"] = 0.0
    
    return stats


def generate_report(stats: Dict, output_path: str) -> None:
    """生成汇总报告"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备输出数据
    report = {
        "summary": {
            "total_samples": stats["total"],
            "sys_acc": round(stats["sys_acc"], 4),
            "mod_top1": round(stats["mod_top1"], 4),
            "mod_top3": round(stats["mod_top3"], 4),
        },
        "by_fault_type": {
            ft: {
                "total": ft_stats["total"],
                "sys_acc": round(ft_stats.get("sys_acc", 0), 4),
                "top1_acc": round(ft_stats.get("top1_acc", 0), 4),
                "top3_acc": round(ft_stats.get("top3_acc", 0), 4),
            }
            for ft, ft_stats in stats["by_fault_type"].items()
        },
        "error_cases_count": len(stats.get("error_cases", [])),
        "error_cases": stats.get("error_cases", [])[:20],  # 前20个错误案例
    }
    
    # 保存 JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] 报告已保存: {output_file}")
    
    # 同时生成 Markdown 报告
    md_path = output_file.with_suffix('.md')
    md_lines = [
        "# 模块级定位验证报告",
        "",
        "## 整体指标",
        f"- 样本总数: {stats['total']}",
        f"- **系统级准确率**: {stats['sys_acc']:.1%}",
        f"- **mod_top1**: {stats['mod_top1']:.1%}",
        f"- **mod_top3**: {stats['mod_top3']:.1%}",
        "",
        "## 按故障类型分组",
        "| 故障类型 | 样本数 | 系统级 | Top1 | Top3 |",
        "|----------|--------|--------|------|------|",
    ]
    
    for ft, ft_stats in stats["by_fault_type"].items():
        md_lines.append(
            f"| {ft} | {ft_stats['total']} | {ft_stats.get('sys_acc', 0):.1%} | {ft_stats.get('top1_acc', 0):.1%} | {ft_stats.get('top3_acc', 0):.1%} |"
        )
    
    md_lines.extend([
        "",
        f"## 错误案例 (前10个，共{len(stats.get('error_cases', []))}个)",
        "| 样本ID | 故障类型 | GT模块 | 预测Top1 |",
        "|--------|----------|--------|----------|",
    ])
    
    for case in stats.get("error_cases", [])[:10]:
        gt_mod = case['gt_module'][:25] if len(case['gt_module']) > 25 else case['gt_module']
        pred_mod = case['pred_top1'][:25] if len(case['pred_top1']) > 25 else case['pred_top1']
        md_lines.append(
            f"| {case['sample_id']} | {case['fault_type']} | {gt_mod} | {pred_mod} |"
        )
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"[INFO] Markdown 报告已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description='批量诊断结果汇总工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/aggregate_batch_diagnosis.py --input_dir Output/batch_diagnosis
  python tools/aggregate_batch_diagnosis.py --input_dir Output/batch_diagnosis --output Output/batch_diagnosis/module_localization_report.json
        """
    )
    
    parser.add_argument('--input_dir', '-i', default='Output/batch_diagnosis',
                        help='批量诊断输出目录 (包含 *_diagnosis.json 文件)')
    parser.add_argument('--output', '-o', default=None,
                        help='输出报告路径 (默认: <input_dir>/module_localization_report.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细输出')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = Path(args.input_dir) / "module_localization_report.json"
    
    print("=" * 60)
    print("批量诊断结果汇总工具")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出文件: {args.output}")
    
    # 加载诊断结果
    results = load_diagnosis_results(args.input_dir)
    print(f"加载诊断文件数: {len(results)}")
    
    if not results:
        print("[错误] 没有找到诊断结果文件")
        return 1
    
    # 评估
    stats = evaluate_module_localization(results)
    
    # 生成报告
    generate_report(stats, args.output)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    print(f"样本总数: {stats['total']}")
    print(f"系统级准确率: {stats['sys_acc']:.1%}")
    print(f"mod_top1: {stats['mod_top1']:.1%}")
    print(f"mod_top3: {stats['mod_top3']:.1%}")
    
    if args.verbose:
        print("\n按故障类型:")
        for ft, ft_stats in stats["by_fault_type"].items():
            print(f"  {ft}: Top1={ft_stats.get('top1_acc', 0):.1%}, Top3={ft_stats.get('top3_acc', 0):.1%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
