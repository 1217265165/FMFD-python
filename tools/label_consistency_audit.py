#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1: 数据-标签一致性审计工具

扫描 labels.json，检测 module_cause != module_v2 的冲突样本，
生成一致性报告和模块命名字典建议。

输出:
- report_label_consistency.md: 一致性审计报告
- module_taxonomy_v1.json: 模块命名字典建议
- conflict_map.csv: 冲突映射对详情
"""
from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tools.label_mapping import (
    normalize_module_name,
    module_v2_from_v1,
    MODULE_V1_TO_V2,
)


def audit_label_consistency(labels_path: Path, output_dir: Path) -> Dict:
    """执行标签一致性审计。
    
    Parameters
    ----------
    labels_path : Path
        labels.json 文件路径
    output_dir : Path
        输出目录
        
    Returns
    -------
    Dict
        审计结果统计
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载标签数据
    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)
    
    total_samples = len(labels_data)
    conflict_samples: List[Dict] = []
    conflict_counter: Counter = Counter()
    template_counter: Dict[str, Counter] = defaultdict(Counter)
    
    # 扫描所有样本
    for sample_id, sample_data in labels_data.items():
        module_cause = sample_data.get("module_cause", "")
        module_v2 = sample_data.get("module_v2", "")
        template_id = sample_data.get("template_id", "")
        system_fault_class = sample_data.get("system_fault_class", "")
        
        # 规范化名称
        cause_normalized = normalize_module_name(module_cause)
        v2_normalized = normalize_module_name(module_v2)
        
        # 检查预期映射
        expected_v2 = module_v2_from_v1(cause_normalized)
        expected_v2_normalized = normalize_module_name(expected_v2)
        
        # 检测冲突
        is_conflict = v2_normalized != expected_v2_normalized
        
        if is_conflict:
            conflict_key = f"{cause_normalized} -> {v2_normalized}"
            conflict_counter[conflict_key] += 1
            template_counter[conflict_key][template_id] += 1
            
            conflict_samples.append({
                "sample_id": sample_id,
                "module_cause": module_cause,
                "module_v2": module_v2,
                "expected_v2": expected_v2,
                "template_id": template_id,
                "system_fault_class": system_fault_class,
            })
    
    # 生成统计
    conflict_count = len(conflict_samples)
    conflict_rate = conflict_count / total_samples if total_samples > 0 else 0
    
    # 按频次排序的冲突映射
    sorted_conflicts = conflict_counter.most_common()
    
    # 生成 conflict_map.csv
    conflict_csv_path = output_dir / "conflict_map.csv"
    with conflict_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conflict_key", "count", "associated_templates"])
        for conflict_key, count in sorted_conflicts:
            templates = ", ".join(
                f"{tid}({c})" 
                for tid, c in template_counter[conflict_key].most_common()
            )
            writer.writerow([conflict_key, count, templates])
    
    # 生成 module_taxonomy_v1.json
    taxonomy = {
        "version": "v1",
        "description": "模块命名字典建议 - 由 label_consistency_audit.py 自动生成",
        "v1_to_v2_mapping": MODULE_V1_TO_V2,
        "conflicts_found": sorted_conflicts,
        "suggested_additions": [],
    }
    
    # 检测需要新增的映射
    seen_causes = set()
    for sample in conflict_samples:
        cause = normalize_module_name(sample["module_cause"])
        v2 = normalize_module_name(sample["module_v2"])
        if cause and cause not in MODULE_V1_TO_V2 and cause not in seen_causes:
            taxonomy["suggested_additions"].append({
                "v1": cause,
                "observed_v2": v2,
                "action": "ADD_TO_MODULE_V1_TO_V2",
            })
            seen_causes.add(cause)
    
    taxonomy_path = output_dir / "module_taxonomy_v1.json"
    with taxonomy_path.open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
    
    # 生成 report_label_consistency.md
    report_path = output_dir / "report_label_consistency.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# T1: 数据-标签一致性审计报告\n\n")
        f.write(f"**审计时间**: {Path(__file__).stat().st_mtime}\n")
        f.write(f"**数据源**: `{labels_path}`\n\n")
        
        f.write("## 1. 概述\n\n")
        f.write(f"- 总样本数: {total_samples}\n")
        f.write(f"- 冲突样本数: {conflict_count}\n")
        f.write(f"- 冲突率: {conflict_rate:.2%}\n\n")
        
        f.write("## 2. 冲突映射统计（按频次排序）\n\n")
        f.write("| 冲突映射 | 计数 | 关联模板 |\n")
        f.write("|----------|------|----------|\n")
        for conflict_key, count in sorted_conflicts[:20]:
            templates = template_counter[conflict_key].most_common(3)
            templates_str = ", ".join(f"{tid}({c})" for tid, c in templates)
            f.write(f"| {conflict_key} | {count} | {templates_str} |\n")
        
        if len(sorted_conflicts) > 20:
            f.write(f"\n*（显示前 20 条，共 {len(sorted_conflicts)} 条冲突映射）*\n")
        
        f.write("\n## 3. 冲突样本详情（前 50 条）\n\n")
        f.write("| sample_id | module_cause | module_v2 | expected_v2 | template_id |\n")
        f.write("|-----------|--------------|-----------|-------------|-------------|\n")
        for sample in conflict_samples[:50]:
            f.write(f"| {sample['sample_id']} | {sample['module_cause']} | {sample['module_v2']} | {sample['expected_v2']} | {sample['template_id']} |\n")
        
        f.write("\n## 4. 建议操作\n\n")
        if not sorted_conflicts:
            f.write("✅ 无冲突发现，标签一致性良好。\n")
        else:
            f.write("### 需要统一的映射\n\n")
            for item in taxonomy["suggested_additions"][:10]:
                f.write(f"- `{item['v1']}` → `{item['observed_v2']}` (建议添加到 MODULE_V1_TO_V2)\n")
        
        f.write("\n## 5. 输出文件\n\n")
        f.write(f"- `{conflict_csv_path.name}`: 冲突映射详情 CSV\n")
        f.write(f"- `{taxonomy_path.name}`: 模块命名字典建议 JSON\n")
    
    return {
        "total_samples": total_samples,
        "conflict_count": conflict_count,
        "conflict_rate": conflict_rate,
        "output_files": {
            "report": str(report_path),
            "taxonomy": str(taxonomy_path),
            "conflict_map": str(conflict_csv_path),
        },
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="T1: 数据-标签一致性审计")
    parser.add_argument("--labels", "-l", default="Output/sim_spectrum/labels.json",
                        help="labels.json 文件路径")
    parser.add_argument("--output", "-o", default="Output/label_audit",
                        help="输出目录")
    
    args = parser.parse_args()
    
    labels_path = Path(args.labels)
    output_dir = Path(args.output)
    
    if not labels_path.exists():
        print(f"[错误] labels.json 不存在: {labels_path}")
        return 1
    
    result = audit_label_consistency(labels_path, output_dir)
    
    print(f"[T1 审计完成]")
    print(f"  总样本数: {result['total_samples']}")
    print(f"  冲突样本数: {result['conflict_count']}")
    print(f"  冲突率: {result['conflict_rate']:.2%}")
    print(f"  输出文件:")
    for name, path in result["output_files"].items():
        print(f"    - {name}: {path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
