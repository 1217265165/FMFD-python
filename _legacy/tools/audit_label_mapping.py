#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3.2: 标签映射审计工具
audit_label_mapping.py

功能：
- 检查 labels.json 中的标签一致性
- 输出 Output/sim_spectrum/data_label_audit_report.md
- 冲突率 > 2% 直接 exit(1)

检查项：
1. fault_type_canonical 是否在 {normal, amp_error, freq_error, ref_error}
2. module_v2_canonical 是否存在且规范化
3. 标签冲突：同一 module 映射到不同 fault_type
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.label_mapping import (
    SYS_CLASS_TO_CN, CN_TO_SYS_CLASS,
    normalize_module_name, module_v2_from_v1,
    HIERARCHY_MAP, expected_system_class_for_module
)


VALID_FAULT_TYPES = {"normal", "amp_error", "freq_error", "ref_error"}


def load_labels(labels_path: Path) -> List[Dict]:
    """Load labels from JSON file."""
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if "samples" in data:
            return data["samples"]
        else:
            # Dict format: convert to list
            return [{"sample_id": k, **v} for k, v in data.items()]
    elif isinstance(data, list):
        return data
    
    return []


def audit_labels(labels: List[Dict]) -> Dict:
    """Audit labels for consistency issues."""
    issues = {
        "missing_fault_type": [],
        "invalid_fault_type": [],
        "missing_module": [],
        "module_fault_type_conflicts": [],
        "unmapped_modules": [],
    }
    
    stats = {
        "total": 0,
        "normal": 0,
        "amp_error": 0,
        "freq_error": 0,
        "ref_error": 0,
    }
    
    # Track module -> fault_type mappings for conflict detection
    module_to_fault_types: Dict[str, Set[str]] = defaultdict(set)
    
    for sample in labels:
        sample_id = sample.get("sample_id", "unknown")
        stats["total"] += 1
        
        # Check fault_type
        fault_type = sample.get("system_fault_class", sample.get("type", ""))
        if not fault_type:
            if sample.get("type") == "normal":
                fault_type = "normal"
            else:
                issues["missing_fault_type"].append(sample_id)
                continue
        
        if fault_type not in VALID_FAULT_TYPES:
            issues["invalid_fault_type"].append({
                "sample_id": sample_id,
                "fault_type": fault_type,
            })
            continue
        
        stats[fault_type] = stats.get(fault_type, 0) + 1
        
        # Check module for non-normal samples
        if fault_type != "normal":
            module_v1 = sample.get("module_cause", sample.get("module", ""))
            module_v2 = sample.get("module_v2", "")
            
            if not module_v2 and not module_v1:
                issues["missing_module"].append(sample_id)
                continue
            
            # Get canonical module name
            if module_v2:
                canonical_module = normalize_module_name(module_v2)
            else:
                canonical_module = module_v2_from_v1(normalize_module_name(module_v1))
            
            # Track module -> fault_type mapping
            module_to_fault_types[canonical_module].add(fault_type)
            
            # Check if module is in known hierarchy
            expected_fault = expected_system_class_for_module(module_v1)
            if not expected_fault:
                issues["unmapped_modules"].append({
                    "sample_id": sample_id,
                    "module": canonical_module,
                })
    
    # Find conflicts: modules mapped to multiple fault types
    for module, fault_types in module_to_fault_types.items():
        if len(fault_types) > 1:
            issues["module_fault_type_conflicts"].append({
                "module": module,
                "fault_types": list(fault_types),
            })
    
    # Calculate conflict rate
    conflict_samples = 0
    for conflict in issues["module_fault_type_conflicts"]:
        # Count samples with this conflicting module
        conflict_samples += sum(
            1 for s in labels
            if normalize_module_name(s.get("module_v2", module_v2_from_v1(s.get("module_cause", "")))) == conflict["module"]
        )
    
    conflict_rate = conflict_samples / stats["total"] if stats["total"] > 0 else 0.0
    
    return {
        "stats": stats,
        "issues": issues,
        "conflict_rate": conflict_rate,
    }


def generate_report(audit_result: Dict, output_path: Path) -> str:
    """Generate markdown report from audit result."""
    stats = audit_result["stats"]
    issues = audit_result["issues"]
    conflict_rate = audit_result["conflict_rate"]
    
    lines = [
        "# 标签审计报告",
        "",
        "## 统计摘要",
        f"- 样本总数: {stats['total']}",
        f"- 正常: {stats['normal']}",
        f"- 幅度失准: {stats['amp_error']}",
        f"- 频率失准: {stats['freq_error']}",
        f"- 参考电平失准: {stats['ref_error']}",
        "",
        "## 冲突率",
        f"- **冲突率: {conflict_rate:.2%}**",
        f"- 阈值: 2%",
        f"- 状态: {'✅ 通过' if conflict_rate <= 0.02 else '❌ 失败'}",
        "",
    ]
    
    # Issues
    lines.append("## 问题详情")
    
    lines.append(f"\n### 缺失 fault_type ({len(issues['missing_fault_type'])})")
    for sid in issues['missing_fault_type'][:10]:
        lines.append(f"- {sid}")
    if len(issues['missing_fault_type']) > 10:
        lines.append(f"- ... 共 {len(issues['missing_fault_type'])} 个")
    
    lines.append(f"\n### 无效 fault_type ({len(issues['invalid_fault_type'])})")
    for item in issues['invalid_fault_type'][:10]:
        lines.append(f"- {item['sample_id']}: {item['fault_type']}")
    
    lines.append(f"\n### 缺失 module ({len(issues['missing_module'])})")
    for sid in issues['missing_module'][:10]:
        lines.append(f"- {sid}")
    
    lines.append(f"\n### 模块-故障类型冲突 ({len(issues['module_fault_type_conflicts'])})")
    for conflict in issues['module_fault_type_conflicts']:
        lines.append(f"- {conflict['module']}: {', '.join(conflict['fault_types'])}")
    
    lines.append(f"\n### 未映射模块 ({len(issues['unmapped_modules'])})")
    seen_modules = set()
    for item in issues['unmapped_modules']:
        if item['module'] not in seen_modules:
            lines.append(f"- {item['module']}")
            seen_modules.add(item['module'])
    
    report = "\n".join(lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='P3.2: 标签映射审计工具'
    )
    parser.add_argument('--labels', default='Output/sim_spectrum/labels.json',
                        help='labels.json 路径')
    parser.add_argument('--output', default='Output/sim_spectrum/data_label_audit_report.md',
                        help='输出报告路径')
    parser.add_argument('--threshold', type=float, default=0.02,
                        help='冲突率阈值 (默认: 0.02)')
    
    args = parser.parse_args()
    
    labels_path = Path(args.labels)
    output_path = Path(args.output)
    
    if not labels_path.exists():
        print(f"[错误] 标签文件不存在: {labels_path}")
        return 1
    
    print("=" * 60)
    print("P3.2: 标签映射审计工具")
    print("=" * 60)
    print(f"标签文件: {labels_path}")
    print(f"输出报告: {output_path}")
    
    # Load and audit labels
    labels = load_labels(labels_path)
    print(f"加载样本数: {len(labels)}")
    
    audit_result = audit_labels(labels)
    
    # Generate report
    report = generate_report(audit_result, output_path)
    print(f"\n报告已保存: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("审计结果摘要")
    print("=" * 60)
    print(f"样本总数: {audit_result['stats']['total']}")
    print(f"冲突率: {audit_result['conflict_rate']:.2%}")
    
    issues = audit_result["issues"]
    print(f"缺失 fault_type: {len(issues['missing_fault_type'])}")
    print(f"无效 fault_type: {len(issues['invalid_fault_type'])}")
    print(f"缺失 module: {len(issues['missing_module'])}")
    print(f"模块-故障冲突: {len(issues['module_fault_type_conflicts'])}")
    
    # Check threshold
    if audit_result["conflict_rate"] > args.threshold:
        print(f"\n❌ 冲突率 {audit_result['conflict_rate']:.2%} > {args.threshold:.0%} 阈值")
        print("   审计失败!")
        return 1
    else:
        print(f"\n✅ 冲突率 {audit_result['conflict_rate']:.2%} <= {args.threshold:.0%} 阈值")
        print("   审计通过!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
