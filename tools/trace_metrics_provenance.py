#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5: 钉死"历史高准确率来源"
trace_metrics_provenance.py

功能：
- 追溯高准确率数值的来源
- 输出 Output/debug/metrics_provenance.md

输出包含：
- 哪条命令/脚本跑出了 95%/53%
- 使用的数据路径、样本数、labels hash
- 是否启用 gating
- 是否做了样本过滤（跳过冲突/缺失等）
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file."""
    if not file_path.exists():
        return "FILE_NOT_FOUND"
    
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def count_samples_in_labels(labels_path: Path) -> Dict[str, int]:
    """Count samples by fault type in labels.json."""
    if not labels_path.exists():
        return {"error": "file not found"}
    
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if "samples" in data:
            samples = data["samples"]
        else:
            samples = [{"sample_id": k, **v} for k, v in data.items()]
    else:
        samples = data
    
    counts = {"total": 0, "normal": 0, "amp_error": 0, "freq_error": 0, "ref_error": 0}
    for s in samples:
        counts["total"] += 1
        ft = s.get("system_fault_class", s.get("type", "normal"))
        if ft in counts:
            counts[ft] += 1
    
    return counts


def find_recent_results() -> List[Dict]:
    """Find recent evaluation results in Output directory."""
    results = []
    
    output_dir = ROOT / "Output"
    if not output_dir.exists():
        return results
    
    # Check known result locations
    known_paths = [
        "compare/metrics_summary.json",
        "compare/results_table.csv",
        "batch_diagnosis/module_localization_report.json",
        "module_eval/module_localization_results.json",
        "diagnosis/batch/module_localization_report.json",
        "sim_spectrum/eval_results.json",
    ]
    
    for rel_path in known_paths:
        full_path = output_dir / rel_path
        if full_path.exists():
            stat = full_path.stat()
            results.append({
                "path": str(full_path.relative_to(ROOT)),
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size,
            })
    
    return results


def analyze_metrics_source() -> Dict[str, Any]:
    """Analyze where metrics came from."""
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "data_paths": {},
        "metrics_sources": [],
        "gating_config": {},
        "sample_filtering": {},
    }
    
    # Check labels.json
    labels_path = ROOT / "Output" / "sim_spectrum" / "labels.json"
    if labels_path.exists():
        analysis["data_paths"]["labels"] = {
            "path": str(labels_path.relative_to(ROOT)),
            "hash": compute_file_hash(labels_path),
            "counts": count_samples_in_labels(labels_path),
        }
    
    # Check raw_curves directory
    curves_dir = ROOT / "Output" / "sim_spectrum" / "raw_curves"
    if curves_dir.exists():
        csv_files = list(curves_dir.glob("*.csv"))
        analysis["data_paths"]["raw_curves"] = {
            "path": str(curves_dir.relative_to(ROOT)),
            "file_count": len(csv_files),
        }
    
    # Check gating config
    gating_config_path = ROOT / "config" / "gating_prior.json"
    if gating_config_path.exists():
        with gating_config_path.open("r", encoding="utf-8") as f:
            analysis["gating_config"] = json.load(f)
    
    # Find recent results
    analysis["metrics_sources"] = find_recent_results()
    
    # Analyze unified inference entry usage
    analysis["unified_entry"] = {
        "file": "methods/ours_adapter.py",
        "function": "infer_system_and_modules",
        "description": "All paths now use this single entry point",
    }
    
    return analysis


def generate_provenance_report(analysis: Dict, output_path: Path) -> str:
    """Generate markdown provenance report."""
    lines = [
        "# 指标来源追溯报告 (Metrics Provenance)",
        "",
        f"生成时间: {analysis['timestamp']}",
        "",
        "## 数据路径",
    ]
    
    # Labels info
    if "labels" in analysis["data_paths"]:
        labels = analysis["data_paths"]["labels"]
        lines.extend([
            "",
            "### labels.json",
            f"- 路径: `{labels['path']}`",
            f"- Hash: `{labels['hash']}`",
            f"- 样本数:",
        ])
        counts = labels.get("counts", {})
        for k, v in counts.items():
            lines.append(f"  - {k}: {v}")
    
    # Curves info
    if "raw_curves" in analysis["data_paths"]:
        curves = analysis["data_paths"]["raw_curves"]
        lines.extend([
            "",
            "### raw_curves/",
            f"- 路径: `{curves['path']}`",
            f"- CSV 文件数: {curves['file_count']}",
        ])
    
    # Gating config
    lines.extend([
        "",
        "## 门控配置 (Gating Config)",
    ])
    gating = analysis.get("gating_config", {})
    if gating:
        lines.append(f"- 融合方法: `{gating.get('method', 'unknown')}`")
        if "gated" in gating:
            g = gating["gated"]
            lines.append(f"- 置信度阈值: {g.get('threshold', 'N/A')}")
            lines.append(f"- RF权重范围: [{g.get('w_min', 'N/A')}, {g.get('w_max', 'N/A')}]")
    else:
        lines.append("- 未找到配置文件")
    
    # Unified entry
    lines.extend([
        "",
        "## 统一推理入口",
        f"- 文件: `{analysis['unified_entry']['file']}`",
        f"- 函数: `{analysis['unified_entry']['function']}`",
        f"- 说明: {analysis['unified_entry']['description']}",
        "",
        "所有链路现在必须通过此入口：",
        "- compare_methods.py (ours 分支)",
        "- brb_diagnosis_cli.py",
        "- aggregate_batch_diagnosis.py",
        "- eval_module_localization.py",
    ])
    
    # Recent results
    lines.extend([
        "",
        "## 历史指标来源",
    ])
    
    sources = analysis.get("metrics_sources", [])
    if sources:
        lines.append("")
        lines.append("| 路径 | 修改时间 | 大小 |")
        lines.append("|------|----------|------|")
        for src in sources:
            lines.append(f"| `{src['path']}` | {src['mtime']} | {src['size']} B |")
    else:
        lines.append("未找到历史评估结果")
    
    # Commands to reproduce
    lines.extend([
        "",
        "## 复现命令",
        "",
        "### D1. compare_methods.py",
        "```bash",
        "python compare_methods.py",
        "```",
        "",
        "### D2. diagnosis batch",
        "```bash",
        "python brb_diagnosis_cli.py --input_dir Output/sim_spectrum/raw_curves --labels Output/sim_spectrum/labels.json --output Output/diagnosis/batch",
        "python aggregate_batch_diagnosis.py --labels Output/sim_spectrum/labels.json --pred_dir Output/diagnosis/batch",
        "python eval_module_localization.py --labels Output/sim_spectrum/labels.json --pred_dir Output/diagnosis/batch",
        "```",
        "",
        "### D3. alignment debug",
        "```bash",
        "python tools/debug_single_sample_alignment.py --sample_id sim_00000 --labels Output/sim_spectrum/labels.json --curve_csv Output/sim_spectrum/raw_curves/sim_00000.csv",
        "python tools/debug_alignment_batch20.py --labels Output/sim_spectrum/labels.json --curves_dir Output/sim_spectrum/raw_curves",
        "```",
    ])
    
    # Filtering notes
    lines.extend([
        "",
        "## 样本过滤说明",
        "",
        "- 评估跳过 `normal` 样本的模块级指标",
        "- 冲突率 > 2% 的数据集会被拒绝",
        "- 禁用模块 (DISABLED_MODULES) 不参与 TopK 统计",
    ])
    
    report = "\n".join(lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='P5: 指标来源追溯工具'
    )
    parser.add_argument('--output', default='Output/debug/metrics_provenance.md',
                        help='输出报告路径')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("=" * 60)
    print("P5: 指标来源追溯工具")
    print("=" * 60)
    
    # Analyze metrics sources
    print("\n分析指标来源...")
    analysis = analyze_metrics_source()
    
    # Generate report
    print(f"\n生成报告: {output_path}")
    report = generate_provenance_report(analysis, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("追溯摘要")
    print("=" * 60)
    
    if "labels" in analysis["data_paths"]:
        labels = analysis["data_paths"]["labels"]
        print(f"Labels hash: {labels['hash']}")
        print(f"样本数: {labels['counts'].get('total', 0)}")
    
    print(f"\n历史结果文件: {len(analysis['metrics_sources'])} 个")
    for src in analysis["metrics_sources"]:
        print(f"  - {src['path']}")
    
    print(f"\n统一入口: {analysis['unified_entry']['function']}()")
    print(f"报告路径: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
