#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render paper section 3.3 markdown from exported metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


def _read_csv(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return list(reader)


def _markdown_table(path: Path) -> str:
    rows = _read_csv(path)
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _metric(metrics: dict, key: str) -> str:
    value = metrics.get(key, {}).get("value")
    if value is None:
        return "未提供"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_ratio(value: object) -> str:
    if value is None:
        return "未提供"
    try:
        return f"{float(value):.2f}×"
    except (TypeError, ValueError):
        return "未提供"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render section 3.3 markdown")
    parser.add_argument("--output_dir", default="Output/paper_v1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics_summary.json"
    if not metrics_path.exists():
        try:
            from tools.export_paper_metrics import main as export_main
            export_main()
        except Exception as exc:
            raise FileNotFoundError(
                f"metrics_summary.json not found at {metrics_path}. "
                f"Run: python tools/export_paper_metrics.py"
            ) from exc
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics_summary.json not found at {metrics_path}. "
            f"Run: python tools/export_paper_metrics.py"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    comparisons = metrics.get("comparisons", {})
    fewshot_summary = metrics.get("fewshot_summary", {}).get("value", {})
    normal_fpr = metrics.get("normal_fpr", {}).get("value")

    tab_3_2 = output_dir / "tables" / "tab_3_2_benchmark.csv"
    tab_3_3 = output_dir / "tables" / "tab_3_3_system_perf.csv"
    tab_3_4 = output_dir / "tables" / "tab_3_4_severity_perf.csv"
    tab_3_5 = output_dir / "tables" / "tab_3_5_examples.csv"
    fewshot = output_dir / "tables" / "fewshot_results.csv"

    section_lines = []
    section_lines.append("## 3.3 诊断性能评估与对比")
    section_lines.append("")
    section_lines.append(
        "对比方法包括 HCF、BRB-P、BRB-MU、DBRB（如有实现 A-IBRB 也纳入），"\
        "并统一使用 compare_methods.py 中的特征与划分设置。"
    )
    section_lines.append("")
    section_lines.append("### 3.3.1 规则库规模与参数复杂度分析")
    section_lines.append(
        "表 3-2 汇总了各方法的规则规模、参数量与系统级准确率，"\
        "同时给出平均推理耗时用于复杂度对比。"
    )
    rule_ratio = _format_ratio(comparisons.get("rule_compression_ratio_vs_brb_p", {}).get("value"))
    speedup = _format_ratio(comparisons.get("speedup_vs_brb_p", {}).get("value"))
    section_lines.append(
        f"相对 BRB-P，本文方法规则规模压缩比为 {rule_ratio}，推理加速比为 {speedup}（若缺失则标注未提供）。"
    )
    section_lines.append("")
    if tab_3_2.exists():
        section_lines.append(_markdown_table(tab_3_2))
    else:
        section_lines.append("表 3-2 未生成。")
    section_lines.append("")

    section_lines.append("### 3.3.2 诊断性能与小样本鲁棒性评估")
    section_lines.append(
        "系统级分类统计如表 3-3，严重度分层结果见表 3-4，"\
        "小样本曲线来自表 3-5 的 few-shot 实验结果。"
    )
    if normal_fpr is not None:
        try:
            section_lines.append(f"正常类的误报率（FPR）为 {float(normal_fpr):.4f}。")
        except (TypeError, ValueError):
            section_lines.append("正常类误报率未提供。")
    section_lines.append("")
    if tab_3_3.exists():
        section_lines.append("**表 3-3 系统级分类性能**")
        section_lines.append("")
        section_lines.append(_markdown_table(tab_3_3))
        section_lines.append("")
    if tab_3_4.exists():
        section_lines.append("**表 3-4 严重度分层性能**")
        section_lines.append("")
        section_lines.append(_markdown_table(tab_3_4))
        section_lines.append("")
    if fewshot.exists():
        section_lines.append("**小样本实验结果（few-shot）**")
        section_lines.append("")
        section_lines.append(_markdown_table(fewshot))
        section_lines.append("")
        if fewshot_summary:
            low = fewshot_summary.get("0.1", {})
            high = fewshot_summary.get("0.8", {})
            if low and high:
                section_lines.append(
                    "小样本训练 10%→80% 时，平均系统准确率从 "
                    f"{low.get('mean_accuracy', 0.0):.4f} 提升至 {high.get('mean_accuracy', 0.0):.4f}。"
                )
                section_lines.append("")

    section_lines.append("### 3.3.3 分层架构优势与案例分析")
    section_lines.append(
        "表 3-5 给出典型诊断输出示例，"\
        "包括正常、幅度失准、频率失准、参考电平失准以及误分样本。"
    )
    section_lines.append("")
    if tab_3_5.exists():
        section_lines.append(_markdown_table(tab_3_5))
    else:
        section_lines.append("表 3-5 未生成。")

    output_path = output_dir / "section_3_3.md"
    output_path.write_text("\n".join(section_lines), encoding="utf-8")
    print(f"Saved section to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
