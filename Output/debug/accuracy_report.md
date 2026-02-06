# 三链路准确率验证报告

**生成时间**: 2026-02-05

## 验证配置
- 总样本数: 400 (balanced)
- 故障样本数: 300 (amp_error=100, freq_error=100, ref_error=100)
- 正常样本数: 100

---

## 三链路评估结果对比

| 链路 | N_eval | sys_acc | mod_top1 | mod_top3 |
|------|--------|---------|----------|----------|
| **Path-A: compare_methods** | 60 (test split) | 18.3% | 1.67% | — |
| **Path-B: brb_diagnosis_cli + aggregate** | 300 (fault only) | 47.7% | 33.0% | 60.3% |
| **Path-C: eval_module_localization** | 300 (fault only) | — | 34.0% | 61.0% |

---

## 详细分析

### Path-A (compare_methods)
- **评估样本数**: 60 (使用 60/20/20 test split)
- **系统级准确率**: 18.3%
- **模块 Top-1 准确率**: 1.67%
- **说明**: 由于使用了 test split，只评估了 60 个测试样本

### Path-B (brb_diagnosis_cli + aggregate)
- **评估样本数**: 300 (排除 normal，只评故障样本)
- **系统级准确率**: 47.7%
- **模块 Top-1 准确率**: 33.0%
- **模块 Top-3 准确率**: 60.3%

### Path-C (eval_module_localization)
- **评估样本数**: 300 (使用 manifest_fault_300)
- **模块 Top-1 准确率**: 34.0%
- **模块 Top-3 准确率**: 61.0%

---

## 一致性分析

### Path-B vs Path-C (模块级定位)
| 指标 | Path-B | Path-C | 差异 |
|------|--------|--------|------|
| mod_top1 | 33.0% | 34.0% | 1.0% (≈3样本) |
| mod_top3 | 60.3% | 61.0% | 0.7% (≈2样本) |

**结论**: Path-B 与 Path-C 的模块级指标基本一致，差异在 3 样本以内。

### Path-A 样本口径不一致
- Path-A 使用 test split (60 样本)
- Path-B/C 使用全量故障集 (300 样本)
- 需要统一评估口径才能进行公平比较

---

## 结论

1. **当前系统级准确率**: ~47.7% (在300故障样本上)
2. **当前模块 Top-1 准确率**: ~33-34% (在300故障样本上)
3. **当前模块 Top-3 准确率**: ~60-61% (在300故障样本上)
4. **三链路一致性**: Path-B 与 Path-C 差异 ≤ 3 样本，基本满足一致性要求

---

## 产物路径
- Path-A 输出: `Output/compare_methods/comparison_summary.json`
- Path-B 输出: `Output/batch_diagnosis/module_localization_report.json`
- Path-C 输出: `Output/module_eval/module_localization_results.json`
