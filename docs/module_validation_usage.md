# T6: 模块级诊断自动验证使用指南

## 概述

`module_validation` 功能用于自动验证 BRB 模块级诊断结果的准确性。通过将诊断输出与 `labels.json` 中的 Ground Truth (GT) 对比，计算 Top1/Top3/Top5 命中率、GT 排名和 GT 概率等指标。

## 使用方法

### 方法1: 通过 `brb_diagnosis_cli.py` 命令行使用

运行诊断时添加 `--labels` 参数指定 `labels.json` 文件路径：

```bash
# 单样本诊断 + 模块验证
python brb_diagnosis_cli.py \
    --input Output/sim_spectrum/raw_curves/sim_00009.csv \
    --output Output/diagnosis_test/result.json \
    --labels Output/sim_spectrum/labels.json \
    --verbose
```

**输出示例** (JSON 格式):

```json
{
  "status": "success",
  "meta": {
    "sample_id": "sim_00009",
    ...
  },
  "system_diagnosis": {
    "predicted_class": "幅度失准",
    "max_prob": 0.4589,
    ...
  },
  "module_diagnosis": {
    "topk": [
      {"module": "混频器", "probability": 0.2341},
      {"module": "ADC", "probability": 0.1856},
      {"module": "低通滤波器", "probability": 0.1523}
    ],
    ...
  },
  "module_validation": {
    "sample_id": "sim_00009",
    "gt_module": "低频段前置低通滤波器",
    "gt_module_v2": "[RF板][RF] 低频通路固定滤波/抑制网络",
    "top1": "混频器",
    "top3": ["混频器", "ADC", "低通滤波器"],
    "top5": ["混频器", "ADC", "低通滤波器", "检波器", "时钟"],
    "gt_rank": 3,
    "gt_prob": 0.1523,
    "top1_hit": false,
    "top3_hit": true,
    "top5_hit": true
  }
}
```

### 方法2: 在 Python 代码中直接调用

```python
from tools.module_validation import (
    validate_module_diagnosis,
    compute_validation_summary,
    run_batch_validation,
    format_validation_report,
    ModuleValidationResult,
    ModuleValidationSummary
)

# 单样本验证
result = validate_module_diagnosis(
    sample_id="sim_00009",
    gt_module="低频段前置低通滤波器",
    gt_module_v2="[RF板][RF] 低频通路固定滤波/抑制网络",
    module_probs={
        "混频器": 0.2341,
        "ADC": 0.1856,
        "低通滤波器": 0.1523,
        "检波器": 0.1102,
        "时钟": 0.0987,
        # ... 其他模块
    },
    disabled_modules=["电源模块"]  # 可选: 禁用的模块
)

print(f"Top1 命中: {result.top1_hit}")  # False
print(f"Top3 命中: {result.top3_hit}")  # True
print(f"GT 排名: {result.gt_rank}")     # 3
print(f"GT 概率: {result.gt_prob:.4f}") # 0.1523
```

### 方法3: 批量验证

```python
from pathlib import Path
from tools.module_validation import run_batch_validation, format_validation_report

# 批量验证
labels_path = Path("Output/sim_spectrum/labels.json")
diagnosis_results = {
    "sim_00001": {"module_probs": {...}},
    "sim_00002": {"module_probs": {...}},
    # ...
}

results, summary = run_batch_validation(
    labels_path=labels_path,
    diagnosis_results=diagnosis_results,
    output_path=Path("Output/validation_report.json"),
    disabled_modules=["电源模块"]
)

# 打印报告
print(format_validation_report(summary))
```

**批量验证报告示例**:

```
============================================================
T6: 模块级诊断自动验证报告
============================================================
总样本数: 100
有效样本数: 95

命中统计:
  Top1 命中率: 62.11% (59/95)
  Top3 命中率: 87.37% (83/95)
  Top5 命中率: 94.74% (90/95)

GT 平均排名: 2.34
GT 平均概率: 0.1856

按系统类别统计:
  [amp_error]
    样本数: 30
    Top1: 63.33%, Top3: 86.67%
    平均排名: 2.20
  [freq_error]
    样本数: 25
    Top1: 60.00%, Top3: 88.00%
    平均排名: 2.45
  [ref_error]
    样本数: 20
    Top1: 65.00%, Top3: 90.00%
    平均排名: 2.10
  [normal]
    样本数: 20
    Top1: 60.00%, Top3: 85.00%
    平均排名: 2.60

验收标准检查 (AC5):
  Top1 >= 60%: ✅ PASS (62.11%)
  Top3 >= 85%: ✅ PASS (87.37%)
  平均排名 <= 3: ✅ PASS (2.34)

整体验收: ✅ PASS
============================================================
```

## 输出字段说明

### `ModuleValidationResult` - 单样本验证结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `sample_id` | `str` | 样本 ID |
| `gt_module` | `str` | Ground Truth 模块名 (V1 版本) |
| `gt_module_v2` | `str` | Ground Truth 模块名 (V2 版本) |
| `top1` | `str` | BRB 输出的 Top1 模块 |
| `top3` | `List[str]` | BRB 输出的 Top3 模块列表 |
| `top5` | `List[str]` | BRB 输出的 Top5 模块列表 |
| `gt_rank` | `int` | GT 模块在排序中的位置 (1-based, -1 表示未找到) |
| `gt_prob` | `float` | GT 模块的后验概率 |
| `top1_hit` | `bool` | Top1 是否命中 GT |
| `top3_hit` | `bool` | GT 是否在 Top3 中 |
| `top5_hit` | `bool` | GT 是否在 Top5 中 |

### `ModuleValidationSummary` - 批量验证汇总

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_samples` | `int` | 总样本数 |
| `valid_samples` | `int` | 有效样本数 (有 GT 的样本) |
| `top1_hit_count` | `int` | Top1 命中数 |
| `top3_hit_count` | `int` | Top3 命中数 |
| `top5_hit_count` | `int` | Top5 命中数 |
| `top1_hit_rate` | `float` | Top1 命中率 |
| `top3_hit_rate` | `float` | Top3 命中率 |
| `top5_hit_rate` | `float` | Top5 命中率 |
| `mean_gt_rank` | `float` | GT 平均排名 |
| `mean_gt_prob` | `float` | GT 平均概率 |
| `per_class_stats` | `Dict` | 按系统类别的统计 |

## 验收标准 (AC5)

根据任务书要求，模块级诊断的验收标准为:

- **Top1 命中率 >= 60%**
- **Top3 命中率 >= 85%**
- **GT 平均排名 <= 3**

## 与 `labels.json` 的关联

`labels.json` 文件包含每个仿真样本的 Ground Truth 信息:

```json
{
  "sim_00009": {
    "type": "amp_error",
    "system_fault_class": "amp_error",
    "module_cause": "低频段前置低通滤波器",
    "module_v2": "[RF板][RF] 低频通路固定滤波/抑制网络",
    "fault_params": {...}
  }
}
```

验证时使用 `module_cause` (V1) 和 `module_v2` 两个字段进行匹配，支持模块名称的标准化和映射。

## 注意事项

1. **sample_id 解析**: CLI 会自动从输入文件名解析 `sim_XXXXX` 格式的 sample_id
2. **模块名称映射**: 使用 `tools/label_mapping.py` 中的 `normalize_module_name()` 和 `module_v2_from_v1()` 进行名称标准化
3. **禁用模块**: 可通过 `disabled_modules` 参数排除某些模块 (如电源模块)
4. **GT 缺失处理**: 如果 `labels.json` 中没有对应样本的 GT，验证结果中 `gt_rank=-1`
