# BRB 解释字段说明

## 1. 概述

本文档说明 BRB 诊断输出中的解释字段格式和语义。

## 2. 系统级诊断字段

### 2.1 基本输出

```json
{
  "system_diagnosis": {
    "probabilities": {
      "正常": 0.65,
      "幅度失准": 0.15,
      "频率失准": 0.12,
      "参考电平失准": 0.08
    },
    "predicted_class": "正常",
    "max_prob": 0.65,
    "is_normal": true
  }
}
```

### 2.2 UNCERTAIN 解释字段

当 `uncertainty.is_uncertain=true` 时输出：

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_uncertain` | bool | 是否为低置信度状态 |
| `reason` | string | 触发原因描述 |
| `top_class` | string | 最高概率类别 |
| `runner_up` | string | 次高概率类别 |
| `max_prob` | float | 最高概率 |
| `second_prob` | float | 次高概率 |
| `prob_gap` | float | 概率差值 |
| `support_top_features` | array | 支持 top_class 的特征列表 |
| `support_runnerup_features` | array | 支持 runner_up 的特征列表 |
| `conflict_features` | array | 冲突特征列表 |
| `suggested_actions` | array | 建议验证动作 |

## 3. 模块级诊断字段

### 3.1 基本输出

```json
{
  "module_diagnosis": {
    "probabilities": {...},
    "topk": [
      {"module": "...", "probability": 0.25},
      {"module": "...", "probability": 0.18},
      {"module": "...", "probability": 0.15}
    ],
    "disabled_modules": ["..."]
  }
}
```

### 3.2 验证字段 (T6)

当提供 labels.json 时输出：

```json
{
  "module_validation": {
    "sample_id": "sim_00005",
    "gt_module": "低频段前置低通滤波器",
    "gt_module_v2": "[RF板][RF] 低频通路固定滤波/抑制网络",
    "top1": "[RF板][RF] 输入连接/匹配/保护",
    "top3": ["...", "...", "..."],
    "top5": ["...", "...", "...", "...", "..."],
    "gt_rank": 3,
    "gt_prob": 0.119,
    "top1_hit": false,
    "top3_hit": true,
    "top5_hit": true
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `sample_id` | string | 样本 ID |
| `gt_module` | string | GT 模块名 (v1) |
| `gt_module_v2` | string | GT 模块名 (v2) |
| `top1` | string | BRB 输出 Top1 模块 |
| `top3` | array | BRB 输出 Top3 模块 |
| `top5` | array | BRB 输出 Top5 模块 |
| `gt_rank` | int | GT 在排序中的位置 (1-based) |
| `gt_prob` | float | GT 的后验概率 |
| `top1_hit` | bool | Top1 是否命中 GT |
| `top3_hit` | bool | GT 是否在 Top3 中 |
| `top5_hit` | bool | GT 是否在 Top5 中 |

## 4. 证据字段

```json
{
  "evidence": {
    "global_offset_db": -0.002,
    "hf_std_db": 0.024,
    "p95_abs_dev_db": 0.059,
    "inside_env_frac": 1.0,
    "peak_freq_mae_hz": 3968.15,
    "peak_freq_outlier_frac": 0.0
  },
  "evidence_detail": {
    "viol_rate": 0.0,
    "envelope_violation": false,
    "violation_max_db": 0.0,
    "violation_energy": 0.0,
    "baseline_coverage": 1.0
  }
}
```

## 5. Ground Truth 字段

当提供 labels.json 时输出：

```json
{
  "ground_truth": {
    "type": "fault",
    "system_class_en": "amp_error",
    "system_class_cn": "幅度失准",
    "module": "低频段前置低通滤波器",
    "module_v2": "[RF板][RF] 低频通路固定滤波/抑制网络",
    "fault_params": {
      "severity": "mid",
      "tier_target": "in_spec_weak",
      "type": "lpf_shift",
      "subtype": "amp_error_band",
      "ripple_amp_db": 0.075,
      "period_norm": 0.089,
      "template_id": "T3"
    }
  }
}
```
