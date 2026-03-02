# T3: 低置信度与解释策略文档

## 1. 概述

本文档定义了系统级诊断输出的低置信度检测机制和解释策略。

当系统级诊断输出的概率分布接近时（如 45% vs 40%），用户会感到困惑。本策略将这种情况标记为 `UNCERTAIN`，并输出解释字段帮助用户理解原因。

## 2. 低置信度门限定义

```python
UNCERTAINTY_CONFIG = {
    "max_prob_threshold": 0.55,  # 最大概率低于此值时标记 UNCERTAIN
    "gap_threshold": 0.10,       # max_prob - second_prob 小于此值时标记 UNCERTAIN
    "top_support_features": 5,   # 输出支持 top_class 的特征数量
    "top_conflict_features": 3,  # 输出冲突特征数量
}
```

## 3. UNCERTAIN 触发条件

当满足以下任一条件时，标记 `UNCERTAIN=true`：

1. **最大概率过低**: `max_prob < 55%`
2. **概率差过小**: `max_prob - second_prob < 10%`

## 4. UNCERTAIN 时的解释输出

当触发 UNCERTAIN 时，必须输出以下字段：

### 4.1 基本信息

| 字段 | 说明 | 示例 |
|------|------|------|
| `is_uncertain` | 是否为低置信度 | `true` |
| `reason` | 触发原因 | `"max_prob(45%) < 55% && prob_gap(5%) < 10%"` |
| `top_class` | 最高概率类别 | `"幅度失准"` |
| `runner_up` | 次高概率类别 | `"频率失准"` |
| `max_prob` | 最高概率 | `0.45` |
| `second_prob` | 次高概率 | `0.40` |
| `prob_gap` | 概率差 | `0.05` |

### 4.2 特征解释

| 字段 | 说明 |
|------|------|
| `support_top_features` | 最支持 top_class 的 Top5 特征 |
| `support_runnerup_features` | 最支持 runner_up 的 Top5 特征 |
| `conflict_features` | 同时拉扯两类的特征（贡献差值/方向） |

### 4.3 建议动作

| 字段 | 说明 |
|------|------|
| `suggested_actions` | 建议的验证动作列表 |

## 5. 示例输出

```json
{
  "is_uncertain": true,
  "reason": "max_prob(45.00%) < 55% && prob_gap(5.00%) < 10%",
  "top_class": "幅度失准",
  "runner_up": "频率失准",
  "max_prob": 0.45,
  "second_prob": 0.40,
  "prob_gap": 0.05,
  "support_top_features": [
    {"feature": "X6", "contribution": 0.234},
    {"feature": "X1", "contribution": 0.189},
    {"feature": "X11", "contribution": 0.156}
  ],
  "support_runnerup_features": [
    {"feature": "X16", "contribution": 0.267},
    {"feature": "X17", "contribution": 0.198}
  ],
  "conflict_features": [
    {"feature": "X4", "top_contrib": 0.12, "runnerup_contrib": 0.11}
  ],
  "suggested_actions": [
    "复测全频段幅度响应",
    "检查前端衰减器/放大器链路",
    "检查本振锁定状态",
    "建议重复测试或更换测试点以提高置信度"
  ]
}
```

## 6. 用户界面建议

当 `is_uncertain=true` 时，前端应：

1. 显示警告标记（如 ⚠️ 图标）
2. 显示 Top1 和 Top2 类别及其概率
3. 展开显示解释字段
4. 提供建议动作列表

## 7. 验收标准

- AC2: 触发 UNCERTAIN 时必须输出冲突解释字段
- 用户能从解释字段看出"为什么接近、哪里冲突"
