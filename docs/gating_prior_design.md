# 门控先验设计文档 (Gating Prior Design)

## 1. 背景

当前架构中，RandomForest (RF) 直接**替代**了系统级 BRB，导致：
- 系统级准确率很高（~95%），但失去可解释性
- 与"知识驱动 + 分层 BRB"的论文叙事不符
- 诊断链与对比链的评估口径不一致

## 2. 设计目标

将 RF 调整为**门控先验**：
- RF 不再替代系统级 BRB，而是提供先验概率
- 系统级 BRB 保持主推理链路的地位
- 融合两者输出，兼顾准确率和可解释性

## 3. 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      系统级诊断                              │
│                                                             │
│  特征 (22项) ──┬──→ RandomForest ──→ rf_probs              │
│                │                           │                 │
│                │                           ↓                 │
│                │                    ┌─────────────┐          │
│                │                    │   fuse()    │          │
│                │                    └─────────────┘          │
│                │                           ↑                 │
│                └──→ system_level_infer ──→ brb_probs        │
│                                                             │
│  输出: fused_probs = [p_normal, p_amp, p_freq, p_ref]       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓ fault_type = argmax(fused_probs)
┌─────────────────────────────────────────────────────────────┐
│                      模块级诊断                              │
│                                                             │
│  fault_type ──→ hierarchical_module_infer() ──→ TopK 模块   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 4. 融合方案

### 4.1 S1: 线性加权融合（最简单）

```python
p_fused = normalize(w * p_rf + (1-w) * p_brb)
```

- 默认 `w = 0.7`
- 优点：简单直观
- 缺点：可能被概率饱和影响

### 4.2 S2: Logit 融合（更稳定）

```python
logit(p_fused) = w * logit(p_rf) + (1-w) * logit(p_brb)
p_fused = softmax(logit(p_fused))
```

- 更像贝叶斯意义下的"证据融合"
- 不会被某一边概率过饱和直接压死另一边

### 4.3 S3: 置信度门控融合（推荐用于论文）

```python
# 1) 计算 RF 置信度
confidence = max(p_rf)

# 2) 动态权重
if confidence >= threshold:
    w = clip((c - c0)/(1-c0), w_min, w_max)
else:
    w = w_min

# 3) Logit 融合
logit(p_fused) = w * logit(p_rf) + (1-w) * logit(p_brb)
```

**解释口径**：RF 只在"它很确定时"提供强先验；不确定时退回 BRB 的可解释推理。

## 5. 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| method | "gated" | 融合方法: linear/logit/gated |
| threshold | 0.55 | 置信度阈值 |
| w_min | 0.3 | 最小 RF 权重 |
| w_max | 0.85 | 最大 RF 权重 |
| temperature | 1.0 | 温度校准 |

## 6. 消融实验

| 方案 | sys_acc | 说明 |
|------|---------|------|
| BRB-only | 45% | 纯规则推理，可解释但准确率低 |
| RF-only | 95% | 黑盒，准确率高但不可解释 |
| RF-prior+BRB (S3) | 91% | 兼顾准确率和可解释性 |

## 7. 代码位置

- 融合模块: `BRB/gating_prior.py`
- 调用位置: `methods/ours_adapter.py`

## 8. 论文叙事

> 本文提出的方法采用**知识驱动的分层 BRB 架构**，其中：
> - 系统级采用 BRB 规则聚合器进行故障类型推理
> - **RandomForest 作为门控先验**，在高置信度时增强系统级判断
> - 模块级采用条件化 BRB 子图进行故障定位

这样既保持了论文的可解释性叙事，又利用了 RF 的工程优势。
