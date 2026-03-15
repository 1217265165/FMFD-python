# 分层 BRB 第二层（模块级定位）架构分析

> 本文档回答论文写作对齐所需的 4 个核心问题，所有描述严格对应代码实现。

---

## Q1：子网拓扑结构 — 是否为三类异常分别实例化了独立的 BRB 子网？

### 回答：是的，代码实现了条件路由的三组独立子图，但不是通过"三个 BRB 类实例"实现的，而是通过一个统一函数 `hierarchical_module_infer()` 中的 `if/elif` 分支实现条件激活。

### 代码事实

**子图映射定义** (`BRB/module_brb.py`, L720-763):

```python
FAULT_TO_SUBGRAPH = {
    "freq_error": "LO_Clock_Network",
    "amp_error":  "RF_IF_ADC_Network",
    "ref_error":  "Calibration_Network",
    "normal":     None               # 正常时不触发模块推理
}

SUBGRAPH_TO_BOARDS = {
    "LO_Clock_Network":    ["LO/时钟板"],
    "RF_IF_ADC_Network":   ["RF板", "数字中频板", "电源板"],
    "Calibration_Network": ["校准链路"]
}
```

**三个独立的候选模块集合** (`BRB/module_brb.py`, L100-118):

| 子网名称 | 对应异常类型 | 候选模块数 | 候选模块 |
|----------|------------|-----------|---------|
| `_AMP_MODULES` | 幅度失准 | 6 | ADC, Mixer1, RF低通滤波, Power, IF, DSP |
| `_FREQ_MODULES` | 频率失准 | 4 | 参考分配, Mixer1, LO1, OCXO |
| `_REF_MODULES` | 参考电平失准 | 3 | 校准源, 校准表/存储, 校准路径开关/耦合 |

**条件路由逻辑** (`hierarchical_module_infer()`, L837-844):

```python
if fault_type == "amp_error":
    filtered_probs = _scale_and_normalize(_BASE_AMP_PRIORS, hp[0:6], _AMP_MODULES)
elif fault_type == "freq_error":
    filtered_probs = _scale_and_normalize(_BASE_FREQ_PRIORS, hp[6:10], _FREQ_MODULES)
elif fault_type == "ref_error":
    filtered_probs = _scale_and_normalize(_BASE_REF_PRIORS, hp[10:13], _REF_MODULES)
```

### 论文写法建议

> ✅ 可以说"根据系统级诊断结果，条件激活对应的模块级 BRB 子网（Amp-BRB / Freq-BRB / Ref-BRB）"。
> ⚠️ 注意：代码中**不是**三个独立的 BRB 类实例，而是同一个函数中的三组独立参数集和候选集。
> 效果等价于三个并行子网，只是实现上是"条件路由 + 参数切换"。

---

## Q2：频率与参考链路的真实推理逻辑

### 回答：**同一 BRB 子网的多个并行输出结论**。每种异常类型的子网将所有候选模块作为一组并行的结论输出（概率因子），不是链式/串行的。

### 代码事实

**以频率失准为例** (`hierarchical_module_infer()`, L915-965):

当 `fault_type == "freq_error"` 时，函数执行以下步骤：

**Step 1 — 先验初始化**（L839-840）：
```python
_BASE_FREQ_PRIORS = [0.37, 0.33, 0.17, 0.13]  # 对应4个模块
_FREQ_MODULES = ["[时钟板][参考分配]", "[RF板][Mixer1]", "[LO/时钟板][LO1] 合成链", "[时钟板][参考域] 10MHz 基准 OCXO"]
filtered_probs = _scale_and_normalize(_BASE_FREQ_PRIORS, hp[6:10], _FREQ_MODULES)
```

**Step 2 — 特征驱动的决策树评分**（L921-965）：
```python
# Mixer1: 高 X13 (>0.7) + 低 X14 (<0.003) → 强证据
mixer_s = 1.0
if x13 > 0.7:  mixer_s += 5.0
if x14 < 0.003: mixer_s += 2.0

# 参考分配: 高 X35 (>0.0002), 低 X36 (<0.86), 高 X7 (>0.07)
ref_dist_s = 1.0
if x35 > 0.0002: ref_dist_s += 2.0
if x36 < 0.86:   ref_dist_s += 1.5

# LO1, OCXO 类似...
```

**Step 3 — 调整 + 归一化**（L964-1006）：
```python
for mod in filtered_probs:
    filtered_probs[mod] *= adj.get(mod, 1.0) ** feat_sens  # feat_sens 由 P-CMA-ES 优化

# 最后归一化确保 Σ=1
total = sum(filtered_probs.values())
filtered_probs = {m: p / total for m, p in filtered_probs.items()}
```

### 融合公式（可用于论文）

$$
P(M_j | \text{freq\_error}) = \frac{\pi_j \cdot s_j \cdot a_j^{\gamma}}{\sum_k \pi_k \cdot s_k \cdot a_k^{\gamma}}
$$

其中：
- $\pi_j$ = 基础先验（训练数据统计）
- $s_j$ = P-CMA-ES 优化的缩放因子
- $a_j$ = 特征驱动的决策树评分
- $\gamma$ = 特征敏感度参数（P-CMA-ES 优化）

**参考电平失准完全一致**（L967-1001）：3 个校准模块按相同方式并行输出。

---

## Q3：门控与路由的具体实现 — 如何传递系统级置信度？

### 回答：代码实现了**三层门控机制**，从粗到细：

### 第一层：RF 置信度门控（GatingPriorFusion）

**位置**: `BRB/gating_prior.py`, L119-155  
**类型**: **连续软门控（非 if/else）**  
**机制**: RF 分类器的最大输出概率决定 RF 与 BRB 的融合权重

```python
confidence = max(rf_probs)  # RF 峰值概率

# 动态权重映射
if confidence >= threshold (0.55):
    w = w_min + (confidence - threshold)/(1 - threshold) * (w_max - w_min)
    w = clip(w, w_min=0.3, w_max=0.85)
else:
    w = w_min (0.3)  # RF 不确定时，让 BRB 主导

# 在 logit 空间进行融合
fused = softmax(w * logit(P_rf) + (1-w) * logit(P_brb))
```

**论文关键**: 这不是简单的 if/else，而是连续可微的门控函数。

### 第二层：异常概率软门控

**位置**: `BRB/module_brb.py`, L417-429  
**类型**: **阈值软门控**  
**机制**: 系统级异常概率 P_abn = 1 - P(正常) 决定是否执行模块推理

```python
SOFT_GATING_CONFIG = {
    "TH_NORMAL_STRONG": 0.85,   # P(正常) >= 0.85 → 标记低置信度
    "TH_P_ABN_FORCE":  0.20,    # P_abn > 0.2 → 强制模块推理
}

p_abn = 1.0 - normal_prob
force_module_infer = (p_abn > 0.20)
# 即使系统判"正常"，只要 P_abn > 0.2 也执行模块推理
```

### 第三层：多假设软门控路由（最关键）

**位置**: `hierarchical_module_infer_soft_gating()`, L1014-1120  
**类型**: **概率加权多路由（非硬切分）**  
**机制**: 当 top-2 异常类型概率差距 < δ 时，同时激活两个子网并加权融合

```python
# 条件：top-1 与 top-2 概率差距 < δ (默认 0.1)
if (top1_prob - top2_prob) < delta:
    use_top2 = True

# 分别运行两个子网
probs_1 = hierarchical_module_infer(top1_ft, features)  # e.g., amp_error
probs_2 = hierarchical_module_infer(top2_ft, features)  # e.g., freq_error

# 概率加权融合
for module in all_modules:
    fused[module] = (P_top1 * probs_1[module] + P_top2 * probs_2[module]) / (P_top1 + P_top2)
```

### 论文关键总结

| 层级 | 门控类型 | 门控信号 | 作用 |
|------|---------|---------|------|
| L1: RF 融合 | 连续可微 | `max(P_rf)` → 动态权重 w | RF 高置信→主导；低置信→让 BRB 主导 |
| L2: 异常检测 | 阈值软门控 | `P_abn > 0.2` | 防止"正常"判定遗漏微弱故障 |
| L3: 子网路由 | 概率加权多路由 | `|P_top1 - P_top2| < 0.1` | 不确定时同时激活多个子网并融合 |

> ⚠️ **重要**: 没有任何层级使用硬 if/else 做二值切分。即使 L3 中的 `if use_top2` 也是"额外叠加"而非"排他选择"。

---

## Q4：模块层的输出形式

### 回答：返回候选模块的故障概率排序列表，包含概率值。不确定性通过返回结构中的元信息间接表达。

### 最终返回结构

**入口函数**: `infer_system_and_modules()` (`methods/ours_adapter.py`, L329-334)

```python
return {
    "system_probs": {
        "normal":     0.02,
        "amp_error":  0.85,
        "freq_error": 0.10,
        "ref_error":  0.03,
    },
    "fault_type_pred": "amp_error",
    "module_topk": [
        {"name": "[数字中频板][ADC] 数字检波与平均", "prob": 0.32},
        {"name": "[RF板][Mixer1]",                   "prob": 0.28},
        {"name": "[RF板][RF] 低频通路固定滤波/抑制网络", "prob": 0.19},
        {"name": "[电源板] 电源管理模块",              "prob": 0.08},
        ...  # Top-10
    ],
    "debug": {
        "rf_probs":    {"normal": 0.01, "amp_error": 0.92, ...},
        "brb_probs":   {"normal": 0.05, "amp_error": 0.70, ...},
        "fused_probs": {"normal": 0.02, "amp_error": 0.85, ...},
        "gating_status": "gated_ok",
        "soft_gating": {
            "used_hypotheses": [("amp_error", 0.85), ("freq_error", 0.10)],
            "single_hypothesis": True  # False 表示激活了多假设融合
        }
    }
}
```

### 模块级子图内部返回

**函数**: `hierarchical_module_infer()` 返回 `Dict[str, float]`:

```python
{
    "[数字中频板][ADC] 数字检波与平均": 0.32,
    "[RF板][Mixer1]":                  0.28,
    "[RF板][RF] 低频通路固定滤波/抑制网络": 0.19,
    "[电源板] 电源管理模块":             0.08,
    "[数字中频板][IF] 中频放大/衰减链":   0.07,
    "[数字中频板][DSP] 数字增益/偏置校准": 0.06,
}
# 总和 = 1.0（归一化保证）
```

### 不确定性（Uncertainty）表达

代码中**没有**显式输出一个标量"总不确定性"值。不确定性通过以下方式间接表达：

1. **概率分布的熵**: 若 top-1 概率仅 0.25（接近均匀），说明 BRB 对模块定位不确定
2. **`single_hypothesis` 标志**: 为 `False` 表示系统级诊断本身就不确定（激活了多假设）
3. **ERBRB 中的无知项 u**: `ERBRB.infer()` (`BRB/utils.py`, L111-113) 显式计算：
   ```python
   u_k = max(0.0, 1.0 - sum(beta_kj))  # 每条规则的无知项
   # 递推合成后: u_final = 最终无知项
   ```
   但当前模块层使用的是 `hierarchical_module_infer()`（非 ERBRB），所以 **实际生产路径中无显式无知项**。

### 论文写法建议

> ✅ "模块层输出各候选模块的后验概率分布 $\{P(M_j)\}_{j=1}^{N_m}$，其中 $\sum_j P(M_j) = 1$"  
> ✅ "当多假设融合被激活时，概率分布反映了系统级不确定性向模块级的传播"  
> ⚠️ 若论文需要显式 Uncertainty 指标，建议用 $H = -\sum P(M_j) \log P(M_j)$ 作为信息熵度量

---

## 附录：完整调用链路图

```
OursAdapter.predict(X)
  └─ infer_system_and_modules(features)               [ours_adapter.py L128]
       ├─ system_level_infer(features)                 [system_brb.py] → P_brb(系统级)
       ├─ GatingPriorFusion.fuse(P_rf, P_brb)         [gating_prior.py L119] → P_fused(系统级)
       │     └─ 动态权重 w = f(max(P_rf))
       ├─ fault_type_pred = argmax(P_fused)
       └─ hierarchical_module_infer_soft_gating(P_fused, features)  [module_brb.py L1014]
             ├─ 检查 |P_top1 - P_top2| < δ ?
             ├─ hierarchical_module_infer("amp_error", features)    [module_brb.py L766]
             │     ├─ 先验初始化: π = BASE_PRIORS × scales (P-CMA-ES)
             │     ├─ 特征评分: adj = decision_tree_scoring(features)
             │     └─ 输出: P(module) = normalize(π × adj^γ)
             ├─ [可选] hierarchical_module_infer("freq_error", features)
             └─ 加权融合: P_final(m) = Σ P(fault_k) × P(m|fault_k) / Σ P(fault_k)
```

---

## ER 算法实现对照

代码中有两套 BRB 合成器（`BRB/utils.py`）：

| 实现 | 类名 | 使用位置 | 特点 |
|------|------|---------|------|
| **SimpleBRB** | `SimpleBRB` | `module_level_infer()` (备用路径) | 加权平均，无冲突系数 |
| **ERBRB** | `ERBRB` | 可替换使用 | 完整 ER 递推，含无知项 u_k 和冲突系数 K |

**生产路径**（`hierarchical_module_infer`）使用的是**自定义的"先验×特征评分"公式**，而非直接调用 `SimpleBRB` 或 `ERBRB` 的 `infer()` 方法。

> 论文中描述 ER 合成时，应注明：系统级 BRB 使用 SimpleBRB/ERBRB 合成器；模块级 BRB 使用基于专家先验的特征门控评分机制（公式见 Q2 部分）。
