# 代码细节检索与公式提取报告 (Code-to-Paper Formula Extraction)

> **生成日期**: 2026-03-06
> **代码版本**: FMFD v4 (copilot/fix-compare-methods-parameters 分支)
> **核心文件**: `BRB/module_brb.py`, `pipelines/optimize_brb.py`, `BRB/utils.py`

---

## 1. 特征子集映射 (Feature Subset Mapping per Fault Type)

> **代码位置**: `BRB/module_brb.py` → `_aggregate_module_score()` (L276-324) 和 `hierarchical_module_infer()` (L851-1001)

当系统级 Sub-BRB1 输出定性结论后，模块级推理使用 **两个层面** 的特征分流：

### 1.1 聚合评分层 (`_aggregate_module_score`, L276-324)

此函数计算模块层的总异常分数，用于传统 BRB 规则的虚拟属性 $V$。

#### 幅度失准 (amp_error) — 12 个特征

| 序号 | 特征 ID | 物理含义 | 归一化范围 `[low, high]` | 代码行号 |
|------|---------|---------|------------------------|---------|
| 1 | `step_score` / `X7` | 步进评分（增益非线性） | [0.2, 1.5] | L260-266 |
| 2 | `ripple_var` / `X6` | 纹波方差 | [0.001, 0.02] | L268 |
| 3 | `gain_bias` | 增益偏差 $\|g-1\|$ | [0.02, 0.2] | L271-274 |
| 4 | `X11` | 包络超出率 (envelope overrun rate) | [0.01, 0.3] | L283 |
| 5 | `X12` | 最大违规幅度 (max violation) | [0.5, 5.0] | L284 |
| 6 | `X13` | 违规能量 (violation energy) | [0.1, 10.0] | L285 |
| 7 | `X19` | 低频段斜率 | [1e-12, 1e-10] | L286 |
| 8 | `X20` | 峰度 (kurtosis) | [0.5, 5.0] | L287 |
| 9 | `X21` | 峰值计数 (peak count) | [1, 20] | L288 |
| 10 | `X22` | 主频占比 (dominant freq ratio) | [0.1, 0.8] | L289 |
| 11 | `X36` | 周期性指数 (periodicity) | [0.1, 0.9] | L290 |
| 12 | `X37` | 线性拟合残差 | [0.001, 0.2] | L291 |

#### 频率失准 (freq_error) — 6 个特征

| 序号 | 特征 ID | 物理含义 | 归一化范围 `[low, high]` | 代码行号 |
|------|---------|---------|------------------------|---------|
| 1 | `df` / `X4` | 频率偏差 | [1e6, 5e7] | L269 |
| 2 | `X14` | 低频段残差能量 | [0.01, 1.0] | L299 |
| 3 | `X15` | 高频段残差能量 | [0.01, 0.5] | L300 |
| 4 | `X16` | 频移量 (freq shift) | [0.001, 0.1] | L301 |
| 5 | `X17` | 缩放因子 (scaling) | [0.001, 0.05] | L302 |
| 6 | `X18` | 偏移量 (bias) | [0.001, 0.05] | L303 |

#### 参考电平失准 (ref_error) — 6 个特征

| 序号 | 特征 ID | 物理含义 | 归一化范围 `[low, high]` | 代码行号 |
|------|---------|---------|------------------------|---------|
| 1 | `res_slope` / `X3` | 残差斜率 | [1e-12, 1e-10] | L267 |
| 2 | `gain_bias` / `X5` | 增益偏差 | [0.02, 0.2] | L271-274 |
| 3 | `X11` | 包络超出率 | [0.01, 0.3] | L312 |
| 4 | `X12` | 最大违规幅度 | [0.5, 5.0] | L313 |
| 5 | `X13` | 违规能量 | [0.1, 10.0] | L314 |
| 6 | `X35` | 差分方差 (diff variance) | [0.001, 0.02] | L315 |

### 1.2 判别特征层 (`hierarchical_module_infer`, L851-1001)

此函数在确定候选模块后，使用 **决策树型** 特征打分来区分同一子图内的不同模块。

#### 幅度子图判别特征 (L858-913)

| 特征 | 用于区分的模块 | 判别逻辑 |
|------|--------------|---------|
| `X7` (step_score) | 电源板 (`>0.12`) vs ADC (`>0.09`) vs 其他 | 阈值越高 → 越倾向电源板 |
| `X13` (violation energy) | 滤波器 (`>0.8` 或 `>0.3`) | 强指标：仅滤波器故障产生高违规能量 |
| `X35` (diff variance) | 电源板 (`>0.002`) vs ADC (`0.0005-0.002`) | 梯度区分 |
| `X36` (periodicity) | 电源板 (`<0.65`) vs 其他 | 低周期性 → 电源噪声 |
| `shape_rmse` | IF/DSP (`<0.001`) vs Mixer (`>0`) vs ADC (`>0.04`) | 全零特征 → IF/DSP |

#### 频率子图判别特征 (L921-965)

| 特征 | 用于区分的模块 | 判别逻辑 |
|------|--------------|---------|
| `X13` | Mixer1 (`>0.7`) vs 参考分配/LO1/OCXO (`~0.45`) | 高 X13 → Mixer 频率泄漏 |
| `X14` | Mixer1 (`<0.003`) vs 校准表 (`>0.14`) | 低频残差区分 |
| `X35` | 参考分配 (`>0.0002`) vs OCXO (`<0.00017`) | 差分方差微小区别 |
| `X36` | OCXO (`>0.89`) vs LO1 (`0.87-0.91`) | 高周期性 → OCXO |
| `X7` | OCXO (`<0.062`) vs 参考分配 (`>0.07`) | 步进分差异 |

#### 参考子图判别特征 (L971-1001)

| 特征 | 用于区分的模块 | 判别逻辑 |
|------|--------------|---------|
| `offset_slope` / `res_slope` | 校准源 (`>+0.005`) vs 开关 (`<-0.005`) | 斜率符号 |
| `band_offset_db_1` | 校准源 (`<-0.03`) vs 开关 (`>+0.03`) | 频段偏移方向 |
| `X14` | 校准表 (`>0.14`) vs 校准源 (`<0.12`) | 残差幅度 |

---

## 2. 数学公式提取

### 2.1 归一化函数 (Feature Normalization)

> **代码位置**: `BRB/utils.py` → `normalize_feature()` (L146-152)

$$
\bar{x} = \text{normalize}(x, l, h) = \begin{cases}
0 & \text{if } x \leq l \\
\frac{x - l}{h - l} & \text{if } l < x < h \\
1 & \text{if } x \geq h
\end{cases}
$$

其中 $l$ 和 $h$ 分别是专家设定的物理下界和上界。

---

### 2.2 加权融合公式 (Multi-Hypothesis Soft Gating Fusion)

> **代码位置**: `BRB/module_brb.py` → `hierarchical_module_infer_soft_gating()` (L1014-1120)

#### 步骤 1: 多假设激活条件 (L1070-1078)

设系统级 Sub-BRB1 输出前两名故障假设的概率为 $P(t_1)$ 和 $P(t_2)$，其中 $P(t_1) \geq P(t_2)$。

双假设激活条件为：

$$
\text{activate\_top2} = \begin{cases}
\text{True} & \text{if } P(t_1) - P(t_2) < \delta \\
\text{False} & \text{otherwise}
\end{cases}
$$

其中 $\delta = 0.1$ 为门控阈值（代码 L1017）。

#### 步骤 2: 子网模块概率计算 (L1084-1095)

对每个激活的故障假设 $t_k$，运行独立的模块级推理：

$$
\mathbf{p}^{(k)} = \text{hierarchical\_module\_infer}(t_k, \mathbf{x})
$$

其中 $p_j^{(k)}$ 表示在假设 $t_k$ 下模块 $M_j$ 的故障概率。

#### 步骤 3: 加权融合 (L1097-1104)

当双假设激活时，对所有候选模块 $M_j \in \mathcal{M}_{t_1} \cup \mathcal{M}_{t_2}$：

$$
\boxed{P_{\text{fused}}(M_j) = \frac{P(t_1) \cdot p_j^{(1)} + P(t_2) \cdot p_j^{(2)}}{P(t_1) + P(t_2)}}
$$

当仅单假设激活时：$P_{\text{fused}}(M_j) = p_j^{(1)}$。

> **论文描述**: 该公式实现了概率加权的多假设融合（Probability-Weighted Multi-Hypothesis Fusion）。当两个系统级故障假设置信度接近（差距 $< \delta$）时，不做硬判决而是同时运行两个模块子网，按系统级概率加权平均其输出，从而在系统级不确定时保留更多诊断信息。

---

### 2.3 条件概率打分公式 (Module Conditional Probability Scoring)

> **代码位置**: `BRB/module_brb.py` → `hierarchical_module_infer()` (L790-1011)

模块级定位分三步完成：

#### 步骤 1: 先验缩放 (Prior Scaling, L828-842)

每个故障类型 $t \in \{amp, freq, ref\}$ 拥有 $K_t$ 个候选模块，每个模块 $M_j$ 有专家先验 $\pi_j^{(0)}$ 和 P-CMA-ES 优化的缩放因子 $\alpha_j$：

$$
\pi_j = \frac{\pi_j^{(0)} \cdot \max(\alpha_j, 0.01)}{\sum_{i=1}^{K_t} \pi_i^{(0)} \cdot \max(\alpha_i, 0.01)}
$$

其中：
- 幅度先验 $\boldsymbol{\pi}^{(0)}_{\text{amp}} = [0.30, 0.30, 0.21, 0.10, 0.05, 0.04]$（L828, 对应 ADC, Mixer1, 滤波器, 电源, IF, DSP）
- 频率先验 $\boldsymbol{\pi}^{(0)}_{\text{freq}} = [0.37, 0.33, 0.17, 0.13]$（L829, 对应 参考分配, Mixer1, LO1, OCXO）
- 参考先验 $\boldsymbol{\pi}^{(0)}_{\text{ref}} = [0.38, 0.32, 0.30]$（L830, 对应 校准源, 校准表, 校准开关）
- 缩放因子 $\boldsymbol{\alpha} = [\alpha_1, ..., \alpha_{13}]$，其中 $\alpha_j \in [0.1, 5.0]$（由 P-CMA-ES 投影保证）

#### 步骤 2: 特征调整因子 (Feature-Based Adjustment, L851-1001)

对每个候选模块，基于判别特征计算调整分数 $a_j$。以幅度子图中的 **电源板模块** 为例（L867-875）：

$$
a_{\text{power}} = 1 + 4 \cdot \mathbb{1}[X_{36} < 0.65] + 3 \cdot \mathbb{1}[X_{35} > 0.002] + 1 \cdot \mathbb{1}[X_7 > 0.12]
$$

其中 $\mathbb{1}[\cdot]$ 为指示函数。$a_j \in [1, 9]$ 范围（取决于命中的条件数量）。

#### 步骤 3: 最终概率 (Final Probability, L912-913 + L1003-1006)

应用特征灵敏度指数 $\gamma_t$（P-CMA-ES 优化），计算未归一化概率，再归一化：

$$
\boxed{P(M_j \mid t, \mathbf{x}) = \frac{\pi_j \cdot a_j^{\gamma_t}}{\sum_{i=1}^{K_t} \pi_i \cdot a_i^{\gamma_t}}}
$$

其中：
- $\pi_j$ 为缩放后先验（步骤 1）
- $a_j$ 为特征调整因子（步骤 2）
- $\gamma_t$ 为故障类型 $t$ 对应的特征灵敏度：$\gamma_{\text{amp}} = h_{13}$，$\gamma_{\text{freq}} = h_{14}$，$\gamma_{\text{ref}} = h_{15}$
- $\gamma_t \in [0.1, 5.0]$（由 P-CMA-ES 投影保证）

> **论文描述**: 此公式实现了"先验-证据"融合范式。专家先验 $\pi_j$ 编码领域知识（哪些模块更可能故障），特征调整因子 $a_j$ 编码实测证据（观测到的物理特征模式），灵敏度指数 $\gamma_t$ 控制证据对先验的修正强度——$\gamma > 1$ 时放大特征区分力，$\gamma < 1$ 时抑制噪声特征的影响。

---

### 2.4 P-CMA-ES 适应度函数 (Fitness Function)

> **代码位置**: `pipelines/optimize_brb.py` → `supervised_objective()` (L209-242) 和 `project_to_feasible()` (L171-184)

#### 投影算子 (Projection Operator, L171-184)

P-CMA-ES 的核心是在每次目标函数求值前，将无约束的 CMA-ES 搜索向量投影到合法参数空间：

$$
\text{proj}(\boldsymbol{\theta}) = \begin{cases}
\theta_j \leftarrow \text{clip}(\theta_j, 0.1, 5.0) & j \in \{0, ..., 12\} \quad \text{(缩放因子)} \\
\theta_j \leftarrow \text{clip}(\theta_j, 0.1, 5.0) & j \in \{13, 14, 15\} \quad \text{(灵敏度)}
\end{cases}
$$

其中 $\text{clip}(x, l, h) = \max(l, \min(h, x))$。

单纯形约束（$\sum \pi_i = 1$）通过推理层的归一化自动保证（L832-834 的 `_scale_and_normalize`），无需显式投影。

#### 监督模式适应度函数 (Supervised, L209-242)

$$
\boxed{f(\boldsymbol{\theta}) = \underbrace{\left(1 - \text{BalAcc}(\boldsymbol{\theta})\right)}_{\text{主损失}} + \underbrace{0.05 \sum_{j=0}^{12} (\theta_j - 1)^2}_{\text{缩放因子正则化}} + \underbrace{0.001 \sum_{j=13}^{15} (\theta_j - 1)^2}_{\text{灵敏度正则化}}}
$$

其中：

**平衡准确率 (Balanced Accuracy)**：

$$
\text{BalAcc} = \frac{1}{C} \sum_{c=1}^{C} \frac{|\{i : \hat{y}_i = y_i = c\}|}{|\{i : y_i = c\}|}
$$

即各类别 Recall 的宏平均（L231-237），$C$ 为模块类别数。使用平衡准确率而非普通准确率，是为了防止多数类开发（majority class exploitation）。

**L2 正则化**：
- 缩放因子项（权重 $\lambda_s = 0.05$）：惩罚先验缩放因子偏离专家初始值 1.0（L240）
- 灵敏度项（权重 $\lambda_f = 0.001$）：弱惩罚灵敏度偏离中性值 1.0（L241）

> **论文描述**: 正则化项体现了"物理可解释性保护"——强正则 ($\lambda_s = 0.05$) 约束优化后的先验分布不偏离专家知识太远，弱正则 ($\lambda_f = 0.001$) 允许灵敏度自由调整以适应数据。

#### 无监督模式适应度函数 (Unsupervised, L245-260)

$$
f_{\text{unsup}}(\boldsymbol{\theta}) = w_H \cdot \overline{H}(\boldsymbol{\theta}) + w_c \cdot (1 - \overline{c}(\boldsymbol{\theta}))
$$

其中：
- $\overline{H} = \frac{1}{N} \sum_{i=1}^{N} H(\mathbf{p}_i)$，$H(\mathbf{p}) = -\sum_j p_j \log p_j$ 为熵
- $\overline{c} = \frac{1}{N} \sum_{i=1}^{N} \max_j p_{ij}$ 为平均最高置信度
- 默认权重：$w_H = 0.6$，$w_c = 0.4$

> **论文描述**: 无监督模式通过最小化输出熵（促进决定性输出）和最大化 Top-1 置信度（促进清晰排序）来优化 BRB 参数，无需模块级标签。

---

## 3. 候选模块清单 (Module Candidate Lists)

> **代码位置**: `BRB/module_brb.py` L100-118

### 幅度子图 ($K_{\text{amp}} = 6$)

| 索引 $j$ | 模块 V2 名称 | 缩放参数 | 专家先验 $\pi_j^{(0)}$ |
|----------|------------|---------|---------------------|
| 0 | [数字中频板][ADC] 数字检波与平均 | $\alpha_0$ | 0.30 |
| 1 | [RF板][Mixer1] | $\alpha_1$ | 0.30 |
| 2 | [RF板][RF] 低频通路固定滤波/抑制网络 | $\alpha_2$ | 0.21 |
| 3 | [电源板] 电源管理模块 | $\alpha_3$ | 0.10 |
| 4 | [数字中频板][IF] 中频放大/衰减链 | $\alpha_4$ | 0.05 |
| 5 | [数字中频板][DSP] 数字增益/偏置校准 | $\alpha_5$ | 0.04 |

### 频率子图 ($K_{\text{freq}} = 4$)

| 索引 $j$ | 模块 V2 名称 | 缩放参数 | 专家先验 $\pi_j^{(0)}$ |
|----------|------------|---------|---------------------|
| 0 | [时钟板][参考分配] | $\alpha_6$ | 0.37 |
| 1 | [RF板][Mixer1] | $\alpha_7$ | 0.33 |
| 2 | [LO/时钟板][LO1] 合成链 | $\alpha_8$ | 0.17 |
| 3 | [时钟板][参考域] 10MHz 基准 OCXO | $\alpha_9$ | 0.13 |

### 参考子图 ($K_{\text{ref}} = 3$)

| 索引 $j$ | 模块 V2 名称 | 缩放参数 | 专家先验 $\pi_j^{(0)}$ |
|----------|------------|---------|---------------------|
| 0 | [校准链路][校准源] | $\alpha_{10}$ | 0.38 |
| 1 | [校准链路][校准表/存储] | $\alpha_{11}$ | 0.32 |
| 2 | [校准链路][校准路径开关/耦合] | $\alpha_{12}$ | 0.30 |

---

## 4. P-CMA-ES 参数空间汇总

| 参数组 | 索引 | 维度 | 约束 | 含义 |
|-------|------|------|------|------|
| 幅度先验缩放 | $\theta_{0:6}$ | 6 | $[0.1, 5.0]$ | 幅度子图 6 个模块的先验权重缩放 |
| 频率先验缩放 | $\theta_{6:10}$ | 4 | $[0.1, 5.0]$ | 频率子图 4 个模块的先验权重缩放 |
| 参考先验缩放 | $\theta_{10:13}$ | 3 | $[0.1, 5.0]$ | 参考子图 3 个模块的先验权重缩放 |
| 特征灵敏度 | $\theta_{13:16}$ | 3 | $[0.1, 5.0]$ | 3 个子图各自的灵敏度指数 $\gamma_t$ |
| **合计** | | **16** | | |

---

## 5. 核心函数调用链

```
OursAdapter.predict(X)
  └─ infer_system_and_modules(features, rf_classifier)
       ├─ [系统层] system_level_infer(features) → sys_probs
       ├─ [RF门控] rf_classifier.predict_proba(X_kd) → rf_probs
       ├─ [概率融合] gated_fuse(rf_probs, brb_probs) → final_probs
       └─ [模块层] hierarchical_module_infer_soft_gating(final_probs, features)
            ├─ 检查 P(t1) - P(t2) < δ → 激活多假设？
            ├─ hierarchical_module_infer(t1, features)
            │    ├─ _scale_and_normalize(π⁰, α, modules) → π
            │    ├─ 判别特征打分 → adj[module]
            │    └─ π_j * adj_j^γ → normalize → P(M_j|t,x)
            ├─ [可选] hierarchical_module_infer(t2, features)
            └─ 加权融合 → P_fused(M_j)
```

---

## 6. 关键常量速查

| 常量 | 值 | 代码位置 | 论文用途 |
|------|---|---------|---------|
| 多假设门控阈值 $\delta$ | 0.1 | `module_brb.py:1017` | 软门控路由灵敏度 |
| RF 先验权重范围 $w$ | [0.82, 0.95] | `gating_prior.json` | 门控区间 |
| 缩放因子正则系数 $\lambda_s$ | 0.05 | `optimize_brb.py:240` | 防止先验漂移 |
| 灵敏度正则系数 $\lambda_f$ | 0.001 | `optimize_brb.py:241` | 允许灵敏度自适应 |
| 投影箱约束 | [0.1, 5.0] | `optimize_brb.py:175-176` | P-CMA-ES 可行域 |
| 最小模块分数 | 0.01 | `module_brb.py:324` | 避免 BRB 全零输出 |
| CMA-ES 初始步长 $\sigma_0$ | 0.5 | `optimize_brb.py:265` | 搜索初始范围 |
