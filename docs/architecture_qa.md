# 频谱分析仪特征解耦与双驱动分层 BRB 故障诊断：五问核实

> 本文档基于代码库实际逻辑编写，所有数据均标注对应源文件与行号。

---

## 问题 1：关于特征维度与解耦分类的核实

### 1.1 特征总数

当前模型共提取 **58 个特征**，分为三个来源：

| 来源函数 | 文件:行号 | 特征数量 |
|----------|-----------|----------|
| `extract_system_features()` | `features/feature_extraction.py:294-810` | X1–X37 (37 个) + 3 个别名 (offset_db, viol_rate_aligned, viol_energy_aligned) + 13 个鲁棒残差特征 = **53 个** |
| `compute_dynamic_threshold_features()` | `features/feature_extraction.py:813-892` | 5 个动态阈值特征 (env_overrun_rate/max/mean, switch_step_mean_abs/std) + 重复的 3 个别名和 13 个 robust_feats（合并时去重）= 新增 **5 个** |
| **合并去重后** (`**sys_feats, **dyn_feats`) | `run_simulation_brb.py:1910,1920-1926` | 53 + 5 = **58 个独立特征** |

> 写入 `features_brb.csv` 时仅包含 `sample_id` + 58 个特征列，无标签列。
> `compare_methods.py` 加载后特征矩阵形状为 `(400, 58)`（`compare_methods.py:572`）。

### 1.2 全部 58 个特征列表

#### A. 核心物理特征 X1–X37（37 个）

| 编号 | 变量名 | 物理含义 | 分类 |
|------|--------|---------|------|
| X1 | `bias` / `amplitude_offset` | 频响曲线整体幅度偏移（相对 RRS 基线） | **局部指纹** |
| X2 | `ripple_var` / `inband_flatness` | 带内平坦度方差（纹波幅度变异性） | **局部指纹** |
| X3 | `res_slope` / `hf_attenuation_slope` | 高频衰减斜率（参考电平漂移指标） | **局部指纹** |
| X4 | `df` / `freq_scale_nonlinearity` | 频率刻度非线性（频率校准误差） | **局部指纹** |
| X5 | `amp_scale_consistency` / `gain_consistency` | 幅度刻度一致性（跨频段增益一致性） | **局部指纹** |
| X6 | `ripple_variance` | 纹波方差（模块级，类似 X2） | **全局宏观** |
| X7 | `step_score` / `gain_nonlinearity` | 增益非线性（阶跃响应幅度跳变） | **局部指纹** |
| X8 | `lo_leakage` | 本振信号泄漏到 RF 输出 | **局部指纹** |
| X9 | `tuning_linearity_residual` | 调谐线性度残差 | **全局宏观** |
| X10 | `band_amplitude_consistency` | 频段幅度一致性 | **全局宏观** |
| X11 | `env_overrun_rate` / `viol_rate` | 包络违规率（超出限值的点比例） | **全局宏观** |
| X12 | `env_overrun_max` | 最大包络违规幅度 | **全局宏观** |
| X13 | `env_violation_energy` | 包络违规能量（超限的积分总量） | **全局宏观** |
| X14 | `band_residual_low` | 低频段残差能量 | **全局宏观** |
| X15 | `band_residual_high_std` | 高频段残差标准差 | **全局宏观** |
| X16 | `corr_shift_bins` | 互相关峰值滞后（频率偏移，以 bin 为单位） | **局部指纹** |
| X17 | `warp_scale` | 频率轴缩放因子（频率压缩/扩展） | **局部指纹** |
| X18 | `warp_bias` | 频率轴平移因子 | **局部指纹** |
| X19 | `slope_low` | 低频段斜率 | **局部指纹** |
| X20 | `kurtosis_detrended` | 去趋势峰度（纹波/阻抗失配指标） | **局部指纹** |
| X21 | `peak_count_residual` | 残差中的峰值计数（纹波密度） | **局部指纹** |
| X22 | `ripple_dom_freq_energy` | 主频能量占比 | **局部指纹** |
| X23 | `warp_residual_energy` | 频率校准残差能量 | **局部指纹** |
| X24 | `phase_slope_diff` | 相位导数斜率差异 | **局部指纹** |
| X25 | `interp_mse_after_shift` | 最优滞后偏移重采样后的 MSE | **局部指纹** |
| X26 | `high_quantile_compress_score` | 上尾斜率变化（参考电平线性度） | **局部指纹** |
| X27 | `piecewise_gain_change` | 分段线性斜率差异（跨频段增益一致性） | **局部指纹** |
| X28 | `residual_upper_tail_asym` | 上尾不对称性（噪声底不对称指标） | **局部指纹** |
| X29 | `hf_lf_energy_ratio` | 高低频能量比（基于 PSD） | **全局宏观** |
| X30 | `high_level_compression_idx` | 高电平压缩指数（幅度非线性） | **全局宏观** |
| X31 | `piecewise_offset_consistency` | 分段偏移一致性（分段方差） | **全局宏观** |
| X32 | `hf_lf_psd_energy_ratio` | 功率谱密度高低频比 | **全局宏观** |
| X33 | `spectrum_smoothness` | 频谱平滑度（频响变化率） | **全局宏观** |
| X34 | `local_spectrum_variance` | 局部频谱特征方差（分段统计量） | **全局宏观** |
| X35 | `diff_variance` | 差分方差（二阶曲率指标） | **局部指纹** |
| X36 | `periodicity` | 周期性指数（谐波含量指标） | **局部指纹** |
| X37 | `linear_fit_residual` | 线性拟合残差（非线性失真度） | **局部指纹** |

#### B. 鲁棒残差特征（13 个）

> 来源：`compute_residual_robust_features()`（`feature_extraction.py:199-287`）
> 被 `extract_system_features()` 和 `compute_dynamic_threshold_features()` 同时调用，dict 合并时去重。

| 编号 | 变量名 | 物理含义 | 分类 |
|------|--------|---------|------|
| 38 | `global_offset_db` | 残差中值（全局偏移量，dB） | **全局宏观** |
| 39 | `shape_rmse` | 去中值后的形状 RMSE | **全局宏观** |
| 40 | `ripple_hp` | 高通纹波（差分标准差） | **全局宏观** |
| 41 | `tail_asym` | 尾部不对称度 (p95−p50)−(p50−p5) | **全局宏观** |
| 42 | `compress_ratio` | 高幅值压缩比（>p80 占比） | **全局宏观** |
| 43 | `compress_ratio_high` | 高频段压缩比 | **全局宏观** |
| 44 | `freq_shift_score` | 频率偏移综合评分 (√(lag²+(1−corr)²)) | **局部指纹** |
| 45 | `offset_slope` | 残差随频率的斜率（×10⁹） | **局部指纹** |
| 46 | `high_low_energy_ratio` | 高低频段能量比 | **全局宏观** |
| 47 | `band_offset_db_1` | 频段 1 偏移量 (dB) | **局部指纹** |
| 48 | `band_offset_db_2` | 频段 2 偏移量 (dB) | **局部指纹** |
| 49 | `band_offset_db_3` | 频段 3 偏移量 (dB) | **局部指纹** |
| 50 | `band_offset_db_4` | 频段 4 偏移量 (dB) | **局部指纹** |

#### C. 别名/冗余特征（3 个，与 X11/X13/global_offset_db 重复）

| 编号 | 变量名 | 等价于 | 分类 |
|------|--------|--------|------|
| 51 | `offset_db` | `global_offset_db`（取中值实现略有差异） | **全局宏观** |
| 52 | `viol_rate_aligned` | `X11`（对齐后的包络违规率） | **全局宏观** |
| 53 | `viol_energy_aligned` | `X13`（对齐后的违规能量） | **全局宏观** |

#### D. 动态阈值 / 频段切换特征（5 个）

> 来源：`compute_dynamic_threshold_features()`（`feature_extraction.py:813-892`）

| 编号 | 变量名 | 物理含义 | 分类 |
|------|--------|---------|------|
| 54 | `env_overrun_rate` | 包络超出率（考虑上下限） | **全局宏观** |
| 55 | `env_overrun_max` | 最大包络超出幅度 | **全局宏观** |
| 56 | `env_overrun_mean` | 平均包络超出量 | **全局宏观** |
| 57 | `switch_step_mean_abs` | 频段切换跳变均值（绝对值） | **局部指纹** |
| 58 | `switch_step_std` | 频段切换跳变标准差 | **全局宏观** |

### 1.3 两类特征汇总

| 类别 | 数量 | 特征编号 |
|------|------|---------|
| **基于物理机理的局部指纹特征** | **27 个** | X1–X5, X7, X8, X16–X28, X35–X37, freq_shift_score, offset_slope, band_offset_db_1~4, switch_step_mean_abs |
| **基于数学统计的全局宏观特征** | **31 个** | X6, X9–X15, X29–X34, global_offset_db, shape_rmse, ripple_hp, tail_asym, compress_ratio, compress_ratio_high, high_low_energy_ratio, offset_db, viol_rate_aligned, viol_energy_aligned, env_overrun_rate, env_overrun_max, env_overrun_mean, switch_step_std |

**划分标准**：
- **局部指纹特征**：直接反映特定物理链路或器件的退化特征，可定位到具体模块（如频率偏移 → 时钟链路，频段偏移 → 滤波器链路）
- **全局宏观特征**：基于统计聚合（方差、分位数、能量比等），反映系统整体状态但不能直接定位模块

---

## 问题 2：关于诊断模块划分、实际仿真情况与数据集的核实

### 2.1 代码中定义的全部底层诊断模块

#### A. V1 模块列表（20 个，`module_brb.py:32-53`）

| 序号 | V1 模块名 | 对应 V2 名称 | 禁用状态 |
|------|----------|-------------|---------|
| 1 | 衰减器 | [RF板][RF] 输入衰减器组 | ⛔ AC耦合模式禁用 |
| 2 | 前置放大器 | 前置放大器 | ⛔ 单频段模式禁用 |
| 3 | 低频段前置低通滤波器 | [RF板][RF] 低频通路固定滤波/抑制网络 | ✅ 启用 |
| 4 | 低频段第一混频器 | [RF板][Mixer1] | ✅ 启用 |
| 5 | 高频段YTF滤波器 | [RF板][YTF] | ⛔ 单频段模式禁用 |
| 6 | 高频段混频器 | [RF板][Mixer2] | ⛔ 单频段模式禁用 |
| 7 | 时钟振荡器 | [时钟板][参考域] 10MHz 基准 OCXO | ✅ 启用 |
| 8 | 时钟合成与同步网络 | [时钟板][参考分配] | ✅ 启用 |
| 9 | 本振源（谐波发生器） | [LO/时钟板][LO1] 合成链 | ✅ 启用 |
| 10 | 本振混频组件 | [RF板][Mixer1] | ✅ 启用 |
| 11 | 校准源 | [校准链路][校准源] | ✅ 启用 |
| 12 | 存储器 | [校准链路][校准表/存储] | ✅ 启用 |
| 13 | 校准信号开关 | [校准链路][校准路径开关/耦合] | ✅ 启用 |
| 14 | 中频放大器 | [数字中频板][IF] 中频放大/衰减链 | ✅ 启用 |
| 15 | ADC | [数字中频板][ADC] 数字检波与平均 | ✅ 启用 |
| 16 | 数字RBW | [数字中频板][IF] RBW数字滤波器 | ✅ 启用 |
| 17 | 数字放大器 | [数字中频板][DSP] 数字增益/偏置校准 | ✅ 启用 |
| 18 | 数字检波器 | [数字中频板][ADC] 数字检波与平均 | ✅ 启用 |
| 19 | VBW滤波器 | [数字中频板][VBW] | ✅ 启用 |
| 20 | 电源模块 | [电源板] 电源管理模块 | ✅ 启用 |

> **禁用配置**（`module_brb.py:64-75`）：当前默认 `SINGLE_BAND=True` + `DISABLE_PREAMP=True` + `AC_COUPLED=True`，共禁用 4 个模块。

#### B. V2 合并后的模块（去重后 18 个独立名称）

由于 V1→V2 存在多对一映射（如 "低频段第一混频器" 和 "本振混频组件" 均映射到 "[RF板][Mixer1]"；"ADC" 和 "数字检波器" 均映射到 "[数字中频板][ADC] 数字检波与平均"），去重后为 **18 个**独立 V2 模块（`label_mapping.py:190-218`，`module_brb.py:55-61`）。

去重后完整 V2 列表：
1. `[RF板][RF] 输入衰减器组` ⛔
2. `前置放大器` ⛔
3. `[RF板][RF] 低频通路固定滤波/抑制网络`
4. `[RF板][Mixer1]`
5. `[RF板][YTF]` ⛔
6. `[RF板][Mixer2]` ⛔
7. `[时钟板][参考域] 10MHz 基准 OCXO`
8. `[时钟板][参考分配]`
9. `[LO/时钟板][LO1] 合成链`
10. `[校准链路][校准源]`
11. `[校准链路][校准表/存储]`
12. `[校准链路][校准路径开关/耦合]`
13. `[数字中频板][IF] 中频放大/衰减链`
14. `[数字中频板][ADC] 数字检波与平均`
15. `[数字中频板][IF] RBW数字滤波器`
16. `[数字中频板][DSP] 数字增益/偏置校准`
17. `[数字中频板][VBW]`
18. `[电源板] 电源管理模块`

### 2.2 实际参与仿真的模块

#### 代码中理论划定但 **全部实际仿真** 的模块

仿真代码（`run_simulation_brb.py:1161-1404`）中，根据 `fault_kind` 和 `target_class` 分配模块。以下列出了每种系统级故障对应的实际仿真模块：

**幅度失准（`amp_error`）— 100 条样本，6 种故障种类**

| 故障种类 (fault_kind) | 抽样概率 | 分配的 V1 模块 | V2 模块 |
|----------------------|----------|---------------|---------|
| `amp` | 25% | 随机选择(注1) | 随机 |
| `lpf` | 18% | 低频段前置低通滤波器 | [RF板][RF] 低频通路固定滤波/抑制网络 |
| `mixer` | 18% | 低频段第一混频器 | [RF板][Mixer1] |
| `adc` | 15% | ADC | [数字中频板][ADC] 数字检波与平均 |
| `vbw` | 12% | 数字检波器 | [数字中频板][ADC] 数字检波与平均 |
| `power` | 12% | 电源模块 | [电源板] 电源管理模块 |

> 注1：`amp` 种类通过 `_choose_module_for_system("amp_error")` 从 MODULE_LIBRARY 中随机选取（`module_library.py:40-45`），可选模块为：低频段前置低通滤波器、低频段第一混频器、本振混频组件、中频放大器、数字放大器、电源模块

**频率失准（`freq_error`）— 100 条样本，3 种故障种类**

| 故障种类 (fault_kind) | 抽样概率 | 分配的 V1 模块 | V2 模块 |
|----------------------|----------|---------------|---------|
| `freq` | 40% | 随机选择(注2) | 随机 |
| `clock` | 30% | 时钟合成与同步网络 | [时钟板][参考分配] |
| `lo` | 30% | 本振混频组件 | [RF板][Mixer1] |

> 注2：`freq` 种类通过 `_choose_module_for_system("freq_error")` 随机选取，可选模块为：时钟振荡器、时钟合成与同步网络、本振源（谐波发生器）

**参考电平失准（`ref_error`）— 100 条样本，2 种故障种类**

| 故障种类 (fault_kind) | 抽样概率 | 分配的 V1 模块 | V2 模块 |
|----------------------|----------|---------------|---------|
| `rl` | 60% | 随机选择(注3) | 随机 |
| `att` | 40% | 随机选择(注3) | 随机 |

> 注3：`rl` 和 `att` 均通过 `_choose_ref_module()` 随机选取（`run_simulation_brb.py:303-308`），可选模块为：校准源、存储器、校准信号开关（衰减器因 AC 耦合被禁用）

**正常（`normal`）— 100 条样本**：模块标签为 `"none"`，不分配模块。

#### 结论：实际参与仿真的 V1 模块（16 个启用中的 14 个）

| V1 模块 | 是否实际仿真 | 仿真路径 |
|---------|-------------|---------|
| 低频段前置低通滤波器 | ✅ | `lpf` + `amp` 随机 |
| 低频段第一混频器 | ✅ | `mixer` + `amp` 随机 |
| 本振混频组件 | ✅ | `lo` + `amp` 随机 |
| 中频放大器 | ✅ | `amp` 随机 |
| 数字放大器 | ✅ | `amp` 随机 |
| 电源模块 | ✅ | `power` + `amp` 随机 |
| 时钟振荡器 | ✅ | `freq` 随机 |
| 时钟合成与同步网络 | ✅ | `clock` + `freq` 随机 |
| 本振源（谐波发生器） | ✅ | `freq` 随机 |
| 校准源 | ✅ | `rl`/`att` 随机 |
| 存储器 | ✅ | `rl`/`att` 随机 |
| 校准信号开关 | ✅ | `rl`/`att` 随机 |
| ADC | ✅ | `adc` |
| 数字检波器 | ✅ | `vbw` |
| 数字RBW | ❌ 未仿真 | MODULE_LIBRARY 中标记为 ref_error，但仿真的 `rl`/`att` 路径仅从 `_choose_ref_module()` 选取 |
| VBW滤波器 | ❌ 未仿真 | MODULE_LIBRARY 中标记为 ref_error，但仿真的 `rl`/`att` 路径仅从 `_choose_ref_module()` 选取 |

> **关键发现**：20 个 V1 模块中，4 个被配置禁用，16 个启用；其中 14 个实际参与仿真，**2 个**（数字RBW、VBW滤波器）虽然在 MODULE_LIBRARY 中标记为 `ref_error` 但未被任何 `fault_kind` 路径分配到（仿真的 `rl`/`att` 路径仅通过 `_choose_ref_module()` 从 `[校准源, 存储器, 校准信号开关]` 中选取）。

### 2.3 数据集划分

**划分比例**（`default_paths.py:30`）：

```python
SPLIT = (0.6, 0.2, 0.2)  # 训练集:验证集:测试集
```

**划分逻辑**（`compare_methods.py:685-718`）：分层采样（Stratified），按系统级标签保持类别比例。

| 集合 | 比例 | 样本数（400 总量） |
|------|------|-------------------|
| 训练集 | 60% | 240 条（每类 60） |
| 验证集 | 20% | 80 条（每类 20） |
| 测试集 | 20% | 80 条（每类 20） |

### 2.4 模块级故障数据分布

- **故障样本总数**：300 条（正常 100 条不含模块标签）
- **模块级标签分布**：由于 `_choose_module_for_system()` 和 `_choose_ref_module()` 是**随机等概率选取**，具体分布因随机种子而异。以下是**基于概率的期望分布**（仅供参考，实际值因随机种子而有偏差）：

| 系统类别 | 样本数 | 分配方式说明 |
|---------|--------|-------------|
| 幅度失准 | 100 | `amp`(25%) 从 6 个 V1 模块中均匀随机选取 + `lpf`(18%)→低频段前置低通滤波器 + `mixer`(18%)→低频段第一混频器 + `adc`(15%)→ADC + `vbw`(12%)→数字检波器 + `power`(12%)→电源模块 |
| 频率失准 | 100 | `freq`(40%) 从 3 个 V1 模块中均匀随机选取 + `clock`(30%)→时钟合成与同步网络 + `lo`(30%)→本振混频组件 |
| 参考电平失准 | 100 | `rl`(60%) + `att`(40%) 均从 `[校准源, 存储器, 校准信号开关]` 中均匀随机选取 |
| 正常 | 100 | 无模块标签 |

---

## 问题 3：关于系统级 RF 与 BRB 双驱动融合机制

### 3.1 融合方式

RF 与 BRB 的融合权重是 **自适应动态调整** 的，采用 **置信度门控融合**（gated fusion）策略。

> 配置文件：`config/gating_prior.json`  
> 实现代码：`BRB/gating_prior.py:119-155`  
> 调用入口：`methods/ours_adapter.py`

### 3.2 融合机制详解

```
输入特征
    ├─ RF分类器.predict_proba()  → P_RF = [p_正常, p_幅度, p_频率, p_参考]
    │
    └─ 系统级BRB.infer()        → P_BRB = [p_正常, p_幅度, p_频率, p_参考]
    
    置信度门控融合 (gating_prior.py:119-155)
    ├─ Step 1: 计算 RF 置信度 c = max(P_RF)
    ├─ Step 2: 动态权重计算
    │     if c ≥ c₀:
    │         w = w_min + (c - c₀) / (1 - c₀) × (w_max - w_min)
    │         w = clip(w, w_min, w_max)
    │     else:
    │         w = w_min    （RF 不确定时退回到最低权重）
    ├─ Step 3: Logit 空间融合
    │     logit_fused = w × logit(P_RF) + (1-w) × logit(P_BRB)
    └─ Step 4: P_final = softmax(logit_fused)
```

### 3.3 配置参数

| 参数 | 值 | 含义 | 来源 |
|------|-----|------|------|
| `method` | `"gated"` | 使用门控融合方法 | `gating_prior.json:5` |
| `threshold` (c₀) | 0.55 | RF 置信度激活阈值 | `gating_prior.json:9` |
| `w_min` | 0.82 | RF 最低权重（不确定时 BRB 占 18%） | `gating_prior.json:10` |
| `w_max` | 0.95 | RF 最高权重（高置信时 BRB 仅占 5%） | `gating_prior.json:11` |
| `temperature` | 1.2 | RF 概率温度校准 | `gating_prior.json:12` |

### 3.4 自适应性质

- 当 RF **高置信**（`max(P_RF)` 接近 1.0）时：`w → 0.95`，RF 主导决策
- 当 RF **中等置信**（0.55 < `max(P_RF)` < 0.8）时：`w` 在 0.82–0.95 之间线性插值
- 当 RF **低置信**（`max(P_RF)` < 0.55）时：`w = 0.82`，BRB 知识库提供更大贡献
- **回退机制**（`gating_prior.json:14-17`）：当 RF 分类器不可用时，退回到纯 BRB 推理

### 3.5 核心公式

$$w(c) = \begin{cases} w_{\min} & \text{if } c < c_0 \\ w_{\min} + \frac{c - c_0}{1 - c_0} \cdot (w_{\max} - w_{\min}) & \text{if } c \geq c_0 \end{cases}$$

$$\text{logit}(p) = \ln\frac{p}{1-p}, \quad P_{\text{final}} = \text{softmax}\big(w \cdot \text{logit}(P_{\text{RF}}) + (1-w) \cdot \text{logit}(P_{\text{BRB}})\big)$$

---

## 问题 4：关于模块级分层特征注入机制的核实

### 4.1 代码现状

代码中 **没有** 严格执行"现象级特征 → 细节特征 → 决策特征"的逐级注入分层架构。

实际采用的是 **"基先验 + 特征决策树一次性平面调整"** 的方式。

### 4.2 实际注入逻辑

`hierarchical_module_infer()`（`module_brb.py:800-1011`）的执行流程：

```
输入: fault_type（系统级诊断结果）, features（全部58个特征）

Step 1: 加载基先验 (line 828-830)
    amp_error:  [0.30, 0.30, 0.21, 0.10, 0.05, 0.04] (6个模块)
    freq_error: [0.37, 0.33, 0.17, 0.13]               (4个模块)
    ref_error:  [0.38, 0.32, 0.30]                      (3个模块)

Step 2: CMA-ES 优化的缩放因子 (line 832-835)
    scaled_prior = base_prior × scale_factor[i]
    归一化: Σ scaled_prior = 1

Step 3: 特征决策树调整 (line 851-1001)
    ★ 所有特征一次性输入，无分层 ★
    
    if amp_error:
        读取 X7, X13, X35, X36, shape_rmse
        对每个候选模块计算调整分数 adj[module]
        例如: Power模块 → if X36<0.65: score+=4; if X35>0.002: score+=3
        
    if freq_error:
        读取 X13, X14, X7, X36, X35
        
    if ref_error:
        读取 offset_slope, band_offset_db_1, X14

Step 4: 应用调整 (line 912-913, 964-965, 1000-1001)
    filtered_probs[mod] *= adj[mod] ^ feat_sensitivity

Step 5: 归一化 → 输出概率分布
```

### 4.3 为什么没有逐级注入

1. **数据规模限制**：每类故障仅 100 条样本，若分 3 层注入，每层可用于训练/调参的数据更少
2. **特征区分度不足**：同一故障类别内的不同模块，特征分布高度重叠（如 ref_error 三个模块的特征几乎无法区分，RF 天花板仅 ~35%）
3. **优化目标统一**：P-CMA-ES 优化器统一优化 16 个参数（13 个缩放因子 + 3 个特征灵敏度），若拆分为 3 层则参数空间维度急剧膨胀
4. **工程可行性**：单层决策树 + 先验调整已能在当前数据量下达到可接受的模块定位精度

### 4.4 代码中的"分层"体现

虽然没有逐级注入，但代码在以下维度体现了分层思想：

1. **故障类型条件分支**（`module_brb.py:837-842`）：根据系统级诊断结果（amp/freq/ref），选择不同的模块候选集和先验分布 — 这是**第一层路由**
2. **特征选择差异化**：不同故障类型使用不同的判别特征（amp 用 X7/X13/X35/X36，freq 用 X13/X14/X36，ref 用 offset_slope/band1）— 这是**特征分流**（`module_brb.py:851-1001`）
3. **Soft-gating 多假设融合**（`module_brb.py:1014-1120`）：当 top-2 故障类型概率差 < δ 时，同时激活两个假设的模块推理并加权融合 — 这是**多假设分层**

---

## 问题 5：关于底层诊断概率输出的核实

### 5.1 代码确实输出了各物理模块的概率分布

`hierarchical_module_infer()` 返回一个 `Dict[str, float]`（`module_brb.py:1004-1011`），键为 V2 模块名称，值为归一化概率。

### 5.2 实际输出格式示例

**模块级输出**（来自 `ours_adapter.py` 和 `brb_diagnosis_cli.py`）：

```json
{
    "module": {
        "topk": [
            {
                "module_id": "[数字中频板][ADC] 数字检波与平均",
                "gamma": 0.42
            },
            {
                "module_id": "[RF板][Mixer1]",
                "gamma": 0.38
            },
            {
                "module_id": "[电源板] 电源管理模块",
                "gamma": 0.20
            }
        ]
    }
}
```

### 5.3 输出内容详解

| 输出项 | 类型 | 说明 | 来源 |
|--------|------|------|------|
| 系统级概率 | `Dict[str, float]` | 4 个类别的完整概率分布，Σ=1.0 | `ours_adapter.py:163-165` |
| 模块级概率 | `Dict[str, float]` | 候选模块的完整概率分布（amp:6个, freq:4个, ref:3个），Σ=1.0 | `module_brb.py:1004-1006` |
| TopK 模块 | `List[Dict]` | 概率最高的前 K 个模块（默认 K=10） | `module_brb.py:1112-1113` |
| Soft-gating 详情 | `Dict` | 多假设激活状态、各假设的模块分布 | `module_brb.py:1115-1120` |

### 5.4 关键特性

1. **输出的是连续概率分布**，而非 Top-1 类别或离散异常分数
2. **概率值满足归一化约束**：`Σ P(M_j) = 1.0`（`module_brb.py:1004-1006`）
3. **不区分故障严重程度**：当前输出仅为"哪个模块最可能导致该故障"，**不包含**"严重/轻微/正常"的分级概率
4. **当系统级诊断为"正常"时**：不执行模块级推理，返回空（`module_brb.py:810-812`）

### 5.5 具体格式示例（幅度失准场景）

```
系统级诊断: 幅度失准 (p=0.75)
模块级输出:
    [数字中频板][ADC] 数字检波与平均:      42.0%
    [RF板][Mixer1]:                         28.0%
    [RF板][RF] 低频通路固定滤波/抑制网络:   15.0%
    [电源板] 电源管理模块:                   8.0%
    [数字中频板][IF] 中频放大/衰减链:        4.5%
    [数字中频板][DSP] 数字增益/偏置校准:     2.5%
    ────────────────────────────────────
    合计:                                  100.0%
```

> **注**：如需增加"各模块在不同故障程度下的概率分布"（如 `[严重: XX%, 轻微: YY%, 正常: ZZ%]`），需扩展 `hierarchical_module_infer()` 的返回结构，当前代码未实现此功能。

---

## 附录：关键代码文件索引

| 文件 | 主要内容 |
|------|---------|
| `features/feature_extraction.py:294-892` | 58 个特征的提取逻辑 |
| `BRB/module_brb.py:32-118` | 模块定义（V1/V2）、候选模块列表 |
| `BRB/module_brb.py:800-1120` | 分层模块推理 + Soft-gating |
| `BRB/gating_prior.py:119-155` | RF-BRB 门控融合 |
| `config/gating_prior.json` | 融合参数配置 |
| `pipelines/simulate/run_simulation_brb.py:1121-1425` | 仿真曲线生成与模块分配 |
| `pipelines/simulate/fault_models/module_library.py` | 模块规格库 |
| `pipelines/compare_methods.py:685-718` | 数据集分层划分 |
| `pipelines/default_paths.py:28-30` | 默认参数（400 样本、60/20/20 划分） |
| `tools/label_mapping.py:190-218` | V1→V2 模块名映射 |
| `pipelines/optimize_brb.py:171-300` | P-CMA-ES 优化 |
