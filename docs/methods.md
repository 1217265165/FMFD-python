# FMFD 方法说明（单频段）

本文档按“真实代码实现”的口径整理 FMFD 单频段流程与方法细节：RRS/包络、仿真、特征提取、方法的特征池划分、推理与校准机制，并尽量对应到实际实现文件。

> 说明：以下内容以当前仓库实现为准，强调“用了哪些特征、如何分层/注入、如何校准”。

## 1. RRS 基准与包络提取

### 1.1 正常曲线对齐与 RRS
- 正常曲线读取自 `normal_response_data/`，在 `baseline/baseline.py` 中通过线性插值对齐到统一频率轴（`N_POINTS`），得到 `traces` 矩阵。 【F:baseline/baseline.py†L1-L82】
- RRS 采用**点位中位数**（默认不平滑），避免过度平滑导致形状漂移；`RRS_SMOOTH_ENABLED=False` 为默认行为。 【F:baseline/baseline.py†L18-L24】
- 为避免“全局偏移”导致包络过宽，先用 `compute_offsets` 计算每条曲线相对 RRS 的中位数偏移量，再在对齐后的残差上估计包络分布。 【F:baseline/baseline.py†L84-L102】

### 1.2 包络（Envelope）
- 包络宽度来自残差分位数搜索（覆盖率网格），并通过平滑/裁剪/扩张确保覆盖率满足目标；滑窗覆盖率也被检查。 【F:baseline/baseline.py†L26-L44】【F:baseline/baseline.py†L118-L168】
- 输出覆盖率指标（整体、单曲线、点位分位数），用来评估包络质量。 【F:baseline/baseline.py†L118-L168】

## 2. 故障仿真（Simulation）

### 2.1 仿真入口与输出
- 仿真入口：`pipelines/simulate/run_simulation_brb.py`，默认无参数生成 400 样本（平衡四类），输出 `raw_curves/*.csv`、`features_brb.csv`、`labels.json` 等。 【F:pipelines/simulate/run_simulation_brb.py†L1-L118】【F:pipelines/simulate/run_simulation_brb.py†L676-L949】
- 仿真流程：读取 RRS/包络后注入故障（幅度/频率/参考电平/正常），再提取系统与模块特征。 【F:pipelines/simulate/run_simulation_brb.py†L676-L949】
- Dataset-M：使用 `--module_driven` 按模块机理采样（module_id → system_label → 模板外观），用于模块定位评测。 【F:pipelines/simulate/run_simulation_brb.py†L890-L1027】

### 2.2 raw_curves 与峰值频率行为
- `raw_curves/*.csv` 三列输出：`freq_injected_hz`、`peak_freq_meas_hz`、`peak_amp_dbm`，用于保留峰值追踪行为并与注入频率对齐。 【F:pipelines/simulate/run_simulation_brb.py†L168-L202】
- 峰值追踪会按故障类型随机生成 `none/spike/dense/hole` 模式，并输出 `peak_freq_mae_hz` 与 `peak_freq_outlier_frac`（>5 MHz）用于审计。 【F:pipelines/simulate/run_simulation_brb.py†L412-L451】【F:pipelines/simulate/run_simulation_brb.py†L819-L926】

### 2.3 标签字段与审计统计
- `labels.json` 记录系统级与模块级标签，同时包含 `fault_template_id`、`module_signature`、`module_v2`、`tier`、`severity`、`global_offset_rrs_db`、`p95_abs_dev_rrs_db`、`inside_env_frac` 等审计字段。 【F:pipelines/simulate/run_simulation_brb.py†L819-L926】
- `real_normal_stats.json` 基于真实正常曲线计算全局偏移与高频噪声统计（p10/p50/p90/p95），用于校验仿真“正常”样本是否偏移。 【F:pipelines/simulate/run_simulation_brb.py†L363-L392】【F:pipelines/simulate/run_simulation_brb.py†L719-L726】

**Normal 语义一致性**
- normal 曲线必须围绕 RRS 生成：`global_offset`、`hf_noise_std`、`p95_abs_dev` 与真实正常统计一致，否则 reject-resample。 【F:pipelines/simulate/sim_constraints.py†L139-L223】

### 2.4 外观模板库（T1–T8）
仿真模板同时约束幅度外观与峰频行为： 【F:pipelines/simulate/fault_models/templates.py†L1-L176】
- T1 刚性平移（smooth_shift）
- T2 连续倾斜（tilt_rolloff）
- T3 稳定波纹（stable_ripple）
- T4 固定台阶（step_fixed）
- T5 稀疏毛刺/离群（spike_sparse）
- T6 全带散布变厚（scatter_thick）
- T7 量化颗粒/锯齿纹理（quant_grain）
- T8 峰值追踪（peak_track：spike/dense/hole）

### 2.5 v1 模块 → 模板绑定（示例表）
| v1 模块 | 模板集合 |
| --- | --- |
| 衰减器 | T1 / T5 |
| 低频段前置低通滤波器 | T2 / T3 |
| 低频段第一混频器 | T2 / T5 |
| 时钟振荡器 | T8 |
| 时钟合成与同步网络 | T8 |
| 本振源（谐波发生器） | T8 |
| 本振混频组件 | T2 / T5 / T8 |
| 校准源 | T1 |
| 存储器 | T1 / T3 / T4 |
| 校准信号开关 | T4 / T5 |
| 中频放大器 | T1 / T2 |
| ADC | T7 / T1 |
| 数字RBW | T3 / T1 |
| 数字放大器 | T1 / T4 |
| 数字检波器 | T1 / T6 |
| VBW滤波器 | T1 / T6 |
| 电源模块 | T6 / T5 |

### 2.6 模块机理库（module_library）
模块机理库定义 `module_id → system_label + 主/辅模板`，用于模块驱动仿真与可解释定位。 【F:pipelines/simulate/fault_models/module_library.py†L1-L62】

## 3. 特征提取（System + Module）

### 3.1 系统特征（X1–X34）
- `features/feature_extraction.py` 中 `extract_system_features` 生成系统级特征：
  - X1–X5：基础偏差/纹波/斜率/频偏等基础特征。
  - X11–X15：包络/残差相关（越界率、最大越界、越界能量、低/高频残差）。
  - X16–X18：频率对齐与形变（相关峰移位、warp scale/bias）。
  - X19–X22：幅度链路细粒度（低频斜率、峰度、峰值数、主频能量占比）。
  - X23–X34：增强频率/参考电平特征与谱结构统计（如相位斜率差、PSDs）。 【F:features/feature_extraction.py†L292-L758】

### 3.2 模块特征与模块顺序
- `extract_module_features` 复用系统特征与传统特征，保持系统层与模块层一致性。 【F:features/feature_extraction.py†L869-L934】
- 模块顺序统一定义在 `MODULES_ORDER`，并与 BRB 规则列表对齐（已移除“未定义/其他”）。 【F:features/feature_extraction.py†L47-L60】

## 4. 方法总览（System-Level）

评估入口为 `pipelines/compare_methods.py`，统一特征/标签顺序与 split，并输出审计与可复现摘要。 【F:pipelines/compare_methods.py†L53-L1150】

### 4.0 Evaluation fairness（共享基础特征）
所有方法共享同一份基础特征表，确保对比公平性；差异来自推理结构/规则门控，而非特征泄漏。独立特征提取只能作为独立实验分支，不得混入主对比管线。 【F:pipelines/compare_methods.py†L432-L520】

| 方法 | 使用的共享特征 | 结构差异 |
| --- | --- | --- |
| Ours | 统一特征表 | 系统级三分支 + 模块级激活门控 |
| DBRB | 统一特征表 | 重要性排序分层 + 逐级注入 |
| BRB-MU | 统一特征表 | 多源不确定性融合 |
| 其他对比方法 | 统一特征表 | 各自的概率/规则结构 |

### 4.1 Ours（层次 BRB + 规则压缩 + 正常锚点）

**核心思想**
- 系统层先做“正常锚点”判别，随后对幅度/频率/参考电平分支进行软门控融合，避免硬阈值误判。 【F:BRB/normal_anchor.py†L1-L239】【F:BRB/aggregator.py†L737-L892】
- 模块层只激活与异常类型相关的模块组，实现规则压缩与可解释推理。 【F:BRB/module_brb.py†L21-L360】

**SUB-BRB1 / SUB-BRB2 语义对齐**
- 系统层输出的概率向量（可视为 G_s / β）作为虚拟先验传入模块层；模块层在此先验下生成模块概率（可视为 G_m / γ）。 【F:BRB/module_brb.py†L200-L258】

**特征池与推理流程**
1. 系统层主要依赖 X1–X22（`OursAdapter.kd_features`），并支持特征别名映射。 【F:methods/ours_adapter.py†L27-L114】
2. `system_level_infer` 输出系统级概率（正常/幅度/频率/参考电平）。 【F:methods/ours_adapter.py†L159-L188】
3. 模块层调用 `module_level_infer_with_activation`，仅激活相关模块组，输出模块概率（20 模块）。 【F:methods/ours_adapter.py†L190-L219】【F:BRB/module_brb.py†L21-L360】

**补偿与校准**
- 校准由 `pipelines/calibrate_ours.py` 网格搜索得到 `best_params.json`/`calibration.json`，`OursAdapter` 加载覆盖 `SystemBRBConfig` 的参数（`alpha`、阈值、权重等）。 【F:pipelines/calibrate_ours.py†L682-L780】【F:methods/ours_adapter.py†L12-L92】

**参数/规则来源**
- 系统层规则与软门控逻辑在 `BRB/aggregator.py` 与 `BRB/normal_anchor.py`；阈值（`T_low/T_high`、`pmax_threshold`、`margin_threshold`）可通过校准覆盖。 【F:BRB/aggregator.py†L777-L892】【F:BRB/normal_anchor.py†L39-L287】
- 模块层规则与模块组定义在 `BRB/module_brb.py`，基于异常类型进行规则压缩。 【F:BRB/module_brb.py†L21-L360】

### 4.2 BRB-MU（多源不确定性融合）

**特征池划分**
- 使用特征名语义把输入分为 amplitude/frequency/noise/switching 四个来源（或均分为 3–4 组）。 【F:methods/brb_mu_adapter.py†L98-L155】

**推理与融合**
1. 对每个来源训练“类条件均值/方差”高斯模型输出概率。 【F:methods/brb_mu_adapter.py†L56-L116】【F:methods/brb_mu_adapter.py†L169-L222】
2. 计算来源不确定度（SNR + SVD），按 `w_s ∝ 1-u_s` 融合来源概率。 【F:methods/brb_mu_adapter.py†L124-L148】

### 4.3 DBRB（深层 BRB / 重要性排序 + 分级注入）

**核心思想**
- 先用 XGBoost/GradientBoosting 得到特征重要性排序，再按排序分成 3 层特征池；每层输出的概率向量作为“隐变量”注入下一层（分级注入）。 【F:methods/dbrb_adapter.py†L29-L141】

**实现流程（按源码）**
1. **特征池**：训练 XGBoost（或回退 GradientBoosting）得到 `feature_importances_`；依赖缺失时回退到方差排序。 【F:methods/dbrb_adapter.py†L29-L63】
2. **重要性排序与分层**：按重要性排序切分 Layer1/Layer2/Layer3。 【F:methods/dbrb_adapter.py†L66-L77】
3. **分级注入**：
   - Layer1：使用 `layer1_features` 推理得到 `z1`。 【F:methods/dbrb_adapter.py†L80-L83】
   - Layer2：拼接 `layer2_features + z1` 推理得到 `z2`。 【F:methods/dbrb_adapter.py†L85-L92】
   - Layer3：拼接 `layer3_features + z2` 输出最终概率。 【F:methods/dbrb_adapter.py†L94-L141】
4. **层内推导**：每层使用高斯似然（类均值/方差 + 先验）推断。 【F:methods/dbrb_adapter.py†L150-L191】

### 4.4 HCF（分层认知框架）

**特征池划分与流程**
- Level‑a：Fisher 分数选择主/次特征。 【F:methods/hcf_adapter.py†L40-L76】
- Level‑b：按语义来源分组，使用 GMM 聚类编码为 one‑hot。 【F:methods/hcf_adapter.py†L75-L117】
- Level‑c：拼接编码特征，逻辑回归输出系统结果。 【F:methods/hcf_adapter.py†L121-L176】

### 4.5 BRB‑P（规则分区 BRB）

**特征池与推理**
- 使用特征分区构造规则并进行 BRB 推理；规则权重在训练中学习。 【F:methods/brb_p_adapter.py†L32-L96】

### 4.6 A‑IBRB（区间规则 BRB）

**特征池与推理**
- 构建区间规则并做区间匹配；规则置信度由区间内样本统计估计。 【F:methods/a_ibrb_adapter.py†L33-L117】

## 5. 规则推导与参数说明（通用）

- 系统/模块层 BRB 规则结构与模块分组定义位于 `BRB/module_brb.py`、`BRB/system_brb*.py`，推理由 `SimpleBRB/ERBRB` 完成归一化融合。 【F:BRB/module_brb.py†L21-L360】【F:BRB/utils.py†L6-L61】
- 系统层“正常锚点 + 软门控”逻辑在 `BRB/normal_anchor.py` 与 `BRB/aggregator.py`，关键阈值可通过校准覆盖。 【F:BRB/normal_anchor.py†L39-L287】【F:BRB/aggregator.py†L777-L892】

## 6. 模块级诊断与评估（Module-Level）

### 6.1 模块级诊断链路（真实诊断）
模块级诊断依赖系统层输出作为先验，核心链路如下：
1. `pipelines/detect.py` 读取待检曲线并抽取系统特征。 【F:pipelines/detect.py†L1-L144】
2. `system_level_infer` 给出系统级概率（正常/幅度/频率/参考电平）。 【F:BRB/system_brb.py†L1-L258】
3. `module_level_infer` 在系统先验下激活模块组，输出模块级概率。 【F:BRB/module_brb.py†L21-L360】
4. `brb_diagnosis_cli.py` 汇总系统级与模块级输出，生成 Top‑K 模块定位结果与证据字段。 【F:brb_diagnosis_cli.py†L200-L610】

该链路强调“系统级类型 + 模块级定位”，用于真实样本的解释性诊断。

### 6.2 模块级评估入口（仿真审计）
模块级的可复现评估通过仿真数据完成：
* `pipelines/simulate/run_simulation_brb.py --module_driven` 生成带模块标签的数据集（Dataset‑M）。 【F:pipelines/simulate/run_simulation_brb.py†L890-L1027】
* 输出的 `labels.json` 同时包含系统标签与模块标签，便于审计模块定位性能。 【F:pipelines/simulate/run_simulation_brb.py†L819-L926】

该路径用于离线评估模块定位质量，补充系统级对比之外的诊断能力审计。

### 6.3 模块级特征与模块顺序一致性
模块级推理沿用系统特征与模块特征的统一顺序：
* `extract_module_features` 复用系统特征并保持模块顺序一致。 【F:features/feature_extraction.py†L869-L934】
* `MODULES_ORDER` 与模块 BRB 的规则顺序对齐，避免模块级输出错位。 【F:features/feature_extraction.py†L47-L60】【F:BRB/module_brb.py†L21-L360】

通过统一的模块顺序与特征池，保证系统级与模块级推理之间的可追溯性。
