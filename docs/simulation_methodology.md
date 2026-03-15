# FMFD 故障仿真方法论：物理退化模型与约束体系

> **文件定位**：本文档面向论文写作与审稿人，系统阐述仿真数据生成的物理模型、参数约束与数学公式。
> 所有数据均直接取自代码，可作为论文的工程补充材料。

---

## 1. 仿真总览

### 1.1 频谱分析仪仿真参数

| 参数 | 值 | 代码位置 |
|------|------|----------|
| 频率起点 | 10 MHz | `baseline/config.py` |
| 频率步长 | 10 MHz | `baseline/config.py` |
| 采样点数 | 820 | `baseline/config.py` |
| 频率范围 | 10 MHz – 8.2 GHz | 单频段模式 |
| 带宽 (BW) | 8.19 GHz | `f[-1] - f[0]` |
| 频段模式 | 单频段 (`SINGLE_BAND_MODE=True`) | `faults.py:6` |

### 1.2 数据集规模与组成

| 系统级故障类型 | 中文 | 样本数 | 故障子类数 |
|---------------|------|--------|-----------|
| `normal` | 正常 | 100 | — |
| `amp_error` | 幅度失准 | 100 | 6 种子故障模式 |
| `freq_error` | 频率失准 | 100 | 3 种子故障模式 |
| `ref_error` | 参考电平失准 | 100 | 2 种子故障模式 |
| **合计** | | **400** | **11 种子故障 + 正常** |

### 1.3 数据集划分

采用**分层随机划分**（`StratifiedShuffleSplit`），保证每类故障在各子集中比例一致：

| 子集 | 比例 | 样本数 |
|------|------|--------|
| 训练集 | 60% | 240 |
| 验证集 | 20% | 80 |
| 测试集 | 20% | 80 |

> **代码位置**：`methods/ours_adapter.py` 中 `train()` 方法，`test_size=0.2` 的两次划分。

---

## 2. 三级严重度体系

所有故障均支持三级严重度分档：

| 级别 | 英文 | 采样概率 | severity_float | 代码 |
|------|------|---------|---------------|------|
| 轻度 | `light` | 55% | 0.3 | `run_simulation_brb.py:1221` |
| 中度 | `mid` | 35% | 0.5 | `run_simulation_brb.py:1221` |
| 重度 | `severe` | 10% | 0.8 | `run_simulation_brb.py:1307` |

```
severity_float = {"light": 0.3, "mid": 0.5, "severe": 0.8}[severity]
```

---

## 3. 故障注入：11 种子故障模式详解

### 3.1 幅度失准类（amp_error）— 6 种子故障

#### (1) `amp` — 校准源幅度失准

**物理背景**：校准信号源的增益漂移、偏置偏差或非线性失真导致整体幅度失准。

**数学模型**（代码 `faults.py:48-89`）：

$$
A' = g \cdot A + b + c \cdot A^2
$$

其中：
- $g \sim \mathcal{N}(1, \sigma_g \cdot \max(1, \hat{\sigma}))$，增益系数
- $b \sim \mathcal{N}(0, \sigma_b \cdot \max(0.2, \hat{\sigma}))$，偏置偏移 (dB)
- $c \sim \text{clip}(\mathcal{N}(\mu_c, \sigma_c), -0.03, +0.03)$，二次非线性系数
- $\hat{\sigma}$ 为曲线局部 MAD 稳健估计：$\hat{\sigma} = 1.4826 \cdot \text{median}(|x_i - \text{median}(x)|)$

**参数约束表**：

| 严重度 | $\sigma_g$ | $\sigma_b$ | $\mu_c$ | $\sigma_c$ | $c$ 范围 |
|--------|-----------|-----------|---------|-----------|---------|
| light | 0.005 | 0.15 | 0.005 | 0.002 | ±0.03 |
| mid | 0.010 | 0.25 | 0.010 | 0.004 | ±0.03 |
| severe | 0.020 | 0.40 | 0.015 | 0.006 | ±0.03 |

**物理核叠加**：通过 `CurveGenerator.apply_global_shift()` 施加全局平移（±0.15–0.30 dB）。

**对应物理模块**：校准源 → CurveGenerator 键 `step_attenuator`

---

#### (2) `lpf` — 低通滤波器截止频率漂移

**物理背景**：低频段前置低通滤波器截止点因老化或温度变化发生偏移，导致带通边沿衰减特性改变。

**数学模型**（代码 `faults.py:285-310`）：

$$
A'(f) = A(f) - D \cdot \left[1 - \frac{1}{1 + \exp\left(\frac{f - f_c}{w}\right)}\right]
$$

其中：
- $f_c = f_0 + 0.85 \cdot \text{BW} + \Delta f_c$，截止频率中心
- $f_c$ 被限制在 $[f_0 + 0.70 \cdot \text{BW},\; f_0 + 0.95 \cdot \text{BW}]$
- $w = 0.05 \cdot \text{BW}$，过渡带宽度
- $D$ 为最大衰减深度 (dB)

**参数约束表**：

| 严重度 | $\Delta f_c$ 范围 | $D$ 范围 (dB) |
|--------|-----------------|-------------|
| light | ±0.5% BW (±40.95 MHz) | 0.04 – 0.08 |
| mid | ±1.0% BW (±81.9 MHz) | 0.08 – 0.14 |
| severe | ±2.0% BW (±163.8 MHz) | 0.14 – 0.18 |

**物理核叠加**：通过 `CurveGenerator.apply_band_insertion_loss()` 施加带通插入损耗。

**对应物理模块**：低频段前置低通滤波器 → `lpf_low_band`

---

#### (3) `mixer` — 混频器带内纹波

**物理背景**：第一混频器转换损耗的频率依赖性导致周期性纹波叠加在频响上。

**数学模型**（代码 `faults.py:312-327`）：

$$
A'(f) = A(f) + R \cdot \sin\left(\frac{2\pi f}{P}\right)
$$

其中：
- $R \sim \mathcal{U}(R_{\min}, R_{\max})$，纹波幅度 (dB)
- $P \sim \mathcal{U}(0.05 \cdot \text{BW}, 0.20 \cdot \text{BW})$，纹波周期 (Hz)

**参数约束表**：

| 严重度 | $R$ 范围 (dB) | $P$ 范围 |
|--------|-------------|---------|
| light | 0.01 – 0.03 | 5% – 20% BW |
| mid | 0.03 – 0.06 | 5% – 20% BW |
| severe | 0.06 – 0.09 | 5% – 20% BW |

**物理核叠加**：通过 `CurveGenerator.apply_linear_slope()` 施加频率依赖的线性斜率（±0.15–0.30 dB）。

**对应物理模块**：低频段第一混频器 → `mixer1`

---

#### (4) `adc` — ADC 量化非线性

**物理背景**：ADC 的微分非线性 (DNL)、积分非线性 (INL) 和采样抖动共同导致量化噪声纹理。

**数学模型**（代码 `faults.py:363-378`）：

$$
A'(f) = A(f) + A_{\text{saw}} \cdot \text{sawtooth}\left(\frac{2\pi C}{N} \cdot i,\; \text{width}=0.5\right)
$$

其中：
- $A_{\text{saw}}$ 为锯齿幅度 (dB)
- $C \sim \mathcal{U}(10, 25)$，锯齿周期数
- $N$ 为曲线长度 (820 点)
- `width=0.5` 表示三角波形态

**参数约束表**：

| 严重度 | $A_{\text{saw}}$ 范围 (dB) | 周期数 $C$ |
|--------|-------------------------|-----------|
| light | 0.02 – 0.05 | 10 – 25 |
| mid | 0.05 – 0.12 | 10 – 25 |
| severe | 0.12 – 0.20 | 10 – 25 |

**物理核叠加**：通过 `CurveGenerator.apply_quantization_noise()` 施加 DNL + INL + 抖动混合噪声：

$$
\text{noise}(i) = 0.6 \cdot A_n \cdot \text{sawtooth}\!\left(\frac{2\pi \cdot C_{\text{DNL}}}{N} \cdot i\right) + 0.3 \cdot A_n \cdot \sin\!\left(\frac{2\pi \cdot C_{\text{INL}}}{N} \cdot i\right) + 0.2 \cdot A_n \cdot \mathcal{U}(-1, 1)
$$

其中 $C_{\text{DNL}} \sim \mathcal{U}(8, 16)$ 为 DNL 锯齿周期数，$C_{\text{INL}} \sim \mathcal{U}(2, 5)$ 为 INL 正弦周期数，$A_n$ 为总噪声幅度，$N = 820$ 为曲线点数。

**对应物理模块**：ADC → `adc_module`

---

#### (5) `vbw` — 数字检波/VBW 平滑

**物理背景**：视频带宽滤波器过度平滑导致频响细节丢失。

**数学模型**（代码 `faults.py:448-460`）：

$$
A'(f_i) = \frac{1}{W} \sum_{j=-\lfloor W/2 \rfloor}^{\lfloor W/2 \rfloor} A(f_{i+j})
$$

即滑动平均卷积，窗口大小 $W$（强制奇数：若 $W$ 为偶数则 $W \leftarrow W+1$），边界使用 edge padding。

**参数约束**：

| 参数 | 范围 |
|------|------|
| 窗口 $W$ | 50 – 800 点（自适应，$W = \max(50, \min(N/10, \mathcal{U}(200, 800)))$，$N=820$ 为曲线点数） |
| 奇偶修正 | $W$ 若为偶数则 $W \leftarrow W + 1$ |
| 边界处理 | edge padding |

**物理核叠加**：通过 `CurveGenerator.apply_smooth_shift()` 施加平滑偏移（0.25–0.50 dB）。

**对应物理模块**：数字检波器 → `dsp_detector`

---

#### (6) `power` — 电源模块噪声

**物理背景**：电源纹波和噪声耦合至信号链，导致全频段随机噪声增大。

**数学模型**（代码 `faults.py:462-496`）：

$$
A'(f) = A(f) + \beta \cdot \hat{\sigma}(f) \cdot \mathcal{N}(0, 1)
$$

其中：
- $\beta$ 为噪声放大系数
- $\hat{\sigma}(f)$ 为局部稳健标准差估计（MAD）
- 噪声去均值处理：$n \leftarrow n - \bar{n}$

**参数约束表**：

| 严重度 | $\beta$ 范围 |
|--------|------------|
| light | 1.0 – 1.5 |
| mid | 1.5 – 2.2 |
| severe | 2.2 – 3.0 |

**物理核叠加**：通过 `CurveGenerator.apply_high_diff_variance()` 施加高频噪声 + 周期性纹波（0.08–0.12 dB RMS）。

**对应物理模块**：电源模块 → `power_management`

---

### 3.2 频率失准类（freq_error）— 3 种子故障

#### (7) `freq` — 时钟振荡器频率漂移

**物理背景**：OCXO 老化或温度漂移导致频率轴系统性偏移。

**数学模型**（代码 `faults.py:91-170`）：

$$
A'(f) = \text{interp}_{\text{cubic}}\left(f + \Delta f,\; A\right)\Big|_{f}
$$

其中：
- $\Delta f = \text{ppm} \times \text{BW}$
- $\text{ppm} \sim \mathcal{U}(-\text{ppm}_{\max}, +\text{ppm}_{\max})$
- 保证最小偏移：$|\text{ppm}| \geq 0.1 \times \text{ppm}_{\max}$
- 插值方法：三次样条（cubic），边界外推使用端点值

**参数约束表**：

| 严重度 | ppm 范围 | 绝对频率偏移 |
|--------|---------|------------|
| light | ±80 ppm | ±655 kHz |
| mid | ±200 ppm | ±1.64 MHz |
| severe | ±500 ppm | ±4.10 MHz |

**物理核叠加**：通过 `CurveGenerator.apply_peak_jitter()` 施加峰值抖动（1–2 点），再进行频率轴 warp：

$$
x_{\text{warp}} = \text{clip}(x \cdot s + b,\; 0,\; 1)
$$

其中 $s \sim 1 \pm \mathcal{U}(0.00008, 0.0005)$ 为 warp 缩放因子，$b \sim \mathcal{U}(-0.001, +0.001)$ 为 warp 偏置。

**对应物理模块**：时钟振荡器 → `ocxo_ref`

---

#### (8) `clock` — 时钟合成与同步网络漂移

**物理背景**：PLL/时钟合成网络参考信号偏差传递至整个频率合成链。

**数学模型**：与 `freq` 完全一致（`faults.py:409-411`）：

```python
def inject_clock_drift(frequency, amp, delta_f=None, rng=None):
    return inject_freq_miscal(frequency, amp, delta_f=delta_f, rng=rng)
```

**参数约束**：同 `freq`（±80/200/500 ppm）

**物理核叠加**：通过 `CurveGenerator.apply_peak_jitter()`（与 freq 同一物理效应，不同的 module_key）。

**对应物理模块**：时钟合成与同步网络 → `ref_distribution`

---

#### (9) `lo` — 本振路径误差

**物理背景**：本振源（谐波发生器）相位噪声或频率合成失锁。

**数学模型**（代码 `faults.py:413-434`）：

逐频段进行频率轴平移重采样：

$$
\forall\; \text{band } (f_s, f_e): \quad A'_{\text{band}}(f) = \text{interp}_{\text{cubic}}\left(f + \Delta f_{\text{band}},\; A_{\text{band}}\right)\Big|_{f}
$$

其中：
- $\Delta f_{\text{band}} = \text{ppm} \times (f_e - f_s)$，每个频段独立偏移
- 在单频段模式下等效于全局 `freq` 加上信号跌落

**参数约束**：同 `freq`（±80/200/500 ppm per band）

**物理核叠加**：通过 `CurveGenerator.apply_signal_drop()` 施加信号跌落 + 相位噪声尖峰（0.3–0.5 dB）。

**对应物理模块**：本振混频组件 → `lo1_synth`

---

### 3.3 参考电平失准类（ref_error）— 2 种子故障

#### (10) `rl` — 参考电平偏移与压缩

**物理背景**：参考电平校准值漂移，同时高幅度区域出现增益压缩效应。

**数学模型**（代码 `faults.py:172-255`）：

**Type-A（全局偏移）**：

$$
A'(f) = A(f) + \Delta_{\text{offset}}
$$

$$
\Delta_{\text{offset}} = \text{clip}\left(\mathcal{N}(0,\; \sigma_o \cdot \max(0.5, \hat{\sigma})),\; -C_o,\; +C_o\right)
$$

最小偏移保证：$|\Delta_{\text{offset}}| \geq 0.10\;\text{dB}$

**Type-B（高幅度压缩）**：

$$
\text{threshold} = P_{80}(A'), \quad A'[A' \geq \text{thr}] = A'[A' \geq \text{thr}] - c_{\text{comp}} \cdot (A'[A' \geq \text{thr}] - \text{thr})
$$

最小压缩系数：$c_{\text{comp}} \geq 0.03$

**参数约束表**：

| 严重度 | $\sigma_o$ | $C_o$ (dB) | $\mu_{\text{comp}}$ | $\sigma_{\text{comp}}$ |
|--------|-----------|-----------|---------------------|----------------------|
| light | 0.2 | ±0.6 | 0.06 | 0.03 |
| mid | 0.4 | ±0.8 | 0.10 | 0.05 |
| severe | 0.7 | ±1.0 | 0.15 | 0.07 |

**物理核叠加**：通过 `CurveGenerator.apply_smooth_shift()` 施加平滑偏移（0.25–0.50 dB）。

**对应物理模块**：校准源 / 校准信号开关 / 存储器（随机选择） → `cal_source`

---

#### (11) `att` — 衰减器状态误差

**物理背景**：步进衰减器切换状态残留误差或机械磨损。

**数学模型**：与 `rl` 完全一致（`faults.py:172-255` 中 `inject_reflevel_miscal`）。

**参数约束**：同 `rl`

**物理核叠加**：通过 `CurveGenerator.apply_global_shift()` 施加全局平移（±0.15–0.30 dB）。

**对应物理模块**：校准源 / 校准信号开关 / 存储器（随机选择） → `step_attenuator`

---

## 4. 正常基线生成

### 4.1 基线约束

正常样本生成围绕 **RRS（参考响应频谱）**，受以下严格约束（代码 `run_simulation_brb.py:1151-1154`）：

| 约束 | 条件 |
|------|------|
| 最大偏差 | $\|x - \text{RRS}\|_\infty \leq 0.4\;\text{dB}$ |
| 包络约束 | $\text{lower\_bound} \leq x \leq \text{upper\_bound}$ |
| 统计约束 | 特征值在真实正常分布的 $\pm 2\sigma$ 内 |

### 4.2 正常样本的状态层级

```python
if fault_kind == "normal":
    tier = choice(["in_spec_weak", "edge"], p=[0.7, 0.3])
```

| 状态 | 概率 | 含义 |
|------|------|------|
| `in_spec_weak` | 70% | 完全在规格内，微弱波动 |
| `edge` | 30% | 接近规格边界（测试分类器鲁棒性） |

---

## 5. 双层仿真架构：数学注入 + 物理核

每条故障曲线的生成采用**双层叠加**架构：

### 第 1 层：数学故障注入（faults.py）

基于解析公式直接修改频响幅度 $A(f)$：
- 幅度类：增益/偏置/非线性（$A' = gA + b + cA^2$）
- 频率类：频率轴重采样（$A'(f) = \text{interp}(f + \Delta f, A)$）
- 参考类：全局偏移 + 高幅度压缩

### 第 2 层：物理退化核（curve_generator.py）

基于物理机理模型叠加退化效应：

| CurveGenerator 函数 | 物理效应 | 幅度范围 |
|---------------------|---------|---------|
| `apply_global_shift()` | 均匀偏移 | ±0.15–0.30 dB |
| `apply_band_insertion_loss()` | 带通插入损耗 | 0.4–0.6 dB |
| `apply_linear_slope()` | 频率依赖斜率 | ±0.15–0.30 dB |
| `apply_quantization_noise()` | DNL+INL+抖动 | 0.1–0.2 dB |
| `apply_smooth_shift()` | 正弦平滑偏移 | 0.25–0.50 dB |
| `apply_high_diff_variance()` | 高频噪声+纹波 | 0.08–0.12 dB RMS |
| `apply_peak_jitter()` | 频率轴抖动 | 1–2 点 |
| `apply_signal_drop()` | 信号跌落+尖峰 | 0.3–0.5 dB |
| `apply_high_pass_filter_effect()` | 低频塌陷 | 0.3–0.5 dB |
| `apply_periodic_ripple()` | 周期性纹波 | 0.2–0.4 dB Vpp |
| `apply_step_discontinuity()` | 阶跃不连续 | 0.1–0.4 dB |

### 纹理缩放

数学注入的纹理在叠加到 RRS 前进行缩放（代码 `run_simulation_brb.py:1298-1302`）：

$$
\text{curve} = \text{RRS} + (\text{curve}_{\text{base}} - \text{RRS}) \times s
$$

| 故障类型 | 缩放因子 $s$ |
|---------|-------------|
| 幅度类 (`amp`) | $\mathcal{U}(0.4, 0.9)$ |
| 其他类 | $\mathcal{U}(0.2, 0.5)$ |

---

## 6. 故障子类概率分布

### 6.1 自由分布模式（无 target_class 约束）

| 子故障 | 概率 | 系统标签 |
|--------|------|---------|
| `amp` | 12% | 幅度失准 |
| `lpf` | 9% | 幅度失准 |
| `mixer` | 9% | 幅度失准 |
| `adc` | 9% | 幅度失准 |
| `vbw` | 8% | 幅度失准 |
| `power` | 8% | 幅度失准 |
| `freq` | 8% | 频率失准 |
| `clock` | 6% | 频率失准 |
| `lo` | 6% | 频率失准 |
| `rl` | 8% | 参考电平失准 |
| `att` | 6% | 参考电平失准 |
| `normal` | 10% | 正常 |

> **代码位置**：`run_simulation_brb.py:1163-1180`

### 6.2 定向生成模式（target_class 约束）

| target_class | 子故障概率分布 |
|-------------|-------------|
| `amp_error` | amp:25%, lpf:18%, mixer:18%, adc:15%, vbw:12%, power:12% |
| `freq_error` | freq:40%, clock:30%, lo:30% |
| `ref_error` | rl:60%, att:40% |
| `normal` | normal:100% |

---

## 7. 故障种类 → 物理模块映射

| 故障种类 | 物理模块中文名 | CurveGenerator 键 | 退化函数 |
|---------|-------------|------------------|---------|
| `amp` | 校准源 | `step_attenuator` | `apply_global_shift()` |
| `lpf` | 低频段前置低通滤波器 | `lpf_low_band` | `apply_band_insertion_loss()` |
| `mixer` | 低频段第一混频器 | `mixer1` | `apply_linear_slope()` |
| `adc` | ADC | `adc_module` | `apply_quantization_noise()` |
| `vbw` | 数字检波器 | `dsp_detector` | `apply_smooth_shift()` |
| `power` | 电源模块 | `power_management` | `apply_high_diff_variance()` |
| `freq` | 时钟振荡器 | `ocxo_ref` | `apply_peak_jitter()` |
| `clock` | 时钟合成与同步网络 | `ref_distribution` | `apply_peak_jitter()` |
| `lo` | 本振混频组件 | `lo1_synth` | `apply_signal_drop()` |
| `rl` | 校准源/开关/存储器 | `cal_source` | `apply_smooth_shift()` |
| `att` | 校准源/开关/存储器 | `step_attenuator` | `apply_global_shift()` |

> **代码位置**：`run_simulation_brb.py:1249-1281`

---

## 8. 验证与重试机制

### 8.1 约束验证

每条曲线生成后通过 `sim_constraints.py` 进行物理一致性验证：

```python
for _ in range(max_attempts=200):
    curve = generate_fault(...)
    result = constraints.validate_fault(curve, fault_kind)
    if result.ok:
        return (curve, label_sys, label_mod, fault_params, ...)
    constraints._record_reject("fault", fault_kind, result.reasons)
```

### 8.2 降级回退

若 200 次重试均失败，使用简单偏移作为降级方案（`run_simulation_brb.py:1453-1494`）：

$$
A'_{\text{fallback}} = \text{RRS} + \mathcal{U}(-0.3, +0.3)
$$

---

## 9. 关键代码入口索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 仿真主入口 | `run_simulation_brb.py` | `simulate_curve()` L1121 |
| 故障概率分布 | `run_simulation_brb.py` | `kind_probs` L1163 |
| 严重度选择 | `run_simulation_brb.py` | L1221 |
| 物理核映射 | `run_simulation_brb.py` | `FAULT_KIND_TO_MODULE_KEY` L1268 |
| 幅度失准注入 | `faults.py` | `inject_amplitude_miscal()` L48 |
| 频率失准注入 | `faults.py` | `inject_freq_miscal()` L91 |
| 参考电平注入 | `faults.py` | `inject_reflevel_miscal()` L172 |
| LPF 漂移注入 | `faults.py` | `inject_lpf_shift()` L285 |
| 混频器纹波注入 | `faults.py` | `inject_mixer_ripple()` L312 |
| ADC 锯齿注入 | `faults.py` | `inject_adc_sawtooth()` L363 |
| VBW 平滑注入 | `faults.py` | `inject_vbw_smoothing()` L448 |
| 电源噪声注入 | `faults.py` | `inject_power_noise()` L462 |
| 本振路径注入 | `faults.py` | `inject_lo_path_error()` L413 |
| 曲线退化引擎 | `curve_generator.py` | `CurveGenerator` L48 |
| 正常基线生成 | `sim_constraints.py` | `generate_normal()` |
| 频谱配置 | `baseline/config.py` | 全局常量 |
