# 旧仿真逻辑审查报告

**审计日期**: 2026-02-06  
**审计范围**: `pipelines/simulate/run_simulation_brb.py` 及相关模块

---

## 1. 核心故障生成代码位置

故障曲线生成的核心逻辑位于 `run_simulation_brb.py` 的 **第 1280-1400 行**。

### 1.1 主要调用的函数

```python
# run_simulation_brb.py:1289-1368 (核心故障注入循环)

elif fault_kind == "freq":
    curve, freq_params = inject_freq_miscal(
        frequency, curve, rng=rng, return_params=True, severity=severity
    )
    # 额外的频率轴变换
    warp_scale = 1.0 + rng.uniform(0.00008, 0.0005) * (1 if rng.random() < 0.5 else -1)
    x_axis = np.linspace(0.0, 1.0, len(curve))
    x_warp = np.clip(x_axis * warp_scale + warp_bias, 0.0, 1.0)
    curve = np.interp(x_axis, x_warp, curve)

elif fault_kind == "lpf":
    curve = inject_lpf_shift(frequency, rrs, rng=rng, severity=severity)

elif fault_kind == "mixer":
    curve = inject_mixer1_slope(frequency, rrs, rng=rng, severity=severity)

elif fault_kind == "adc":
    curve = inject_adc_sawtooth(frequency, rrs, rng=rng, severity=severity)

elif fault_kind == "power":
    curve = inject_power_noise_rrs(rrs, sigma, rng=rng, severity=severity)

# 模板系统叠加
if template_id:
    template_result = apply_template(template_id, curve, frequency, rrs, rng, severity)
    curve = template_result.curve
```

---

## 2. 各函数的数学实现分析

### 2.1 `inject_amplitude_miscal()` - 幅度失准 ⚠️ 简单数学

```python
# faults.py:48-80
result = gain * amp + bias + comp * (amp ** 2)
```

**分析**: 使用 **线性变换 + 二次项**，纯数学运算，无物理模型。
- `gain * amp`: 增益缩放
- `bias`: 偏置偏移
- `comp * (amp ** 2)`: 二次压缩项

---

### 2.2 `inject_freq_miscal()` - 频率失准 ⚠️ 简单数学

```python
# faults.py:91-130
# 核心逻辑：频率轴平移 + 插值重采样
shifted = frequency + delta_f
result = np.interp(frequency, shifted, amp)
```

**分析**: 使用 **频率轴偏移 + 线性插值**，是数学变换而非物理模型。

---

### 2.3 `inject_lpf_shift()` - 低通滤波器漂移 ✅ 半物理模型

```python
# faults.py:285-310
trans = 1 / (1 + np.exp((f - center) / width))  # Sigmoid 滚降
attenuation = drop_db * (1.0 - trans)
return amp - attenuation
```

**分析**: 使用 **Sigmoid 函数** 模拟滤波器滚降特性，具有一定的物理意义，但并非真正的滤波器模型。

---

### 2.4 `inject_mixer1_slope()` - 混频器斜率 ⚠️ 简单数学

```python
# faults.py:348-360
f_norm = (frequency - frequency[0]) / (frequency[-1] - frequency[0] + 1e-12)
return rrs + slope_db * f_norm
```

**分析**: 纯 **线性斜坡叠加**，无混频器物理模型。

---

### 2.5 `inject_adc_sawtooth()` - ADC 量化噪声 ⚠️ 简单数学

```python
# faults.py:363-378
phase = np.linspace(0.0, 2 * np.pi * cycles, len(frequency))
texture = amplitude_db * sawtooth(phase, width=0.5)
return rrs + texture
```

**分析**: 使用 `scipy.signal.sawtooth` 生成 **周期锯齿波**，模拟量化纹理，但并非真实的 ADC 量化模型。

---

### 2.6 `inject_power_noise_rrs()` - 电源噪声 ⚠️ 简单数学

```python
# faults.py:381-394
noise = rng.normal(0, noise_std, size=len(rrs))
return rrs + noise
```

**分析**: 纯 **高斯白噪声叠加**，无电源纹波的物理模型。

---

### 2.7 模板系统 (`templates.py`) ⚠️ 简单数学

所有模板都是基础数学变换：

| 模板ID | 功能 | 实现方式 |
|--------|------|----------|
| T1 | 平滑偏移 | `base + offset` |
| T2 | 斜率滚降 | `base + slope * f_norm` |
| T3 | 稳定纹波 | `base + amplitude * np.sin(...)` |
| T4 | 阶跃跳变 | `curve[idx:] += step` |

**分析**: 全部是 **np.add / np.sin / 线性变换**，无物理模型。

---

## 3. 结论

### 3.1 答案

**问题**: 这段代码是使用了 `np.random / np.add` 进行简单的数学加噪/偏移，还是调用了具有物理意义的函数（如滤波器、调制模型）？

**回答**: **主要是简单数学运算，少量半物理模型**。

| 类别 | 函数 | 物理意义 |
|------|------|----------|
| ⚠️ 简单数学 | `inject_amplitude_miscal` | 线性变换 + 二次项 |
| ⚠️ 简单数学 | `inject_freq_miscal` | 频率平移 + 插值 |
| ⚠️ 简单数学 | `inject_mixer1_slope` | 线性斜坡 |
| ⚠️ 简单数学 | `inject_adc_sawtooth` | 锯齿波叠加 |
| ⚠️ 简单数学 | `inject_power_noise_rrs` | 高斯噪声 |
| ⚠️ 简单数学 | 模板 T1-T9 | 偏移/斜率/纹波 |
| ✅ 半物理模型 | `inject_lpf_shift` | Sigmoid 滚降 |

### 3.2 缺失的物理模型

1. **真正的滤波器模型**: 应使用 `scipy.signal.butter` + `filtfilt` 实现高/低通效果
2. **调制/混频模型**: 应模拟 LO 信号与输入的混频过程
3. **ADC 真实量化**: 应模拟位深限制、DNL/INL 误差
4. **电源纹波**: 应包含 50/60Hz 工频分量 + 开关噪声

### 3.3 建议

建议使用新的 `pipelines/simulate/curve_generator.py`（已实现的物理退化函数库），替换现有的简单数学模型。新版本包含：

- `apply_high_pass_filter_effect()` - 真实高通滤波器效应
- `apply_quantization_noise()` - 真实量化噪声模型
- `apply_signal_drop()` - LO 失锁黑洞效应
- 等 23 个物理退化函数

---

## 附录：代码位置速查

| 文件 | 行号 | 功能 |
|------|------|------|
| `run_simulation_brb.py` | 1280-1400 | 故障生成主循环 |
| `faults.py` | 48-89 | `inject_amplitude_miscal` |
| `faults.py` | 91-165 | `inject_freq_miscal` |
| `faults.py` | 285-310 | `inject_lpf_shift` |
| `faults.py` | 348-360 | `inject_mixer1_slope` |
| `faults.py` | 363-378 | `inject_adc_sawtooth` |
| `faults.py` | 381-394 | `inject_power_noise_rrs` |
| `fault_models/templates.py` | 32-200 | 模板系统 T1-T9 |
| `curve_generator.py` | 1-839 | **新版物理退化函数（推荐）** |
