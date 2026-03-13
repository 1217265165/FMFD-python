# FMFD 项目标签与特征定义一致性核查报告 (Consistency Audit)

**审计日期**: 2026-02-06  
**审计目标**: 核查现有代码是否与用户提供的 37 维特征定义 (X1-X37) 和完整模块列表保持一致

---

## 1. 模块标签 (Module Labels) 一致性核查

### 1.1 代码中的模块定义

**BRB/module_brb.py 中的 MODULE_LABELS (V1 版本)**:
```python
# BRB/module_brb.py:32-53
MODULE_LABELS: List[str] = [
    "衰减器",
    "前置放大器",  # DISABLED in single-band mode
    "低频段前置低通滤波器",
    "低频段第一混频器",
    "高频段YTF滤波器",
    "高频段混频器",
    "时钟振荡器",
    "时钟合成与同步网络",
    "本振源（谐波发生器）",
    "本振混频组件",
    "校准源",
    "存储器",
    "校准信号开关",
    "中频放大器",
    "ADC",
    "数字RBW",
    "数字放大器",
    "数字检波器",
    "VBW滤波器",
    "电源模块",
]  # 共 20 个模块
```

**BRB/module_brb.py 中的 BOARD_MODULES (V2 版本)**:
```python
# BRB/module_brb.py:631-659
BOARD_MODULES = {
    "RF板": [
        "[RF板][RF] 输入连接/匹配/保护",
        "[RF板][RF] 输入衰减器组",
        "[RF板][RF] 低频通路固定滤波/抑制网络",
        "[RF板][RF] 高频通路固定滤波/抑制网络",
        "[RF板][Mixer1]"
    ],
    "数字中频板": [
        "[数字中频板][IF] 中频放大/衰减链",
        "[数字中频板][ADC] 数字检波与平均",
        "[数字中频板][ADC] 采样时钟",
        "[数字中频板][DSP] 数字增益/偏置校准",
        "[数字中频板][IF] RBW数字滤波器"
    ],
    "LO/时钟板": [
        "[LO/时钟板][LO1] 合成链",
        "[时钟板][参考域] 10MHz 基准 OCXO",
        "[时钟板][参考分配]"
    ],
    "校准链路": [
        "[校准链路][校准源]",
        "[校准链路][校准路径开关/耦合]",
        "[校准链路][校准表/存储]"
    ],
    "电源板": [
        "[电源板] 电源管理模块"
    ]
}  # 共 16 个 V2 模块
```

### 1.2 模块差异表

| 用户要求的模块类别 | 代码中的状态 | 是否缺失 | 备注 |
|------------------|-------------|---------|------|
| RF 板系列 | ✅ 存在 | 否 | `[RF板][RF]` 前缀的 5 个模块 |
| Digital IF 板系列 | ✅ 存在 | 否 | `[数字中频板]` 前缀的 5 个模块 |
| LO/时钟板系列 | ✅ 存在 | 否 | `[LO/时钟板]` 和 `[时钟板]` 前缀的 3 个模块 |
| 校准(Calibration)系列 | ✅ 存在 | 否 | `[校准链路]` 前缀的 3 个模块 |
| ADC 板系列 | ⚠️ 部分存在 | 部分 | ADC 功能合并在 `[数字中频板][ADC]` 中，**无独立 `[ADC板]` 前缀** |
| 电源板 | ✅ 存在 | 否 | `[电源板] 电源管理模块` |

**缺失项检查结论**:
- ⚠️ 代码中**没有独立的 `[校准板][CAL]` 前缀**，而是使用 `[校准链路]` 表示校准相关模块
- ⚠️ 代码中**没有独立的 `[ADC板][ADC]` 前缀**，ADC 功能合并在 `[数字中频板][ADC]` 下

### 1.3 命名一致性

**诊断输出使用的语言**: 代码输出使用**中文 V2 名称**（如 `[数字中频板][IF] 中频放大/衰减链`），而非英文 Key。

**证据** (`tools/aggregate_batch_diagnosis.py:89-91`):
```python
# 获取预测结果
sys_pred = result.get("system_diagnosis", {}).get("predicted_class", "")
module_topk = result.get("module_diagnosis", {}).get("topk", [])
```

输出 JSON 中的 `topk` 字段包含的是**模块名称**（中文），而非模块 ID。

### 1.4 评估对齐机制

**`tools/eval_module_localization.py` 的匹配逻辑**:
```python
# tools/eval_module_localization.py:36-38
from metrics.module_localization_metrics import compute_mod_topk, compute_mod_metrics
from utils.canonicalize import modules_match as unified_modules_match
```

**`utils/canonicalize.py::modules_match()` 函数** (行 142-207):
- 使用 `canonical_module_v2()` 将预测和真值都转换为标准 V2 格式
- 实现了**模糊匹配**机制，支持关键词匹配（如 "中频放大" ↔ "中频放大"）
- ✅ **已实现 V1→V2 映射**，减少了因中英文不匹配导致误判的风险

```python
# utils/canonicalize.py:142-207
def modules_match(pred: str, gt: str) -> bool:
    # Canonicalize both
    pred_canonical = canonical_module_v2(pred)
    gt_canonical = canonical_module_v2(gt)
    
    # Exact match
    if pred_canonical == gt_canonical:
        return True
    
    # Fuzzy matching with keywords
    keywords = [
        ("中频放大", "中频放大"),
        ("检波", "检波"),
        ("ADC", "ADC"),
        # ...
    ]
```

---

## 2. 特征实现 (Feature Implementation) 现状核查

### 2.1 特征提取完整性

**`features/feature_extraction.py::extract_system_features()` 返回的特征**:
```python
# features/feature_extraction.py:795-810
return {
    "X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5,
    "X6": x6, "X7": x7, "X8": x8, "X9": x9, "X10": x10,
    "X11": x11, "X12": x12, "X13": x13, "X14": x14, "X15": x15,
    "X16": x16, "X17": x17, "X18": x18,
    "X19": x19, "X20": x20, "X21": x21, "X22": x22,
    "X23": x23, "X24": x24, "X25": x25,  # New freq features
    "X26": x26, "X27": x27, "X28": x28,  # New ref features
    "X29": x29, "X30": x30, "X31": x31,  # v8: Amp vs Ref features
    "X32": x32, "X33": x33, "X34": x34,  # v9: Enhanced spectrum analysis features
    "X35": x35, "X36": x36, "X37": x37,  # v10: Diff variance + periodicity + linear residual
    "offset_db": offset_db,
    # ...
}
```

**结论**: ✅ **feature_extraction.py 已完整实现 X1-X37**

| 特征范围 | 实现状态 | 备注 |
|---------|---------|------|
| X1-X5 | ✅ 已实现 | 系统级基础特征 |
| X6-X10 | ✅ 已实现 | 模块级症状特征 |
| X11-X15 | ✅ 已实现 | 包络/残差特征 |
| X16-X18 | ✅ 已实现 | 频率对齐特征 |
| X19-X22 | ✅ 已实现 | 幅度链路细粒度特征 |
| X23-X25 | ✅ 已实现 | 新增频率特征 (v7+) |
| X26-X28 | ✅ 已实现 | 新增参考电平特征 (v7+) |
| X29-X31 | ✅ 已实现 | v8: Amp vs Ref 区分特征 |
| X32-X34 | ✅ 已实现 | v9: 频谱分析特征 (PSD比等) |
| X35-X37 | ✅ 已实现 | v10: 形态特征 (差分方差、周期性、线性拟合残差) |

### 2.2 子图 BRB 特征使用对比

**用户指定的子图特征分配**:
- amp_error: X1, X2, X5, X7, X29, X30, X35
- freq_error: X4, X16, X17, X23, X24, X25
- ref_error: X1, X3, X5, X11, X26, X27, X28, X31

**代码中的实际实现**:

| 子图 BRB | 文件位置 | 实际使用特征 | 与用户要求对比 |
|---------|---------|-------------|---------------|
| **Amp 子BRB** | `BRB/system_brb_amp.py:42-56` | X1, X2, X5, X6, X7, X10, X11, X12, X13, X19, X20, X21, X22 | ⚠️ **缺少 X29, X30, X35** (X36,X37 替代 X35 是错配) |
| **Freq 子BRB** | `BRB/system_brb_freq.py:42-51` | X4, X8, X9, X14, X15, X16, X17, X18 | ⚠️ **缺少 X23, X24, X25** |
| **Ref 子BRB** | `BRB/system_brb_ref.py:42-50` | X1, X3, X5, X10, X11, X12, X13 | ⚠️ **缺少 X26, X27, X28, X31** |

**`features/feature_router.py` 中的分流定义**:
```python
# features/feature_router.py:74-83
SYSTEM_BRANCH_FEATURES = {
    'amp': ['X1', 'X2', 'X5', 'X6', 'X7', 'X10', 'X11', 'X12', 'X13', 'X19', 'X20', 'X21', 'X22', 'X36', 'X37'],
    'freq': ['X4', 'X8', 'X9', 'X14', 'X15', 'X16', 'X17', 'X18'],
    'ref': ['X1', 'X3', 'X5', 'X10', 'X11', 'X12', 'X13', 'X35'],
}
```

**特征滞后差异表**:

| 子图 | 用户要求特征 | 代码实际使用 (feature_router.py) | 缺失特征 | 额外/错配特征 |
|-----|------------|--------------------------------|---------|--------------|
| **amp_error** | X1,X2,X5,X7,X29,X30,X35 | X1,X2,X5,X6,X7,X10-X13,X19-X22,X36,X37 | **X29, X30, X35** | X6,X10-X13,X19-X22; X36,X37 替代 X35 (编号错配) |
| **freq_error** | X4,X16,X17,X23,X24,X25 | X4,X8,X9,X14,X15,X16,X17,X18 | **X23, X24, X25** | X8,X9,X14,X15,X18 |
| **ref_error** | X1,X3,X5,X11,X26,X27,X28,X31 | X1,X3,X5,X10-X13,X35 | **X26, X27, X28, X31** | X10,X12,X13; X35 应在 amp 分支 |

**注意**: 代码中 `feature_router.py` 将 X35 放在 ref 分支，将 X36,X37 放在 amp 分支，与用户要求的 X35 应在 amp_error 分支不一致。这可能是特征分配错配问题。

### 2.3 索引偏移风险

**代码特征索引规则**: 从 **1 开始** (X1-X37)，而非 0 开始

```python
# features/feature_extraction.py:319
return {f"X{i}": 0.0 for i in range(1, 23)}  # 注意 range(1, 23) 从 1 开始
```

**utils/feature_pool.py:135 的索引转换**:
```python
def get_feature_indices(pool: List[str]) -> List[int]:
    indices = []
    for name in pool:
        if name.startswith("X") and name[1:].isdigit():
            indices.append(int(name[1:]) - 1)  # X1 -> 0, X2 -> 1, etc.
    return sorted(indices)
```

**结论**: 
- 特征字典 key 使用 X1-X37 (1-based 命名)
- 转换为数组索引时减 1 (0-based)
- ✅ 索引逻辑一致，无偏移风险

---

## 3. 三链路评估口径核查

### 3.1 系统级标签对比

| 链路 | 文件 | 使用的标签格式 | 是否有映射转换 |
|-----|------|--------------|--------------|
| **Path A (仿真)** | `pipelines/simulate/run_simulation_brb.py` | 中文 ("正常", "幅度失准"...) | 通过 `tools/label_mapping.py` 转换 |
| **Path B (对比)** | `pipelines/compare_methods.py:76` | 中文 `['正常', '幅度失准', '频率失准', '参考电平失准']` | 内置定义 |
| **Path C (诊断)** | `brb_diagnosis_cli.py` + `aggregate_batch_diagnosis.py` | 混合使用 | 有映射逻辑 (行 92-98) |
| **Path D (评估)** | `tools/eval_module_localization.py` | 英文 (`amp_error`, `freq_error`...) | 通过 `utils/canonicalize.py` 转换 |

**标签映射代码证据**:

```python
# tools/aggregate_batch_diagnosis.py:92-98
sys_correct = (sys_pred == gt_fault_type) or (
    # 处理中英文映射
    (sys_pred == "正常" and gt_fault_type == "normal") or
    (sys_pred == "幅度失准" and gt_fault_type == "amp_error") or
    (sys_pred == "频率失准" and gt_fault_type == "freq_error") or
    (sys_pred == "参考电平失准" and gt_fault_type == "ref_error")
)
```

```python
# tools/label_mapping.py:24-36
SYS_CLASS_TO_CN = {
    "normal": "正常",
    "amp_error": "幅度失准",
    "freq_error": "频率失准",
    "ref_error": "参考电平失准",
}
CN_TO_SYS_CLASS = {v: k for k, v in SYS_CLASS_TO_CN.items()}
```

```python
# utils/canonicalize.py:27-65
def canonical_fault_type(name: str) -> str:
    # Normalize to English: normal/amp_error/freq_error/ref_error
```

### 3.2 Top-K 输出格式

**`aggregate_batch_diagnosis.py` 输出的 JSON 格式**:
```json
{
  "module_diagnosis": {
    "topk": [
      {"module": "[数字中频板][IF] 中频放大/衰减链", "probability": 0.35},
      {"module": "[RF板][RF] 低频通路固定滤波/抑制网络", "probability": 0.25},
      ...
    ]
  }
}
```

**结论**: `top_modules` 字段包含的是**模块名称**（V2 中文格式），而非模块 ID。

### 3.3 口径风险汇总

| 风险类型 | 风险等级 | 说明 | 受影响链路 |
|---------|---------|------|----------|
| **系统标签中英文不一致** | 🟡 中 | Path B 使用中文，Path D 使用英文，但有映射机制 | Path B ↔ Path D |
| **模块名 V1/V2 混用** | 🟡 中 | MODULE_LABELS 是 V1，BOARD_MODULES 是 V2，需要通过 `module_v2_from_v1()` 转换 | 全部链路 |
| **Top-K 格式依赖** | 🟢 低 | `topk` 使用名称而非 ID，`modules_match()` 支持模糊匹配 | Path C → Path D |
| **子图特征滞后** | 🔴 高 | 子BRB 未使用最新特征 (X23-X31)，可能影响诊断准确率 | 全部链路 |

---

## 4. 差异汇总表

### 4.1 模块差异表

| 模块类别 | 用户期望格式 | 代码实际格式 | 差异说明 |
|---------|------------|-------------|---------|
| 校准模块 | `[校准板][CAL] xxx` | `[校准链路][校准xxx]` | 前缀命名不一致 |
| ADC 模块 | `[ADC板][ADC] xxx` | `[数字中频板][ADC] xxx` | 板级归属不同 |

### 4.2 特征差异表

| 子图 | 未实现/未使用的特征 | 影响 |
|-----|-------------------|------|
| **amp_error** | X29 (HF/LF能量比), X30 (压缩指数) | 无法区分 Amp vs Ref |
| **freq_error** | X23 (Warp残差能量), X24 (相位斜率差), X25 (位移后MSE) | 频率失准检测精度下降 |
| **ref_error** | X26 (高分位压缩分), X27 (分段增益变化), X28 (上尾不对称), X31 (分段偏移一致性) | 参考电平失准检测精度下降 |

### 4.3 口径风险表

| 链路 A (仿真) → 链路 B (对比) | 链路 C (诊断) → 链路 D (评估) |
|------------------------------|------------------------------|
| 🟢 使用相同的特征提取函数 `extract_system_features()` | 🟡 依赖 `modules_match()` 做模糊匹配 |
| 🟢 系统标签使用中文 | 🟡 系统标签需要中英文转换 |
| 🟢 模块使用 V1 名称 | 🟡 模块需要 V1→V2 转换 |

---

## 5. 建议修复措施

### 5.1 高优先级

1. **更新子BRB特征使用**:
   - `BRB/system_brb_amp.py`: 添加 X29, X30, X35
   - `BRB/system_brb_freq.py`: 添加 X23, X24, X25
   - `BRB/system_brb_ref.py`: 添加 X26, X27, X28, X31

2. **同步 `features/feature_router.py`**:
   ```python
   SYSTEM_BRANCH_FEATURES = {
       'amp': ['X1', 'X2', 'X5', 'X7', 'X29', 'X30', 'X35'],  # 按用户要求
       'freq': ['X4', 'X16', 'X17', 'X23', 'X24', 'X25'],
       'ref': ['X1', 'X3', 'X5', 'X11', 'X26', 'X27', 'X28', 'X31'],
   }
   ```

### 5.2 中优先级

1. **统一模块命名前缀**:
   - 如需支持 `[校准板][CAL]` 格式，在 `utils/canonicalize.py` 中添加映射
   - 如需支持 `[ADC板][ADC]` 格式，更新 `BOARD_MODULES` 定义

2. **增强标签映射覆盖**:
   - 确保所有入口都使用 `canonical_fault_type()` 和 `canonical_module_v2()`

### 5.3 低优先级

1. **文档更新**: 更新 `docs/` 中的特征定义说明，确保与代码一致
2. **测试覆盖**: 添加特征一致性检查的单元测试

---

## 附录: 关键代码位置

| 功能 | 文件 | 行号 |
|-----|------|------|
| MODULE_LABELS (V1) | `BRB/module_brb.py` | 32-53 |
| BOARD_MODULES (V2) | `BRB/module_brb.py` | 631-659 |
| V1→V2 映射 | `tools/label_mapping.py` | 190-219 |
| 特征提取 (X1-X37) | `features/feature_extraction.py` | 294-810 |
| 系统标签映射 | `tools/label_mapping.py` | 24-36 |
| 模块匹配函数 | `utils/canonicalize.py` | 142-207 |
| 特征分流规则 | `features/feature_router.py` | 74-83 |
| Amp 子BRB | `BRB/system_brb_amp.py` | 42-56 |
| Freq 子BRB | `BRB/system_brb_freq.py` | 42-51 |
| Ref 子BRB | `BRB/system_brb_ref.py` | 42-50 |
