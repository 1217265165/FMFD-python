# FMFD 项目架构真实逻辑审计报告

**审计日期**: 2026-02-06  
**审计目标**: 通过代码分析还原当前系统级门控、特征分池、BRB推理的真实实现方式

---

## 1. 真实架构图解

### 1.1 调用链概览

```
                 ┌────────────────────────────────────────────────────────────┐
                 │                      入口层                                 │
                 │   run_simulation_brb.py    brb_diagnosis_cli.py            │
                 │            │                        │                       │
                 │            ▼                        ▼                       │
                 │   pipelines/simulate/        infer_system_and_modules()    │
                 │   run_simulation_brb.py      (methods/ours_adapter.py)     │
                 │   (直接调用 system_level_infer)       │                    │
                 │            │                        │                       │
                 └────────────┼────────────────────────┼───────────────────────┘
                              │                        │
                              ▼                        ▼
                 ┌────────────────────────────────────────────────────────────┐
                 │                    推理层                                   │
                 │                                                             │
                 │   ┌─────────────────────────────────────────────────────┐  │
                 │   │        infer_system_and_modules() 统一入口           │  │
                 │   │  (methods/ours_adapter.py:119-316)                  │  │
                 │   │                                                      │  │
                 │   │   Step 0: 自动加载 RF classifier (从 artifacts/)    │  │
                 │   │   Step 1: BRB 系统级推理 (mode="sub_brb")           │  │
                 │   │   Step 2: RF + BRB 门控融合 (GatingPriorFusion)     │  │
                 │   │   Step 3: argmax(final_probs) 得到 fault_type_pred  │  │
                 │   │   Step 4: 模块级 BRB 推理                            │  │
                 │   └─────────────────────────────────────────────────────┘  │
                 │                                                             │
                 └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────────────────────────────────────────┐
                 │                    BRB 核心层                               │
                 │                                                             │
                 │   ┌───────────────────┐    ┌───────────────────────────┐  │
                 │   │ system_level_infer │    │ hierarchical_module_infer │  │
                 │   │ (BRB/system_brb.py)│    │ _soft_gating              │  │
                 │   │                    │    │ (BRB/module_brb.py)       │  │
                 │   │  mode="sub_brb":   │    │                           │  │
                 │   │   ├─ amp_brb_infer │    │  根据 fault_type 激活     │  │
                 │   │   ├─ freq_brb_infer│    │  对应子图,然后调用        │  │
                 │   │   └─ ref_brb_infer │    │  module_level_infer       │  │
                 │   └───────────────────┘    └───────────────────────────┘  │
                 │                                                             │
                 └─────────────────────────────────────────────────────────────┘
```

### 1.2 双链路汇合点

| 链路 | 入口文件 | 最终调用的推理类/函数 | 定义位置 |
|------|---------|---------------------|---------|
| Path A | `run_simulation_brb.py` | `system_level_infer()` + `module_level_infer()` | `BRB/system_brb.py`, `BRB/module_brb.py` |
| Path B | `brb_diagnosis_cli.py` | `infer_system_and_modules()` (统一入口) | `methods/ours_adapter.py` |

**汇合点分析**:

- **Path A** (`run_simulation_brb.py`): 直接导入并调用 `system_level_infer` 和 `module_level_infer`:
  ```python
  # pipelines/simulate/run_simulation_brb.py:54-55
  from BRB.module_brb import DISABLED_MODULES, MODULE_LABELS_V2, module_level_infer
  from BRB.system_brb import system_level_infer
  ```

- **Path B** (`brb_diagnosis_cli.py`): 使用统一入口 `infer_system_and_modules`:
  ```python
  # brb_diagnosis_cli.py:180
  from methods.ours_adapter import infer_system_and_modules
  
  # brb_diagnosis_cli.py:337-342
  unified_result = infer_system_and_modules(
      features,
      use_gating=True,
      rf_classifier=None,  # Auto-load from artifacts/
      allow_fallback=getattr(args, 'allow_fallback', True),
  )
  ```

**结论**: 两条链路**不完全一致**。Path B 使用统一入口包含 RF 门控融合逻辑,而 Path A 直接调用 BRB。

---

## 2. 特征分池 (Feature Pooling) 现状

### 2.1 存在硬编码的特征池定义

在 `utils/feature_pool.py` 中存在明确的特征池定义:

```python
# utils/feature_pool.py:50-82
COMMON_POOL = ["X1", "X5"]

AMP_POOL = ["X2", "X3", "X4", "X6", "X7"]

FREQ_POOL = ["X8", "X9", "X10"]

REF_POOL = ["X2", "X11", "X12", "X13"]

FEATURE_POOLS = {
    "common": COMMON_POOL,
    "amp_error": COMMON_POOL + AMP_POOL,
    "freq_error": COMMON_POOL + FREQ_POOL,
    "ref_error": COMMON_POOL + REF_POOL,
    "normal": COMMON_POOL,
}
```

### 2.2 实际推理中的使用情况

**关键发现**: `utils/feature_pool.py` 中定义的 FEATURE_POOLS **未被推理代码实际使用**。

搜索 `from utils.feature_pool` 的结果显示:只有 `feature_pool.py` 自身有此导入语句(用于示例代码),**没有其他推理模块导入使用这些常量**。

### 2.3 实际的特征分流实现

特征分流**确实存在**,但实现在 BRB 模块内部,而非使用集中定义的特征池:

**系统级子BRB** (`BRB/system_brb_amp.py`, `system_brb_freq.py`, `system_brb_ref.py`):

```python
# BRB/system_brb_amp.py:42-56 - 幅度子BRB使用的特征
AMP_FEATURE_PARAMS = {
    'X1': (0.02, 0.5),      # 整体幅度偏移
    'X2': (0.002, 0.05),    # 带内平坦度
    'X5': (0.01, 0.35),     # 幅度缩放一致性
    'X6': (0.001, 0.03),    # 纹波
    'X7': (0.05, 2.0),      # 增益非线性
    'X10': (0.02, 0.5),     # 频段幅度一致性
    'X11': (0.01, 0.3),     # 包络超出率
    'X12': (0.5, 5.0),      # 最大包络违规
    'X13': (0.1, 10.0),     # 包络违规能量
    'X19': (1e-12, 1e-10),  # 低频段斜率
    'X20': (0.5, 5.0),      # 去趋势残差峰度
    'X21': (1, 20),         # 残差峰值数
    'X22': (0.1, 0.8),      # 残差主频能量占比
}
```

```python
# BRB/system_brb_freq.py:42-51 - 频率子BRB使用的特征
FREQ_FEATURE_PARAMS = {
    'X4': (5e5, 3e7),       # 频率标度非线性度
    'X8': (0.01, 1.0),      # 本振泄漏
    'X9': (1e3, 1e5),       # 调谐线性度残差
    'X14': (0.01, 1.0),     # 低频段残差均值
    'X15': (0.01, 0.5),     # 高频段残差标准差
    'X16': (0.001, 0.1),    # 互相关滞后/频移
    'X17': (0.001, 0.05),   # 频轴缩放因子
    'X18': (0.001, 0.05),   # 频轴平移因子
}
```

**模块级特征分流** (`BRB/module_brb.py:193-238`):

```python
# BRB/module_brb.py:194-210
if anomaly_type == "幅度失准":
    # 幅度模块：使用幅度相关特征X1,X2,X5,X11-X13,X19-X22
    amp_features = [
        md_step, md_ripple, md_gain_bias,
        normalize_feature(features.get("X11", 0.0), 0.01, 0.3),
        normalize_feature(features.get("X12", 0.0), 0.5, 5.0),
        # ... (其他幅度相关特征)
    ]
    result = _mean(amp_features)
```

### 2.4 特征池现状结论

| 项目 | 状态 |
|------|------|
| 集中式特征池定义 | **存在** (`utils/feature_pool.py`) |
| 实际推理中使用 | **未使用** |
| 分散式特征分流 | **存在** (在各子BRB中硬编码) |
| X1-X37 全量传递 | **是** (特征字典整体传入,内部按需提取) |

---

## 3. 软激活与门控逻辑

### 3.1 系统级分类器 - RandomForest 门控

**当前实现**: 使用 RandomForest 作为**门控先验**,与 BRB 输出融合。

```python
# methods/ours_adapter.py:200-244
if use_gating:
    if rf_classifier is not None and hasattr(rf_classifier, 'predict_proba'):
        # Prepare feature vector for RF
        feature_vector = _features_to_array(features)
        rf_proba = rf_classifier.predict_proba(feature_vector.reshape(1, -1))[0]
        
        # ... 映射概率到类别名称 ...
        
        # Fuse RF and BRB
        gating_config = _load_gating_prior_config()
        fusion = GatingPriorFusion(gating_config)
        fused_array = fusion.fuse(rf_array, brb_array)
```

### 3.2 是 argmax 还是保留概率向量?

**关键发现**: 系统级使用 **argmax 硬选择** 确定最终故障类型:

```python
# methods/ours_adapter.py:270
fault_type_pred = max(final_probs, key=final_probs.get)  # 等价于 argmax
```

但概率向量**被保留**用于模块级推理:

```python
# methods/ours_adapter.py:273-286
sys_probs_cn = {
    "正常": final_probs.get("normal", 0.0),
    "幅度失准": final_probs.get("amp_error", 0.0),
    "频率失准": final_probs.get("freq_error", 0.0),
    "参考电平失准": final_probs.get("ref_error", 0.0),
    # ...
}
```

### 3.3 子图激活逻辑 - 硬选择 vs 软加权

**模块级推理采用混合策略**:

**硬切(硬逻辑)存在** - 子图激活基于 fault_type:

```python
# BRB/module_brb.py:623-628
FAULT_TO_SUBGRAPH = {
    "freq_error": "LO_Clock_Network",
    "amp_error": "RF_IF_ADC_Network",
    "ref_error": "Calibration_Network",
    "normal": None
}
```

```python
# BRB/module_brb.py:693-706
subgraph = FAULT_TO_SUBGRAPH.get(fault_type, None)

if subgraph is None or fault_type == "normal":
    # Normal 样本，返回均匀分布
    return {m: uniform_prob for m in all_modules}

# 获取激活的板级
active_boards = SUBGRAPH_TO_BOARDS.get(subgraph, list(BOARD_MODULES.keys()))
```

**软加权也存在** - `hierarchical_module_infer_soft_gating`:

```python
# BRB/module_brb.py:906-938
if use_top2 and top2_ft:
    # Top-2 hypothesis
    probs_2 = hierarchical_module_infer(top2_ft, features, use_board_prior)
    
    # Weighted fusion: score(module) = P(t1)*score_t1(module) + P(t2)*score_t2(module)
    all_modules = set(probs_1.keys()) | set(probs_2.keys())
    total_weight = top1_prob + top2_prob
    fused_probs = {}
    for m in all_modules:
        p1 = probs_1.get(m, 0.0)
        p2 = probs_2.get(m, 0.0)
        fused_probs[m] = (top1_prob * p1 + top2_prob * p2) / total_weight
```

### 3.4 软激活现状结论

| 项目 | 实现方式 |
|------|----------|
| 系统级故障类型判定 | **argmax 硬选择** (`max(final_probs, key=...)`) |
| RF 与 BRB 融合 | **软加权** (GatingPriorFusion: logit 或 gated 融合) |
| 子图激活 | **硬切** (`FAULT_TO_SUBGRAPH` 映射) |
| 模块级多假设 | **软加权** (Top-2 假设加权融合, delta < 0.1 时激活) |

**注意**: 不存在 `amp_prob * ref_module_score` 这种乘法形式的软门控。实际实现是**先硬选择子图,再在子图内做概率计算**。

这种设计选择的原因:
1. **效率考虑**: 硬选择子图减少了需要计算的模块数量
2. **物理链路对应**: 子图激活直接对应物理故障传播路径
3. **兜底机制**: `hierarchical_module_infer_soft_gating` 提供了多假设软融合作为补充

---

## 4. BRB 内部结构

### 4.1 BRB 组合器实现

代码中存在两种 BRB 实现:

**SimpleBRB** (`BRB/utils.py:12-40`):
```python
class SimpleBRB:
    """
    现有的简化版 BRB 组合器：
    - 不区分不同规则的匹配度，仅用一个 matching_degrees 列表
    - 通过 (规则权重 × 匹配度乘积) 作为激活度，再对结论做加权平均
    """
    def infer(self, matching_degrees: List[float]) -> Dict[str, float]:
        activations = []
        for r in self.rules:
            act = r.weight * math.prod(matching_degrees)
            activations.append((act, r.belief))

        total = sum(a for a, _ in activations) + 1e-9
        out = {lab: 0.0 for lab in self.labels}

        for a, bel in activations:
            for lab in self.labels:
                out[lab] += (a / total) * bel.get(lab, 0.0)
        # 归一化输出
```

**ERBRB** (`BRB/utils.py:43-143`) - 更接近论文的 ER 递推合成:
```python
class ERBRB:
    """
    更接近论文 2.2 节"基于证据推理的递推合成"形式的 BRB 组合器。
    
    特点：
    - 每条规则有自己的匹配度向量 alpha_k^i
    - 计算规则激活权重 w_k
    - 显式引入"无知项" u_k = 1 - sum_j beta_{k,j}
    - 按规则逐条递推合成，并计算简化版冲突系数 K
    """
```

### 4.2 实际使用的 BRB 类

模块级推理主要使用 **SimpleBRB**:

```python
# BRB/module_brb.py:389-390
brb = SimpleBRB(MODULE_LABELS, rules)
result = brb.infer([md])
```

### 4.3 是否存在层(Layer)概念?

**答案: 不存在典型的分层递推结构**

代码中没有"上一层输出加到下一层输入"的逻辑。当前架构是:

1. **系统级推理**: 三个并行子BRB (amp/freq/ref),结果通过 softmax 聚合
2. **模块级推理**: 接收系统级结果作为先验,但不是递归输入

```python
# BRB/aggregator.py (系统级聚合) - 并行而非递推
from .system_brb_amp import amp_brb_infer
from .system_brb_freq import freq_brb_infer
from .system_brb_ref import ref_brb_infer

# 三个子BRB并行执行,结果聚合
amp_result = amp_brb_infer(features, alpha)
freq_result = freq_brb_infer(features, alpha)
ref_result = ref_brb_infer(features, alpha)
```

### 4.4 BRB 结构现状结论

| 项目 | 状态 |
|------|------|
| 主要组合器 | **SimpleBRB** (加权平均,非递推ER) |
| ER 递推版本 | **ERBRB 存在但未主要使用** |
| 层级结构 | **无** (系统→模块是顺序调用,非层级递推) |
| 子BRB架构 | **存在** (amp/freq/ref 三个并行子BRB) |

---

## 5. 审计总结

| 审计项 | 现状描述 |
|--------|---------|
| **双链路汇合** | 不完全一致。Path B 使用统一入口含 RF 门控,Path A 直接调用 BRB |
| **特征分池** | 定义存在但未被推理使用。实际分流在各子BRB内硬编码 |
| **软激活门控** | 混合模式: 系统级 argmax 硬选,RF-BRB 融合软加权,模块级子图硬切+多假设软融合 |
| **BRB 结构** | 扁平结构,使用 SimpleBRB 加权平均,有子BRB但无层级递推 |

---

## 附录: 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 统一推理入口 | `methods/ours_adapter.py` | 119-316 |
| 系统级 BRB | `BRB/system_brb.py` | 全文 |
| 子 BRB (幅度) | `BRB/system_brb_amp.py` | 全文 |
| 子 BRB (频率) | `BRB/system_brb_freq.py` | 全文 |
| 模块级 BRB | `BRB/module_brb.py` | 288-398 |
| 分层模块推理 | `BRB/module_brb.py` | 669-844 |
| 软门控模块推理 | `BRB/module_brb.py` | 847-953 |
| BRB 组合器 | `BRB/utils.py` | 12-143 |
| RF 门控融合 | `BRB/gating_prior.py` | 60-186 |
| 特征池定义 | `utils/feature_pool.py` | 50-82 |
