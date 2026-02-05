# Architecture Snapshot - 当前代码真实架构审计

生成时间: 2026-02-05
审计范围: methods/ours_adapter.py, BRB/module_brb.py, pipelines/compare_methods.py

---

## 1. 系统级推理架构

### 1.1 当前架构类型: **RF+BRB 门控融合 (Gating Prior Fusion)**

```
                    ┌──────────────┐
                    │   Features   │
                    │   X1-X22     │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │ RF Classifier   │       │ BRB System      │
    │ (Prior)         │       │ (Posterior)     │
    │ rf_probs        │       │ brb_probs       │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             └──────────┬──────────────┘
                        ▼
              ┌─────────────────┐
              │ GatingPriorFusion│
              │ fused_probs      │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ fault_type_pred │
              │ = argmax(fused) │
              └─────────────────┘
```

### 1.2 融合公式

位置: `BRB/gating_prior.py::GatingPriorFusion.fuse()`

```python
# 默认配置 (gating_prior_config.json)
alpha = 0.6  # BRB权重
beta = 0.4   # RF权重

fused = alpha * brb_probs + beta * rf_probs
fused = fused / sum(fused)  # 归一化
```

### 1.3 Fallback 逻辑

| 条件 | 结果 | gating_status |
|------|------|---------------|
| RF artifact 存在且加载成功 | RF+BRB融合 | "gated_ok" |
| RF artifact 不存在, allow_fallback=True | 仅BRB | "fallback_brb_only" |
| RF artifact 不存在, allow_fallback=False | 抛出异常 | - |
| use_gating=False | 仅BRB | "disabled" |

---

## 2. 模块级推理架构

### 2.1 当前架构类型: **分层BRB + 软门控 (Soft-Gating)**

```
              ┌─────────────────┐
              │ fused_probs     │
              │ (系统级概率)    │
              └────────┬────────┘
                       │
        ┌──────────────┴──────────────┐
        │ top1 - top2 < delta (0.1)?  │
        └──────────────┬──────────────┘
                       │
        ┌──────────YES─┴─NO────────────┐
        ▼                              ▼
┌───────────────────┐        ┌───────────────────┐
│ 激活 Top-2 假设   │        │ 仅激活 Top-1 假设 │
│ 运行两个子图      │        │ 运行一个子图      │
└────────┬──────────┘        └────────┬──────────┘
         │                            │
         ▼                            ▼
┌───────────────────┐        ┌───────────────────┐
│ 加权融合:         │        │ 直接使用:         │
│ score = Σ P(t)*   │        │ score = score_t1  │
│   score_t(module) │        │                   │
└────────┬──────────┘        └────────┬──────────┘
         │                            │
         └──────────────┬─────────────┘
                        ▼
              ┌─────────────────┐
              │ module_topk     │
              │ (模块排名)      │
              └─────────────────┘
```

### 2.2 子图选择逻辑: **Soft Gate (软门控)**

位置: `BRB/module_brb.py::hierarchical_module_infer_soft_gating()`

```python
MIN_FAULT_PROBABILITY = 0.01
delta = 0.1  # 软门控阈值

# 过滤 normal 和低概率假设
fault_hypotheses = [(ft, p) for ft, p in sorted_faults 
                    if ft != "normal" and p > MIN_FAULT_PROBABILITY]

top1_ft, top1_prob = fault_hypotheses[0]
top2_ft, top2_prob = fault_hypotheses[1] if len(fault_hypotheses) >= 2 else (None, 0)

# 软门控决策
if (top1_prob - top2_prob) < delta:
    # 激活两个假设，加权融合
    use_top2 = True
else:
    # 仅激活 top1
    use_top2 = False
```

### 2.3 Fallback 逻辑

| 条件 | 结果 |
|------|------|
| soft_gating 成功 | 返回 fused_topk |
| soft_gating 异常 | 回退到 module_level_infer_with_activation() |

---

## 3. Feature Pool 使用情况

### 3.1 当前状态: **Feature Pool 未在推理代码中显式分离**

| 组件 | 是否使用分池特征 | 说明 |
|------|------------------|------|
| RF Classifier | 否 | 使用完整 X1-X22 |
| BRB System | 否 | 使用完整特征字典 |
| BRB Module | 否 | 使用完整特征字典 |

### 3.2 特征列表 (X1-X22)

当前推理使用的特征索引:
- X1-X22: 全部22个特征
- 无显式 AMP_POOL / FREQ_POOL / REF_POOL 分离

**结论**: Feature Pool 仅存在于设计文档/构想中，尚未在推理代码中生效。

---

## 4. 推理入口调用链

### 4.1 三条链路统一入口

**唯一合法入口**: `methods/ours_adapter.py::infer_system_and_modules()`

| 链路 | 入口脚本 | 调用路径 |
|------|----------|----------|
| 右键链路 | compare_methods.py | OursAdapter.predict() → infer_system_and_modules() |
| 诊断链路1 | brb_diagnosis_cli.py | 直接调用 infer_system_and_modules() |
| 诊断链路2 | eval_module_localization.py | 直接调用 infer_system_and_modules() |

### 4.2 调用链详情

```
compare_methods.py
└── main()
    └── evaluate_method(OursAdapter)
        └── OursAdapter.predict(X_test)
            └── infer_system_and_modules(features)
                ├── system_level_infer()         # BRB系统级
                ├── load_rf_artifact()           # RF分类器
                ├── GatingPriorFusion.fuse()     # 融合
                └── hierarchical_module_infer_soft_gating()  # 模块级

brb_diagnosis_cli.py
└── main()
    └── _infer_one(input_path)
        └── infer_system_and_modules(features)
            └── (同上)

tools/eval_module_localization.py
└── main()
    └── evaluate_module_localization(samples)
        └── infer_system_and_modules(features)
            └── (同上)
```

---

## 5. 样本数差异来源分析

### 5.1 样本数裁剪路径

| 原始样本数 | 裁剪后 | 来源 | 裁剪规则 |
|-----------|--------|------|----------|
| 400 | 80 | compare_methods.py | train/val/test = 60%/20%/20% 划分，测试集 80 样本 |
| 400 | 300 | aggregate_batch / eval_module | 过滤 normal 样本 (100个)，仅评估故障样本 |
| 400 | 240/80/80 | compare_methods.py | 完整划分：train=240, val=80, test=80 |

### 5.2 裁剪代码位置

**compare_methods.py 的 test split**:
```python
# pipelines/compare_methods.py 约 line 660
indices = np.arange(n_samples)
train_idx, temp_idx = train_test_split(indices, train_size=0.6, stratify=y_sys, random_state=seed)
val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, stratify=y_sys[temp_idx], random_state=seed)
# 结果: train=60%, val=20%, test=20%
```

**aggregate_batch / eval_module 的 normal 过滤**:
```python
# tools/aggregate_batch_diagnosis.py
if fault_type == "normal" or not gt_module:
    continue  # 跳过 normal 样本
```

---

## 6. 标签与模块名规范化

### 6.1 系统级标签映射

| 中文标签 | 英文标签 (canonical) |
|----------|---------------------|
| 正常 | normal |
| 幅度失准 | amp_error |
| 频率失准 | freq_error |
| 参考电平失准 | ref_error |

### 6.2 模块名规范化

使用 `utils/canonicalize.py::canonical_module_v2()` 进行 V1→V2 映射。

### 6.3 模块匹配策略

所有链路统一使用: `utils/canonicalize.py::modules_match()`
- 策略: alias + contains (关键词匹配)

---

## 7. 总结

| 问题 | 答案 |
|------|------|
| 系统级架构 | RF+BRB 门控融合 (Gating Prior Fusion) |
| 模块级架构 | 分层BRB + 软门控 (Hierarchical BRB with Soft-Gating) |
| Feature Pool | 未在代码中生效，仅存在于设计文档 |
| 400→80 | compare_methods test split (20%) |
| 400→300 | 过滤 normal 样本 |
| 推理入口 | 统一使用 infer_system_and_modules() |
| 标签规范 | 统一使用 canonical_fault_type() / canonical_module_v2() |
