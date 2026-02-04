# 评估口径契约 (Evaluation Consistency Contract)

## 1. 目的

确保系统级与模块级指标计算的一致性，避免：
- compare 表与模块验证报告的指标不一致
- mod_top1/mod_top3 出现 0 但实际有值的情况
- 标签映射不一致导致的误差

## 2. 数据源约定

### 2.1 系统级标签
- 字段: `labels.json` 中的 `system_fault_class`
- 取值: `normal`, `amp_error`, `freq_error`, `ref_error`
- 映射: 直接使用，无需转换

### 2.2 模块级标签
- 字段: `labels.json` 中的 `module_v2`（优先）或 `module_cause`
- 映射: 通过 `config/module_v1_to_v2.json` 统一

### 2.3 样本 ID
- 格式: `sim_XXXXX`（不含 `.csv` 后缀）
- 匹配: labels 与 predictions 必须使用相同的 sample_id

## 3. 指标计算规则

### 3.1 系统级指标

| 指标 | 计算方式 |
|------|---------|
| sys_acc | `correct / total` |
| macro_f1 | `mean(F1_per_class)` |
| confusion | 4x4 矩阵 |

### 3.2 模块级指标

| 指标 | 计算方式 |
|------|---------|
| mod_top1 | `sum(gt in top1) / valid_samples` |
| mod_top3 | `sum(gt in top3) / valid_samples` |
| mrr | `mean(1/rank_of_gt)` |

**注意**:
- `valid_samples` = 排除 `system=normal` 的样本
- 标签映射冲突样本单独统计

### 3.3 冲突样本处理

若 `module_cause` 与 `module_v2` 不一致：
1. 优先使用 `module_v2`
2. 记录为冲突样本
3. 在报告中输出: `mod_top1_all` 和 `mod_top1_clean`

## 4. 输出结构

### 4.1 compare_methods 输出

```json
{
  "method": "ours",
  "sys_acc": 0.913,
  "macro_f1": 0.912,
  "mod_top1": 0.467,
  "mod_top3": 0.767,
  "n_samples": 400,
  "n_valid_module_samples": 300
}
```

### 4.2 diagnosis_result.json 输出

```json
{
  "sample_id": "sim_00001",
  "system": {
    "rf_probs": [0.1, 0.7, 0.1, 0.1],
    "brb_probs": [0.2, 0.5, 0.2, 0.1],
    "fused_probs": [0.15, 0.6, 0.15, 0.1],
    "pred_class": "amp_error",
    "fusion_weight": 0.72
  },
  "module": {
    "topk": [
      ["[RF板][IF] 中频放大器", 0.35],
      ["[RF板][ADC] 模数转换", 0.25],
      ["[RF板][Mixer1] 第一混频器", 0.15]
    ],
    "fault_type": "amp_error"
  }
}
```

## 5. 一致性检查

### 5.1 自检项

- [ ] compare 表的 mod_top1 与 `eval_module_localization.py` 输出一致
- [ ] 同一批样本上，两个脚本的 top1_hit 完全相同
- [ ] 标签映射冲突样本数量一致

### 5.2 验证命令

```bash
# 运行 compare
python compare_methods.py

# 运行模块验证
python tools/eval_module_localization.py

# 对比两个输出的 mod_top1/mod_top3
```

## 6. 变更历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-02-04 | v1.0 | 初始版本 |
