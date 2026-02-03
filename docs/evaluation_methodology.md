# 评估方法差异说明 (Evaluation Methodology Difference)

## 问题背景

用户观察到两种评估结果存在显著差异：
- **compare_methods.py**: 系统级准确率 97.5%
- **brb_diagnosis_cli.py**: 系统级看起来不高

这不是造假，而是两种完全不同的评估范式。

## 评估方法对比

### 方法1: compare_methods.py (监督学习方法)

**工作原理**:
1. 从 `features_brb.csv` 加载22个预提取特征
2. 从 `labels.json` 加载真实标签
3. 划分数据集: 60%训练 / 20%验证 / 20%测试
4. 训练分类器（如决策树、随机森林等）
5. 在测试集上评估准确率

**特点**:
- ✅ 使用训练数据学习特征模式
- ✅ 特征是精心设计的，具有良好的可分性
- ✅ 测试集样本数量足够（80个样本）
- ⚠️ 需要训练数据
- ⚠️ 可能过拟合到特定数据分布

**输出文件**:
- `comparison_summary.json`: 所有方法的指标汇总
- `performance_table.csv`: 性能对比表
- `confusion_matrix_*.png`: 混淆矩阵图

### 方法2: brb_diagnosis_cli.py (BRB规则推理)

**工作原理**:
1. 输入单条幅频曲线
2. 提取特征
3. 应用 BRB (Belief Rule Base) 规则推理
4. 输出系统级和模块级诊断结果

**特点**:
- ✅ 无需训练数据
- ✅ 基于专家知识和物理规则
- ✅ 可解释性强
- ⚠️ 规则可能不够完善
- ⚠️ 对边界样本敏感

**输出文件**:
- `cli_trace_*.jsonl`: 诊断追踪日志（每行一个样本）
- 诊断结果JSON

## 为什么 cli_trace.jsonl 看起来准确率低？

### 原因1: 样本数量太少

`cli_trace_*.jsonl` 只记录了 **5条** 手动测试的记录，而不是完整的400个样本评估。

```bash
$ wc -l Output/diagnosis_audit/cli_trace_*.jsonl
5 Output/diagnosis_audit/cli_trace_20260201.jsonl
```

### 原因2: 测试样本选择偏差

手动测试时可能选择了边界样本或难样本来验证系统行为。

### 原因3: 不同的推理机制

- compare_methods: 基于统计学习的模式识别
- BRB诊断: 基于规则的逻辑推理

两种方法对同一样本可能给出不同结论。

## 如何获得 BRB 的真实准确率？

运行完整的 BRB 评估脚本，对所有400个样本进行诊断：

```bash
# 运行完整 BRB 评估
python tools/eval_module_metrics.py --input Output/sim_spectrum --output Output/brb_full_eval
```

## 结论

1. **两种方法都是有效的**，但评估范式不同
2. **没有造假**，97.5%是监督学习在测试集上的真实结果
3. **cli_trace只有5个样本**，不具有统计代表性
4. 建议进行完整的 BRB 评估以获得可比较的结果

## 数据一致性保证

两种方法使用相同的底层数据：
- 仿真曲线: `Output/sim_spectrum/raw_curves/`
- 标签文件: `Output/sim_spectrum/labels.json`
- 特征文件: `Output/sim_spectrum/features_brb.csv`

区别仅在于评估方法和推理机制。
