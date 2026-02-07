# FMFD — 频谱分析仪故障诊断系统

基于知识驱动规则优化与分层 BRB 推理的频谱分析仪故障诊断方法。

详细文档见 [docs/README.md](docs/README.md)。

## 快速开始：对比实验

按顺序运行以下三个脚本即可复现论文对比实验（IDE 中右键运行亦可）：

```bash
# 步骤 1: 生成基线
python pipelines/run_baseline.py

# 步骤 2: 生成仿真数据（自动提取特征，同时生成 features_brb.csv 和 labels.json）
python pipelines/simulate/run_simulation_brb.py

# 步骤 3: 运行对比实验
python pipelines/compare_methods.py
```

结果输出在 `Output/compare_methods/performance_table.csv`。

> **只需这三步，不需要额外运行其他文件。**
> 步骤 2 会自动提取特征并生成 `features_brb.csv`，无需手动运行特征提取。

### 可选：校准 "ours" 方法参数

在步骤 2 和步骤 3 之间，可运行校准脚本优化参数（通常可提升 5-10% 准确率）：

```bash
python pipelines/calibrate_ours.py --data_dir Output/sim_spectrum
```

### 可选：只运行部分方法

```bash
python pipelines/compare_methods.py --methods ours,hcf
```

## 快速开始：现场诊断

```bash
# 1. 生成基线
python pipelines/run_baseline.py

# 2. 批量检测（将待检CSV放入 to_detect/ 目录）
python pipelines/detect.py

# 3. 单样本诊断
python brb_diagnosis_cli.py -i to_detect/sample.csv -o result.json
```

## 许可证

MIT License
