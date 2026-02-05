# 指标来源追溯报告 (Metrics Provenance)

生成时间: 2026-02-05T03:39:50.470284

## 数据路径

### labels.json
- 路径: `Output/sim_spectrum/labels.json`
- Hash: `821e28e311d1cc94`
- 样本数:
  - total: 100
  - normal: 25
  - amp_error: 25
  - freq_error: 25
  - ref_error: 25

### raw_curves/
- 路径: `Output/sim_spectrum/raw_curves`
- CSV 文件数: 100

## 门控配置 (Gating Config)
- 融合方法: `gated`
- 置信度阈值: 0.55
- RF权重范围: [0.3, 0.85]

## 统一推理入口
- 文件: `methods/ours_adapter.py`
- 函数: `infer_system_and_modules`
- 说明: All paths now use this single entry point

所有链路现在必须通过此入口：
- compare_methods.py (ours 分支)
- brb_diagnosis_cli.py
- aggregate_batch_diagnosis.py
- eval_module_localization.py

## 历史指标来源
未找到历史评估结果

## 复现命令

### D1. compare_methods.py
```bash
python compare_methods.py
```

### D2. diagnosis batch
```bash
python brb_diagnosis_cli.py --input_dir Output/sim_spectrum/raw_curves --labels Output/sim_spectrum/labels.json --output Output/diagnosis/batch
python aggregate_batch_diagnosis.py --labels Output/sim_spectrum/labels.json --pred_dir Output/diagnosis/batch
python eval_module_localization.py --labels Output/sim_spectrum/labels.json --pred_dir Output/diagnosis/batch
```

### D3. alignment debug
```bash
python tools/debug_single_sample_alignment.py --sample_id sim_00000 --labels Output/sim_spectrum/labels.json --curve_csv Output/sim_spectrum/raw_curves/sim_00000.csv
python tools/debug_alignment_batch20.py --labels Output/sim_spectrum/labels.json --curves_dir Output/sim_spectrum/raw_curves
```

## 样本过滤说明

- 评估跳过 `normal` 样本的模块级指标
- 冲突率 > 2% 的数据集会被拒绝
- 禁用模块 (DISABLED_MODULES) 不参与 TopK 统计