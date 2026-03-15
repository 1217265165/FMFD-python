# FMFD — 频谱分析仪故障模式与故障定位诊断系统

**Frequency-domain Measurement Fault Diagnosis (FMFD) v4**

基于 **先验门控随机森林 + 分层 BRB 推理 + P-CMA-ES 进化优化** 的两级故障诊断框架。

---

## 核心架构 (Architecture)

```
输入特征 X ──┬──▶ [RF 分类器] ──▶ P_rf (系统级概率)
             │
             └──▶ [分层 BRB]  ──▶ P_brb (系统级概率)
                       │
                       ├──▶ [门控融合] ──▶ P_fused = Gate(P_rf, P_brb) ──▶ 系统诊断结果
                       │      (Gated Fusion)
                       └──▶ [模块级 BRB] ──▶ 故障模块排名 (Top-K)
                              (Hierarchical Module Inference)
```

- **第一层 — 系统级 (System-Level)**：RF 提供高精度先验，BRB 提供物理机理似然，通过门控机制融合
- **第二层 — 模块级 (Module-Level)**：基于故障类型路由到对应子图，利用领域知识定位具体故障模块
- **参数优化**：P-CMA-ES（投影型协方差矩阵自适应进化策略）保证 BRB 参数的物理约束

---

## 目录结构 (Directory Tree)

```
FMFD-python/
├── BRB/                          # BRB 推理引擎核心
│   ├── gating_prior.py           #   门控先验融合 (Gated Fusion)
│   ├── module_brb.py             #   模块级分层推理 (Hierarchical Module Inference)
│   ├── system_brb.py             #   系统级 BRB 推理
│   ├── engines/layered_engine.py #   分层推理引擎
│   └── ...
├── methods/                      # 对比方法适配器
│   ├── ours_adapter.py           #   本方法 (RF + BRB 融合)
│   ├── hcf_adapter.py            #   HCF 基线
│   ├── dbrb_adapter.py           #   D-BRB 基线
│   ├── brb_mu_adapter.py         #   BRB-MU 基线
│   ├── brb_p_adapter.py          #   BRB-P 基线
│   └── a_ibrb_adapter.py         #   A-IBRB 基线
├── pipelines/                    # 流水线入口
│   ├── run_baseline.py           #   Step 1: 生成基线数据
│   ├── simulate/                 #   Step 2: 仿真数据生成
│   │   └── run_simulation_brb.py
│   ├── optimize_brb.py           #   Step 3: P-CMA-ES 参数优化
│   ├── compare_methods.py        #   Step 4: 方法对比评估
│   └── evaluate_baselines_module.py  # 模块级基线评估
├── features/                     # 特征提取与路由
├── baseline/                     # 基线构建模块
├── metrics/                      # 评估指标
├── config/                       # 配置文件 (BRB 规则、特征定义、门控参数)
├── tools/                        # 活跃工具 (仅保留 3 个核心工具)
│   ├── label_mapping.py          #   标签映射 (V1↔V2 模块名)
│   ├── module_validation.py      #   模块验证
│   └── check_features_integrity.py  # 特征完整性检查
├── utils/                        # 通用工具函数
├── tests/                        # 验证测试
├── normal_response_data/         # 真实测量数据 (基线源)
├── artifacts/                    # 训练产物 (RF 模型)
├── docs/                         # 技术文档
│   ├── README.md                 #   详细技术手册
│   ├── methods.md                #   对比方法说明
│   ├── brb_inference_formulation.md  # BRB 推理公式
│   ├── gating_prior_design.md    #   门控先验设计
│   └── comparison_methods_details.md # 对比方法实现细节
├── _legacy/                      # 归档的旧版代码和文档
├── brb_diagnosis_cli.py          # [关键] Qt 上位机诊断接口
├── baseline.py                   # 便捷入口: 运行基线
├── run_simulation_brb.py         # 便捷入口: 运行仿真
├── calibrate_ours.py             # 便捷入口: 参数校准
├── compare_methods.py            # 便捷入口: 方法对比
├── brb_rules.yaml                # BRB 规则定义
├── brb_chains_generated.yaml     # 生成的 BRB 推理链
├── thresholds.json               # 阈值配置
└── requirements.txt              # Python 依赖
```

---

## 核心文件清单 (File Glossary)

| 文件 | 职责 |
|------|------|
| `brb_diagnosis_cli.py` | **C++/Qt 上位机调用接口** — 接收 JSON 输入，执行单样本/批量诊断，返回系统级+模块级结果 |
| `BRB/gating_prior.py` | 门控先验融合类 `GatingPriorFusion` — 实现 RF 与 BRB 的概率融合 |
| `BRB/module_brb.py` | 分层模块推理 `hierarchical_module_infer()` — 按故障类型路由到子图进行模块定位 |
| `methods/ours_adapter.py` | 本方法适配器 — 训练 RF (仅 X1-X37 特征)，通过融合引擎预测 |
| `pipelines/compare_methods.py` | 学术对比评估 — 系统+模块级指标，自动泄漏检测，基线模块级平坦评估 |
| `pipelines/optimize_brb.py` | P-CMA-ES 优化器 — 投影型进化策略，优化 BRB 分层推理参数 |
| `config/gating_prior.json` | 门控融合配置 — 融合方法、权重范围、温度参数 |
| `config/module_taxonomy_v2.json` | 模块分类法 — 23 个 V2 模块定义 |
| `tools/label_mapping.py` | V1↔V2 模块标签映射 — 被 CLI、对比、优化脚本共同依赖 |

---

## 学术复现流 (Standard Execution Flow)

### 方式一：IDE 右键运行 (推荐)

依次右键运行以下文件：

```
1. baseline.py            → 生成基线特征 (Output/baseline_features)
2. run_simulation_brb.py  → 生成仿真数据 (Output/sim_spectrum)
3. compare_methods.py     → 运行方法对比 (Output/compare_methods)
```

### 方式二：命令行运行

```bash
# Step 1: 生成基线数据
python pipelines/run_baseline.py

# Step 2: 生成仿真数据 (400 样本)
python pipelines/simulate/run_simulation_brb.py --samples 400

# Step 3: P-CMA-ES 优化 BRB 参数 (可选，提升模块准确率)
python pipelines/optimize_brb.py \
  --data_dir Output/sim_spectrum \
  --output_dir Output/optimization_results \
  --supervised --generations 100 --population 24

# Step 4: 方法对比评估
python pipelines/compare_methods.py \
  --data_dir Output/sim_spectrum \
  --output_dir Output/compare_methods \
  --methods ours,hcf,dbrb,brb_mu,brb_p,a_ibrb \
  --load_params Output/optimization_results/best_params.json
```

### 工程诊断流 (Qt 上位机)

```bash
# 单样本诊断 (由 C++/Qt 前端调用)
python brb_diagnosis_cli.py --input <sample.json> --output <result.json>

# 批量诊断
python brb_diagnosis_cli.py --batch --input_dir <dir> --output_dir <dir>
```

---

## 当前性能 (Current Results)

| 方法 | 系统准确率 | 系统 Macro-F1 | 模块 Top-1 | 模块 Top-3 |
|------|-----------|--------------|-----------|-----------|
| **Ours** | **97.5%** | **98.7%** | **63.3%** | **88.3%** |
| HCF | 85.0% | 83.7% | 45.0% | 88.3% |
| BRB-MU | 55.0% | 52.0% | 53.3% | 91.7% |
| DBRB | 81.3% | 78.9% | 41.7% | 81.7% |
| A-IBRB | 58.8% | 42.0% | 21.7% | 51.7% |
| BRB-P | 38.8% | 29.1% | 8.3% | 30.0% |

---

## 依赖安装

```bash
pip install -r requirements.txt
```

## 详细文档

请参阅 [`docs/README.md`](docs/README.md) 获取完整的技术手册。

## 许可证

MIT License
