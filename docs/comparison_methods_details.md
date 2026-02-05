# 对比方法详细说明 (Comparison Methods Details)

本文档详细说明了 `compare_methods.py` 中每个对比方法的具体实现原理。

## 方法总览

| 方法名 | 论文出处 | 核心算法 | 学习类型 | 特征数 | 规则数 |
|--------|---------|---------|---------|--------|--------|
| **ours** | 本文提出 | RandomForest + 分层BRB | 监督学习 | 22 | 48 |
| hcf | Zhang 2022 | GMM聚类 + LogisticRegression | 监督学习 | 6 | 90 |
| brb_p | Ming 2023 | 概率表BRB + 约束优化 | 监督学习 | 15 | 81 |
| brb_mu | Feng 2024 | 多源不确定性融合 | 监督学习 | 6 | 72 |
| dbrb | Zhao 2024 | XGBoost特征重要性 + 3层BRB | 监督学习 | 动态 | 60 |
| a_ibrb | Wan 2025 | 区间BRB + CMA-ES优化 | 监督学习 | 5 | 50 |

---

## 1. HCF (Hierarchical Cognitive Framework) - Zhang 2022

### 核心思想
模拟人类认知过程的三层框架：
- **Level-a (特征认知)**: 选择主要/次要特征源
- **Level-b (模式认知)**: 对每个特征源进行GMM聚类
- **Level-c (气候认知)**: 融合编码特征进行分类

### 实现步骤

```python
# Level-a: 特征选择 (Fisher评分)
f_scores = f_classif(X_train, y_train)  # Fisher分数评估
primary_idx = np.argsort(f_scores)[-6:]  # 选择top-6特征

# Level-b: GMM聚类编码
for source_name, source_idx in sources.items():
    X_source = X_train[:, source_idx]
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_source)
    cluster_labels = gmm.predict(X_source)  # 聚类标签作为编码

# Level-c: 逻辑回归分类
classifier = LogisticRegression()
classifier.fit(X_encoded, y_train)  # 基于编码特征训练
```

### 特点
- 将特征按语义分组：amplitude, frequency, noise
- 每组使用GMM学习模式
- 最终使用逻辑回归融合

---

## 2. BRB-P (Probability-constrained BRB) - Ming 2023

### 核心思想
带概率约束的BRB优化：
1. **概率表初始化**: 从训练数据统计初始化置信度
2. **可解释性约束**: 限制参数偏离初始值的程度
3. **CMA-ES优化**: 在约束下优化参数

### 实现步骤

```python
# (1) 概率表初始化
for r in range(n_rules):
    y_neighborhood = y[in_neighborhood]  # 规则邻域内的样本
    for c in range(n_classes):
        beta_init[r, c] = count(y_neighborhood == c) / len(y_neighborhood)

# (2) 可解释性约束优化
# 目标函数: cross_entropy + λ1 * ||β - β_init||² + λ2 * semantic_penalty
for iteration in range(n_iter):
    loss = cross_entropy + 0.5 * ||beta - beta_init||² + 0.1 * semantic_penalty
    beta = beta - lr * gradient(loss)
```

### 特点
- 初始置信度来自数据分布
- 优化时保持与初始值接近（可解释性）
- 确保行和为1（语义约束）

---

## 3. BRB-MU (Multi-source Uncertainty Fusion) - Feng 2024

### 核心思想
多源特征的不确定性建模与融合：
- **多源分组**: 至少3个特征源
- **不确定性估计**: u_s = f(SNR, SVD)
- **加权融合**: w_s ∝ (1 - u_s)

### 实现步骤

```python
# (1) 定义特征源
sources = {
    'amplitude': [amp_idx],
    'frequency': [freq_idx],
    'noise': [noise_idx],
    'switching': [switch_idx]
}

# (2) 估计每个源的不确定性
for source_name, source_idx in sources.items():
    X_source = X_train[:, source_idx]
    
    # SNR组件: u_snr = 1 / (1 + SNR)
    snr = mean(|x|) / std(x)
    u_snr = 1 / (1 + snr)
    
    # SVD组件: u_svd = 1 - (σ₁ / Σσᵢ)
    U, S, Vt = svd(X_source)
    u_svd = 1 - S[0] / sum(S)
    
    u_combined = 0.6 * u_snr + 0.4 * u_svd  # 融合不确定性

# (3) 计算融合权重
weights[s] = (1 - u_s) / Σ(1 - u_j)

# (4) 预测时加权融合
final_prob = Σ weights[s] * source_prob[s]
```

### 特点
- 自动估计每个特征源的可靠性
- 低不确定性的源获得更高权重
- 使用高斯似然进行源内预测

---

## 4. DBRB (Deep BRB) - Zhao 2024

### 核心思想
深度BRB，使用XGBoost进行特征重要性排序，然后构建3层BRB网络：
- **Layer1**: 最重要的top特征 → z1
- **Layer2**: 次重要特征 + z1 → z2
- **Layer3**: 剩余特征 + z2 → 最终输出

### 实现步骤

```python
# (1) XGBoost特征重要性
xgb = XGBClassifier(n_estimators=50, max_depth=3)
xgb.fit(X_train, y_train)
importance = xgb.feature_importances_

# (2) 按重要性分层
sorted_idx = argsort(importance)[::-1]  # 降序
layer1_features = sorted_idx[:5]   # top-5
layer2_features = sorted_idx[5:10] # next-5
layer3_features = sorted_idx[10:]  # rest

# (3) 逐层训练
# Layer 1
X_layer1 = X_train[:, layer1_features]
layer1_model = train_gaussian_model(X_layer1, y_train)
z1 = predict_layer(X_layer1, layer1_model)  # 输出是概率向量

# Layer 2 (输入包含z1)
X_layer2 = hstack([X_train[:, layer2_features], z1])
layer2_model = train_gaussian_model(X_layer2, y_train)
z2 = predict_layer(X_layer2, layer2_model)

# Layer 3 (最终输出)
X_layer3 = hstack([X_train[:, layer3_features], z2])
layer3_model = train_gaussian_model(X_layer3, y_train)
```

### 特点
- 重要特征先处理，形成高层抽象
- 层间传递概率向量
- 类似残差网络的设计

---

## 5. A-IBRB (Automatic Interval BRB) - Wan 2025

### 核心思想
自动构建区间BRB：
1. **区间构建**: 误差约束的k-means++聚类
2. **规则生成**: 只为观察到的组合生成规则
3. **GIBM初始化**: 基于区间样本分布
4. **P-CMA-ES优化**: 约束参数调优

### 实现步骤

```python
# (1) 区间构建 (1D k-means++)
for feat_idx in range(n_features):
    centers = kmeans_1d(X[:, feat_idx], k)
    boundaries = [min] + midpoints(sorted_centers) + [max]

# (2) 区间规则生成
for sample in X:
    combo = tuple(interval_index(sample, feat) for feat in features)
    rules.add(combo)  # 只保留观察到的组合

# (3) GIBM置信度初始化
for rule_combo in rules:
    samples_in_rule = find_samples_in_interval(rule_combo)
    for c in n_classes:
        belief[rule, c] = count(y[samples_in_rule] == c) / len(samples_in_rule)

# (4) P-CMA-ES优化
for iteration in range(n_iter):
    loss = -log_likelihood + 0.3 * ||beliefs - beliefs_init||²
    beliefs = beliefs - lr * gradient(loss)
```

### 特点
- 区间数量自适应（基于重构误差）
- 只生成实际存在的规则组合（避免指数爆炸）
- 软匹配：区间外的样本使用高斯衰减

---

## 6. Ours (知识驱动规则压缩 + 分层BRB)

### 核心思想
两阶段混合架构：
- **阶段1 (系统级)**: RandomForest 实现高准确率分类
- **阶段2 (模块级)**: BRB分层推理提供可解释诊断

### 实现步骤

```python
# 阶段1: 系统级分类 (监督学习)
classifier = RandomForestClassifier(n_estimators=100, max_depth=10)
classifier.fit(X_train, y_sys_train)
sys_pred = classifier.predict(X_test)

# 阶段2: 模块级诊断 (BRB推理)
for sample in X_test:
    features = array_to_dict(sample)
    
    # 系统级结果作为条件
    sys_result = {
        'predicted_class': sys_labels[sys_pred[i]],
        'probabilities': sys_proba[i]
    }
    
    # 分层BRB推理 (基于系统类型激活不同子图)
    mod_probs = module_level_infer_with_activation(
        features, 
        sys_result, 
        only_activate_relevant=True  # 条件激活
    )
```

### 特点
- **RandomForest**: 100棵树，深度10，处理非线性关系
- **分层BRB**: 3个子BRB对应不同故障类型
- **条件激活**: 根据系统诊断结果激活对应子图

---

## 准确率对比

基于 400 个仿真样本的测试集（80 样本）评估：

| 方法 | 系统准确率 | Macro-F1 | 推理时间(ms/样本) |
|------|-----------|----------|------------------|
| **ours** | **95.0%** | **95.0%** | 2.5 |
| brb_mu | 76.3% | 76.5% | 0.8 |
| dbrb | 70.0% | 69.5% | 1.2 |
| hcf | 61.3% | 53.6% | 1.0 |
| a_ibrb | 43.8% | 39.7% | 0.6 |
| brb_p | 38.8% | 38.8% | 1.5 |

---

## 关键区别总结

| 特性 | ours | 其他方法 |
|------|------|---------|
| 系统级分类器 | RandomForest (强) | 高斯/逻辑回归 (弱) |
| 特征数量 | 22个知识引导特征 | 5-15个 |
| 模块级诊断 | 有 (BRB分层) | 无 |
| 条件推理 | 有 (子图激活) | 无 |
| 可解释性 | 系统级+模块级 | 仅系统级 |

---

## 代码位置

- `methods/ours_adapter.py` - 我们的方法
- `methods/hcf_adapter.py` - HCF
- `methods/brb_p_adapter.py` - BRB-P
- `methods/brb_mu_adapter.py` - BRB-MU
- `methods/dbrb_adapter.py` - DBRB
- `methods/a_ibrb_adapter.py` - A-IBRB
