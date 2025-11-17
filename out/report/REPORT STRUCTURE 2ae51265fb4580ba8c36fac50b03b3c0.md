# REPORT STRUCTURE

### 1. Abstract

**内容要点：**

- 简要背景：Kepler KOIs、false positives、Robovetter、`koi_score`。
- 目标：用 PCA + ML 回归来近似预测 `koi_score`。
- 方法：数据集、PCA、Linear Regression、MLP（可提一句还有其它模型的话）。
- 主要结果：哪类模型效果最好，PCA 是否有帮助。
- 一句话总结意义：提供了一个轻量级的 KOI 质量估计工具。

---

### 2. Introduction

**应包含：**

1. **科学背景**
    - Kepler 任务、KOIs 是什么、false positives 的问题。
2. **`koi_score` 的角色**
    - 真正定义：它如何表示 vetting 置信度，如何用于选择高质量候选。
3. **质量工程视角**
    - 把 `koi_score` 看成“质量/可靠性指标”，KOI 特征看成“过程特征”，非常贴合课程主题。
4. **研究动机 & 问题定义**
    - Robovetter & scoring 计算昂贵且非公开；
    - 目标：用 PCA + ML 预测 `koi_score`，构建 surrogate。
5. **贡献概述**
    - 使用 KOI 数据建立 PCA + 回归模型；
    - 比较线性 vs 非线性模型（Linear Regression vs MLP）；
    - 分析 PCA 对模型性能与解释性的影响。

---

### 3. Data Description

**应包含：**

1. **数据源说明**
    - NASA Exoplanet Archive KOI cumulative table；
    - 总样本数（~9564），使用多少条（清洗后）。
2. **主要变量**
    - 列出关键特征（period, depth, duration, SNR, stellar params 等）及其物理意义；
    - 解释 `koi_score`，以及为什么不把 `koi_disposition` / `koi_pdisposition` 作为特征。
3. **数据清洗**
    - 缺失值处理策略；
    - 异常值过滤；
    - 最终保留的特征数量与样本数量。
4. **基本统计 / 可视化（可选）**
    - `koi_score` 的分布；
    - 一些特征的直方图 / 相关矩阵。

---

### 4. Methodology

可以分成两个小 subsection：

### 4.1 Principal Component Analysis

- 简述 PCA 理论（协方差矩阵、特征值分解、主成分）。
- 说明：
    - 对哪些标准化后的特征做 PCA；
    - 如何选择主成分数量（解释方差比例）。
- 提到 PCA 的目的：
    - 降维；
    - 去除相关性；
    - 提供可解释的“综合物理方向”。

### 4.2 Regression and Machine Learning Models

- 介绍你要用的模型：
    - Linear Regression（基线）；
    - MLP（以及是否加其它模型，如 Random Forest / Gradient Boosting）。
- 数据划分策略：
    - Training / Test split（例如 80/20 或 90/10）；
    - 是否使用 cross-validation。
- 特征设置：
    - 原始特征 vs PCA 主成分的两种输入方案。
- 评价指标：
    - R²、MSE、RMSE、MAE 等。

---

### 5. Experimental Setup

（也可以并入 Methodology，视篇幅而定）

**应包含：**

- 具体实现细节：
    - 使用的 Python 库（scikit-learn 等）；
    - PCA 保留的主成分数量；
    - MLP 的网络结构（层数、节点数、激活函数）；
    - 训练参数（学习率、epoch 数、early stopping 等）。
- 训练/测试划分比例、随机种子；
- 如有的话：超参数选择方法（简单 grid search / 手动调参）。

---

### 6. Results

**应包含：**

1. **PCA 结果**
    - Scree plot & explained variance；
    - 前几个主成分的载荷表（或图）；
    - 对主成分的物理解释。
2. **回归模型表现**
    - 表格比较：
        - Linear Regression (Original) vs Linear Regression (PCA)
        - MLP (Original) vs MLP (PCA)
        - 如有其它模型，一并列出。
    - 测试集上的 R²、MSE、RMSE 等。
3. **可视化**
    - 真实 `koi_score` vs 预测 `koi_score` 的散点图；
    - 误差分布图（residuals）。

---

### 7. Discussion

**应包含：**

- 对结果的解读：
    - 哪个模型在近似 `koi_score` 上表现最好？
    - PCA 是否提升了性能，还是只是带来更好的解释性？
- 对“质量控制”角度的分析：
    - 模型在高 `koi_score` 区间（高质量候选）是否更准确？
    - 是否可以用模型输出作为快速淘汰低质量 KOI 的工具？
- 局限性：
    - `koi_score` 本身的噪声和不确定性；
    - 模型对训练样本分布的依赖；
    - 未用到的更复杂特征（如像元级信息）等。

---

### 8. Conclusion and Future Work

**应包含：**

- 总结：
    - 再次回顾：
        - 为什么选择 KOI & `koi_score`；
        - PCA + Linear Regression + MLP 的主要发现。
- 主要结论：
    - 是否能够较好地近似 `koi_score`；
    - 哪些特征/主成分最重要；
    - PCA & 非线性模型在质量评估中的作用。
- Future work：
    - 引入更多特征（比如 false positive flags、更多 stellar/planet parameters）；
    - 尝试更强的模型（Gradient Boosting, XGBoost）；
    - 扩展到“直接分类高/低质量 KOI”或“类地行星筛选”。

---

### 9. References

- 列出你引用的：
    - Kepler/Robovetter/koi_score 相关论文；
    - NASA Exoplanet Archive 文档；
    - PCA / Regression / MLP 的教科书或论文。