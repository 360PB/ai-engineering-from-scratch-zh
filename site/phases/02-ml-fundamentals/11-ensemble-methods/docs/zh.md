# 集成方法

> 一组弱学习器，正确组合，就成为一个强学习器。这不是比喻。这是一个定理。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 2, Lesson 10（偏差-方差权衡）
**时间：** 约 120 分钟

## 学习目标

- 从零实现 AdaBoost 和梯度提升，解释 Boosting 如何顺序降低偏差
- 构建 Bagging 集成，并证明对去相关模型取平均如何在不增加偏差的情况下降低方差
- 比较 Bagging、Boosting 和 Stacking 在各方法针对哪种误差分量上的差异
- 评估集成多样性，并解释为什么多数投票精度随着更多独立弱学习器而提高

## 问题

单个决策树训练快速且易于解释，但它会过拟合。单线性模型在复杂边界上欠拟合。你可能花费数天工程化完美的模型架构。或者你可以组合一堆不完美的模型，得到比任何单独模型都更好的结果。

集成方法正是做这个的。它们是在表格数据上赢得 Kaggle 竞赛最可靠的技术，为大多数生产 ML 系统提供动力，并且展示了偏差-方差权衡的实际运作。Bagging 降低方差。Boosting 降低偏差。Stacking 学习在哪些输入上信任哪些模型。

## 概念

### 为什么集成有效

假设你有 N 个独立分类器，每个准确率 p > 0.5。多数投票的准确率为：

```
P(多数正确) = sum over k > N/2 of C(N,k) * p^k * (1-p)^(N-k)
```

对于 21 个准确率各为 60% 的分类器，多数投票准确率约为 74%。有 101 个分类器时，上升到 84%。当模型犯不同错误时，误差相互抵消。

关键要求是**多样性**。如果所有模型犯同样的错误，组合它们毫无帮助。集成有效是因为它们通过以下方式产生多样化模型：

- 不同的训练子集（Bagging）
- 不同的特征子集（随机森林）
- 顺序误差修正（Boosting）
- 不同的模型家族（Stacking）

### Bagging（Bootstrap 聚合）

Bagging 通过在不同的 bootstrap 样本上训练每个模型来创造多样性。

```mermaid
flowchart TD
    D[训练数据] --> B1[Bootstrap 样本 1]
    D --> B2[Bootstrap 样本 2]
    D --> B3[Bootstrap 样本 3]
    D --> BN[Bootstrap 样本 N]

    B1 --> M1[模型 1]
    B2 --> M2[模型 2]
    B3 --> M3[模型 3]
    BN --> MN[模型 N]

    M1 --> V[平均或多数投票]
    M2 --> V
    M3 --> V
    MN --> V

    V --> P[最终预测]
```

Bootstrap 样本是从原始数据有放回抽取的，大小与原始数据相同。每个 bootstrap 大约包含 63.2% 的唯一样本。剩余的 36.8%（袋外样本）提供了一个免费的验证集。

Bagging 在不大幅增加偏差的情况下降低方差。每个单独的树都过拟合其 bootstrap 样本，但每棵树的过拟合不同，所以平均抵消了噪声。

**随机森林**是 Bagging 加一个额外技巧：在每个分裂处，只考虑特征的随机子集。这强制树之间产生更多多样性。典型的候选特征数量：分类用 `sqrt(n_features)`，回归用 `n_features / 3`。

### Boosting（顺序误差修正）

Boosting 顺序训练模型。每个新模型专注于前一个模型犯错的样本。

```mermaid
flowchart LR
    D[带权重的数据] --> M1[模型 1]
    M1 --> E1[找出错误]
    E1 --> W1[增加错误样本的权重]
    W1 --> M2[模型 2]
    M2 --> E2[找出错误]
    E2 --> W2[增加错误样本的权重]
    W2 --> M3[模型 3]
    M3 --> F[所有模型的加权和]
```

Boosting 降低偏差。每个新模型修正到目前为止集成的系统性错误。最终预测是所有模型的加权和，表现更好的模型获得更高权重。

权衡：Boosting 如果运行太多轮可能会过拟合，因为它不断拟合更难的样本，其中一些可能是噪声。

### AdaBoost

AdaBoost（自适应提升）是第一个实用的 Boosting 算法。它可以与任何基学习器一起工作，通常使用决策树桩（深度为 1 的树）。

算法：

```
1. 初始化样本权重：w_i = 1/N 对所有 i

2. 对于 t = 1 到 T：
   a. 在加权数据上训练弱学习器 h_t
   b. 计算加权误差：
      err_t = sum(w_i * I(h_t(x_i) != y_i)) / sum(w_i)
   c. 计算模型权重：
      alpha_t = 0.5 * ln((1 - err_t) / err_t)
   d. 更新样本权重：
      w_i = w_i * exp(-alpha_t * y_i * h_t(x_i))
   e. 归一化权重使和为 1

3. 最终预测：H(x) = sign(sum(alpha_t * h_t(x)))
```

误差更低的模型获得更高的 alpha。错误分类的样本获得更高权重，这样下一个模型会专注于它们。

### 梯度提升

梯度提升将 Boosting 泛化到任意损失函数。不是重新加权样本，而是将每个新模型拟合到当前集成的残差（损失的负梯度）。

```
1. 初始化：F_0(x) = argmin_c sum(L(y_i, c))

2. 对于 t = 1 到 T：
   a. 计算伪残差：
      r_i = -dL(y_i, F_{t-1}(x_i)) / dF_{t-1}(x_i)
   b. 将树 h_t 拟合到残差 r_i
   c. 找到最优步长：
      gamma_t = argmin_gamma sum(L(y_i, F_{t-1}(x_i) + gamma * h_t(x_i)))
   d. 更新：
      F_t(x) = F_{t-1}(x) + learning_rate * gamma_t * h_t(x)

3. 最终预测：F_T(x)
```

对于平方误差损失，伪残差就是实际残差：`r_i = y_i - F_{t-1}(x_i)`。每棵树实际上拟合前一个集成的误差。

学习率（收缩）控制每棵树的贡献程度。更小的学习率需要更多树但泛化更好。典型值：0.01 到 0.3。

### XGBoost：为什么它主导表格数据

XGBoost（极端梯度提升）是带工程优化的梯度提升，使其快速、准确且抗过拟合：

- **正则化目标：** 叶权重的 L1 和 L2 惩罚，防止单棵树过于自信
- **二阶近似：** 使用损失的一阶和二阶导数，给出更好的分裂决策
- **稀疏感知分裂：** 原生处理缺失值，在每个分裂处学习缺失数据的最佳方向
- **列采样：** 像随机森林一样，在每个分裂处采样特征以增加多样性
- **加权分位数草图：** 在分布式数据上高效寻找连续特征的分裂点
- **缓存感知块结构：** 为 CPU 缓存行优化的内存布局

对于表格数据，XGBoost（及其后继者 LightGBM）持续优于神经网络。这不会很快改变。如果你的数据可以用行和列放入表格，从梯度提升开始。

### Stacking（元学习）

Stacking 使用多个基模型的预测作为元学习器的特征。

```mermaid
flowchart TD
    D[训练数据] --> M1[模型 1: 随机森林]
    D --> M2[模型 2: SVM]
    D --> M3[模型 3: 逻辑回归]

    M1 --> P1[预测 1]
    M2 --> P2[预测 2]
    M3 --> P3[预测 3]

    P1 --> META[元学习器]
    P2 --> META
    P3 --> META

    META --> F[最终预测]
```

元学习器学习在哪些输入上信任哪个基模型。如果随机森林在某些区域更好而 SVM 在其他区域更好，元学习器会学会相应地路由。

为避免数据泄露，基模型预测必须通过在训练集上的交叉验证生成。你永远不会在相同数据上训练基模型并生成元特征。

### 投票

最简单的集成。直接组合预测。

- **硬投票：** 对类别标签进行多数投票。
- **软投票：** 平均预测概率，选择平均概率最高的类别。通常更好，因为它使用了置信度信息。

## 构建

### 步骤 1：决策树桩（基学习器）

`code/ensembles.py` 中的代码从头实现一切。我们从一个决策树桩开始：一棵只有一次分裂的树。

```python
class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float("inf")

        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for thresh in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    pred[polarity * X[:, f] < polarity * thresh] = -1
                    error = np.sum(weights[pred != y])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = f
                        self.threshold = thresh
                        self.polarity = polarity

    def predict(self, X):
        n = X.shape[0]
        pred = np.ones(n)
        idx = self.polarity * X[:, self.feature_idx] < self.polarity * self.threshold
        pred[idx] = -1
        return pred
```

### 步骤 2：从零实现 AdaBoost

```python
class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.full(n, 1 / n)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            pred = stump.predict(X)

            err = np.sum(weights[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)
            weights *= np.exp(-alpha * y * pred)
            weights /= weights.sum()

            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        total = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.sign(total)
```

### 步骤 3：从零实现梯度提升

```python
class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        current_pred = np.full(len(y), self.initial_pred)

        for _ in range(self.n_estimators):
            residuals = y - current_pred
            tree = SimpleRegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            current_pred += self.lr * update
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
```

### 步骤 4：与 sklearn 比较

代码验证我们的从头实现产生的准确率与 sklearn 的 `AdaBoostClassifier` 和 `GradientBoostingClassifier` 相近，并比较所有方法的并排表现。

## 使用

### 何时使用每种方法

| 方法 | 降低 | 最适合 | 注意事项 |
|------|------|--------|---------|
| Bagging / 随机森林 | 方差 | 噪声数据，特征多 | 不能帮助偏差 |
| AdaBoost | 偏差 | 干净数据，简单基学习器 | 对异常值和噪声敏感 |
| 梯度提升 | 偏差 | 表格数据，竞赛 | 训练慢，调参不当易过拟合 |
| XGBoost / LightGBM | 两者 | 生产表格 ML | 超参数多 |
| Stacking | 两者 | 追求最后 1-2% 准确率 | 复杂，元学习器有过拟合风险 |
| 投票 | 方差 | 快速组合多样化模型 | 只有模型多样化时才有效 |

### 表格数据的生产栈

对于大多数表格预测问题，按此顺序尝试：

1. **LightGBM 或 XGBoost** 使用默认参数
2. 调参 n_estimators、learning_rate、max_depth、min_child_weight
3. 如果需要最后 0.5%，用 3-5 个多样化模型构建 Stacking 集成
4. 全程使用交叉验证

神经网络在表格数据上几乎总是比梯度提升差，尽管持续有研究尝试。TabNet、NODE 及类似架构偶尔匹配，但很少能打败调优良好的 XGBoost。

## 上线

本课产出 `outputs/prompt-ensemble-selector.md` -- 一个帮助你为给定数据集选择正确集成方法的提示。描述你的数据（大小、特征类型、噪声水平、类别平衡）和你要解决的问题。提示会引导你完成决策检查表，推荐一种方法，建议起始超参数，并警告该方法的常见错误。还产出 `outputs/skill-ensemble-builder.md`，其中包含完整的选型指南。

## 练习

1. 修改 AdaBoost 实现以追踪每轮后的训练准确率。绘制准确率 vs 估计器数量。它何时收敛？

2. 通过添加随机特征子采样从头实现随机森林。用 `max_features=sqrt(n_features)` 训练 100 棵树并平均预测。比较与单棵树的方差减少。

3. 在梯度提升实现中添加早停：追踪每轮后的验证损失，当连续 10 轮没有改善时停止。它实际需要多少棵树？

4. 构建一个 Stacking 集成，包含三个基模型（逻辑回归、决策树、K 近邻）和一个逻辑回归元学习器。使用 5 折交叉验证生成元特征。与每个单独的基模型比较。

5. 在相同数据集上使用默认参数运行 XGBoost。与从头实现的梯度提升比较准确率。两者都计时。速度差异有多大？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Bagging | "在随机子集上训练" | Bootstrap 聚合：在 bootstrap 样本上训练模型，平均预测以降低方差 |
| Boosting | "专注于难样本" | 顺序训练模型，每个修正集成到目前为止的错误，以降低偏差 |
| AdaBoost | "重新加权数据" | 通过样本权重更新进行提升；错误分类的点为下一个学习器获得更高权重 |
| 梯度提升 | "拟合残差" | 通过将每个新模型拟合到损失函数的负梯度来进行提升 |
| XGBoost | "Kaggle 武器" | 带正则化、二阶优化和系统级加速技巧的梯度提升 |
| Stacking | "模型之上的模型" | 使用基模型的预测作为元学习器的输入特征 |
| 随机森林 | "许多随机化的树" | 用决策树的 Bagging，在每个分裂处添加随机特征子采样以增加多样性 |
| 集成多样性 | "犯不同错误" | 模型在错误上必须不相关，集成才能改进个体 |
| 袋外误差 | "免费验证" | 不在 bootstrap 抽取中的样本（约 36.8%）用作验证集，无需留出法 |

## 扩展阅读

- [Schapire & Freund: Boosting: Foundations and Algorithms](https://mitpress.mit.edu/9780262526036/) -- AdaBoost 创建者的著作
- [Friedman: Greedy Function Approximation: A Gradient Boosting Machine (2001)](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) -- 原始梯度提升论文
- [Chen & Guestrin: XGBoost (2016)](https://arxiv.org/abs/1603.02754) -- XGBoost 论文
- [Wolpert: Stacked Generalization (1992)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) -- 原始 Stacking 论文
- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) -- 实践参考