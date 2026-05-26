---
name: skill-svd
description: Apply SVD to real problems including compression, denoising, recommendations, and least-squares solving
phase: 1
lesson: 11
---

你是一位将奇异值分解（SVD）应用于实际工程问题的专家。当遇到涉及矩阵、数据压缩、噪声、缺失数据或线性系统的任务时，判断 SVD 是否是合适的工具，并说明如何应用它。

## 决策框架

### 第一步：确定问题类型

- **数据压缩 / 降维**：使用截断 SVD。保留前 k 个奇异值。通过能量阈值（常用 95%）或下游任务性能来选择 k。
- **去噪**：计算完整 SVD。在奇异值谱中寻找间隙。在间隙下方截断。间隙将信号与噪声分开。
- **缺失数据 / 推荐**：填充缺失条目（行均值或零），计算 SVD，用低秩重构填充。在生产环境中使用原生处理缺失数据的 ALS 或增量 SVD。
- **最小二乘 / 伪逆**：计算 SVD。对非零奇异值求逆。用 V Sigma⁺ U^T 乘以目标向量。比正规方程更稳定。
- **文本相似性 / 主题建模**：构建词-文档矩阵。应用 SVD（这就是 LSA/LSI）。将文档和词投影到低秩空间。使用余弦相似度进行比较。
- **数值秩确定**：计算 SVD。统计超过阈值（相对于最大奇异值）的奇异值数量。这比行化简更可靠。
- **矩阵范数计算**：谱范数 = 最大奇异值。Frobenius 范数 = 奇异值平方和的平方根。核范数 = 奇异值之和。
- **条件数**：sigma_max / sigma_min。告诉你系统对扰动的敏感程度。

### 第二步：选择正确的变体

| 情况 | 方法 | 原因 |
|------|------|------|
| 稠密矩阵，需要完整分解 | `np.linalg.svd(A)` / Julia 中的 `svd(A)` | 标准算法，数值稳定 |
| 只需要前 k 个分量 | `scipy.sparse.linalg.svds(A, k)` | k 小时比完整 SVD 更快 |
| 稀疏矩阵 | `scipy.sparse.linalg.svds` | 高效处理稀疏存储 |
| 流式数据 | 增量 SVD / 在线 SVD | 无需从头重新计算即可更新分解 |
| 缺失数据（推荐） | ALS、Funk SVD 或 NMF | 标准 SVD 需要完整矩阵 |
| 超大矩阵（数百万行） | 随机 SVD（`sklearn.utils.extmath.randomized_svd`） | O(mn log k) 而非 O(mn min(m,n)) |
| 中心化数据的 PCA | 对中心化数据矩阵的 SVD | 等价于协方差矩阵的特征分解，但更稳定 |

### 第三步：选择秩 k

- **能量阈值**：计算累计能量 = sum(sigma_1^2 ... sigma_k^2) / sum(所有 sigma^2)。当能量超过 0.95 时停止（高精度任务用 0.99）。
- **间隙检测**：绘制奇异值。寻找急剧下降。间隙指示信号与噪声之间的边界。
- **交叉验证**：对于下游任务，扫描 k 并测量在留出数据上的性能。
- **肘部法**：绘制重构误差与 k 的关系图。拐点处添加更多分量不再有帮助。
- **领域知识**：如果你知道数据有 d 个底层因子，使用 k = d。

### 第四步：验证结果

- **重构误差**：计算 ‖A - A_k‖ / ‖A‖。如果截断有意义，应该很小。
- **解释方差**：对于 PCA/压缩，报告捕获的总方差（能量）比例。
- **下游任务性能**：如果 SVD 是预处理步骤，测量端到端指标。
- **目视检查**：对于图像，比较原始图像和重构图像。对于推荐，检查预测与已知评分的对比。

## 常见错误

- 通过 A^T A 的特征分解来计算 SVD。这会将条件数平方并损失数值精度。使用专用 SVD 例程。
- 当只需要前 k 个分量时使用完整 SVD。对于大矩阵，使用截断或随机 SVD。
- 直接对有缺失条目的矩阵应用 SVD。标准 SVD 需要完整矩阵。使用矩阵补全方法（ALS、Funk SVD）。
- 忽略中心化。对于 PCA，数据必须先中心化（减去均值）。不中心化的话，第一分量捕获的是均值而非方差。
- 过度截断。如果你保留的奇异值太少，会丢失信号。如果你保留太多，会保留噪声。使用能量阈值或交叉验证。
- 混淆 SVD 与特征分解。SVD 适用于任意矩阵（任意形状、任意秩）。特征分解需要方阵且需要一组完整的特征向量。对于对称半正定矩阵，两者相同。

## 代码模式

### 快速压缩
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
k = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.95) + 1
A_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 用于最小二乘的伪逆
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.array([1/s if s > 1e-10 else 0 for s in S])
x = Vt.T @ np.diag(S_inv) @ U.T @ b
```

### 去噪
```python
U, S, Vt = np.linalg.svd(noisy_data, full_matrices=False)
k = find_gap(S)
clean_data = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 大规模 PCA
```python
from sklearn.utils.extmath import randomized_svd
U, S, Vt = randomized_svd(X_centered, n_components=50, random_state=42)
explained_variance = S**2 / (n_samples - 1)
```

## 何时不使用 SVD

- 矩阵非常稀疏且你只需要几个分量。直接使用稀疏特征求解器。
- 你需要非负因子（主题建模、光谱解混）。改用 NMF。
- 数据有强非线性结构，线性方法无法捕获。使用自编码器或流形学习。
- 你需要流式数据的实时更新且矩阵不断变化。使用增量/在线 SVD 或近似方法。
- 矩阵在内存中但太大，即使随机 SVD 也太慢。考虑草图方法或采样方法。

## 计算成本

| 方法 | 时间 | 空间 |
|------|------|------|
| m×n 矩阵的完整 SVD | O(mn min(m,n)) | O(mn) |
| 截断 SVD（前 k 个） | O(mnk) | O((m+n)k) |
| 随机 SVD（前 k 个） | O(mn log k) | O((m+n)k) |
| 幂迭代（1 个向量） | O(mn * iters) | O(m+n) |

对于 10000×5000 的矩阵：
- 完整 SVD：约 2500 亿次运算
- 截断 SVD（k=50）：约 25 亿次运算
- 随机 SVD（k=50）：约 5 亿次运算

选择与你的规模和精度要求相匹配的方法。