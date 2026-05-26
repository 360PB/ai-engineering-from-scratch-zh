import numpy as np


def gaussian_elimination(A, b):
    """带部分选主元的高斯消元

    将增广矩阵 [A|b] 化简为上三角形式，然后回代求解。
    时间复杂度：O(n^3)
    """
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        # 部分选主元：找到当前列中绝对值最大的元素所在行
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(f"矩阵在第 {k} 个枢轴处奇异或近乎奇异")

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i + 1 : n] @ x[i + 1 : n]) / Ab[i, i]

    return x


def lu_decompose(A):
    """LU 分解（带部分选主元）

    计算 PA = LU，其中 P 是置换矩阵，L 是下三角，U 是上三角。
    时间复杂度：O(n^3)
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    P = np.eye(n)

    for k in range(n):
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


def lu_solve(P, L, U, b):
    """使用 LU 分解求解 Ax = b

    步骤：
      1. 应用置换：Pb
      2. 前向代入求解 Ly = Pb
      3. 回代求解 Ux = y
    时间复杂度：O(n^2)
    """
    n = len(b)
    Pb = P @ b.astype(float)

    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]

    return x


def cholesky(A):
    """Cholesky 分解：对对称正定矩阵 A = LL^T

    仅适用于对称正定矩阵。
    时间复杂度：O(n^3/3)，是 LU 的一半。
    """
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("矩阵不是正定的")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L


def cholesky_solve(L, b):
    """使用 Cholesky 分解求解 Ax = b（其中 A = LL^T）

    步骤：
      1. 前向代入求解 Ly = b
      2. 回代求解 L^T x = y
    """
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

    x = np.zeros(n)
    Lt = L.T
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - Lt[i, i + 1 :] @ x[i + 1 :]) / Lt[i, i]

    return x


def least_squares_normal(A, b):
    """通过正规方程的最小二乘法

    求解 A^T A x = A^T b
    仅适用于小规模问题（n < 1000）
    """
    AtA = A.T @ A
    Atb = A.T @ b
    return gaussian_elimination(AtA, Atb)


def ridge_regression(A, b, lam):
    """岭回归：通过 Cholesky 求解正则化最小二乘

    求解 (A^T A + lambda*I) x = A^T b
    正则化使矩阵总是对称正定的，所以可以使用 Cholesky。
    """
    n = A.shape[1]
    AtA = A.T @ A + lam * np.eye(n)
    Atb = A.T @ b
    L = cholesky(AtA)
    return cholesky_solve(L, Atb)


def condition_number(A):
    """计算矩阵的条件数：kappa = sigma_max / sigma_min

    通过 SVD 计算。
    kappa < 100：良态，解可靠
    kappa ~ 10^6：开始丢失精度
    kappa ~ 10^16：矩阵实际上奇异，解无意义
    """
    _, S, _ = np.linalg.svd(A)
    if S[-1] < 1e-15:
        return float("inf")
    return S[0] / S[-1]


def conjugate_gradient(A, b, tol=1e-10, max_iter=None):
    """共轭梯度法：求解 Ax = b（其中 A 对称正定）

    迭代求解器，比直接方法更快地收敛到大规模稀疏系统。
    在精确算术中最多 n 步找到精确解。
    通常在远少于 n 步时收敛（尤其是条件数较小时）。

    Args:
        A: 对称正定矩阵
        b: 右端向量
        tol: 收敛容差
        max_iter: 最大迭代次数（默认 n）
    """
    n = len(b)
    if max_iter is None:
        max_iter = n

    x = np.zeros(n)
    r = b.astype(float) - A @ x
    p = r.copy()
    rs_old = r @ r

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            return x, k + 1
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter


def demo_gaussian_elimination():
    print("=" * 60)
    print("带部分选主元的高斯消元")
    print("=" * 60)

    A = np.array([[2, 1, 1], [4, 3, 3], [2, 3, 1]], dtype=float)
    b = np.array([8, 20, 12], dtype=float)

    x_ours = gaussian_elimination(A, b)
    x_numpy = np.linalg.solve(A, b)

    print(f"A =\n{A}")
    print(f"b = {b}")
    print(f"解（我们的）:  {x_ours}")
    print(f"解（numpy）: {x_numpy}")
    print(f"最大差异: {np.max(np.abs(x_ours - x_numpy)):.2e}")

    residual = A @ x_ours - b
    print(f"残差 ||Ax - b||: {np.linalg.norm(residual):.2e}")
    print()


def demo_lu():
    print("=" * 60)
    print("LU 分解")
    print("=" * 60)

    A = np.array([[2, 1, 1], [4, 3, 3], [2, 3, 1]], dtype=float)
    b = np.array([8, 20, 12], dtype=float)

    P, L, U = lu_decompose(A)

    print(f"P =\n{P}")
    print(f"L =\n{L}")
    print(f"U =\n{U}")

    reconstructed = P.T @ L @ U
    print(f"PA = LU 重建误差: {np.max(np.abs(A - reconstructed)):.2e}")

    x = lu_solve(P, L, U, b)
    print(f"解: {x}")

    print("\n用同一个 LU 求解 3 个不同的右端向量:")
    for b_i in [np.array([1, 0, 0.0]), np.array([0, 1, 0.0]), np.array([0, 0, 1.0])]:
        x_i = lu_solve(P, L, U, b_i)
        print(f"  b = {b_i} -> x = {np.round(x_i, 4)}")
    print()


def demo_cholesky():
    print("=" * 60)
    print("Cholesky 分解")
    print("=" * 60)

    A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=float)

    L = cholesky(A)
    print(f"A =\n{A}")
    print(f"L =\n{np.round(L, 4)}")
    print(f"L @ L^T =\n{np.round(L @ L.T, 4)}")
    print(f"重建误差: {np.max(np.abs(A - L @ L.T)):.2e}")

    L_numpy = np.linalg.cholesky(A)
    print(f"与 numpy cholesky 的最大差异: {np.max(np.abs(L - L_numpy)):.2e}")

    b = np.array([7, 10, 10], dtype=float)
    x = cholesky_solve(L, b)
    x_direct = np.linalg.solve(A, b)
    print(f"\n求解 Ax = b:")
    print(f"  x（我们的）:  {np.round(x, 4)}")
    print(f"  x（numpy）: {np.round(x_direct, 4)}")

    print("\n通过 Cholesky 计算对数行列式:")
    log_det = 2 * np.sum(np.log(np.diag(L)))
    log_det_np = np.log(np.linalg.det(A))
    print(f"  2 * sum(log(diag(L))) = {log_det:.6f}")
    print(f"  log(det(A))           = {log_det_np:.6f}")
    print()


def demo_least_squares():
    print("=" * 60)
    print("最小二乘 = 线性回归")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100
    n_features = 3
    w_true = np.array([2.0, -1.0, 0.5])

    X_raw = np.random.randn(n_samples, n_features)
    noise = np.random.randn(n_samples) * 0.1
    y = X_raw @ w_true + noise

    X = np.column_stack([np.ones(n_samples), X_raw])
    w_true_with_bias = np.array([0.0, 2.0, -1.0, 0.5])

    w_ols = least_squares_normal(X, y)
    w_numpy = np.linalg.lstsq(X, y, rcond=None)[0]

    print(f"真实权重:          {w_true_with_bias}")
    print(f"OLS 权重（我们的）:    {np.round(w_ols, 4)}")
    print(f"OLS 权重（numpy）:   {np.round(w_numpy, 4)}")
    print(f"最大差异: {np.max(np.abs(w_ols - w_numpy)):.2e}")

    residual = X @ w_ols - y
    print(f"残差范数: {np.linalg.norm(residual):.4f}")
    print()


def demo_ridge():
    print("=" * 60)
    print("岭回归（正则化最小二乘）")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100
    n_features = 3
    w_true = np.array([2.0, -1.0, 0.5])

    X_raw = np.random.randn(n_samples, n_features)
    noise = np.random.randn(n_samples) * 0.1
    y = X_raw @ w_true + noise

    X = np.column_stack([np.ones(n_samples), X_raw])

    for lam in [0.0, 0.1, 1.0, 10.0]:
        if lam == 0.0:
            w = least_squares_normal(X, y)
        else:
            w = ridge_regression(X, y, lam)
        r = np.linalg.norm(X @ w - y)
        wnorm = np.linalg.norm(w)
        print(f"lambda={lam:>5.1f}  w={np.round(w, 3)}  ||w||={wnorm:.3f}  ||Xw-y||={r:.3f}")

    try:
        from sklearn.linear_model import Ridge

        print("\n与 sklearn Ridge 比较:")
        for lam in [0.1, 1.0, 10.0]:
            w_ours = ridge_regression(X, y, lam)
            ridge_sk = Ridge(alpha=lam, fit_intercept=False)
            ridge_sk.fit(X, y)
            diff = np.max(np.abs(w_ours - ridge_sk.coef_))
            print(f"  lambda={lam:>5.1f}  与 sklearn 的最大差异: {diff:.2e}")
    except ImportError:
        print("\n安装 scikit-learn 以进行 sklearn 比较：pip install scikit-learn")
    print()


def demo_condition_number():
    print("=" * 60)
    print("条件数")
    print("=" * 60)

    A_good = np.array([[2, 0], [0, 1]], dtype=float)
    print(f"良态： kappa = {condition_number(A_good):.1f}")

    A_bad = np.array([[1, 1], [1, 1 + 1e-10]], dtype=float)
    print(f"病态：  kappa = {condition_number(A_bad):.2e}")

    np.random.seed(42)
    X = np.random.randn(100, 5)
    print(f"\n随机 100x5 矩阵:")
    print(f"  kappa(X)     = {condition_number(X):.2f}")
    print(f"  kappa(X^T X) = {condition_number(X.T @ X):.2f}")

    X_collinear = X.copy()
    X_collinear[:, 4] = X_collinear[:, 0] + 1e-8 * np.random.randn(100)
    print(f"\n加入近似共线特征后:")
    print(f"  kappa(X)     = {condition_number(X_collinear):.2e}")
    print(f"  kappa(X^T X) = {condition_number(X_collinear.T @ X_collinear):.2e}")

    lam = 0.01
    XtX_reg = X_collinear.T @ X_collinear + lam * np.eye(5)
    print(f"\n正则化后（lambda={lam}）:")
    print(f"  kappa(X^T X + lambda I) = {condition_number(XtX_reg):.2f}")
    print()


def demo_conjugate_gradient():
    print("=" * 60)
    print("共轭梯度法")
    print("=" * 60)

    np.random.seed(42)
    n = 50
    M = np.random.randn(n, n)
    A = M.T @ M + 0.1 * np.eye(n)
    b = np.random.randn(n)

    x_cg, iters = conjugate_gradient(A, b, tol=1e-10)
    x_direct = np.linalg.solve(A, b)

    print(f"系统规模: {n}")
    print(f"CG 迭代次数: {iters}（最多 {n} 次）")
    print(f"与直接求解的最大差异: {np.max(np.abs(x_cg - x_direct)):.2e}")
    print(f"残差范数: {np.linalg.norm(A @ x_cg - b):.2e}")
    print(f"条件数: {condition_number(A):.2f}")

    A_well = np.eye(n) + 0.1 * M.T @ M / n
    b_well = np.random.randn(n)
    x_cg2, iters2 = conjugate_gradient(A_well, b_well, tol=1e-10)
    print(f"\n条件更好的系统:")
    print(f"  kappa = {condition_number(A_well):.2f}")
    print(f"  CG 迭代次数: {iters2}")
    print()


def demo_equivalence():
    print("=" * 60)
    print("所有方法一致：高斯、LU、Cholesky、正规方程、NumPy")
    print("=" * 60)

    np.random.seed(42)
    n = 5
    M = np.random.randn(n, n)
    A = M.T @ M + np.eye(n)
    b = np.random.randn(n)

    x_gauss = gaussian_elimination(A, b)

    P, L, U = lu_decompose(A)
    x_lu = lu_solve(P, L, U, b)

    Lc = cholesky(A)
    x_chol = cholesky_solve(Lc, b)

    x_numpy = np.linalg.solve(A, b)

    x_cg, _ = conjugate_gradient(A, b, tol=1e-12)

    print(f"高斯消元:  {np.round(x_gauss, 6)}")
    print(f"LU:        {np.round(x_lu, 6)}")
    print(f"Cholesky:  {np.round(x_chol, 6)}")
    print(f"NumPy:     {np.round(x_numpy, 6)}")
    print(f"CG:        {np.round(x_cg, 6)}")
    print(f"\n所有方法在容差范围内一致:")
    for name, x in [("LU", x_lu), ("Cholesky", x_chol), ("NumPy", x_numpy), ("CG", x_cg)]:
        print(f"  高斯消元 vs {name:>10s}: {np.max(np.abs(x_gauss - x)):.2e}")
    print()


def demo_linear_regression_full():
    print("=" * 60)
    print("完整流程：从零实现的线性回归")
    print("=" * 60)

    np.random.seed(0)
    n_samples = 200
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.random.uniform(0, 5, n_samples)
    noise = np.random.randn(n_samples) * 0.5
    y = 3.0 * x1 - 2.0 * x2 + 7.0 + noise

    X = np.column_stack([np.ones(n_samples), x1, x2])

    print(f"数据: {n_samples} 个样本, {X.shape[1]} 个特征（含截距）")
    print(f"真实权重: [7.0, 3.0, -2.0]")
    print(f"X^T X 的条件数: {condition_number(X.T @ X):.2f}")

    w_normal = least_squares_normal(X, y)
    print(f"\n正规方程:     {np.round(w_normal, 4)}")

    AtA = X.T @ X
    Lc = cholesky(AtA)
    w_chol = cholesky_solve(Lc, X.T @ y)
    print(f"Cholesky:             {np.round(w_chol, 4)}")

    w_numpy = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"NumPy lstsq:          {np.round(w_numpy, 4)}")

    try:
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression(fit_intercept=False)
        lr.fit(X, y)
        print(f"sklearn:              {np.round(lr.coef_, 4)}")
    except ImportError:
        print("sklearn:              （安装 scikit-learn 以进行比较）")

    y_pred = X @ w_normal
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(f"\nMSE:  {mse:.4f}")
    print(f"R^2:  {r2:.4f}")
    print()


if __name__ == "__main__":
    demo_gaussian_elimination()
    demo_lu()
    demo_cholesky()
    demo_least_squares()
    demo_ridge()
    demo_condition_number()
    demo_conjugate_gradient()
    demo_equivalence()
    demo_linear_regression_full()
