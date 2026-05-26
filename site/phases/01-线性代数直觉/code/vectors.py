"""
线性代数直觉 — 从零实现向量与矩阵运算
Phase 01, Lesson 01 — 中文版

包含：
  - Vector 类：向量加法、减法、点积、模长、归一化、余弦相似度、夹角、投影
  - Matrix 类：矩阵乘法、转置、秩计算
  - 线性无关判断、格拉姆-施密特正交化
  - 神经网络层演示（矩阵 × 向量）
"""


class Vector:
    """n 维向量，支持基本线性运算。"""

    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        """向量加法：对应分量相加。"""
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        """向量减法。"""
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        """标量乘法。"""
        return Vector([x * scalar for x in self.components])

    def dot(self, other):
        """点积（内积）：衡量两个向量的相似度。"""
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        """模长（L2 范数）：sqrt(x1² + x2² + ... + xn²)。"""
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        """归一化：将向量缩放到长度为 1。"""
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        """余弦相似度：[-1, 1]，1 表示同方向，-1 表示反方向。"""
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def angle_between(self, other):
        """计算两个向量之间的夹角（度数）。"""
        import math
        cos_theta = self.cosine_similarity(other)
        # 数值稳定性：防止浮点误差导致 acos 报错
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))

    def project_onto(self, other):
        """将当前向量投影到另一个向量上。"""
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    def __repr__(self):
        return f"Vector({self.components})"


def is_independent(vectors):
    """
    通过高斯行约简判断一组向量是否线性无关。
    思路：将向量作为行构成矩阵，求秩。若秩 == 向量个数，则无关。
    """
    n = len(vectors)
    if n == 0:
        return True
    dim = vectors[0].dim
    rows = [v.components[:] for v in vectors]
    rank = 0
    for col in range(dim):
        # 找主元
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        # 交换到当前行
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        # 归一化主元行
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        # 消去其他行在该列的分量
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def gram_schmidt(vectors):
    """
    格拉姆-施密特正交化：将一组线性无关向量转换为一组标准正交向量。
    输出向量的模长均为 1，且两两点积为 0。
    """
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


class Matrix:
    """矩阵，支持矩阵乘法、转置、秩计算。"""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        """矩阵乘法。支持 Matrix @ Vector 和 Matrix @ Matrix。"""
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        """矩阵转置：行列互换。"""
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def rank(self):
        """通过高斯行约简计算矩阵的秩。"""
        rows = [row[:] for row in self.rows]
        m, n = self.shape
        r = 0
        for col in range(n):
            pivot = None
            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break
            if pivot is None:
                continue
            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            rows[r] = [x / scale for x in rows[r]]
            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [rows[row][j] - factor * rows[r][j] for j in range(n)]
            r += 1
        return r

    def __repr__(self):
        return f"Matrix({self.rows})"


if __name__ == "__main__":
    print("=== 向量运算 ===")
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * 3 = {a * 3}")
    print(f"a · b = {a.dot(b)}")
    print(f"|a| = {a.magnitude():.4f}")
    print(f"â (单位向量) = {a.normalize()}")
    print(f"余弦相似度(a, b) = {a.cosine_similarity(b):.4f}")

    print("\n=== 矩阵变换 ===")
    rotation_90 = Matrix([[0, -1], [1, 0]])  # 90° 逆时针旋转矩阵
    point = Vector([3, 1])
    rotated = rotation_90 @ point
    print(f"将 {point} 旋转 90° → {rotated}")

    print("\n=== 向量夹角 ===")
    v1 = Vector([1, 0])
    v2 = Vector([0, 1])
    v3 = Vector([1, 1])
    print(f"{v1} 与 {v2} 的夹角: {v1.angle_between(v2):.1f}°")
    print(f"{v1} 与 {v3} 的夹角: {v1.angle_between(v3):.1f}°")
    print(f"{v1} 与 {v1} 的夹角: {v1.angle_between(v1):.1f}°")

    print("\n=== 投影 ===")
    a = Vector([3, 4])
    b = Vector([1, 0])
    proj = a.project_onto(b)
    residual = a - proj
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"proj_b(a) = {proj}")
    print(f"残差 = {residual}")
    print(f"残差 · b = {residual.dot(b):.6f}  （应为 0，表示垂直）")

    print("\n=== 线性无关 ===")
    e1 = Vector([1, 0, 0])
    e2 = Vector([0, 1, 0])
    e3 = Vector([0, 0, 1])
    dep = Vector([2, 1, 0])
    print(f"{{e1, e2, e3}} 线性无关: {is_independent([e1, e2, e3])}")
    print(f"{{e1, e2, 2*e1+e2}} 线性无关: {is_independent([e1, e2, dep])}")

    print("\n=== 格拉姆-施密特正交化 ===")
    u1 = Vector([1, 1, 0])
    u2 = Vector([1, 0, 1])
    u3 = Vector([0, 1, 1])
    basis = gram_schmidt([u1, u2, u3])
    for i, vec in enumerate(basis):
        print(f"u{i+1} = {vec}")
    print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
    print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
    print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
    for i, vec in enumerate(basis):
        print(f"|u{i+1}| = {vec.magnitude():.6f}")

    print("\n=== 矩阵的秩 ===")
    full_rank = Matrix([[1, 0], [0, 1]])
    rank_deficient = Matrix([[1, 2], [2, 4]])
    rectangular = Matrix([[1, 0, 0], [0, 1, 0]])
    print(f"单位矩阵 2x2 的秩: {full_rank.rank()}")
    print(f"[[1,2],[2,4]] 的秩: {rank_deficient.rank()}")
    print(f"[[1,0,0],[0,1,0]] 的秩: {rectangular.rank()}")

    print("\n=== 神经网络层（矩阵 × 向量）===")
    import random
    random.seed(42)
    weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
    input_vec = Vector([1.0, 0.5, -0.3])
    output = weights @ input_vec
    print(f"输入 (3D):  {input_vec}")
    print(f"输出 (2D): {output}")
    print("^ 这就是神经网络层在做的事。")
