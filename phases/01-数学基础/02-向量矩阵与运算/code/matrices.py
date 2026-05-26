import random


class Vector:
    """n 维向量，支持基本线性运算。"""

    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        """向量加法：对应分量相加。"""
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        """向量减法。"""
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        """标量乘法。"""
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        """点积（内积）。"""
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        """模长（L2 范数）。"""
        return sum(x ** 2 for x in self.data) ** 0.5

    def normalize(self):
        """归一化为单位向量。"""
        mag = self.magnitude()
        return Vector([x / mag for x in self.data])


class Matrix:
    """矩阵，支持逐元素运算、矩阵乘法、转置、行列式、逆矩阵。"""

    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        """美化打印：带边框的矩阵格式。"""
        col_widths = []
        for j in range(self.cols):
            width = max(len(f"{self.data[i][j]:.4f}") for i in range(self.rows))
            col_widths.append(width)
        lines = []
        for i in range(self.rows):
            row_str = "  ".join(
                f"{self.data[i][j]:{col_widths[j]}.4f}" for j in range(self.cols)
            )
            bracket_l = "|" if 0 < i < self.rows - 1 else ("/" if i == 0 else "\\")
            bracket_r = "|" if 0 < i < self.rows - 1 else ("\\" if i == 0 else "/")
            lines.append(f"  {bracket_l} {row_str} {bracket_r}")
        header = f"Matrix {self.rows}x{self.cols}:"
        return header + "\n" + "\n".join(lines)

    def __add__(self, other):
        """矩阵加法，支持广播：
        - 同形状矩阵：逐元素相加
        - (1, n) 行向量：沿列方向广播到所有行
        - (m, 1) 列向量：沿行方向广播到所有列
        """
        if isinstance(other, Matrix):
            if other.shape == self.shape:
                return Matrix([
                    [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                    for i in range(self.rows)
                ])
            if other.rows == 1 and other.cols == self.cols:
                return Matrix([
                    [self.data[i][j] + other.data[0][j] for j in range(self.cols)]
                    for i in range(self.rows)
                ])
            if other.cols == 1 and other.rows == self.rows:
                return Matrix([
                    [self.data[i][j] + other.data[i][0] for j in range(self.cols)]
                    for i in range(self.rows)
                ])
        raise ValueError(f"无法相加形状 {self.shape} 和 {other.shape}")

    def __sub__(self, other):
        """矩阵减法（同形状）。"""
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        """标量乘法：每个元素乘以 scalar。"""
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        """逐元素乘法（Hadamard 积）：对应位置相乘。"""
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        """矩阵乘法：self @ other。中间维度必须匹配。"""
        if self.cols != other.rows:
            raise ValueError(
                f"无法相乘形状 {self.shape} 和 {other.shape}: "
                f"中间维度 {self.cols} != {other.rows}"
            )
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def __matmul__(self, other):
        """支持 Python @ 运算符。"""
        return self.matmul(other)

    def transpose(self):
        """矩阵转置：行列互换。"""
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    @property
    def T(self):
        """转置的简写属性（如 A.T）。"""
        return self.transpose()

    def determinant(self):
        """行列式：递归按第一行展开（Laplace 展开）。"""
        if self.rows != self.cols:
            raise ValueError("行列式只对方阵有定义")
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            # 构造余子式：去掉第 0 行和第 j 列
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        """2x2 矩阵的逆：adj(A) / det(A)。"""
        if self.shape != (2, 2):
            raise ValueError("此方法只适用于 2x2 矩阵")
        det = self.determinant()
        if abs(det) < 1e-10:
            raise ValueError("矩阵是奇异的，不存在逆矩阵")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        """n 阶单位矩阵：对角线为 1，其余为 0。"""
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])

    @staticmethod
    def zeros(rows, cols):
        """零矩阵。"""
        return Matrix([[0] * cols for _ in range(rows)])

    @staticmethod
    def random(rows, cols, low=-1.0, high=1.0):
        """随机矩阵，元素均匀分布在 [low, high]。"""
        return Matrix([
            [random.uniform(low, high) for _ in range(cols)]
            for _ in range(rows)
        ])


def relu_matrix(m):
    """对矩阵逐元素应用 ReLU：负数变 0，正数不变。"""
    return Matrix([[max(0, val) for val in row] for row in m.data])


def demo_basic_operations():
    print("=" * 60)
    print("矩阵基本运算")
    print("=" * 60)

    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    print("\nA =")
    print(A)
    print("\nB =")
    print(B)

    print("\nA + B =")
    print(A + B)

    print("\nA - B =")
    print(A - B)

    print("\nA * 3 (标量乘法) =")
    print(A.scalar_multiply(3))

    print("\nA * B (逐元素乘法) =")
    print(A.element_wise_multiply(B))

    print("\nA @ B (矩阵乘法) =")
    print(A @ B)

    print("\nA^T =")
    print(A.T)


def demo_determinant_inverse():
    print("\n" + "=" * 60)
    print("行列式与逆矩阵")
    print("=" * 60)

    A = Matrix([[4, 7], [2, 6]])
    print("\nA =")
    print(A)
    print(f"\ndet(A) = {A.determinant()}")

    A_inv = A.inverse_2x2()
    print("\nA^-1 =")
    print(A_inv)

    print("\nA @ A^-1 (应该是单位矩阵) =")
    print(A @ A_inv)

    I = Matrix.identity(3)
    print("\n3x3 单位矩阵 =")
    print(I)


def demo_broadcasting():
    print("\n" + "=" * 60)
    print("广播 (Broadcasting)")
    print("=" * 60)

    output = Matrix([[1, 2, 3], [4, 5, 6]])
    bias = Matrix([[10, 20, 30]])

    print("\n输出 =")
    print(output)
    print("\n偏置 =")
    print(bias)
    print("\n输出 + 偏置 (广播) =")
    print(output + bias)


def demo_neural_network_layer():
    print("\n" + "=" * 60)
    print("神经网络前向传播")
    print("=" * 60)

    random.seed(42)

    input_size = 3
    hidden_size = 4
    output_size = 2

    # 输入向量 (3x1)
    x = Matrix([[0.5], [0.8], [0.2]])
    # 第一层权重 (4x3)
    W1 = Matrix.random(hidden_size, input_size)
    # 第一层偏置 (4x1)
    b1 = Matrix([[0.0]] * hidden_size)
    # 第二层权重 (2x4)
    W2 = Matrix.random(output_size, hidden_size)
    # 第二层偏置 (2x1)
    b2 = Matrix([[0.0]] * output_size)

    print(f"\n输入 x: {x.shape}")
    print(f"W1: {W1.shape}")
    print(f"W2: {W2.shape}")

    # 第一层：z1 = W1 @ x + b1
    z1 = (W1 @ x) + b1
    h1 = relu_matrix(z1)
    print(f"\n隐藏层预激活 z1: {z1.shape}")
    print(z1)
    print(f"\n隐藏层 ReLU 后 h1: {h1.shape}")
    print(h1)

    # 第二层：z2 = W2 @ h1 + b2
    z2 = (W2 @ h1) + b2
    print(f"\n输出 z2: {z2.shape}")
    print(z2)

    print("\n这是一个完整的两层神经网络前向传播。")
    print("第一层: (4x3) @ (3x1) + (4x1) -> (4x1) -> ReLU -> (4x1)")
    print("第二层: (2x4) @ (4x1) + (2x1) -> (2x1)")


def demo_vectors():
    print("\n" + "=" * 60)
    print("向量运算")
    print("=" * 60)

    v = Vector([3, 4])
    w = Vector([1, 2])

    print(f"\nv = {v}")
    print(f"w = {w}")
    print(f"v + w = {v + w}")
    print(f"v - w = {v - w}")
    print(f"v * 2 = {v * 2}")
    print(f"v . w = {v.dot(w)}")
    print(f"|v| = {v.magnitude()}")
    print(f"v 归一化 = {v.normalize()}")
    print(f"|v 归一化| = {v.normalize().magnitude()}")


def demo_weight_matrix_intuition():
    print("\n" + "=" * 60)
    print("权重矩阵的直觉")
    print("=" * 60)

    print("\n权重矩阵把输入特征变换成输出特征。")
    print("每一行从输入中提取一种模式。\n")

    W = Matrix([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ])
    x = Matrix([[0.8], [0.6], [0.1]])

    print("权重矩阵 W (3 个检测器, 3 个输入):")
    print(W)
    print("\n输入 x:")
    print(x)
    print("\nW @ x =")
    result = W @ x
    print(result)
    print("\nW 的第 0 行 = [1, 0, 0]: 直接复制输入特征 0")
    print("W 的第 1 行 = [0, 1, 0]: 直接复制输入特征 1")
    print("W 的第 2 行 = [0.5, 0.5, 0]: 把特征 0 和 1 取平均")


if __name__ == "__main__":
    demo_vectors()
    demo_basic_operations()
    demo_determinant_inverse()
    demo_broadcasting()
    demo_weight_matrix_intuition()
    demo_neural_network_layer()
