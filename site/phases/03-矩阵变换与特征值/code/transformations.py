import math


def rotation_2d(theta):
    """二维旋转矩阵：绕原点逆时针旋转 theta 弧度。"""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]


def rotation_3d_z(theta):
    """三维旋转矩阵：绕 z 轴旋转 theta 弧度。"""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def rotation_3d_x(theta):
    """三维旋转矩阵：绕 x 轴旋转 theta 弧度。"""
    c, s = math.cos(theta), math.sin(theta)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rotation_3d_y(theta):
    """三维旋转矩阵：绕 y 轴旋转 theta 弧度。"""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def scaling_2d(sx, sy):
    """二维缩放矩阵：x 方向缩放 sx 倍，y 方向缩放 sy 倍。"""
    return [[sx, 0], [0, sy]]


def shearing_2d(kx, ky):
    """二维剪切矩阵：
    kx - 沿 x 方向的剪切量（y 每增加 1，x 偏移 kx）
    ky - 沿 y 方向的剪切量（x 每增加 1，y 偏移 ky）
    """
    return [[1, kx], [ky, 1]]


def reflection_x():
    """关于 x 轴的反射矩阵。"""
    return [[1, 0], [0, -1]]


def reflection_y():
    """关于 y 轴的反射矩阵。"""
    return [[-1, 0], [0, 1]]


def mat_vec_mul(matrix, vector):
    """矩阵乘以向量。"""
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]


def mat_mul(a, b):
    """矩阵乘法：a @ b。"""
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]


def det_2x2(m):
    """2x2 矩阵的行列式。"""
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def det_3x3(m):
    """3x3 矩阵的行列式（按第一行展开）。"""
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def eigenvalues_2x2(matrix):
    """计算 2x2 矩阵的特征值。
    特征方程：lambda^2 - trace*lambda + det = 0
    """
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        # 复数特征值：表示变换包含旋转
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)


def eigenvector_2x2(matrix, eigenvalue):
    """计算 2x2 矩阵对应于某个特征值的特征向量（单位化）。"""
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]


def fmt(v, decimals=4):
    """格式化数值或列表，保留指定小数位。"""
    if isinstance(v, list):
        return [round(x, decimals) for x in v]
    return round(v, decimals)


def demo_basic_transformations():
    print("=" * 60)
    print("基本变换")
    print("=" * 60)

    point = [1.0, 0.0]
    theta = math.pi / 4

    rotated = mat_vec_mul(rotation_2d(theta), point)
    print(f"\n把 (1,0) 旋转 45°: {fmt(rotated)}")

    scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
    print(f"把 (1,1) 缩放 (2,3): {fmt(scaled)}")

    sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
    print(f"把 (1,1) 沿 x 剪切 k=1: {fmt(sheared)}")

    reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
    print(f"把 (2,1) 关于 y 轴反射: {fmt(reflected)}")

    reflected_x = mat_vec_mul(reflection_x(), [2.0, 1.0])
    print(f"把 (2,1) 关于 x 轴反射: {fmt(reflected_x)}")


def demo_unit_square():
    print("\n" + "=" * 60)
    print("单位正方形上的变换")
    print("=" * 60)

    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    labels = ["原点", "右边", "右上角", "上边"]

    print("\n原始正方形:")
    for label, pt in zip(labels, square):
        print(f"  {label}: {pt}")

    transforms = [
        ("旋转 45°", rotation_2d(math.pi / 4)),
        ("缩放 (2, 0.5)", scaling_2d(2, 0.5)),
        ("剪切 kx=0.5", shearing_2d(0.5, 0)),
        ("关于 y 轴反射", reflection_y()),
    ]

    for name, matrix in transforms:
        print(f"\n{name}:")
        for label, pt in zip(labels, square):
            result = mat_vec_mul(matrix, pt)
            print(f"  {label}: {pt} -> {fmt(result)}")
        print(f"  det = {fmt(det_2x2(matrix))}")


def demo_composition():
    print("\n" + "=" * 60)
    print("变换的组合")
    print("=" * 60)

    R = rotation_2d(math.pi / 2)
    S = scaling_2d(2, 0.5)

    rotate_then_scale = mat_mul(S, R)
    scale_then_rotate = mat_mul(R, S)

    point = [1.0, 0.0]

    result1 = mat_vec_mul(rotate_then_scale, point)
    result2 = mat_vec_mul(scale_then_rotate, point)

    print(f"\n点: {point}")
    print(f"先旋转 90° 再缩放 (2, 0.5): {fmt(result1)}")
    print(f"先缩放 (2, 0.5) 再旋转 90°: {fmt(result2)}")
    print("顺序不同，结果不同。")

    print(f"\ndet(R) = {fmt(det_2x2(R))}")
    print(f"det(S) = {fmt(det_2x2(S))}")
    print(f"det(S @ R) = {fmt(det_2x2(rotate_then_scale))}")
    print(f"det(S) * det(R) = {fmt(det_2x2(S) * det_2x2(R))}")
    print("组合变换的行列式 = 各行列式的乘积。")


def demo_3d_rotations():
    print("\n" + "=" * 60)
    print("三维旋转")
    print("=" * 60)

    point = [1.0, 0.0, 0.0]
    theta = math.pi / 2

    rz = mat_vec_mul(rotation_3d_z(theta), point)
    rx = mat_vec_mul(rotation_3d_x(theta), point)
    ry = mat_vec_mul(rotation_3d_y(theta), point)

    print(f"\n点: {point}")
    print(f"绕 z 轴旋转 90°: {fmt(rz)}")
    print(f"绕 x 轴旋转 90°: {fmt(rx)}")
    print(f"绕 y 轴旋转 90°: {fmt(ry)}")

    print(f"\ndet(Rz) = {fmt(det_3x3(rotation_3d_z(theta)))}")
    print(f"det(Rx) = {fmt(det_3x3(rotation_3d_x(theta)))}")
    print(f"det(Ry) = {fmt(det_3x3(rotation_3d_y(theta)))}")
    print("所有旋转矩阵的行列式 = 1（体积不变）。")


def demo_eigenvalues_from_scratch():
    print("\n" + "=" * 60)
    print("特征值与特征向量（手写实现，2x2）")
    print("=" * 60)

    matrices = [
        ("对称矩阵", [[2, 1], [1, 2]]),
        ("上三角矩阵", [[3, 1], [0, 2]]),
        ("缩放矩阵", [[3, 0], [0, 5]]),
        ("旋转 90°", [[0, -1], [1, 0]]),
    ]

    for name, A in matrices:
        vals = eigenvalues_2x2(A)
        print(f"\n{name}: {A}")
        print(f"  特征值: {vals[0]}, {vals[1]}")

        if all(isinstance(v, (int, float)) for v in vals):
            for val in vals:
                vec = eigenvector_2x2(A, val)
                result = mat_vec_mul(A, vec)
                scaled = [val * vec[0], val * vec[1]]
                print(f"  lambda={fmt(val)}, v={fmt(vec)}")
                print(f"    A @ v = {fmt(result)}")
                print(f"    l * v = {fmt(scaled)}")
        else:
            print("  复数特征值：纯旋转，没有实特征向量。")


def demo_eigendecomposition():
    print("\n" + "=" * 60)
    print("特征分解（手写实现，2x2）")
    print("=" * 60)

    A = [[3, 1], [0, 2]]
    vals = eigenvalues_2x2(A)

    v0 = eigenvector_2x2(A, vals[0])
    v1 = eigenvector_2x2(A, vals[1])

    V = [[v0[0], v1[0]], [v0[1], v1[1]]]
    D = [[vals[0], 0], [0, vals[1]]]

    det_v = det_2x2(V)
    V_inv = [
        [V[1][1] / det_v, -V[0][1] / det_v],
        [-V[1][0] / det_v, V[0][0] / det_v],
    ]

    reconstructed = mat_mul(mat_mul(V, D), V_inv)

    print(f"\nA = {A}")
    print(f"特征值: {fmt(vals[0])}, {fmt(vals[1])}")
    print(f"V (特征向量作为列):")
    for row in V:
        print(f"  {fmt(row)}")
    print(f"D (特征值在对角线):")
    for row in D:
        print(f"  {fmt(row)}")
    print(f"重构 A = V @ D @ V^-1:")
    for row in reconstructed:
        print(f"  {fmt(row)}")


def demo_determinant_meaning():
    print("\n" + "=" * 60)
    print("行列式作为面积缩放因子")
    print("=" * 60)

    cases = [
        ("旋转 45°", rotation_2d(math.pi / 4)),
        ("缩放 (2, 3)", scaling_2d(2, 3)),
        ("剪切 kx=1", shearing_2d(1, 0)),
        ("关于 y 轴反射", reflection_y()),
        ("奇异矩阵 [[1,2],[2,4]]", [[1, 2], [2, 4]]),
    ]

    print()
    for name, m in cases:
        d = det_2x2(m)
        if d == 0:
            meaning = "空间坍塌，不可逆"
        elif d < 0:
            meaning = "方向被翻转"
        elif abs(d - 1.0) < 1e-10:
            meaning = "面积不变"
        else:
            meaning = f"面积缩放 {abs(d):.1f} 倍"
        print(f"det({name}) = {fmt(d):>8}  ({meaning})")


def demo_numpy_comparison():
    print("\n" + "=" * 60)
    print("NumPy 对比")
    print("=" * 60)

    try:
        import numpy as np
    except ImportError:
        print("\nNumPy 未安装，跳过。")
        return

    theta = math.pi / 4
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])

    point = np.array([1.0, 0.0])
    print(f"\n把 (1,0) 旋转 45°: {R @ point}")

    A = np.array([[2, 1], [1, 2]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\nA = {A.tolist()}")
    print(f"特征值 (numpy): {eigenvalues}")
    print(f"特征向量 (numpy, 列):\n{eigenvectors}")

    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

    B = np.array([[3, 1], [0, 2]], dtype=float)
    vals, vecs = np.linalg.eig(B)
    D = np.diag(vals)
    V = vecs
    reconstructed = V @ D @ np.linalg.inv(V)
    print(f"\n{B.tolist()} 的特征分解:")
    print(f"  重构: {reconstructed.tolist()}")

    Rz = np.array(rotation_3d_z(math.pi / 2))
    point_3d = np.array([1.0, 0.0, 0.0])
    print(f"\n三维点 (1,0,0) 绕 z 轴旋转 90°: {np.round(Rz @ point_3d, 4)}")

    cov = np.array([[2.0, 1.0], [1.0, 3.0]])
    vals, vecs = np.linalg.eig(cov)
    print(f"\n协方差矩阵: {cov.tolist()}")
    print(f"主成分 (特征向量): 列向量\n{vecs}")
    print(f"各方向方差 (特征值): {vals}")
    print("PCA 选取特征值最大的特征向量。")


if __name__ == "__main__":
    demo_basic_transformations()
    demo_unit_square()
    demo_composition()
    demo_3d_rotations()
    demo_eigenvalues_from_scratch()
    demo_eigendecomposition()
    demo_determinant_meaning()
    demo_numpy_comparison()
