import math
import os


# 复数类，支持基本算术、模、相位和直角坐标/极坐标转换
class Complex:
    def __init__(self, real=0.0, imag=0.0):
        self.real = float(real)
        self.imag = float(imag)

    # 加法
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real + other, self.imag)
        return Complex(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self.__add__(other)

    # 减法
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real - other, self.imag)
        return Complex(self.real - other.real, self.imag - other.imag)

    def __rsub__(self, other):
        return Complex(other - self.real, -self.imag)

    # 乘法
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Complex(self.real * other, self.imag * other)
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __rmul__(self, other):
        return self.__mul__(other)

    # 除法
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Complex(other)
        denom = other.real ** 2 + other.imag ** 2
        if denom == 0:
            raise ZeroDivisionError("复数除以零")
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)

    # 负数
    def __neg__(self):
        return Complex(-self.real, -self.imag)

    # 模（幅值）
    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    # 相位（辐角）
    def phase(self):
        return math.atan2(self.imag, self.real)

    # 共轭
    def conjugate(self):
        return Complex(self.real, -self.imag)

    def __repr__(self):
        if abs(self.imag) < 1e-12:
            return f"{self.real:.6f}"
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.6f} {sign} {abs(self.imag):.6f}i"

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = Complex(other)
        return (abs(self.real - other.real) < 1e-10 and
                abs(self.imag - other.imag) < 1e-10)


# 直角坐标转极坐标
def to_polar(z):
    return z.magnitude(), z.phase()


# 极坐标转直角坐标
def from_polar(r, theta):
    return Complex(r * math.cos(theta), r * math.sin(theta))


# 欧拉公式：e^(i*theta) = cos(theta) + i*sin(theta)
def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))


# 离散傅里叶变换（DFT）
def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            xn = signal[n] if isinstance(signal[n], Complex) else Complex(signal[n])
            total = total + xn * euler(angle)
        result.append(total)
    return result


# 逆离散傅里叶变换（IDFT）
def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result


# N 次单位根
def roots_of_unity(N):
    return [euler(2 * math.pi * k / N) for k in range(N)]


def demo_arithmetic():
    print("=" * 65)
    print("  复数算术运算")
    print("=" * 65)
    print()

    z1 = Complex(3, 2)
    z2 = Complex(1, 4)

    print(f"  z1 = {z1}")
    print(f"  z2 = {z2}")
    print()

    print(f"  z1 + z2  = {z1 + z2}")
    print(f"  z1 - z2  = {z1 - z2}")
    print(f"  z1 * z2  = {z1 * z2}")
    print(f"  z1 / z2  = {z1 / z2}")
    print()

    print(f"  |z1|     = {z1.magnitude():.6f}")
    print(f"  相位(z1)= {z1.phase():.6f} 弧度 ({math.degrees(z1.phase()):.2f} 度)")
    print(f"  共轭(z1) = {z1.conjugate()}")
    print()

    product = z1 * z1.conjugate()
    expected = z1.real ** 2 + z1.imag ** 2
    print(f"  z1 * 共轭(z1) = {product}")
    print(f"  a^2 + b^2     = {expected:.6f}")
    print(f"  匹配：{abs(product.real - expected) < 1e-10}")
    print()

    z3 = Complex(5, 2)
    z4 = Complex(1, -3)
    quotient = z3 / z4
    reconstructed = quotient * z4
    print(f"  除法验证：(5+2i) / (1-3i) = {quotient}")
    print(f"  还原：    结果 * (1-3i)  = {reconstructed}")
    print(f"  匹配原始：{abs(reconstructed.real - 5) < 1e-10 and abs(reconstructed.imag - 2) < 1e-10}")


def demo_polar_conversion():
    print()
    print()
    print("=" * 65)
    print("  极坐标形式与转换")
    print("=" * 65)
    print()

    test_cases = [
        Complex(1, 0),
        Complex(0, 1),
        Complex(-1, 0),
        Complex(0, -1),
        Complex(3, 4),
        Complex(-2, 3),
    ]

    print(f"  {'直角坐标':<25s} {'r':>8s}  {'角度（度）':>12s}  {'还原结果':<25s}")
    print(f"  {'-' * 25} {'-' * 8}  {'-' * 12}  {'-' * 25}")

    for z in test_cases:
        r, theta = to_polar(z)
        z_back = from_polar(r, theta)
        print(f"  {str(z):<25s} {r:>8.4f}  {math.degrees(theta):>12.2f}  {str(z_back):<25s}")


def demo_euler_formula():
    print()
    print()
    print("=" * 65)
    print("  欧拉公式：e^(i*theta) = cos(theta) + i*sin(theta)")
    print("=" * 65)
    print()

    angles = [0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2,
              math.pi, 3 * math.pi / 2, 2 * math.pi]
    labels = ["0", "pi/6", "pi/4", "pi/3", "pi/2", "pi", "3pi/2", "2pi"]

    print(f"  {'theta':<8s} {'cos(theta)':>12s} {'sin(theta)':>12s} "
          f"{'e^(i*theta)':>25s} {'|e^(i*theta)|':>14s}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 25} {'-' * 14}")

    for label, theta in zip(labels, angles):
        e = euler(theta)
        print(f"  {label:<8s} {math.cos(theta):>12.6f} {math.sin(theta):>12.6f} "
              f"  {str(e):>23s} {e.magnitude():>14.10f}")

    print()
    e_pi = euler(math.pi)
    result = e_pi + Complex(1, 0)
    print(f"  欧拉恒等式：e^(i*pi) + 1 = {result}")
    print(f"  |e^(i*pi) + 1| = {result.magnitude():.2e}（应接近 0）")


def demo_rotation():
    print()
    print()
    print("=" * 65)
    print("  复数乘法实现的旋转")
    print("=" * 65)
    print()

    point = Complex(3, 4)
    print(f"  原始点：{point}")
    print(f"  模：{point.magnitude():.4f}")
    print(f"  相位：{math.degrees(point.phase()):.2f} 度")
    print()

    rotation_angles = [45, 90, 180, 270, 360]

    print(f"  {'旋转角度':<12s} {'结果':<30s} {'模':>10s} {'相位（度）':>12s}")
    print(f"  {'-' * 12} {'-' * 30} {'-' * 10} {'-' * 12}")

    for deg in rotation_angles:
        rad = math.radians(deg)
        rotated = point * euler(rad)
        r, theta = to_polar(rotated)
        print(f"  {deg:>3d} 度     {str(rotated):<30s} {r:>10.4f} {math.degrees(theta):>12.2f}")

    print()
    print("  所有旋转中模保持不变。")
    print("  360 度回到原始点。")
    print()
    print("  旋转矩阵等价性验证：")
    print()

    test_angles = [math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi]
    test_points = [Complex(1, 0), Complex(3, 4), Complex(-2, 5)]

    max_error = 0.0
    for theta in test_angles:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for p in test_points:
            complex_result = p * euler(theta)
            matrix_x = cos_t * p.real - sin_t * p.imag
            matrix_y = sin_t * p.real + cos_t * p.imag

            err = math.sqrt((complex_result.real - matrix_x) ** 2 +
                            (complex_result.imag - matrix_y) ** 2)
            max_error = max(max_error, err)

    print(f"  复数乘法与")
    print(f"  旋转矩阵之间的最大差异：{max_error:.2e}")


def demo_roots_of_unity():
    print()
    print()
    print("=" * 65)
    print("  单位根")
    print("=" * 65)
    print()

    for N in [4, 8]:
        roots = roots_of_unity(N)
        print(f"  {N} 次单位根：")
        print(f"  {'k':<4s} {'根':<30s} {'|根|':>8s}")
        print(f"  {'-' * 4} {'-' * 30} {'-' * 8}")

        total = Complex(0, 0)
        for k, root in enumerate(roots):
            total = total + root
            print(f"  {k:<4d} {str(root):<30s} {root.magnitude():>8.6f}")

        print(f"  所有根的和：{total}")
        print(f"  |和| = {total.magnitude():.2e}（应接近 0）")
        print()

    print("  单位根的总和始终为零。")
    print("  每个根的模恰好为 1。")


def demo_dft():
    print()
    print()
    print("=" * 65)
    print("  简单信号的 DFT")
    print("=" * 65)
    print()

    N = 32
    freq1 = 3
    freq2 = 7
    amp1 = 1.0
    amp2 = 0.5

    signal = []
    for n in range(N):
        t = n / N
        val = amp1 * math.sin(2 * math.pi * freq1 * t) + amp2 * math.sin(2 * math.pi * freq2 * t)
        signal.append(val)

    print(f"  信号：{amp1}*sin(2*pi*{freq1}*t) + {amp2}*sin(2*pi*{freq2}*t)")
    print(f"  {N} 个采样点")
    print()

    spectrum = dft(signal)

    print(f"  {'频率仓':<10s} {'|X[k]|':>10s} {'相位（度）':>12s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 12}")

    for k in range(N // 2 + 1):
        mag = spectrum[k].magnitude()
        if mag > 0.01:
            phase_deg = math.degrees(spectrum[k].phase())
            print(f"  k={k:<6d} {mag:>10.4f} {phase_deg:>12.2f}")

    print()
    print(f"  预期峰值在 k={freq1}（幅度 {amp1 * N / 2:.1f}）")
    print(f"  和 k={freq2}（幅度 {amp2 * N / 2:.1f}）")
    print()

    reconstructed = idft(spectrum)
    max_err = max(abs(reconstructed[n].real - signal[n]) for n in range(N))
    print(f"  IDFT 重建误差：{max_err:.2e}")
    print(f"  完美重建：{max_err < 1e-10}")


def demo_phasor():
    print()
    print()
    print("=" * 65)
    print("  相量：作为信号的旋转复数")
    print("=" * 65)
    print()

    omega = 2 * math.pi * 3
    N = 16

    print(f"  相量：e^(i*{3}*2*pi*t)，在 {N} 个点上采样")
    print()
    print(f"  {'t':>6s} {'实部（cos）':>12s} {'虚部（sin）':>12s} {'模':>10s}")
    print(f"  {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 10}")

    for n in range(N):
        t = n / N
        phasor = euler(omega * t)
        print(f"  {t:>6.3f} {phasor.real:>12.6f} {phasor.imag:>12.6f} {phasor.magnitude():>10.6f}")

    print()
    print("  实部描绘 cos(6*pi*t)。")
    print("  虚部描绘 sin(6*pi*t)。")
    print("  模始终为 1 —— 相量保持在单位圆上。")


def demo_positional_encoding():
    print()
    print()
    print("=" * 65)
    print("  Transformer 位置编码频率")
    print("=" * 65)
    print()

    d_model = 8
    max_pos = 10

    print(f"  d_model = {d_model}，显示前 {max_pos} 个位置")
    print()
    print(f"  频率（1/10000^(2i/d)）：")
    freqs = []
    for i in range(d_model // 2):
        freq = 1.0 / (10000 ** (2 * i / d_model))
        freqs.append(freq)
        print(f"    维度对 {i}：频率 = {freq:.6f}")

    print()
    print(f"  PE 矩阵（每个位置的 sin/cos 对）：")
    print()

    header = "  pos"
    for i in range(d_model // 2):
        header += f"  sin_{i:d}     cos_{i:d}  "
    print(header)
    print(f"  {'-' * (5 + d_model // 2 * 20)}")

    for pos in range(max_pos):
        line = f"  {pos:>3d}"
        for i in range(d_model // 2):
            angle = pos * freqs[i]
            line += f"  {math.sin(angle):>7.4f}  {math.cos(angle):>7.4f}"
        print(line)

    print()
    print("  每对 (sin, cos) 是 e^(i * pos * freq) 的实部和虚部。")
    print("  不同频率赋予每个位置在复平面中独特的"指纹"。")


def write_skill_output():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "outputs", "skill-complex-arithmetic.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write("name: skill-complex-arithmetic\n")
            f.write("description: 机器学习和信号处理中复数运算的快速参考\n")
            f.write("phase: 1\n")
            f.write("lesson: 19\n")
            f.write("---\n\n")
            f.write("你是复数算术在机器学习和信号处理方面的专家。\n\n")
            f.write("当有人问及复数、傅里叶变换、旋转或位置编码时：\n\n")
            f.write("1. 确定最佳表示形式：直角坐标 (a + bi) 适合加法，极坐标 (r * e^(i*theta)) 适合乘法和旋转。\n\n")
            f.write("2. 关键转换：\n")
            f.write("   - 直角坐标转极坐标：r = sqrt(a^2 + b^2)，theta = atan2(b, a)\n")
            f.write("   - 极坐标转直角坐标：a = r*cos(theta)，b = r*sin(theta)\n")
            f.write("   - 欧拉公式：e^(i*theta) = cos(theta) + i*sin(theta)\n\n")
            f.write("3. 常见运算及其几何含义：\n")
            f.write("   - 加法：复平面中的向量加法\n")
            f.write("   - 乘法：按 arg(z2) 旋转并按 |z2| 缩放\n")
            f.write("   - 共轭：关于实轴镜像\n")
            f.write("   - 除法：反向旋转并重新缩放\n\n")
            f.write("4. 机器学习联系：\n")
            f.write("   - DFT 使用单位根：e^(-2*pi*i*k*n/N)\n")
            f.write("   - 位置编码：sin/cos 对是复指数的实部/虚部\n")
            f.write("   - RoPE：Query/Key 向量的位置相关旋转的显式复数乘法\n")
            f.write("   - FFT：利用单位根对称性的递归 DFT，O(N log N)\n\n")
            f.write("5. 快速检查：\n")
            f.write("   - |e^(i*theta)| = 1 始终成立\n")
            f.write("   - z * conj(z) = |z|^2（始终为实数）\n")
            f.write("   - N 次单位根之和 = 0\n")
            f.write("   - e^(i*pi) + 1 = 0（欧拉恒等式）\n")
            f.write("   - 乘以 e^(i*theta) 旋转 theta 弧度\n\n")
            f.write("6. Python 快速参考：\n")
            f.write("   - 内置：z = 3+2j，abs(z)，z.conjugate()，z.real，z.imag\n")
            f.write("   - cmath：cmath.phase(z)，cmath.exp(1j*theta)，cmath.polar(z)\n")
            f.write("   - numpy：np.abs(z)，np.angle(z)，np.conj(z)，np.fft.fft(signal)\n")
        print(f"\n  技能输出已写入 {output_path}")
    except OSError:
        print("\n  无法写入技能输出（从课程目录运行）")


def print_summary():
    print()
    print()
    print("=" * 65)
    print("  总结")
    print("=" * 65)
    print()
    print("  1. 复数 z = a + bi 是平面上一个点 (a, b)。")
    print("  2. 乘法旋转并缩放。除法反转它。")
    print("  3. 欧拉公式：e^(i*theta) = cos(theta) + i*sin(theta)。")
    print("  4. 乘以 e^(i*theta) 旋转 theta 弧度。")
    print("  5. 复数乘法就是二维旋转（与旋转矩阵相同）。")
    print("  6. DFT 将信号分解为旋转相量（单位根）。")
    print("  7. Transformer 位置编码是不同频率的复指数。")
    print("  8. RoPE 使用显式复数乘法实现位置。")
    print()


if __name__ == "__main__":
    demo_arithmetic()
    demo_polar_conversion()
    demo_euler_formula()
    demo_rotation()
    demo_roots_of_unity()
    demo_dft()
    demo_phasor()
    demo_positional_encoding()
    write_skill_output()
    print_summary()