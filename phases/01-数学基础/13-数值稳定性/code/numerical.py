import math
import struct
import random


def softmax_naive(logits):
    """朴素 softmax（有溢出风险）。"""
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def softmax_stable(logits):
    """数值稳定的 softmax：先减去最大值避免溢出。"""
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def logsumexp_naive(values):
    """朴素的 log-sum-exp（有溢出风险）。"""
    return math.log(sum(math.exp(v) for v in values))


def logsumexp_stable(values):
    """数值稳定的 log-sum-exp：先减去最大值。"""
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))


def log_softmax_stable(logits):
    """数值稳定的 log-softmax。"""
    c = max(logits)
    lse = c + math.log(sum(math.exp(z - c) for z in logits))
    return [z - lse for z in logits]


def cross_entropy_naive(true_class, logits):
    """朴素的交叉熵（有溢出风险）。"""
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])


def cross_entropy_stable(true_class, logits):
    """数值稳定的交叉熵。"""
    log_probs = log_softmax_stable(logits)
    return -log_probs[true_class]


def sigmoid_naive(x):
    """朴素 sigmoid（有溢出风险）。"""
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_stable(x):
    """数值稳定的 sigmoid。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def binary_cross_entropy_naive(y_true, y_pred):
    """朴素的二元交叉熵（有 log(0) 风险）。"""
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


def binary_cross_entropy_stable(y_true, logit):
    """数值稳定的二元交叉熵（在 logit 空间计算）。"""
    max_val = max(0.0, -logit)
    return max_val + math.log(math.exp(-max_val) + math.exp(-logit - max_val)) - y_true * logit


def numerical_gradient(f, x, h=1e-5):
    """使用中心差分法计算数值梯度。"""
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad


def check_gradient(analytical, numerical, tolerance=1e-5):
    """比较解析梯度和数值梯度。"""
    all_ok = True
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")
    return all_ok


def clip_by_value(gradients, max_val):
    """按值裁剪梯度：将每个元素 clamp 到 [-max_val, max_val]。"""
    return [max(-max_val, min(max_val, g)) for g in gradients]


def clip_by_norm(gradients, max_norm):
    """按范数裁剪梯度：缩放向量使其范数不超过阈值，保留梯度方向。"""
    total_norm = math.sqrt(sum(g ** 2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return list(gradients)


def check_tensor(name, values):
    """检查张量中是否有 NaN 或 Inf。"""
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    n_nan = sum(1 for v in values if math.isnan(v))
    n_inf = sum(1 for v in values if math.isinf(v))
    if has_nan or has_inf:
        print(f"  警告 {name}: {n_nan} 个 NaN, {n_inf} 个 Inf，共 {len(values)} 个值")
        return False
    print(f"  OK {name}: 所有 {len(values)} 个值都是有限的")
    return True


def simulate_bfloat16(x):
    """模拟 bfloat16 截断（保留 8 位指数，丢弃 7 位尾数精度）。"""
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    # 丢弃低 16 位（尾数精度）
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]


def simulate_float16(x):
    """模拟 float16 转换（可能有溢出）。"""
    try:
        packed = struct.pack('e', x)
        return struct.unpack('e', packed)[0]
    except (OverflowError, struct.error):
        return float('inf') if x > 0 else float('-inf')


def kahan_sum(values):
    """Kahan 求和：减少浮点累加中的舍入误差。"""
    total = 0.0
    compensation = 0.0
    for v in values:
        y = v - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total


def welford_variance(values):
    """Welford 在线算法：数值稳定的单遍方差计算（避免灾难性抵消）。"""
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in values:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
    if n < 2:
        return 0.0
    return m2 / n


def variance_naive(values):
    """朴素方差计算（有灾难性抵消风险）。"""
    n = len(values)
    mean_x = sum(values) / n
    mean_x2 = sum(v ** 2 for v in values) / n
    return mean_x2 - mean_x ** 2


def layer_norm(values, epsilon=1e-5, gamma=1.0, beta=0.0):
    """层归一化：重新中心化和重新缩放激活，作为数值稳定器。"""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var + epsilon)
    return [(v - mean) / std * gamma + beta for v in values]


def demo_float_precision():
    print("=" * 60)
    print("演示 1：浮点精度限制")
    print("=" * 60)

    print(f"\n  0.1 + 0.2 = {0.1 + 0.2}")
    print(f"  0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
    print(f"  与 0.3 的差异: {(0.1 + 0.2) - 0.3:.2e}")
    print(f"  math.isclose(0.1 + 0.2, 0.3): {math.isclose(0.1 + 0.2, 0.3)}")

    print(f"\n  Float32 最大值: ~{3.4028235e+38:.2e}")
    print(f"  Float32 最小正数（正规）: ~{1.175e-38:.2e}")
    print(f"  Float32 epsilon: ~{1.1920929e-07:.2e}")

    print(f"\n  1.0 + 1e-7 == 1.0?  {1.0 + 1e-7 == 1.0}")
    print(f"  1.0 + 1e-8 == 1.0?  {1.0 + 1e-8 == 1.0}")
    print(f"  （这些是 Python 中的 float64。在 float32 中，epsilon 约为 1.19e-7）")

    # 演示累积误差
    total_naive = 0.0
    for _ in range(1_000_000):
        total_naive += 1e-7
    total_kahan = kahan_sum([1e-7] * 1_000_000)
    true_value = 1e-7 * 1_000_000

    print(f"\n  一百万次累加 1e-7：")
    print(f"  真值:  {true_value}")
    print(f"  朴素求和:   {total_naive:.10f}  （误差: {abs(total_naive - true_value):.2e}）")
    print(f"  Kahan 求和:   {total_kahan:.10f}  （误差: {abs(total_kahan - true_value):.2e}）")
    print()


def demo_catastrophic_cancellation():
    print("=" * 60)
    print("演示 2：灾难性抵消")
    print("=" * 60)

    data = [1_000_000.0, 1_000_001.0, 1_000_002.0]
    true_var = 2.0 / 3.0

    var_naive = variance_naive(data)
    var_welford = welford_variance(data)

    print(f"\n  数据: {data}")
    print(f"  真实方差: {true_var:.10f}")
    print(f"  朴素（E[x^2] - E[x]^2）: {var_naive:.10f}")
    print(f"  Welford（在线）:         {var_welford:.10f}")
    print(f"  朴素误差:   {abs(var_naive - true_var):.2e}")
    print(f"  Welford 误差: {abs(var_welford - true_var):.2e}")

    a = 1.0000001
    b = 1.0000000
    true_diff = 1e-7
    computed_diff = a - b
    rel_error = abs(computed_diff - true_diff) / true_diff * 100

    print(f"\n  减去几乎相等的数：")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  真实 a - b = {true_diff}")
    print(f"  计算值:    {computed_diff}")
    print(f"  相对误差: {rel_error:.1f}%")
    print()


def demo_overflow_underflow():
    print("=" * 60)
    print("演示 3：exp() 和 log() 中的溢出与下溢")
    print("=" * 60)

    print("\n  exp() 溢出边界（Python 中的 float64）：")
    for x in [700, 709, 709.78, 710]:
        try:
            result = math.exp(x)
            print(f"  exp({x}) = {result:.4e}")
        except OverflowError:
            print(f"  exp({x}) = 溢出")

    print("\n  exp() 下溢（结果变为 0.0）：")
    for x in [-700, -745, -746]:
        result = math.exp(x)
        print(f"  exp({x}) = {result}")

    print("\n  log() 边界情况：")
    for x in [1.0, 1e-300, 1e-323, 0.0]:
        try:
            if x == 0.0:
                print(f"  log(0.0) = -inf  （数学上如此）")
                result = math.log(1e-323)
                print(f"  log(1e-323) = {result:.2f}  （最接近的能达到）")
            else:
                result = math.log(x)
                print(f"  log({x}) = {result:.4f}")
        except ValueError:
            print(f"  log({x}) = 域错误")

    print("\n  Float16 溢出边界：")
    for val in [65000.0, 65504.0, 65520.0, 70000.0]:
        f16 = simulate_float16(val)
        print(f"  float16({val}) = {f16}")
    print()


def demo_softmax_stability():
    print("=" * 60)
    print("演示 4：朴素 vs 稳定 Softmax")
    print("=" * 60)

    safe_logits = [2.0, 1.0, 0.1]
    print(f"\n  安全 logits: {safe_logits}")
    naive_result = softmax_naive(safe_logits)
    stable_result = softmax_stable(safe_logits)
    print(f"  朴素:  {[f'{p:.6f}' for p in naive_result]}")
    print(f"  稳定: {[f'{p:.6f}' for p in stable_result]}")
    print(f"  匹配: {all(abs(a - b) < 1e-10 for a, b in zip(naive_result, stable_result))}")

    moderate_logits = [100.0, 101.0, 102.0]
    print(f"\n  中等 logits: {moderate_logits}")
    stable_result = softmax_stable(moderate_logits)
    print(f"  稳定: {[f'{p:.6f}' for p in stable_result]}")
    try:
        naive_result = softmax_naive(moderate_logits)
        print(f"  朴素:  {[f'{p:.6f}' for p in naive_result]}")
    except OverflowError:
        print("  朴素:  溢出（exp(100) 太大）")

    extreme_logits = [1000.0, 1001.0, 1002.0]
    print(f"\n  极端 logits: {extreme_logits}")
    stable_result = softmax_stable(extreme_logits)
    print(f"  稳定: {[f'{p:.6f}' for p in stable_result]}")
    print("  朴素:  会是 [nan, nan, nan] 或溢出")

    negative_logits = [-1000.0, -999.0, -998.0]
    print(f"\n  非常负的 logits: {negative_logits}")
    stable_result = softmax_stable(negative_logits)
    print(f"  稳定: {[f'{p:.6f}' for p in stable_result]}")
    print("  朴素:  会是 [0/0 = nan]（所有 exp() 下溢到 0）")
    print()


def demo_logsumexp():
    print("=" * 60)
    print("演示 5：Log-Sum-Exp 技巧")
    print("=" * 60)

    safe = [1.0, 2.0, 3.0]
    print(f"\n  安全值: {safe}")
    print(f"  朴素:  {logsumexp_naive(safe):.10f}")
    print(f"  稳定: {logsumexp_stable(safe):.10f}")

    large = [500.0, 501.0, 502.0]
    print(f"\n  大值: {large}")
    print(f"  稳定: {logsumexp_stable(large):.10f}")
    try:
        naive = logsumexp_naive(large)
        print(f"  朴素:  {naive}")
    except OverflowError:
        print("  朴素:  溢出")

    very_negative = [-1000.0, -999.0, -998.0]
    print(f"\n  非常负的值: {very_negative}")
    print(f"  稳定: {logsumexp_stable(very_negative):.10f}")

    equal = [5.0, 5.0, 5.0]
    print(f"\n  相等值: {equal}")
    expected = 5.0 + math.log(3.0)
    print(f"  稳定:   {logsumexp_stable(equal):.10f}")
    print(f"  期望: {expected:.10f} （= 5.0 + ln(3)）")

    one_dominant = [100.0, 1.0, 1.0]
    print(f"\n  一个主导值: {one_dominant}")
    print(f"  稳定: {logsumexp_stable(one_dominant):.10f}")
    print(f"  约为 100.0（被 exp(100) 主导）")
    print()


def demo_cross_entropy():
    print("=" * 60)
    print("演示 6：稳定的交叉熵损失")
    print("=" * 60)

    logits = [2.0, 5.0, 1.0]
    true_class = 1

    print(f"\n  Logits: {logits}, 真实类别: {true_class}")
    ce_naive = cross_entropy_naive(true_class, logits)
    ce_stable = cross_entropy_stable(true_class, logits)
    print(f"  朴素:  {ce_naive:.10f}")
    print(f"  稳定: {ce_stable:.10f}")
    print(f"  匹配:  {abs(ce_naive - ce_stable) < 1e-10}")

    large_logits = [100.0, 105.0, 99.0]
    true_class = 1
    print(f"\n  大 logits: {large_logits}, 真实类别: {true_class}")
    ce_stable = cross_entropy_stable(true_class, large_logits)
    print(f"  稳定: {ce_stable:.10f}")
    try:
        ce_naive = cross_entropy_naive(true_class, large_logits)
        print(f"  朴素:  {ce_naive:.10f}")
    except (OverflowError, ValueError):
        print("  朴素:  溢出或 NaN")

    confident_logits = [0.0, 0.0, 50.0]
    true_class = 2
    ce = cross_entropy_stable(true_class, confident_logits)
    print(f"\n  非常自信的预测：")
    print(f"  Logits: {confident_logits}, 真实类别: {true_class}")
    print(f"  损失: {ce:.10f}  （接近零，模型正确且自信）")

    wrong_logits = [0.0, 0.0, 50.0]
    true_class = 0
    ce = cross_entropy_stable(true_class, wrong_logits)
    print(f"\n  非常错误的预测：")
    print(f"  Logits: {wrong_logits}, 真实类别: {true_class}")
    print(f"  损失: {ce:.4f}  （非常大，模型自信但错了）")
    print()


def demo_sigmoid_stability():
    print("=" * 60)
    print("演示 7：稳定的 Sigmoid")
    print("=" * 60)

    test_values = [0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0, 500.0, -500.0, 710.0, -710.0]
    print(f"\n  {'x':>8s}  {'朴素':>14s}  {'稳定':>14s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}")
    for x in test_values:
        try:
            naive = sigmoid_naive(x)
            naive_str = f"{naive:.10f}"
        except OverflowError:
            naive_str = "溢出"
        stable = sigmoid_stable(x)
        print(f"  {x:>8.1f}  {naive_str:>14s}  {stable:.10f}")
    print()


def demo_gradient_checking():
    print("=" * 60)
    print("演示 8：梯度检验")
    print("=" * 60)

    print("\n  测试 1: f(x,y) = x^2 + 3xy + y^3")

    def f1(params):
        x, y = params
        return x ** 2 + 3 * x * y + y ** 3

    def f1_grad(params):
        x, y = params
        return [2 * x + 3 * y, 3 * x + 3 * y ** 2]

    point = [2.0, 1.0]
    analytical = f1_grad(point)
    numerical = numerical_gradient(f1, point)
    print(f"  点: {point}")
    check_gradient(analytical, numerical)

    print("\n  测试 2: f(x) = softmax cross-entropy")

    def f2(logits):
        return cross_entropy_stable(0, logits)

    logits = [2.0, 1.0, 0.5]
    probs = softmax_stable(logits)
    analytical_ce = [probs[i] - (1.0 if i == 0 else 0.0) for i in range(len(logits))]
    numerical_ce = numerical_gradient(f2, logits)
    print(f"  Logits: {logits}")
    check_gradient(analytical_ce, numerical_ce)

    print("\n  测试 3：故意错误的梯度（应该失败）")

    def f3(params):
        x, y = params
        return x ** 2 + y ** 2

    wrong_grad = [1.0, 1.0]
    numerical_f3 = numerical_gradient(f3, [3.0, 4.0])
    print(f"  错误的解析梯度: {wrong_grad}")
    print(f"  正确的数值梯度: {[f'{g:.4f}' for g in numerical_f3]}")
    check_gradient(wrong_grad, numerical_f3)
    print()


def demo_nan_inf():
    print("=" * 60)
    print("演示 9：NaN 和 Inf 检测与传播")
    print("=" * 60)

    print("\n  inf 是如何产生的：")
    print(f"  1.0 / 0.0    = {float('inf')}")
    print(f"  exp(710)     = 溢出 -> inf")
    print(f"  1e308 * 10   = {1e308 * 10}")

    print("\n  nan 是如何产生的：")
    print(f"  0.0 / 0.0        = {float('nan')}")
    print(f"  inf - inf        = {float('inf') - float('inf')}")
    print(f"  inf * 0          = {float('inf') * 0}")
    print(f"  nan + 1          = {float('nan') + 1}")
    print(f"  nan == nan       = {float('nan') == float('nan')}")
    print(f"  nan < 0          = {float('nan') < 0}")
    print(f"  nan > 0          = {float('nan') > 0}")

    print("\n  NaN 传播（一个 nan 毁掉一切）：")
    values = [1.0, 2.0, float('nan'), 4.0, 5.0]
    print(f"  values = {values}")
    print(f"  sum    = {sum(values)}")
    print(f"  max    = nan（与 nan 的比较总是 False）")
    print(f"  mean   = {sum(values) / len(values)}")

    print("\n  张量健康检查：")
    check_tensor("weights", [0.1, -0.3, 0.5, 0.2])
    check_tensor("logits_bad", [1.0, float('inf'), -2.0])
    check_tensor("grads_bad", [0.01, float('nan'), -0.03])
    check_tensor("activations", [0.0, 0.5, 1.0, 0.3])
    print()


def demo_gradient_clipping():
    print("=" * 60)
    print("演示 10：梯度裁剪")
    print("=" * 60)

    grads = [10.0, 20.0, 30.0]
    norm = math.sqrt(sum(g ** 2 for g in grads))

    print(f"\n  梯度: {grads}")
    print(f"  范数: {norm:.4f}")

    clipped_val = clip_by_value(grads, max_val=15.0)
    clipped_norm = clip_by_norm(grads, max_norm=5.0)

    print(f"\n  按值裁剪（max=15.0）: {clipped_val}")
    print(f"  按值裁剪改变方向: "
          f"{[g/grads[0] for g in grads]} vs {[g/clipped_val[0] for g in clipped_val]}")

    print(f"\n  按范数裁剪（max=5.0）: {[f'{g:.4f}' for g in clipped_norm]}")
    clipped_norm_val = math.sqrt(sum(g ** 2 for g in clipped_norm))
    print(f"  裁剪后范数: {clipped_norm_val:.4f}")
    print(f"  方向保持: "
          f"{[round(g/grads[0], 4) for g in grads]} == "
          f"{[round(g/clipped_norm[0], 4) for g in clipped_norm]}")

    print("\n  梯度爆炸模拟：")
    grad_val = 1.0
    max_norm = 1.0
    for step in range(8):
        grad_val *= 3.5
        clipped = clip_by_norm([grad_val], max_norm)[0]
        print(f"  步 {step}: raw_grad={grad_val:>12.2f}  clipped={clipped:>8.4f}")
    print()


def demo_mixed_precision():
    print("=" * 60)
    print("演示 11：混合精度与损失缩放")
    print("=" * 60)

    print("\n  bfloat16 vs float16 精度：")
    test_values = [1.0, 0.1, 3.14159, 100.0, 65504.0, 65536.0, 100000.0]
    print(f"  {'值':>12s}  {'float16':>12s}  {'bfloat16':>12s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}")
    for v in test_values:
        f16 = simulate_float16(v)
        bf16 = simulate_bfloat16(v)
        f16_str = f"{f16:.4f}" if not math.isinf(f16) else "inf"
        bf16_str = f"{bf16:.4f}" if not math.isinf(bf16) else "inf"
        print(f"  {v:>12.4f}  {f16_str:>12s}  {bf16_str:>12s}")

    print("\n  损失缩放模拟：")
    random.seed(42)
    n_grads = 1000
    tiny_grads = [random.uniform(1e-9, 1e-5) for _ in range(n_grads)]

    zeros_without_scaling = sum(1 for g in tiny_grads if simulate_float16(g) == 0.0)

    scale = 1024.0
    scaled_grads = [g * scale for g in tiny_grads]
    zeros_with_scaling = sum(1 for g in scaled_grads if simulate_float16(g) == 0.0)

    scaled_back = [simulate_float16(g * scale) / scale for g in tiny_grads]
    zeros_after_roundtrip = sum(1 for g in scaled_back if g == 0.0)

    print(f"  {n_grads} 个梯度在范围 [1e-9, 1e-5] 内")
    print(f"  无缩放时为零: {zeros_without_scaling}/{n_grads} "
          f"({zeros_without_scaling/n_grads*100:.1f}%)")
    print(f"  缩放（×{scale:.0f}）后为零: {zeros_with_scaling}/{n_grads} "
          f"({zeros_with_scaling/n_grads*100:.1f}%)")
    print(f"  缩放+转换+反缩放后为零: {zeros_after_roundtrip}/{n_grads} "
          f"({zeros_after_roundtrip/n_grads*100:.1f}%)")

    print("\n  动态损失缩放模拟：")
    scale_factor = 65536.0
    no_overflow_steps = 0
    growth_interval = 100

    print(f"  {'步':>6s}  {'比例':>12s}  {'事件':s}")
    for step in range(500):
        grad = random.gauss(0, 1)
        scaled = grad * scale_factor
        if math.isinf(simulate_float16(scaled)):
            scale_factor /= 2
            no_overflow_steps = 0
            if step < 20 or step % 100 == 0:
                print(f"  {step:>6d}  {scale_factor:>12.0f}  溢出 -> 减半")
        else:
            no_overflow_steps += 1
            if no_overflow_steps >= growth_interval:
                scale_factor *= 2
                no_overflow_steps = 0
                if step < 100 or step % 100 == 0:
                    print(f"  {step:>6d}  {scale_factor:>12.0f}  稳定 -> 加倍")
    print(f"  最终比例因子: {scale_factor:.0f}")
    print()


def demo_layer_norm():
    print("=" * 60)
    print("演示 12：归一化作为数值稳定器")
    print("=" * 60)

    print("\n  无归一化（值通过层增长）：")
    values = [1.0, 0.5, -0.3, 0.8, -0.1]
    for layer in range(10):
        values = [max(0, v * 2.5 + 0.1) for v in values]
        max_val = max(abs(v) for v in values)
        if layer % 2 == 0:
            print(f"  层 {layer:>2d}: max={max_val:>12.2f}  values={[f'{v:.2f}' for v in values[:3]]}...")

    print("\n  有层归一化（值保持有界）：")
    values = [1.0, 0.5, -0.3, 0.8, -0.1]
    for layer in range(10):
        values = [max(0, v * 2.5 + 0.1) for v in values]
        values = layer_norm(values)
        max_val = max(abs(v) for v in values)
        if layer % 2 == 0:
            print(f"  层 {layer:>2d}: max={max_val:>6.4f}  values={[f'{v:.4f}' for v in values[:3]]}...")
    print()


def demo_common_bugs():
    print("=" * 60)
    print("演示 13：常见 ML 数值 Bug")
    print("=" * 60)

    print("\n  Bug 1：来自自信错误预测的 log(0)")
    logits = [100.0, -100.0, -100.0]
    probs = softmax_stable(logits)
    print(f"  Softmax: {[f'{p:.2e}' for p in probs]}")
    print(f"  如果真实类别是 1: log({probs[1]:.2e}) = ", end="")
    if probs[1] == 0.0:
        print("log(0) = -inf（崩溃）")
    else:
        print(f"{math.log(probs[1]):.2f}")
    print(f"  稳定交叉熵处理这种情况: {cross_entropy_stable(1, logits):.4f}")

    print("\n  Bug 2：朴素 softmax 中的 exp() 溢出")
    logits = [800.0, 801.0, 802.0]
    try:
        naive = softmax_naive(logits)
        print(f"  朴素 softmax: {naive}")
    except OverflowError:
        print("  朴素 softmax: OverflowError（exp(800) 太大）")
    stable = softmax_stable(logits)
    print(f"  稳定 softmax: {[f'{p:.6f}' for p in stable]}")

    print("\n  Bug 3：大方差数据时的方差下溢")
    data = [1e8 + 1, 1e8 + 2, 1e8 + 3, 1e8 + 4, 1e8 + 5]
    var_naive = variance_naive(data)
    var_welford = welford_variance(data)
    true_var = 2.0
    print(f"  数据: [{data[0]:.0f}, ..., {data[-1]:.0f}]")
    print(f"  真实方差: {true_var}")
    print(f"  朴素:   {var_naive:.6f}  （误差: {abs(var_naive - true_var):.2e}）")
    print(f"  Welford: {var_welford:.6f}  （误差: {abs(var_welford - true_var):.2e}）")

    print("\n  Bug 4：训练循环中的浮点比较")
    loss = 0.0
    for _ in range(10):
        loss += 0.1
    print(f"  10 次累加 0.1 后: loss = {loss}")
    print(f"  loss == 1.0? {loss == 1.0}（错误）")
    print(f"  math.isclose(loss, 1.0)? {math.isclose(loss, 1.0)}（正确）")

    print("\n  Bug 5：归一化中 0/0 产生 NaN")
    values = [5.0, 5.0, 5.0, 5.0]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    print(f"  常数输入: {values}")
    print(f"  方差: {var}")
    print(f"  1/sqrt(var) = 1/sqrt(0) = ", end="")
    try:
        result = 1.0 / math.sqrt(var)
        print(f"{result}")
    except ZeroDivisionError:
        print("ZeroDivisionError")
    safe = 1.0 / math.sqrt(var + 1e-5)
    print(f"  1/sqrt(var + 1e-5) = {safe:.2f}（带 epsilon 安全）")
    print()


def demo_format_comparison():
    print("=" * 60)
    print("演示 14：浮点格式比较总结")
    print("=" * 60)

    print(f"""
  格式      位数  指数  尾数    约位数  最大值           最佳用于
  -------    ----  ---  --------  -------  ----------      --------
  float64    64    11   52        15-16    1.8e308         CPU 训练，累加
  float32    32    8    23        7-8      3.4e38          默认训练
  float16    16    5    10        3-4      65,504          推理
  bfloat16   16    8    7         2-3      3.4e38          GPU/TPU 训练
  float8     8     4    3         1-2      240             仅前向传递（H100+）
""")

    print("  精度测试（表示 pi）：")
    pi = math.pi
    f16_pi = simulate_float16(pi)
    bf16_pi = simulate_bfloat16(pi)
    print(f"  float64:  {pi}")
    print(f"  float16:  {f16_pi}  （误差: {abs(f16_pi - pi):.6f}）")
    print(f"  bfloat16: {bf16_pi}  （误差: {abs(bf16_pi - pi):.6f}）")

    print("\n  范围测试（大值）：")
    for val in [100.0, 1000.0, 10000.0, 65504.0, 100000.0]:
        f16 = simulate_float16(val)
        bf16 = simulate_bfloat16(val)
        f16_ok = "ok" if not math.isinf(f16) else "inf"
        bf16_ok = "ok" if not math.isinf(bf16) else "inf"
        print(f"  {val:>10.0f}  float16={f16_ok:>4s}  bfloat16={bf16_ok:>4s}")
    print()


if __name__ == "__main__":
    demo_float_precision()
    demo_catastrophic_cancellation()
    demo_overflow_underflow()
    demo_softmax_stability()
    demo_logsumexp()
    demo_cross_entropy()
    demo_sigmoid_stability()
    demo_gradient_checking()
    demo_nan_inf()
    demo_gradient_clipping()
    demo_mixed_precision()
    demo_layer_norm()
    demo_common_bugs()
    demo_format_comparison()
    print("所有演示完成。")