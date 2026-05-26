import math
# 优化器：普通梯度下降、带动量的 SGD、Adam
# 作为可调用类实现，在 Rosenbrock 函数和鞍点函数上运行
# 展示收敛、发散和鞍点逃脱行为
# 仅使用标准库


def rosenbrock(params):
    """Rosenbrock 函数：全局最小值在 (1, 1) 处，f = 0"""
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(params):
    """Rosenbrock 函数的梯度"""
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]


class GradientDescent:
    """普通梯度下降优化器"""

    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]


class SGDMomentum:
    """带动量随机梯度下降"""

    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]


class Adam:
    """Adam (自适应矩估计) 优化器"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]


def optimize(optimizer, func, grad_func, start, steps=5000):
    """运行优化器并返回参数历史"""
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        try:
            grads = grad_func(params)
            if any(math.isnan(g) or math.isinf(g) or abs(g) > 1e15 for g in grads):
                break
            params = optimizer.step(params, grads)
            if any(math.isnan(p) or math.isinf(p) or abs(p) > 1e15 for p in params):
                break
            history.append(params[:])
        except (OverflowError, ValueError):
            break
    return history


def distance_to_minimum(params, target=(1.0, 1.0)):
    """计算到最小值的欧氏距离"""
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(params, target)))


def find_convergence_step(history, func, threshold=1e-4):
    """找到损失低于阈值的第一步"""
    for i, params in enumerate(history):
        if func(params) < threshold:
            return i
    return len(history)


def print_trajectory(name, history, func, steps_to_show=10):
    """打印优化轨迹"""
    total = len(history) - 1
    interval = max(1, total // steps_to_show)
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  {'Step':>6s}  {'x':>10s}  {'y':>10s}  {'Loss':>14s}  {'Dist':>8s}")
    print(f"  {'-' * 52}")
    for i in range(0, total + 1, interval):
        p = history[i]
        loss = func(p)
        dist = distance_to_minimum(p)
        print(f"  {i:6d}  {p[0]:10.6f}  {p[1]:10.6f}  {loss:14.8f}  {dist:8.4f}")
    final = history[-1]
    if total % interval != 0:
        loss = func(final)
        dist = distance_to_minimum(final)
        print(f"  {total:6d}  {final[0]:10.6f}  {final[1]:10.6f}  {loss:14.8f}  {dist:8.4f}")


def print_ascii_convergence(results, func, steps=5000):
    """ASCII 图形展示收敛过程（log10 损失 vs 步数）"""
    print(f"\n{'=' * 60}")
    print("  收敛比较（log10 损失随步数变化）")
    print(f"{'=' * 60}")

    width = 50
    sample_points = 40
    interval = max(1, steps // sample_points)

    for name, history in results:
        losses = []
        for i in range(0, min(len(history), steps + 1), interval):
            loss = func(history[i])
            losses.append(loss)

        if not losses:
            continue

        max_log = 5.0
        min_log = -8.0
        log_range = max_log - min_log

        bar = []
        for loss in losses:
            log_loss = math.log10(loss + 1e-15)
            log_loss = max(min_log, min(max_log, log_loss))
            normalized = (log_loss - min_log) / log_range
            pos = int(normalized * (width - 1))
            bar.append(pos)

        print(f"\n  {name}:")
        print(f"  loss 1e-8 {'.' * width} 1e+5")
        for i, pos in enumerate(bar):
            step_num = i * interval
            line = [' '] * width
            line[pos] = '*'
            print(f"  {step_num:5d} |{''.join(line)}|")

        final_loss = func(history[-1])
        conv_step = find_convergence_step(history, func)
        conv_msg = f"step {conv_step}" if conv_step < len(history) else "未收敛"
        print(f"  最终损失: {final_loss:.2e}, 收敛 (< 1e-4): {conv_msg}")


def demo_comparison():
    """比较三种优化器"""
    print("优化方法比较")
    print("最小化 Rosenbrock 函数: f(x,y) = (1-x)^2 + 100(y-x^2)^2")
    print("全局最小值在 (1, 1)，f = 0")
    print(f"起点: (-1.0, 1.0), f = {rosenbrock([-1.0, 1.0]):.1f}")

    start = [-1.0, 1.0]
    steps = 5000

    configs = [
        ("梯度下降", GradientDescent(lr=0.0005)),
        ("SGD + 动量",   SGDMomentum(lr=0.0001, momentum=0.9)),
        ("Adam",             Adam(lr=0.01)),
    ]

    results = []
    for name, optimizer in configs:
        history = optimize(optimizer, rosenbrock, rosenbrock_gradient, start, steps)
        results.append((name, history))
        print_trajectory(name, history, rosenbrock)

    print_ascii_convergence(results, rosenbrock, steps)

    print(f"\n{'=' * 60}")
    print("  最终结果")
    print(f"{'=' * 60}")
    print(f"  {'方法':<22s}  {'x':>10s}  {'y':>10s}  {'Loss':>14s}")
    print(f"  {'-' * 58}")
    for name, history in results:
        final = history[-1]
        loss = rosenbrock(final)
        print(f"  {name:<22s}  {final[0]:10.6f}  {final[1]:10.6f}  {loss:14.8f}")

    print(f"\n  目标: x=1.000000, y=1.000000, loss=0.00000000")


def demo_learning_rate_effect():
    """展示学习率对梯度下降的影响"""
    print(f"\n\n{'=' * 60}")
    print("  学习率对梯度下降的影响")
    print(f"{'=' * 60}")

    start = [-1.0, 1.0]
    rates = [0.0001, 0.0005, 0.001, 0.005]

    print(f"\n  {'LR':>8s}  {'Final x':>10s}  {'Final y':>10s}  {'Loss':>14s}  {'状态'}")
    print(f"  {'-' * 60}")

    for lr in rates:
        gd = GradientDescent(lr=lr)
        history = optimize(gd, rosenbrock, rosenbrock_gradient, start, 5000)
        final = history[-1]
        loss = rosenbrock(final)
        diverged = loss > 1e10 or math.isnan(loss) or math.isinf(loss)
        status = "发散" if diverged else ("收敛" if loss < 0.01 else "慢")
        if diverged:
            print(f"  {lr:8.4f}  {'nan':>10s}  {'nan':>10s}  {'inf':>14s}  {status}")
        else:
            print(f"  {lr:8.4f}  {final[0]:10.6f}  {final[1]:10.6f}  {loss:14.8f}  {status}")


def demo_momentum_effect():
    """展示动量对 SGD 的影响"""
    print(f"\n\n{'=' * 60}")
    print("  动量对 SGD 的影响")
    print(f"{'=' * 60}")

    start = [-1.0, 1.0]
    betas = [0.0, 0.5, 0.9, 0.99]

    print(f"\n  {'Beta':>6s}  {'Final x':>10s}  {'Final y':>10s}  {'Loss':>14s}")
    print(f"  {'-' * 46}")

    for beta in betas:
        sgd = SGDMomentum(lr=0.0001, momentum=beta)
        history = optimize(sgd, rosenbrock, rosenbrock_gradient, start, 5000)
        final = history[-1]
        loss = rosenbrock(final)
        if math.isnan(loss) or math.isinf(loss):
            print(f"  {beta:6.2f}  {'nan':>10s}  {'nan':>10s}  {'inf':>14s}")
        else:
            print(f"  {beta:6.2f}  {final[0]:10.6f}  {final[1]:10.6f}  {loss:14.8f}")


def demo_saddle_point():
    """鞍点逃脱演示：f(x,y) = x^2 - y^2"""
    print(f"\n\n{'=' * 60}")
    print("  鞍点逃脱: f(x,y) = x^2 - y^2")
    print(f"{'=' * 60}")

    def saddle(params):
        x, y = params
        return x ** 2 - y ** 2

    def saddle_gradient(params):
        x, y = params
        return [2 * x, -2 * y]

    start = [0.01, 0.01]
    steps = 200

    configs = [
        ("梯度下降", GradientDescent(lr=0.01)),
        ("SGD + 动量",   SGDMomentum(lr=0.01, momentum=0.9)),
        ("Adam",             Adam(lr=0.01)),
    ]

    print(f"\n  起点: x=0.01, y=0.01（接近原点鞍点）")
    print(f"\n  {'方法':<22s}  {'x':>10s}  {'y':>10s}  {'f(x,y)':>12s}  {'逃出?'}")
    print(f"  {'-' * 62}")

    for name, optimizer in configs:
        history = optimize(optimizer, saddle, saddle_gradient, start, steps)
        final = history[-1]
        val = saddle(final)
        escaped = abs(final[1]) > 1.0
        print(f"  {name:<22s}  {final[0]:10.6f}  {final[1]:10.6f}  {val:12.6f}  {'是' if escaped else '否'}")


if __name__ == "__main__":
    demo_comparison()
    demo_learning_rate_effect()
    demo_momentum_effect()
    demo_saddle_point()