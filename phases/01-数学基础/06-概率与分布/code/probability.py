import math
import random

random.seed(42)


# ============================================================
# 概率与分布 — 从零实现
# ============================================================


def factorial(n):
    """阶乘：n! = 1 * 2 * ... * n"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def combinations(n, k):
    """组合数：C(n,k) = n! / (k! * (n-k)!)"""
    return factorial(n) // (factorial(k) * factorial(n - k))


def conditional_probability(p_a_and_b, p_b):
    """条件概率：P(A|B) = P(A且B) / P(B)"""
    return p_a_and_b / p_b


# ============================================================
# PMF 与 PDF
# ============================================================

def bernoulli_pmf(k, p):
    """伯努利分布 PMF：一次试验，两个结果"""
    return p if k == 1 else (1 - p)


def categorical_pmf(k, probs):
    """分类分布 PMF：一次试验，k 个结果"""
    return probs[k]


def poisson_pmf(k, lam):
    """泊松分布 PMF：稀有事件的计数"""
    return (lam ** k) * math.exp(-lam) / factorial(k)


def uniform_pdf(x, a, b):
    """均匀分布 PDF：区间内等可能"""
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def normal_pdf(x, mu, sigma):
    """正态分布 PDF：钟形曲线"""
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


# ============================================================
# 期望值与方差
# ============================================================

def expected_value(values, probabilities):
    """期望值：概率加权的平均结果"""
    return sum(v * p for v, p in zip(values, probabilities))


def variance(values, probabilities):
    """方差：围绕均值的期望平方偏差"""
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))


# ============================================================
# 采样
# ============================================================

def sample_bernoulli(p, n=1):
    """从伯努利分布采样"""
    return [1 if random.random() < p else 0 for _ in range(n)]


def sample_categorical(probs, n=1):
    """从分类分布采样（累积分布法）"""
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples


def sample_uniform(a, b, n=1):
    """从均匀分布采样"""
    return [a + (b - a) * random.random() for _ in range(n)]


def sample_normal_box_muller(mu, sigma, n=1):
    """Box-Muller 变换：从均匀分布生成正态分布样本"""
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples


# ============================================================
# Softmax 与对数概率
# ============================================================

def softmax(logits):
    """Softmax：将 logits 转换为概率分布（数值稳定版）"""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]  # 减去最大 logit 防止溢出
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(logits):
    """Log-softmax：softmax 的对数，数值稳定"""
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]


def cross_entropy_loss(logits, target_index):
    """交叉熵损失：负对数似然"""
    log_probs = log_softmax(logits)
    return -log_probs[target_index]


# ============================================================
# 联合分布与边缘分布
# ============================================================

def joint_to_marginals(joint):
    """从联合分布计算边缘分布"""
    rows = len(joint)
    cols = len(joint[0])
    marginal_x = [sum(joint[i][j] for j in range(cols)) for i in range(rows)]
    marginal_y = [sum(joint[i][j] for i in range(rows)) for j in range(cols)]
    return marginal_x, marginal_y


def check_independence(joint, marginal_x, marginal_y, tol=1e-9):
    """检验联合分布是否独立：P(X,Y) == P(X)*P(Y) ?"""
    for i in range(len(marginal_x)):
        for j in range(len(marginal_y)):
            if abs(joint[i][j] - marginal_x[i] * marginal_y[j]) > tol:
                return False
    return True


# ============================================================
# 中心极限定理演示
# ============================================================

def demonstrate_clt(dist_fn, n_per_sample, n_averages):
    """演示中心极限定理：样本平均趋向正态"""
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_per_sample)]
        averages.append(sum(samples) / len(samples))
    return averages


# ============================================================
# 主程序：演示所有概念
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("概率与分布")
    print("=" * 60)

    print("\n--- 条件概率 ---")
    p_king_given_face = conditional_probability(4 / 52, 12 / 52)
    print(f"P(King | 人头牌) = {p_king_given_face:.4f}")

    print("\n--- PMF：伯努利 (p=0.7) ---")
    for k in [0, 1]:
        print(f"  P(X={k}) = {bernoulli_pmf(k, 0.7):.4f}")

    print("\n--- PMF：分类分布 ---")
    cat_probs = [0.1, 0.3, 0.4, 0.2]
    for k, p in enumerate(cat_probs):
        print(f"  P(X={k}) = {categorical_pmf(k, cat_probs):.4f}")

    print("\n--- PMF：泊松 (lambda=3) ---")
    for k in range(10):
        print(f"  P(X={k}) = {poisson_pmf(k, 3):.4f}")

    print("\n--- PDF：正态 (mu=0, sigma=1) ---")
    for x in [-3, -2, -1, 0, 1, 2, 3]:
        print(f"  f({x:+d}) = {normal_pdf(x, 0, 1):.4f}")

    print("\n--- 期望值与方差 ---")
    die_values = [1, 2, 3, 4, 5, 6]
    die_probs = [1 / 6] * 6
    mu = expected_value(die_values, die_probs)
    var = variance(die_values, die_probs)
    print(f"  公平骰子: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var ** 0.5:.4f}")

    print("\n--- 采样：伯努利 (p=0.3, n=20) ---")
    bern_samples = sample_bernoulli(0.3, 20)
    print(f"  样本: {bern_samples}")
    print(f"  经验均值: {sum(bern_samples) / len(bern_samples):.4f} (期望 0.3)")

    print("\n--- 采样：分类分布 ---")
    cat_samples = sample_categorical([0.1, 0.3, 0.4, 0.2], 1000)
    counts = [cat_samples.count(i) for i in range(4)]
    print(f"  1000 次采样计数: {counts}")
    print(f"  经验频率: {[c / 1000 for c in counts]}")
    print(f"  理论概率:  [0.1, 0.3, 0.4, 0.2]")

    print("\n--- 采样：正态 (Box-Muller) ---")
    norm_samples = sample_normal_box_muller(0, 1, 10000)
    sample_mean = sum(norm_samples) / len(norm_samples)
    sample_var = sum((x - sample_mean) ** 2 for x in norm_samples) / len(norm_samples)
    print(f"  10000 个 N(0,1) 样本:")
    print(f"  样本均值: {sample_mean:.4f} (期望 0)")
    print(f"  样本方差: {sample_var:.4f} (期望 1)")

    print("\n--- Softmax ---")
    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)
    print(f"  Logits:  {logits}")
    print(f"  Softmax: [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"  求和:    {sum(probs):.4f}")

    print("\n--- 大 logits 的 Softmax (稳定性测试) ---")
    large_logits = [100, 101, 102]
    probs_large = softmax(large_logits)
    print(f"  Logits:  {large_logits}")
    print(f"  Softmax: [{', '.join(f'{p:.4f}' for p in probs_large)}]")
    print(f"  (无溢出，因为我们先减去了最大值)")

    print("\n--- 对数概率 ---")
    log_probs = log_softmax(logits)
    print(f"  Logits:      {logits}")
    print(f"  Log-softmax: [{', '.join(f'{lp:.4f}' for lp in log_probs)}]")
    print(f"  验证 exp:    [{', '.join(f'{math.exp(lp):.4f}' for lp in log_probs)}]")

    print("\n--- 交叉熵损失 ---")
    ce = cross_entropy_loss([2.0, 1.0, 0.1], target_index=0)
    print(f"  Logits: [2.0, 1.0, 0.1], 目标: 0")
    print(f"  交叉熵损失: {ce:.4f}")

    print("\n--- 对数概率为什么重要 ---")
    word_prob = 0.01
    n_words = 50
    raw_product = word_prob ** n_words
    log_sum = n_words * math.log(word_prob)
    print(f"  P(词)^{n_words} = {word_prob}^{n_words}")
    print(f"  原始乘积: {raw_product:.2e} (更多项会下溢)")
    print(f"  对数和:     {log_sum:.4f} (稳定)")
    print(f"  恢复概率:   {math.exp(log_sum):.2e}")

    print("\n--- 联合分布与边缘分布 ---")
    joint = [
        [0.40, 0.10],
        [0.05, 0.45],
    ]
    marginal_x, marginal_y = joint_to_marginals(joint)
    print(f"  联合分布 (天气 x 带伞):")
    print(f"    晴天, 不带伞: {joint[0][0]}")
    print(f"    晴天, 带伞:    {joint[0][1]}")
    print(f"    雨天, 不带伞: {joint[1][0]}")
    print(f"    雨天, 带伞:    {joint[1][1]}")
    print(f"  边缘 X (天气):  {marginal_x}")
    print(f"  边缘 Y (带伞):  {marginal_y}")
    print(f"  是否独立? {check_independence(joint, marginal_x, marginal_y)}")

    print("\n--- 中心极限定理 ---")
    print("  对均匀 [0,1) 样本取平均:")
    for n in [1, 2, 5, 30]:
        avgs = demonstrate_clt(random.random, n, 10000)
        avg_mean = sum(avgs) / len(avgs)
        avg_std = (sum((x - avg_mean) ** 2 for x in avgs) / len(avgs)) ** 0.5
        print(f"    n={n:2d}: 均值={avg_mean:.4f}, 标准差={avg_std:.4f}")
    print("  随着 n 增大，标准差缩小，分布趋向正态。")

    print("\n--- 可视化 ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        ax = axes[0][0]
        ax.set_title("伯努利 PMF (p=0.7)")
        ax.bar([0, 1], [bernoulli_pmf(0, 0.7), bernoulli_pmf(1, 0.7)],
               color=["#4a90d9", "#d94a4a"], width=0.4)
        ax.set_xlabel("k")
        ax.set_ylabel("P(X=k)")
        ax.set_xticks([0, 1])

        ax = axes[0][1]
        ax.set_title("泊松 PMF (lambda=3)")
        ks = list(range(12))
        ax.bar(ks, [poisson_pmf(k, 3) for k in ks], color="#4a90d9", width=0.6)
        ax.set_xlabel("k")
        ax.set_ylabel("P(X=k)")

        ax = axes[0][2]
        ax.set_title("正态 PDF")
        xs = [i * 0.01 - 5 for i in range(1001)]
        for mu_val, sigma_val, label in [(0, 1, "N(0,1)"), (0, 2, "N(0,2)"), (2, 0.5, "N(2,0.5)")]:
            ys = [normal_pdf(x, mu_val, sigma_val) for x in xs]
            ax.plot(xs, ys, label=label, linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()

        ax = axes[1][0]
        ax.set_title("均匀 PDF [a=1, b=4]")
        xs_u = [i * 0.01 - 1 for i in range(701)]
        ys_u = [uniform_pdf(x, 1, 4) for x in xs_u]
        ax.plot(xs_u, ys_u, color="#4a90d9", linewidth=2)
        ax.fill_between(xs_u, ys_u, alpha=0.3, color="#4a90d9")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_ylim(0, 0.5)

        ax = axes[1][1]
        ax.set_title("中心极限定理")
        for n_val, color in [(1, "#aaaaaa"), (2, "#88aacc"), (5, "#4a90d9"), (30, "#d94a4a")]:
            avgs = demonstrate_clt(random.random, n_val, 10000)
            ax.hist(avgs, bins=50, alpha=0.5, color=color, label=f"n={n_val}", density=True)
        ax.set_xlabel("样本均值")
        ax.set_ylabel("密度")
        ax.legend()

        ax = axes[1][2]
        ax.set_title("Softmax 输出")
        logit_sets = [
            ([1, 1, 1], "相等 [1,1,1]"),
            ([2, 1, 0], "分散 [2,1,0]"),
            ([10, 1, 0], "尖锐 [10,1,0]"),
        ]
        x_positions = range(3)
        width = 0.25
        for idx, (lg, label) in enumerate(logit_sets):
            sm = softmax(lg)
            offset = (idx - 1) * width
            ax.bar([x + offset for x in x_positions], sm, width=width, label=label)
        ax.set_xlabel("类别")
        ax.set_ylabel("概率")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(["类别 0", "类别 1", "类别 2"])
        ax.legend()

        plt.tight_layout()
        plt.savefig("probability_distributions.png", dpi=150)
        print("  已保存: probability_distributions.png")
        plt.close()

    except ImportError:
        print("  matplotlib 不可用，跳过可视化。")

    print("\n" + "=" * 60)
    print("所有概率计算完成。")
    print("=" * 60)
