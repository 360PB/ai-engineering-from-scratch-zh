import math
import random

random.seed(42)


def sample_uniform(a, b):
    """从 [a, b] 范围内均匀采样"""
    return a + (b - a) * random.random()


def sample_exponential_inverse_cdf(lam):
    """使用逆 CDF 方法从指数分布采样

    CDF: F(x) = 1 - exp(-lambda * x)
    逆 CDF: F^(-1)(u) = -ln(u) / lambda
    """
    u = random.random()
    return -math.log(u) / lam


def verify_inverse_cdf(lam, n=10000):
    """验证逆 CDF 采样的正确性"""
    samples = [sample_exponential_inverse_cdf(lam) for _ in range(n)]
    empirical_mean = sum(samples) / len(samples)
    theoretical_mean = 1.0 / lam
    print(f"  指数分布(lambda={lam}): 经验均值={empirical_mean:.4f}, "
          f"理论均值={theoretical_mean:.4f}")
    return samples


def normal_pdf(x, mu, sigma):
    """计算正态分布的概率密度函数"""
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


def sample_normal_box_muller(mu=0.0, sigma=1.0):
    """使用 Box-Muller 变换从正态分布采样

    不依赖库函数实现
    """
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + sigma * z


def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    """拒绝采样

    从提议分布中采样，以与 target/proposal 成比例的概率接受。

    Args:
        target_pdf: 目标分布的概率密度函数（可非规范化）
        proposal_sample: 提议分布的采样函数
        proposal_pdf: 提议分布的概率密度函数
        M: 上界，要求 target(x) <= M * proposal(x) 对所有 x 成立
    """
    attempts = 0
    while True:
        x = proposal_sample()
        u = random.random()
        attempts += 1
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x, attempts


def rejection_sample_batch(target_pdf, proposal_sample, proposal_pdf, M, n):
    """批量拒绝采样：返回 n 个样本和接受率"""
    samples = []
    total_attempts = 0
    for _ in range(n):
        x, attempts = rejection_sample(target_pdf, proposal_sample, proposal_pdf, M)
        samples.append(x)
        total_attempts += attempts
    acceptance_rate = n / total_attempts
    return samples, acceptance_rate


def truncated_normal_demo(mu, sigma, a, b, n=5000):
    """截断正态分布演示：在 [a, b] 范围内采样

    使用均匀提议 + 拒绝采样
    """
    norm_const = sum(
        normal_pdf(a + (b - a) * i / 1000, mu, sigma) * (b - a) / 1000
        for i in range(1001)
    )
    M_val = max(
        normal_pdf(x, mu, sigma) / (1.0 / (b - a))
        for x in [a + (b - a) * i / 200 for i in range(201)]
    )

    def target(x):
        if a <= x <= b:
            return normal_pdf(x, mu, sigma)
        return 0.0

    def proposal():
        return sample_uniform(a, b)

    def proposal_pdf(x):
        if a <= x <= b:
            return 1.0 / (b - a)
        return 0.0

    samples, acc_rate = rejection_sample_batch(target, proposal, proposal_pdf, M_val, n)
    return samples, acc_rate


def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    """重要性采样：估计 target 分布下的期望

    用于当你无法直接从 target 采样但可以评估其密度时。
    """
    weighted_sum = 0.0
    weight_sum = 0.0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        weighted_sum += f(x) * w
        weight_sum += w
    unnormalized = weighted_sum / n
    self_normalized = weighted_sum / weight_sum
    return unnormalized, self_normalized


def importance_sampling_demo():
    """重要性采样演示：估计 E[X^2] under N(mu, sigma^2)"""
    mu, sigma = 2.0, 1.0
    a, b = -3.0, 7.0

    def f(x):
        return x ** 2

    def target(x):
        return normal_pdf(x, mu, sigma)

    def proposal():
        return sample_uniform(a, b)

    def proposal_pdf(x):
        if a <= x <= b:
            return 1.0 / (b - a)
        return 0.0

    est_unnorm, est_selfnorm = importance_sampling_estimate(
        f, target, proposal_pdf, proposal, n=50000
    )
    theoretical = mu ** 2 + sigma ** 2
    print(f"  N({mu},{sigma}) 下的 E[X^2]:")
    print(f"    未归一化 IS: {est_unnorm:.4f}")
    print(f"    自归一化 IS: {est_selfnorm:.4f}")
    print(f"    理论值: {theoretical:.4f}")


def monte_carlo_pi(n):
    """蒙特卡洛方法估计 pi

    在 [-1, 1] x [-1, 1] 中均匀采样点，
    落在单位圆内的比例趋近于 pi/4。
    """
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x * x + y * y <= 1:
            inside += 1
    return 4 * inside / n


def monte_carlo_integral(f, a, b, n):
    """蒙特卡洛积分估计：f(x) 在 [a, b] 上的积分"""
    total = 0.0
    for _ in range(n):
        x = sample_uniform(a, b)
        total += f(x)
    return (b - a) * total / n


def metropolis_hastings(target_log_pdf, x0, n_samples, burn_in, proposal_std=1.0):
    """Metropolis-Hastings MCMC：一维版本

    构建马尔可夫链，其平稳分布是目标分布。

    Args:
        target_log_pdf: 目标分布的对数概率密度函数（只需非规范化）
        x0: 初始值
        n_samples: 采样数量
        burn_in: 预烧期（丢弃的初始样本数）
        proposal_std: 提议分布的标准差（均匀随机游走）
    """
    samples = []
    x = x0
    accepted = 0
    total = n_samples + burn_in

    for i in range(total):
        x_new = x + random.gauss(0, proposal_std)
        log_alpha = target_log_pdf(x_new) - target_log_pdf(x)

        if math.log(random.random() + 1e-300) < log_alpha:
            x = x_new
            if i >= burn_in:
                accepted += 1

        if i >= burn_in:
            samples.append(x)

    acceptance_rate = accepted / n_samples
    return samples, acceptance_rate


def metropolis_hastings_2d(target_log_pdf, x0, y0, n_samples, burn_in, proposal_std=0.5):
    """Metropolis-Hastings MCMC：二维版本（独立高斯提议）"""
    samples = []
    x, y = x0, y0
    accepted = 0
    total = n_samples + burn_in

    for i in range(total):
        x_new = x + random.gauss(0, proposal_std)
        y_new = y + random.gauss(0, proposal_std)
        log_alpha = target_log_pdf(x_new, y_new) - target_log_pdf(x, y)

        if math.log(random.random() + 1e-300) < log_alpha:
            x, y = x_new, y_new
            if i >= burn_in:
                accepted += 1

        if i >= burn_in:
            samples.append((x, y))

    acceptance_rate = accepted / n_samples
    return samples, acceptance_rate


def bimodal_log_pdf(x):
    """双峰分布的对数密度：两个高斯的混合（0.4 和 0.6）"""
    p1 = 0.4 * normal_pdf(x, -3, 1)
    p2 = 0.6 * normal_pdf(x, 3, 1)
    return math.log(p1 + p2 + 1e-300)


def gibbs_sampling_2d(rho, n_samples, burn_in):
    """吉布斯采样：二维高斯（相关系数为 rho）

    交替从两个条件分布中采样。
    条件分布都是一维高斯。
    """
    x, y = 0.0, 0.0
    samples = []

    for i in range(n_samples + burn_in):
        x = random.gauss(rho * y, math.sqrt(1 - rho ** 2))
        y = random.gauss(rho * x, math.sqrt(1 - rho ** 2))
        if i >= burn_in:
            samples.append((x, y))

    return samples


def softmax(logits):
    """将 logit 向量转换为概率分布"""
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def sample_from_probs(probs):
    """根据概率向量采样：返回索引

    使用累积分布函数（CDF）逆变换
    """
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1


def temperature_sample(logits, temperature):
    """温度采样：用于 LLM token 生成

    T = 1.0: 标准 softmax
    T < 1:  更有信心（更确定），减少多样性
    T > 1:  更没信心，增加多样性
    T = 0:  贪婪（总是选最高 logit）
    """
    if temperature <= 0:
        return logits.index(max(logits))
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)


def temperature_distribution(logits, temperature):
    """获取温度缩放后的概率分布"""
    if temperature <= 0:
        result = [0.0] * len(logits)
        result[logits.index(max(logits))] = 1.0
        return result
    scaled = [z / temperature for z in logits]
    return softmax(scaled)


def top_k_sample(logits, k):
    """Top-k 采样：限制为概率最高的 k 个 token

    k = 1:  贪婪解码
    k = V:  不过滤
    """
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]


def top_k_distribution(logits, k):
    """Top-k 采样返回的概率分布"""
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    result = [0.0] * len(logits)
    top_indices = [idx for idx, _ in indexed[:k]]
    top_probs = [probs[idx] for idx in top_indices]
    total = sum(top_probs)
    for idx in top_indices:
        result[idx] = probs[idx] / total
    return result


def top_p_sample(logits, p):
    """Top-p（核）采样：动态调整候选集合

    保留累积概率超过 p 的最小 token 集合。
    p = 0.9:  保留覆盖 90% 概率质量的 token
    p = 1.0:  不过滤
    """
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0.0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]


def top_p_distribution(logits, p):
    """Top-p 采样返回的概率分布"""
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0.0
    selected_indices = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected_indices.append(token_idx)
        if cumsum >= p:
            break
    result = [0.0] * len(logits)
    total = sum(probs[i] for i in selected_indices)
    for i in selected_indices:
        result[i] = probs[i] / total
    return result


def reparam_sample(mu, sigma):
    """重参数化采样：z = mu + sigma * epsilon

    将随机性移到一个参数无关的来源（epsilon），
    使采样操作可微分。
    """
    epsilon = random.gauss(0, 1)
    z = mu + sigma * epsilon
    return z, epsilon


def reparam_gradient(epsilon):
    """重参数化采样路径上的梯度

    z = mu + sigma * epsilon
    dz/dmu = 1
    dz/dsigma = epsilon
    """
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma


def vae_forward_demo(mu, log_var):
    """VAE 前向传播演示：展示梯度如何流经重参数化采样

    使用 log_var 而不是 variance，直接在 reparametrization 中使用 sigma。
    """
    sigma = math.exp(0.5 * log_var)
    z, epsilon = reparam_sample(mu, sigma)

    dz_dmu = 1.0
    dz_dsigma = epsilon
    dsigma_dlogvar = 0.5 * sigma

    dz_dmu_total = dz_dmu
    dz_dlogvar = dz_dsigma * dsigma_dlogvar

    return z, epsilon, dz_dmu_total, dz_dlogvar


def gumbel_sample():
    """Gumbel 分布采样：g = -ln(-ln(u)), u ~ Uniform(0, 1)"""
    u = random.random()
    while u == 0:
        u = random.random()
    return -math.log(-math.log(u))


def gumbel_max_sample(log_probs):
    """Gumbel-Max 技巧：精确分类采样（不可微分）

    对每个类别的 log 概率加上 Gumbel 噪声，选择最大值。
    """
    gumbels = [lp + gumbel_sample() for lp in log_probs]
    return gumbels.index(max(gumbels))


def gumbel_softmax_sample(log_probs, temperature):
    """Gumbel-Softmax：可微分分类采样近似

    tau -> 0:  趋近硬 one-hot
    tau -> inf: 趋近均匀分布
    """
    gumbels = [math.log(p) + gumbel_sample() for p in log_probs]
    scaled = [g / temperature for g in gumbels]
    return softmax(scaled)


def gumbel_softmax_straight_through(log_probs, temperature):
    """直通估计器版本的 Gumbel-Softmax

    前向传播使用硬 argmax，反向传播使用软 softmax 梯度
    """
    soft = gumbel_softmax_sample(log_probs, temperature)
    hard_idx = soft.index(max(soft))
    hard = [0.0] * len(log_probs)
    hard[hard_idx] = 1.0
    return hard, soft


def stratified_sample_1d(n):
    """分层采样：强制均匀覆盖的采样策略

    将 [0, 1] 分成 n 个等层，每层中均匀采样一个点。
    方差始终小于或等于标准蒙特卡洛。
    """
    samples = []
    for i in range(n):
        u = random.random()
        samples.append((i + u) / n)
    return samples


def compare_sampling_variance(f, n, n_trials=200):
    """比较标准采样与分层采样的方差"""
    standard_estimates = []
    stratified_estimates = []

    for _ in range(n_trials):
        standard_samples = [random.random() for _ in range(n)]
        standard_est = sum(f(x) for x in standard_samples) / n
        standard_estimates.append(standard_est)

        strat_samples = stratified_sample_1d(n)
        strat_est = sum(f(x) for x in strat_samples) / n
        stratified_estimates.append(strat_est)

    std_mean = sum(standard_estimates) / len(standard_estimates)
    std_var = sum((e - std_mean) ** 2 for e in standard_estimates) / len(standard_estimates)

    strat_mean = sum(stratified_estimates) / len(stratified_estimates)
    strat_var = sum((e - strat_mean) ** 2 for e in stratified_estimates) / len(stratified_estimates)

    return std_var, strat_var


def text_generation_demo(vocab, logits, length, method, **kwargs):
    """文本生成演示：使用不同采样策略生成 token 序列"""
    tokens = []
    for _ in range(length):
        if method == "greedy":
            idx = logits.index(max(logits))
        elif method == "temperature":
            idx = temperature_sample(logits, kwargs.get("temperature", 1.0))
        elif method == "top_k":
            idx = top_k_sample(logits, kwargs.get("k", 5))
        elif method == "top_p":
            idx = top_p_sample(logits, kwargs.get("p", 0.9))
        else:
            idx = sample_from_probs(softmax(logits))
        tokens.append(vocab[idx])
    return " ".join(tokens)


if __name__ == "__main__":
    print("=" * 65)
    print("采样方法")
    print("=" * 65)

    print("\n--- 1. 逆 CDF 采样（指数分布） ---")
    for lam in [0.5, 1.0, 2.0]:
        verify_inverse_cdf(lam)

    print("\n--- 2. 拒绝采样（截断正态） ---")
    trunc_samples, acc = truncated_normal_demo(0, 1, -1, 2, n=5000)
    trunc_mean = sum(trunc_samples) / len(trunc_samples)
    print(f"  截断 N(0,1) 在 [-1, 2]: 均值={trunc_mean:.4f}, "
          f"接受率={acc:.4f}")

    print("\n--- 3. 重要性采样 ---")
    importance_sampling_demo()

    print("\n--- 4. 蒙特卡洛估计 ---")
    print("  估计 pi:")
    for n in [1000, 10000, 100000]:
        pi_est = monte_carlo_pi(n)
        error = abs(pi_est - math.pi)
        print(f"    N={n:>7d}: pi ~ {pi_est:.6f}, 误差={error:.6f}")

    print("  估计 sin(x) 从 0 到 pi 的积分（真实值 = 2.0）:")
    for n in [1000, 10000, 100000]:
        est = monte_carlo_integral(math.sin, 0, math.pi, n)
        error = abs(est - 2.0)
        print(f"    N={n:>7d}: 估计={est:.6f}, 误差={error:.6f}")

    print("\n--- 5. Metropolis-Hastings MCMC ---")
    print("  从双峰分布采样（高斯混合）:")
    for std in [0.5, 1.0, 3.0]:
        samples_mh, acc_rate = metropolis_hastings(
            bimodal_log_pdf, x0=0.0, n_samples=10000, burn_in=2000,
            proposal_std=std
        )
        mh_mean = sum(samples_mh) / len(samples_mh)
        mh_std = (sum((x - mh_mean) ** 2 for x in samples_mh) / len(samples_mh)) ** 0.5
        print(f"    proposal_std={std}: 均值={mh_mean:.4f}, 标准差={mh_std:.4f}, "
              f"接受率={acc_rate:.4f}")

    print("\n  从二维高斯采样:")
    def gaussian_2d_log_pdf(x, y):
        return -0.5 * (x ** 2 + y ** 2)

    samples_2d, acc_2d = metropolis_hastings_2d(
        gaussian_2d_log_pdf, x0=5.0, y0=5.0,
        n_samples=10000, burn_in=2000, proposal_std=1.0
    )
    xs = [s[0] for s in samples_2d]
    ys = [s[1] for s in samples_2d]
    print(f"    均值_x={sum(xs)/len(xs):.4f}, 均值_y={sum(ys)/len(ys):.4f}, "
          f"接受率={acc_2d:.4f}")

    print("\n--- 6. 吉布斯采样（相关二维高斯） ---")
    for rho in [0.0, 0.5, 0.9]:
        gibbs_samples = gibbs_sampling_2d(rho, n_samples=10000, burn_in=1000)
        gx = [s[0] for s in gibbs_samples]
        gy = [s[1] for s in gibbs_samples]
        empirical_corr = (
            sum(a * b for a, b in gibbs_samples) / len(gibbs_samples)
            - (sum(gx) / len(gx)) * (sum(gy) / len(gy))
        )
        print(f"  rho={rho}: 经验相关系数={empirical_corr:.4f}")

    print("\n--- 7. 温度采样 ---")
    token_logits = [3.0, 2.0, 1.5, 0.5, -1.0, -2.0]
    vocab = ["the", "a", "this", "one", "that", "some"]
    print("  Logit 值:", token_logits)
    print("  词表:", vocab)
    for temp in [0.1, 0.5, 1.0, 1.5, 3.0]:
        dist = temperature_distribution(token_logits, temp)
        formatted = [f"{p:.4f}" for p in dist]
        print(f"  T={temp:.1f}: [{', '.join(formatted)}]")

    print("\n  不同温度下的生成样本:")
    for temp in [0.1, 0.7, 1.0, 2.0]:
        tokens_out = []
        for _ in range(10):
            idx = temperature_sample(token_logits, temp)
            tokens_out.append(vocab[idx])
        print(f"    T={temp}: {' '.join(tokens_out)}")

    print("\n--- 8. Top-k 采样 ---")
    print("  Logit 值:", token_logits)
    for k in [1, 2, 3, 6]:
        dist = top_k_distribution(token_logits, k)
        formatted = [f"{p:.4f}" for p in dist]
        print(f"  k={k}: [{', '.join(formatted)}]")

    print("\n--- 9. Top-p（核）采样 ---")
    print("  Logit 值:", token_logits)
    for p in [0.5, 0.8, 0.9, 0.95, 1.0]:
        dist = top_p_distribution(token_logits, p)
        nonzero = sum(1 for d in dist if d > 0)
        formatted = [f"{d:.4f}" for d in dist]
        print(f"  p={p:.2f}: [{', '.join(formatted)}] ({nonzero} 个 token)")

    print("\n--- 10. 重参数化技巧 ---")
    mu_val, log_var_val = 2.0, 0.5

    print(f"  VAE 编码器输出: mu={mu_val}, log_var={log_var_val}")
    z_samples = []
    for trial in range(5):
        z, eps, dz_dmu, dz_dlogvar = vae_forward_demo(mu_val, log_var_val)
        z_samples.append(z)
        print(f"    试验 {trial+1}: z={z:.4f}, eps={eps:.4f}, "
              f"dz/dmu={dz_dmu:.4f}, dz/dlog_var={dz_dlogvar:.4f}")

    print(f"  z 样本的均值: {sum(z_samples)/len(z_samples):.4f} "
          f"(期望约 {mu_val})")
    print("  梯度存在因为 z = mu + sigma * epsilon 是可微分的。")

    print("\n  验证重参数化与直接采样一致:")
    sigma_val = math.exp(0.5 * log_var_val)
    direct_samples = [sample_normal_box_muller(mu_val, sigma_val) for _ in range(10000)]
    reparam_samples = [reparam_sample(mu_val, sigma_val)[0] for _ in range(10000)]
    d_mean = sum(direct_samples) / len(direct_samples)
    r_mean = sum(reparam_samples) / len(reparam_samples)
    d_std = (sum((x - d_mean)**2 for x in direct_samples) / len(direct_samples)) ** 0.5
    r_std = (sum((x - r_mean)**2 for x in reparam_samples) / len(reparam_samples)) ** 0.5
    print(f"    直接采样:  均值={d_mean:.4f}, 标准差={d_std:.4f}")
    print(f"    重参数化:  均值={r_mean:.4f}, 标准差={r_std:.4f}")

    print("\n--- 11. Gumbel-Softmax ---")
    probs = [0.5, 0.3, 0.15, 0.05]
    log_probs = [math.log(p) for p in probs]
    labels = ["cat", "dog", "bird", "fish"]
    print(f"  真实概率: {probs}")

    print("\n  Gumbel-Max（精确分类）验证:")
    counts = [0] * len(probs)
    n_gumbel = 10000
    for _ in range(n_gumbel):
        idx = gumbel_max_sample(log_probs)
        counts[idx] += 1
    empirical = [c / n_gumbel for c in counts]
    print(f"    经验分布: [{', '.join(f'{p:.4f}' for p in empirical)}]")
    print(f"    真实分布: [{', '.join(f'{p:.4f}' for p in probs)}]")

    print("\n  不同温度下的 Gumbel-Softmax:")
    for tau in [0.1, 0.5, 1.0, 5.0]:
        soft = gumbel_softmax_sample(log_probs, tau)
        formatted = [f"{s:.4f}" for s in soft]
        max_idx = soft.index(max(soft))
        print(f"    tau={tau:.1f}: [{', '.join(formatted)}] -> {labels[max_idx]}")

    print("\n  直通估计器:")
    hard, soft = gumbel_softmax_straight_through(log_probs, temperature=0.5)
    print(f"    硬（forward）: {hard}")
    print(f"    软（backward）: [{', '.join(f'{s:.4f}' for s in soft)}]")

    print("\n--- 12. 分层采样 ---")
    def test_fn(x):
        return math.sin(math.pi * x)

    print("  比较标准 vs 分层蒙特卡洛:")
    print(f"  函数: sin(pi*x) 在 [0,1], 真实积分 = 2/pi = {2/math.pi:.6f}")
    for n in [10, 50, 100]:
        std_var, strat_var = compare_sampling_variance(test_fn, n)
        ratio = std_var / strat_var if strat_var > 0 else float('inf')
        print(f"    N={n:3d}: 标准方差={std_var:.8f}, "
              f"分层方差={strat_var:.8f}, 比值={ratio:.2f}x")

    print("\n--- 13. 文本生成演示 ---")
    gen_vocab = ["the", "cat", "sat", "on", "mat", "a", "dog", "ran", "big", "red"]
    gen_logits = [3.0, 2.5, 2.0, 1.8, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0]

    print(f"  词表: {gen_vocab}")
    print(f"  Logit 值: {gen_logits}")
    print()

    methods = [
        ("greedy", {}),
        ("temperature", {"temperature": 0.5}),
        ("temperature", {"temperature": 1.0}),
        ("temperature", {"temperature": 2.0}),
        ("top_k", {"k": 3}),
        ("top_p", {"p": 0.8}),
    ]

    for method_name, params in methods:
        label = method_name
        if params:
            label += f"({', '.join(f'{k}={v}' for k, v in params.items())})"
        sequences = []
        for run in range(3):
            seq = text_generation_demo(gen_vocab, gen_logits, length=8,
                                       method=method_name, **params)
            sequences.append(seq)
        print(f"  {label}:")
        for i, seq in enumerate(sequences):
            print(f"    运行 {i+1}: {seq}")
        unique = len(set(sequences))
        print(f"    唯一序列数: {unique}/3")
        print()

    print("\n--- 14. 可视化 ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))

        ax = axes[0][0]
        ax.set_title("逆 CDF：指数分布样本")
        exp_samples = [sample_exponential_inverse_cdf(1.0) for _ in range(10000)]
        ax.hist(exp_samples, bins=60, density=True, alpha=0.7, color="#4a90d9",
                label="样本")
        xs_exp = [i * 0.05 for i in range(160)]
        ys_exp = [math.exp(-x) for x in xs_exp]
        ax.plot(xs_exp, ys_exp, "r-", linewidth=2, label="真实 PDF")
        ax.set_xlabel("x")
        ax.set_ylabel("密度")
        ax.legend()

        ax = axes[0][1]
        ax.set_title("拒绝采样：截断正态")
        rej_samples, _ = truncated_normal_demo(0, 1, -1, 2, n=5000)
        ax.hist(rej_samples, bins=50, density=True, alpha=0.7, color="#4a90d9",
                label="样本")
        xs_tn = [-1 + 3 * i / 200 for i in range(201)]
        ys_tn = [normal_pdf(x, 0, 1) for x in xs_tn]
        area = sum(ys_tn) * 3 / 200
        ys_tn_norm = [y / area for y in ys_tn]
        ax.plot(xs_tn, ys_tn_norm, "r-", linewidth=2, label="真实 PDF（归一化）")
        ax.set_xlabel("x")
        ax.set_ylabel("密度")
        ax.legend()

        ax = axes[0][2]
        ax.set_title("蒙特卡洛：估计 pi")
        n_mc_vis = 5000
        mc_x = [random.uniform(-1, 1) for _ in range(n_mc_vis)]
        mc_y = [random.uniform(-1, 1) for _ in range(n_mc_vis)]
        inside_x = [mc_x[i] for i in range(n_mc_vis) if mc_x[i]**2 + mc_y[i]**2 <= 1]
        inside_y = [mc_y[i] for i in range(n_mc_vis) if mc_x[i]**2 + mc_y[i]**2 <= 1]
        outside_x = [mc_x[i] for i in range(n_mc_vis) if mc_x[i]**2 + mc_y[i]**2 > 1]
        outside_y = [mc_y[i] for i in range(n_mc_vis) if mc_x[i]**2 + mc_y[i]**2 > 1]
        ax.scatter(inside_x, inside_y, s=1, c="#4a90d9", alpha=0.5)
        ax.scatter(outside_x, outside_y, s=1, c="#d94a4a", alpha=0.5)
        theta = [2 * math.pi * i / 200 for i in range(201)]
        circle_x = [math.cos(t) for t in theta]
        circle_y = [math.sin(t) for t in theta]
        ax.plot(circle_x, circle_y, "k-", linewidth=1.5)
        ax.set_aspect("equal")
        pi_est = 4 * len(inside_x) / n_mc_vis
        ax.set_xlabel(f"pi ~ {pi_est:.4f}")

        ax = axes[1][0]
        ax.set_title("MCMC：双峰分布")
        mcmc_samples, _ = metropolis_hastings(
            bimodal_log_pdf, x0=0.0, n_samples=20000, burn_in=5000, proposal_std=2.0
        )
        ax.hist(mcmc_samples, bins=80, density=True, alpha=0.7, color="#4a90d9",
                label="MCMC 样本")
        xs_bm = [-8 + 16 * i / 400 for i in range(401)]
        ys_bm = [math.exp(bimodal_log_pdf(x)) for x in xs_bm]
        area_bm = sum(ys_bm) * 16 / 400
        ys_bm_norm = [y / area_bm for y in ys_bm]
        ax.plot(xs_bm, ys_bm_norm, "r-", linewidth=2, label="真实密度")
        ax.set_xlabel("x")
        ax.set_ylabel("密度")
        ax.legend()

        ax = axes[1][1]
        ax.set_title("吉布斯采样：二维高斯（rho=0.8）")
        gibbs_vis = gibbs_sampling_2d(0.8, n_samples=3000, burn_in=500)
        gvx = [s[0] for s in gibbs_vis]
        gvy = [s[1] for s in gibbs_vis]
        ax.scatter(gvx, gvy, s=2, alpha=0.3, c="#4a90d9")
        ax.plot(gvx[:100], gvy[:100], "r-", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

        ax = axes[1][2]
        ax.set_title("温度缩放")
        temps = [0.1, 0.5, 1.0, 2.0, 5.0]
        bar_width = 0.15
        positions = list(range(len(token_logits)))
        for t_idx, temp in enumerate(temps):
            dist = temperature_distribution(token_logits, temp)
            offset = (t_idx - 2) * bar_width
            bars = [pos + offset for pos in positions]
            ax.bar(bars, dist, bar_width, label=f"T={temp}", alpha=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(vocab, rotation=45)
        ax.set_ylabel("概率")
        ax.legend(fontsize=8)

        ax = axes[2][0]
        ax.set_title("Top-k vs Top-p 分布")
        k_dist = top_k_distribution(token_logits, k=3)
        p_dist = top_p_distribution(token_logits, p=0.9)
        full_dist = softmax(token_logits)
        x_pos = list(range(len(token_logits)))
        w = 0.25
        ax.bar([x - w for x in x_pos], full_dist, w, label="完整", alpha=0.8, color="#aaaaaa")
        ax.bar(x_pos, k_dist, w, label="Top-3", alpha=0.8, color="#4a90d9")
        ax.bar([x + w for x in x_pos], p_dist, w, label="Top-p=0.9", alpha=0.8, color="#d94a4a")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(vocab, rotation=45)
        ax.set_ylabel("概率")
        ax.legend(fontsize=8)

        ax = axes[2][1]
        ax.set_title("Gumbel-Softmax：温度效应")
        taus = [0.1, 0.5, 1.0, 5.0]
        g_log_probs = [math.log(p) for p in [0.5, 0.3, 0.15, 0.05]]
        n_trials_vis = 500
        for tau in taus:
            max_vals = []
            for _ in range(n_trials_vis):
                soft = gumbel_softmax_sample(g_log_probs, tau)
                max_vals.append(max(soft))
            ax.hist(max_vals, bins=30, alpha=0.5, label=f"tau={tau}", density=True)
        ax.set_xlabel("最大分量值")
        ax.set_ylabel("密度")
        ax.legend(fontsize=8)

        ax = axes[2][2]
        ax.set_title("分层采样 vs 标准采样")
        n_strat_vis = 20
        standard_pts = sorted([random.random() for _ in range(n_strat_vis)])
        stratified_pts = sorted(stratified_sample_1d(n_strat_vis))
        ax.scatter(standard_pts, [1] * n_strat_vis, s=30, c="#d94a4a", label="标准",
                   zorder=3)
        ax.scatter(stratified_pts, [0] * n_strat_vis, s=30, c="#4a90d9", label="分层",
                   zorder=3)
        for i in range(n_strat_vis + 1):
            ax.axvline(i / n_strat_vis, color="#cccccc", linewidth=0.5, linestyle="--")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["分层", "标准"])
        ax.set_xlabel("样本值")
        ax.legend()
        ax.set_ylim(-0.5, 1.5)

        plt.tight_layout()
        plt.savefig("sampling_methods.png", dpi=150)
        print("  已保存: sampling_methods.png")
        plt.close()

    except ImportError:
        print("  matplotlib 不可用，跳过可视化。")

    print("\n" + "=" * 65)
    print("所有采样方法完成。")
    print("=" * 65)
