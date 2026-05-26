import math
import random

random.seed(42)


def mean(data):
    """计算数据集的均值"""
    return sum(data) / len(data)


def median(data):
    """计算数据集的中位数"""
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def mode(data):
    """计算数据集的众数（出现频率最高的值）"""
    counts = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    modes.sort()
    return modes[0]


def variance(data, sample=True):
    """计算数据集的方差

    Args:
        data: 数据列表
        sample: 如果为 True，使用贝塞尔校正（除以 n-1），返回样本方差；
                如果为 False，返回总体方差
    """
    n = len(data)
    m = mean(data)
    total = sum((x - m) ** 2 for x in data)
    if sample and n > 1:
        return total / (n - 1)
    return total / n


def std_dev(data, sample=True):
    """计算数据集的标准差"""
    return math.sqrt(variance(data, sample))


def percentile(data, p):
    """计算数据集的第 p 百分位数

    Args:
        data: 数据列表
        p: 百分位（0-100）
    """
    s = sorted(data)
    n = len(s)
    k = (p / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def iqr(data):
    """计算数据集的四分位距（IQR = Q3 - Q1）"""
    return percentile(data, 75) - percentile(data, 25)


def covariance(x, y, sample=True):
    """计算两个数据集的协方差

    Args:
        x, y: 两个等长的数据列表
        sample: 如果为 True，使用贝塞尔校正（除以 n-1）
    """
    n = len(x)
    mx = mean(x)
    my = mean(y)
    total = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if sample and n > 1:
        return total / (n - 1)
    return total / n


def pearson_correlation(x, y):
    """计算皮尔逊相关系数（衡量线性相关性）

    返回值范围 [-1, 1]：
    - 1 表示完美正相关
    - -1 表示完美负相关
    - 0 表示无线性相关
    """
    n = len(x)
    mx = mean(x)
    my = mean(y)
    sx = std_dev(x, sample=False)
    sy = std_dev(y, sample=False)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    return cov / (sx * sy)


def rank_data(data):
    """将数据转换为秩次（处理并列值）

    并列值的秩是它们排位的平均值
    例如两个并列第3名的值，秩都是 (3+4)/2 = 3.5
    """
    indexed = sorted(enumerate(data), key=lambda pair: pair[1])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x, y):
    """计算斯皮尔曼秩相关系数（衡量单调相关性）

    先将数据转换为秩次，再计算皮尔逊相关。
    能捕捉任何单调关系，而不仅仅是线性的。
    """
    rx = rank_data(x)
    ry = rank_data(y)
    return pearson_correlation(rx, ry)


def covariance_matrix(data):
    """计算协方差矩阵

    Args:
        data: d 个特征列表的列表，每个内部列表是一个特征的所有样本值
    Returns:
        d x d 的协方差矩阵，对角线是各特征的方差
    """
    d = len(data)
    n = len(data[0])
    means = [mean(data[i]) for i in range(d)]
    matrix = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            cov = sum(
                (data[i][k] - means[i]) * (data[j][k] - means[j])
                for k in range(n)
            ) / (n - 1)
            matrix[i][j] = cov
            matrix[j][i] = cov
    return matrix


def t_statistic_one_sample(data, mu_0):
    """单样本 t 统计量：检验样本均值是否与假设值 mu_0 不同"""
    n = len(data)
    m = mean(data)
    s = std_dev(data, sample=True)
    return (m - mu_0) / (s / math.sqrt(n))


def t_statistic_two_sample(data1, data2):
    """双样本 t 统计量（Welch's t 检验）：检验两组均值是否不同

    不假设两组方差相等
    """
    n1 = len(data1)
    n2 = len(data2)
    m1 = mean(data1)
    m2 = mean(data2)
    v1 = variance(data1, sample=True)
    v2 = variance(data2, sample=True)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0
    return (m1 - m2) / se


def welch_df(data1, data2):
    """计算 Welch's t 检验的自由度"""
    n1 = len(data1)
    n2 = len(data2)
    v1 = variance(data1, sample=True)
    v2 = variance(data2, sample=True)
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    if denom == 0:
        return n1 + n2 - 2
    return num / denom


def t_cdf_approx(t_val, df):
    """使用正则化 beta 函数近似 t 分布的 CDF"""
    x = df / (df + t_val * t_val)
    if t_val < 0:
        return 0.5 * _regularized_beta(x, df / 2, 0.5)
    return 1.0 - 0.5 * _regularized_beta(x, df / 2, 0.5)


def _regularized_beta(x, a, b):
    """正则化不完全 beta 函数（用于 t 分布近似）"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    n_steps = 200
    total = 0.0
    dt = x / n_steps
    for i in range(n_steps):
        t = (i + 0.5) * dt
        total += t ** (a - 1) * (1 - t) ** (b - 1) * dt
    beta_val = _beta_function(a, b)
    if beta_val == 0:
        return 0.0
    return total / beta_val


def _beta_function(a, b):
    """Beta 函数 B(a, b) = Gamma(a) * Gamma(b) / Gamma(a+b)"""
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def p_value_two_sided(t_val, df):
    """双侧 p 值：给定 t 统计量和自由度"""
    p_left = t_cdf_approx(abs(t_val), df)
    return 2.0 * (1.0 - p_left)


def one_sample_ttest(data, mu_0=0):
    """单样本 t 检验：检验样本均值是否与假设值 mu_0 不同

    Returns:
        包含 t_statistic、df、p_value 的字典
    """
    n = len(data)
    t = t_statistic_one_sample(data, mu_0)
    df = n - 1
    p = p_value_two_sided(t, df)
    return {"t_statistic": t, "df": df, "p_value": p}


def two_sample_ttest(data1, data2):
    """双样本 t 检验（Welch's）：检验两组均值是否不同

    Returns:
        包含 t_statistic、df、p_value 的字典
    """
    t = t_statistic_two_sample(data1, data2)
    df = welch_df(data1, data2)
    p = p_value_two_sided(t, df)
    return {"t_statistic": t, "df": df, "p_value": p}


def paired_ttest(data1, data2):
    """配对 t 检验：在相同样本上评估的两个模型的比较

    计算差值 d_i = data1_i - data2_i，然后检验 d_i 的均值是否显著不同于 0
    """
    diffs = [a - b for a, b in zip(data1, data2)]
    return one_sample_ttest(diffs, mu_0=0)


def chi_squared_test(observed, expected):
    """卡方检验：检验观察到的频率是否与期望频率不同

    Args:
        observed: 观察到的频数列表
        expected: 期望频数列表
    Returns:
        包含 chi2、df、p_value 的字典
    """
    chi2 = sum(
        (o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0
    )
    df = len(observed) - 1
    p = chi_squared_p_value(chi2, df)
    return {"chi2": chi2, "df": df, "p_value": p}


def chi_squared_p_value(chi2, df):
    """卡方分布的 p 值（使用下不完全 gamma 函数近似）"""
    if chi2 <= 0:
        return 1.0
    return 1.0 - _lower_incomplete_gamma_ratio(df / 2.0, chi2 / 2.0)


def _lower_incomplete_gamma_ratio(a, x):
    """下不完全 gamma 函数的比值 Gamma(a, 0, x) / Gamma(a)"""
    if x <= 0:
        return 0.0
    n_steps = 500
    dt = x / n_steps
    total = 0.0
    for i in range(n_steps):
        t = (i + 0.5) * dt
        if t > 0:
            total += math.exp((a - 1) * math.log(t) - t) * dt
    gamma_a = math.exp(math.lgamma(a))
    if gamma_a == 0:
        return 0.0
    return total / gamma_a


def bootstrap_statistic(data, stat_func, n_bootstrap=5000, ci=95):
    """自助法（Bootstrap）置信区间

    通过有放回重采样构造任意统计量的置信区间，无需假设分布。

    Args:
        data: 原始数据
        stat_func: 要估计的统计量函数
        n_bootstrap: 重采样次数
        ci: 置信水平（默认 95%）
    """
    n = len(data)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # 有放回地重采样 n 个数据点
        sample = [data[random.randint(0, n - 1)] for _ in range(n)]
        bootstrap_stats.append(stat_func(sample))
    bootstrap_stats.sort()
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    ci_lower = percentile(bootstrap_stats, lower_pct)
    ci_upper = percentile(bootstrap_stats, upper_pct)
    return {
        "estimate": stat_func(data),       # 原始数据的点估计
        "ci_lower": ci_lower,              # 置信区间下界
        "ci_upper": ci_upper,              # 置信区间上界
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "std_error": std_dev(bootstrap_stats, sample=True),  # 标准误差
    }


def bootstrap_compare(data1, data2, stat_func, n_bootstrap=5000, ci=95):
    """两个样本的 Bootstrap 比较

    比较两个模型或两组数据的统计量差异，提供差异的置信区间。
    如果置信区间不包含 0，差异显著。
    """
    n1 = len(data1)
    n2 = len(data2)
    diffs = []
    for _ in range(n_bootstrap):
        s1 = [data1[random.randint(0, n1 - 1)] for _ in range(n1)]
        s2 = [data2[random.randint(0, n2 - 1)] for _ in range(n2)]
        diffs.append(stat_func(s2) - stat_func(s1))
    diffs.sort()
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    ci_lower = percentile(diffs, lower_pct)
    ci_upper = percentile(diffs, upper_pct)
    observed_diff = stat_func(data2) - stat_func(data1)
    significant = ci_lower > 0 or ci_upper < 0
    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,  # CI 不包含 0 时为 True
        "ci_level": ci,
    }


def cohens_d(data1, data2):
    """计算 Cohen's d：衡量两组之间差异的效应量

    与样本量无关，衡量差异的实际大小。

    解释：
      d < 0.2   可忽略
      0.2 <= d < 0.5  小效应
      0.5 <= d < 0.8  中效应
      d >= 0.8  大效应
    """
    m1 = mean(data1)
    m2 = mean(data2)
    n1 = len(data1)
    n2 = len(data2)
    v1 = variance(data1, sample=True)
    v2 = variance(data2, sample=True)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (m2 - m1) / pooled


def interpret_cohens_d(d):
    """解释 Cohen's d 的实际意义"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni 校正：控制多重比较中的假阳性率

    当同时检验多个假设时，将显著性阈值 alpha 除以检验数量。
    """
    m = len(p_values)
    adjusted_alpha = alpha / m
    results = []
    for p in p_values:
        results.append({
            "original_p": p,
            "adjusted_alpha": adjusted_alpha,
            "significant": p < adjusted_alpha,
        })
    return results


def generate_normal(n, mu=0, sigma=1):
    """使用 Box-Muller 变换生成 n 个正态分布随机数

    Args:
        n: 生成数量
        mu: 均值
        sigma: 标准差
    """
    samples = []
    for _ in range(n // 2 + 1):
        u1 = random.random()
        u2 = random.random()
        while u1 == 0:
            u1 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        samples.append(mu + sigma * z0)
        samples.append(mu + sigma * z1)
    return samples[:n]


def ab_test_simulator(
    n_per_group=100,
    true_effect=0.0,
    base_mean=50,
    base_std=10,
    alpha=0.05,
):
    """A/B 测试模拟器：模拟一次 A/B 测试实验

    生成两组数据，执行 t 检验，计算 Cohen's d 和 Bootstrap 比较。
    """
    group_a = generate_normal(n_per_group, base_mean, base_std)
    group_b = generate_normal(n_per_group, base_mean + true_effect, base_std)

    result = two_sample_ttest(group_a, group_b)
    d = cohens_d(group_a, group_b)
    boot = bootstrap_compare(group_a, group_b, mean, n_bootstrap=2000)

    return {
        "group_a_mean": mean(group_a),
        "group_b_mean": mean(group_b),
        "observed_diff": mean(group_b) - mean(group_a),
        "true_effect": true_effect,
        "t_test": result,
        "cohens_d": d,
        "effect_interpretation": interpret_cohens_d(d),
        "bootstrap": boot,
        "significant_ttest": result["p_value"] < alpha,
        "significant_bootstrap": boot["significant"],
    }


def run_multiple_ab_tests(
    n_tests=20,
    n_per_group=100,
    true_effect=0.0,
    alpha=0.05,
):
    """多重比较演示：展示多重比较问题

    运行多次 A/B 测试（每次都是零效果），展示未校正和 Bonferroni 校正后的结果。
    揭示：如果不做校正，20 次检验中约 1 次会偶然产生"显著"结果。
    """
    p_values = []
    significant_count = 0
    for _ in range(n_tests):
        group_a = generate_normal(n_per_group, 50, 10)
        group_b = generate_normal(n_per_group, 50 + true_effect, 10)
        result = two_sample_ttest(group_a, group_b)
        p_values.append(result["p_value"])
        if result["p_value"] < alpha:
            significant_count += 1

    corrected = bonferroni_correction(p_values, alpha)
    corrected_significant = sum(1 for r in corrected if r["significant"])

    return {
        "n_tests": n_tests,
        "true_effect": true_effect,
        "false_positive_rate": significant_count / n_tests if true_effect == 0 else None,
        "uncorrected_significant": significant_count,
        "corrected_significant": corrected_significant,
        "p_values": p_values,
    }


def statistical_vs_practical_significance(small_n=30, large_n=10000, effect=0.1):
    """统计显著性 vs 实际显著性演示

    展示大样本量如何使微小差异变得"统计显著"，
    但这些差异实际上没有实际意义。
    """
    small_a = generate_normal(small_n, 50, 10)
    small_b = generate_normal(small_n, 50 + effect, 10)
    small_result = two_sample_ttest(small_a, small_b)
    small_d = cohens_d(small_a, small_b)

    large_a = generate_normal(large_n, 50, 10)
    large_b = generate_normal(large_n, 50 + effect, 10)
    large_result = two_sample_ttest(large_a, large_b)
    large_d = cohens_d(large_a, large_b)

    return {
        "small_sample": {
            "n": small_n,
            "p_value": small_result["p_value"],
            "cohens_d": small_d,
            "significant": small_result["p_value"] < 0.05,
            "interpretation": interpret_cohens_d(small_d),
        },
        "large_sample": {
            "n": large_n,
            "p_value": large_result["p_value"],
            "cohens_d": large_d,
            "significant": large_result["p_value"] < 0.05,
            "interpretation": interpret_cohens_d(large_d),
        },
        "true_effect": effect,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("描述性统计")
    print("=" * 60)
    data = [23, 45, 12, 67, 34, 89, 21, 56, 43, 78, 31, 64, 19, 52, 41]
    print(f"数据: {data}")
    print(f"均值:     {mean(data):.2f}")
    print(f"中位数:   {median(data):.2f}")
    print(f"众数:     {mode(data)}")
    print(f"标准差:  {std_dev(data):.2f}")
    print(f"方差: {variance(data):.2f}")
    print(f"P25:      {percentile(data, 25):.2f}")
    print(f"P50:      {percentile(data, 50):.2f}")
    print(f"P75:      {percentile(data, 75):.2f}")
    print(f"IQR:      {iqr(data):.2f}")

    skewed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]
    print(f"\n偏态数据: {skewed}")
    print(f"均值:   {mean(skewed):.2f}  （被异常值拉高）")
    print(f"中位数: {median(skewed):.2f}  （对异常值稳健）")

    print("\n" + "=" * 60)
    print("相关性")
    print("=" * 60)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_linear = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.1, 18.0, 20.2]
    print(f"线性关系:")
    print(f"  皮尔逊:  {pearson_correlation(x, y_linear):.4f}")
    print(f"  斯皮尔曼: {spearman_correlation(x, y_linear):.4f}")

    y_quadratic = [xi ** 2 for xi in x]
    print(f"二次关系 (y = x^2):")
    print(f"  皮尔逊:  {pearson_correlation(x, y_quadratic):.4f}  （不完美，关系是非线性的）")
    print(f"  斯皮尔曼: {spearman_correlation(x, y_quadratic):.4f}  （完美，关系是单调的）")

    y_none = [random.gauss(0, 1) for _ in x]
    print(f"无线性关系（随机）:")
    print(f"  皮尔逊:  {pearson_correlation(x, y_none):.4f}")
    print(f"  斯皮尔曼: {spearman_correlation(x, y_none):.4f}")

    print("\n" + "=" * 60)
    print("协方差矩阵")
    print("=" * 60)
    feature1 = [random.gauss(0, 1) for _ in range(100)]
    feature2 = [f + random.gauss(0, 0.5) for f in feature1]
    feature3 = [random.gauss(0, 1) for _ in range(100)]
    cov_mat = covariance_matrix([feature1, feature2, feature3])
    print("三特征协方差矩阵:")
    for row in cov_mat:
        print(f"  [{row[0]:7.3f}  {row[1]:7.3f}  {row[2]:7.3f}]")
    print("特征 1 和 2 是相关的（构造时设计的）。")
    print("特征 3 是独立的。")

    print("\n" + "=" * 60)
    print("假设检验：单样本 t 检验")
    print("=" * 60)
    sample = generate_normal(50, mu=52, sigma=10)
    result = one_sample_ttest(sample, mu_0=50)
    print(f"检验总体均值是否为 50（真实均值 = 52）")
    print(f"  样本均值: {mean(sample):.2f}")
    print(f"  t 统计量: {result['t_statistic']:.4f}")
    print(f"  自由度:          {result['df']}")
    print(f"  p 值:     {result['p_value']:.4f}")
    print(f"  alpha=0.05 下显著: {result['p_value'] < 0.05}")

    print("\n" + "=" * 60)
    print("假设检验：双样本 t 检验")
    print("=" * 60)
    model_a_scores = generate_normal(30, mu=0.85, sigma=0.05)
    model_b_scores = generate_normal(30, mu=0.88, sigma=0.05)
    result = two_sample_ttest(model_a_scores, model_b_scores)
    d = cohens_d(model_a_scores, model_b_scores)
    print(f"模型 A 均值: {mean(model_a_scores):.4f}")
    print(f"模型 B 均值: {mean(model_b_scores):.4f}")
    print(f"  t 统计量: {result['t_statistic']:.4f}")
    print(f"  p 值:     {result['p_value']:.4f}")
    print(f"  Cohen's d:   {d:.4f} ({interpret_cohens_d(d)})")
    print(f"  显著: {result['p_value'] < 0.05}")

    print("\n" + "=" * 60)
    print("配对 t 检验（交叉验证）")
    print("=" * 60)
    cv_a = [0.82, 0.85, 0.81, 0.84, 0.83, 0.86, 0.80, 0.84, 0.82, 0.85]
    cv_b = [0.84, 0.87, 0.83, 0.86, 0.85, 0.88, 0.83, 0.86, 0.85, 0.87]
    result = paired_ttest(cv_a, cv_b)
    print(f"模型 A 折: {cv_a}")
    print(f"模型 B 折: {cv_b}")
    print(f"  均值差:   {mean([b - a for a, b in zip(cv_a, cv_b)]):.4f}")
    print(f"  t 统计量: {result['t_statistic']:.4f}")
    print(f"  p 值:     {result['p_value']:.4f}")
    print(f"  显著: {result['p_value'] < 0.05}")

    print("\n" + "=" * 60)
    print("卡方检验")
    print("=" * 60)
    observed = [120, 80, 95, 105]
    expected = [100, 100, 100, 100]
    result = chi_squared_test(observed, expected)
    print(f"观察值: {observed}")
    print(f"期望值: {expected}")
    print(f"  卡方值: {result['chi2']:.4f}")
    print(f"  自由度:          {result['df']}")
    print(f"  p 值:     {result['p_value']:.4f}")
    print(f"  显著: {result['p_value'] < 0.05}")

    print("\n" + "=" * 60)
    print("自助法置信区间")
    print("=" * 60)
    data = generate_normal(50, mu=100, sigma=15)
    boot_mean = bootstrap_statistic(data, mean, n_bootstrap=5000)
    boot_median = bootstrap_statistic(data, median, n_bootstrap=5000)
    print(f"样本量: 50, 真实均值: 100")
    print(f"自助法均值:   {boot_mean['estimate']:.2f}  "
          f"95% CI: [{boot_mean['ci_lower']:.2f}, {boot_mean['ci_upper']:.2f}]  "
          f"SE: {boot_mean['std_error']:.2f}")
    print(f"自助法中位数: {boot_median['estimate']:.2f}  "
          f"95% CI: [{boot_median['ci_lower']:.2f}, {boot_median['ci_upper']:.2f}]  "
          f"SE: {boot_median['std_error']:.2f}")

    print("\n自助法模型比较:")
    scores_a = generate_normal(40, mu=0.85, sigma=0.04)
    scores_b = generate_normal(40, mu=0.88, sigma=0.04)
    comp = bootstrap_compare(scores_a, scores_b, mean, n_bootstrap=5000)
    print(f"  模型 A 均值: {mean(scores_a):.4f}")
    print(f"  模型 B 均值: {mean(scores_b):.4f}")
    print(f"  差异:         {comp['observed_diff']:.4f}")
    print(f"  95% CI:       [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]")
    print(f"  显著:  {comp['significant']}（CI 不包含 0）")

    print("\n" + "=" * 60)
    print("A/B 测试模拟器")
    print("=" * 60)
    print("\n测试 1：无真实效果（true_effect = 0）")
    ab1 = ab_test_simulator(n_per_group=200, true_effect=0.0)
    print(f"  A 组均值: {ab1['group_a_mean']:.2f}")
    print(f"  B 组均值: {ab1['group_b_mean']:.2f}")
    print(f"  观察差异: {ab1['observed_diff']:.2f}")
    print(f"  p 值: {ab1['t_test']['p_value']:.4f}")
    print(f"  显著（t 检验）: {ab1['significant_ttest']}")
    print(f"  Cohen's d: {ab1['cohens_d']:.4f} ({ab1['effect_interpretation']})")

    print("\n测试 2：有真实效果（true_effect = 5）")
    ab2 = ab_test_simulator(n_per_group=200, true_effect=5.0)
    print(f"  A 组均值: {ab2['group_a_mean']:.2f}")
    print(f"  B 组均值: {ab2['group_b_mean']:.2f}")
    print(f"  观察差异: {ab2['observed_diff']:.2f}")
    print(f"  p 值: {ab2['t_test']['p_value']:.4f}")
    print(f"  显著（t 检验）: {ab2['significant_ttest']}")
    print(f"  Cohen's d: {ab2['cohens_d']:.4f} ({ab2['effect_interpretation']})")

    print("\n" + "=" * 60)
    print("多重比较问题")
    print("=" * 60)
    print("\n20 次检验且无真实效果（所有零假设为真）:")
    multi = run_multiple_ab_tests(n_tests=20, true_effect=0.0)
    print(f"  未校正的显著检验数: {multi['uncorrected_significant']}/20")
    print(f"  Bonferroni 校正后的显著数:  {multi['corrected_significant']}/20")
    print(f"  alpha=0.05 下的预期假阳性: 约 1 个")
    print(f"  Bonferroni 调整后 alpha: {0.05/20:.4f}")

    print("\n" + "=" * 60)
    print("统计显著性 vs 实际显著性")
    print("=" * 60)
    result = statistical_vs_practical_significance(
        small_n=30, large_n=10000, effect=0.1
    )
    print(f"\n真实效果: {result['true_effect']}（极小）")
    print(f"\n小样本 (n={result['small_sample']['n']}):")
    print(f"  p 值:  {result['small_sample']['p_value']:.4f}")
    print(f"  Cohen's d: {result['small_sample']['cohens_d']:.4f} ({result['small_sample']['interpretation']})")
    print(f"  显著: {result['small_sample']['significant']}")
    print(f"\n大样本 (n={result['large_sample']['n']}):")
    print(f"  p 值:  {result['large_sample']['p_value']:.4f}")
    print(f"  Cohen's d: {result['large_sample']['cohens_d']:.4f} ({result['large_sample']['interpretation']})")
    print(f"  显著: {result['large_sample']['significant']}")
    print(f"\n教训：大 n 可以使可忽略的效果变得'显著'。")
    print("总是检查效应量，而不仅仅是 p 值。")

    print("\n" + "=" * 60)
    print("功效分析模拟")
    print("=" * 60)
    print("\n我们能以多大频率检测到真实效果（true_effect=3）？")
    n_sims = 200
    detected = 0
    for _ in range(n_sims):
        a = generate_normal(50, 50, 10)
        b = generate_normal(50, 53, 10)
        res = two_sample_ttest(a, b)
        if res["p_value"] < 0.05:
            detected += 1
    print(f"  功效（n=50, 效果=3, 标准差=10）: {detected/n_sims:.2f}")
    print(f"  ({detected}/{n_sims} 次模拟检测到效果）")

    detected_large = 0
    for _ in range(n_sims):
        a = generate_normal(200, 50, 10)
        b = generate_normal(200, 53, 10)
        res = two_sample_ttest(a, b)
        if res["p_value"] < 0.05:
            detected_large += 1
    print(f"  功效（n=200, 效果=3, 标准差=10）: {detected_large/n_sims:.2f}")
    print(f"  ({detected_large}/{n_sims} 次模拟检测到效果）")
    print("  更大样本给出更强功效来检测真实效果。")
