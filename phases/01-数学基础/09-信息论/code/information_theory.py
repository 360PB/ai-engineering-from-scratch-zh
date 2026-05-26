import math
import random
# 信息论：信息量、熵、交叉熵、KL 散度、互信息
# 演示标签平滑和特征选择


def information_content(p, base=2):
    """信息量：-log(p)，单位为 base 指定的单位"""
    if p <= 0:
        return float('inf')
    if p >= 1:
        return 0.0
    return -math.log(p) / math.log(base)


def entropy(probs, base=2):
    """熵：概率分布的平均惊讶度"""
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )


def cross_entropy(p, q, base=2):
    """交叉熵：使用分布 q 编码来自分布 p 的事件时的平均惊讶度"""
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total


def kl_divergence(p, q, base=2):
    """KL 散度：交叉熵减去熵（p 的熵是常数）"""
    return cross_entropy(p, q, base) - entropy(p, base)


def mutual_information(joint_probs, base=2):
    """互信息：I(X;Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y)"""
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0 and margin_x[i] > 0 and margin_y[j] > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi


def softmax(logits):
    """Softmax：将实数值向量映射到概率分布"""
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_loss(true_class, logits):
    """交叉熵损失：用于分类"""
    probs = softmax(logits)
    return -math.log(probs[true_class])


def negative_log_likelihood(labels, all_logits):
    """负对数似然：与交叉熵损失相同"""
    return sum(
        cross_entropy_loss(label, logits)
        for label, logits in zip(labels, all_logits)
    ) / len(labels)


def perplexity(avg_cross_entropy, base="e"):
    """困惑度：交叉熵的指数"""
    if base == "e":
        return math.exp(avg_cross_entropy)
    return 2 ** avg_cross_entropy


def conditional_entropy(joint_probs, base=2):
    """条件熵：H(Y|X) = H(X,Y) - H(X)"""
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]

    h_yx = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0 and margin_x[i] > 0:
                p_y_given_x = pxy / margin_x[i]
                h_yx -= pxy * math.log(p_y_given_x) / math.log(base)
    return h_yx


def joint_entropy(joint_probs, base=2):
    """联合熵：H(X,Y)"""
    total = 0.0
    for row in joint_probs:
        for pxy in row:
            if pxy > 0:
                total -= pxy * math.log(pxy) / math.log(base)
    return total


def label_smoothing_demo():
    """标签平滑演示"""
    print()
    print("=" * 60)
    print("标签平滑与交叉熵")
    print("=" * 60)

    num_classes = 4
    true_class = 2
    logits = [1.0, 0.5, 3.0, 0.2]
    probs = softmax(logits)

    hard_target = [0.0] * num_classes
    hard_target[true_class] = 1.0

    epsilons = [0.0, 0.05, 0.1, 0.2]
    print(f"\n  Logits:  {logits}")
    print(f"  Softmax: [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"  真实类别: {true_class}")
    print()

    for eps in epsilons:
        soft_target = [eps / num_classes] * num_classes
        soft_target[true_class] = (1 - eps) + eps / num_classes

        ce = cross_entropy(soft_target, probs, base=math.e)
        target_entropy = entropy(soft_target, base=math.e)
        label = "硬目标" if eps == 0.0 else f"eps={eps}"
        print(f"  {label:>8s}  target={[f'{t:.3f}' for t in soft_target]}  "
              f"H(target)={target_entropy:.4f}  CE={ce:.4f}")

    print()
    print("  epsilon 越高 -> 目标熵越高 -> 充当正则化")


def feature_selection_mi_demo():
    """基于互信息的特征选择演示"""
    print()
    print("=" * 60)
    print("基于互信息的特征选择")
    print("=" * 60)

    random.seed(42)
    n = 200

    target = [random.choice([0, 1]) for _ in range(n)]

    features = {}
    features["强信号"] = [t ^ (1 if random.random() < 0.1 else 0) for t in target]
    features["弱信号"] = [t ^ (1 if random.random() < 0.35 else 0) for t in target]
    features["噪声"] = [random.choice([0, 1]) for _ in range(n)]
    features["常量"] = [0] * n

    print(f"\n  样本数: {n}")
    print(f"  目标平衡: {sum(target)}/{n - sum(target)}")
    print()

    mi_scores = []
    for name, feat in features.items():
        joint = [[0, 0], [0, 0]]
        for f, t in zip(feat, target):
            joint[f][t] += 1
        joint_p = [[c / n for c in row] for row in joint]
        mi = mutual_information(joint_p, base=2)
        mi_scores.append((name, mi))

    mi_scores.sort(key=lambda x: x[1], reverse=True)
    print("  特征 MI 排名:")
    for name, mi in mi_scores:
        bar = "#" * int(mi * 200)
        print(f"    {name:>16s}  MI = {mi:.4f} bits  {bar}")

    print()
    print("  强信号具有最高 MI。噪声和常量接近 0。")


if __name__ == "__main__":

    print("=" * 60)
    print("信息量（惊讶）")
    print("=" * 60)

    events = [
        ("硬币正面", 0.5),
        ("掷出 6 点", 1 / 6),
        ("千分之一事件", 0.001),
        ("确定事件", 1.0),
    ]
    for name, p in events:
        print(f"  {name:20s}  p={p:<8.4f}  惊讶={information_content(p):.4f} bits")

    print()
    print("=" * 60)
    print("熵")
    print("=" * 60)

    distributions = {
        "公平硬币": [0.5, 0.5],
        "偏倚硬币 (99/1)": [0.99, 0.01],
        "公平骰子 (6 面)": [1 / 6] * 6,
        "加载骰子": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
    }
    for name, probs in distributions.items():
        print(f"  {name:25s}  H = {entropy(probs):.4f} bits")

    print()
    print("=" * 60)
    print("交叉熵和 KL 散度")
    print("=" * 60)

    true_dist = [0.7, 0.2, 0.1]
    good_model = [0.6, 0.25, 0.15]
    bad_model = [0.1, 0.1, 0.8]

    h_true = entropy(true_dist)
    ce_good = cross_entropy(true_dist, good_model)
    ce_bad = cross_entropy(true_dist, bad_model)
    kl_good = kl_divergence(true_dist, good_model)
    kl_bad = kl_divergence(true_dist, bad_model)

    print(f"  真实分布:      {true_dist}")
    print(f"  好模型:          {good_model}")
    print(f"  坏模型:          {bad_model}")
    print()
    print(f"  H(true):          {h_true:.4f} bits")
    print(f"  H(true, good):    {ce_good:.4f} bits")
    print(f"  H(true, bad):     {ce_bad:.4f} bits")
    print(f"  KL(true || good): {kl_good:.4f} bits")
    print(f"  KL(true || bad):   {kl_bad:.4f} bits")
    print()
    print(f"  验证: H(P,Q) = H(P) + KL(P||Q)")
    print(f"  好: {h_true:.4f} + {kl_good:.4f} = {h_true + kl_good:.4f}  (CE = {ce_good:.4f})")
    print(f"  坏: {h_true:.4f} + {kl_bad:.4f} = {h_true + kl_bad:.4f}  (CE = {ce_bad:.4f})")

    print()
    print("=" * 60)
    print("KL 散度不是对称的")
    print("=" * 60)

    p = [0.9, 0.1]
    q = [0.5, 0.5]
    print(f"  P = {p},  Q = {q}")
    print(f"  KL(P || Q) = {kl_divergence(p, q):.4f} bits")
    print(f"  KL(Q || P) = {kl_divergence(q, p):.4f} bits")
    print(f"  它们不同，因为 KL 不是真正的距离度量。")

    print()
    print("=" * 60)
    print("分类的交叉熵损失")
    print("=" * 60)

    logits = [2.0, 1.0, 0.1]
    true_class = 0
    probs = softmax(logits)
    loss = cross_entropy_loss(true_class, logits)

    print(f"  Logits:       {logits}")
    print(f"  Softmax:      [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"  真实类别:   {true_class}")
    print(f"  CE 损失:      {loss:.4f} nats")
    print(f"  困惑度:       {perplexity(loss):.2f}")

    print()
    print("  用相同 logits 尝试不同真实类别:")
    for c in range(3):
        l = cross_entropy_loss(c, logits)
        print(f"    类别 {c}: 损失={l:.4f}  概率={probs[c]:.4f}")

    print()
    print("=" * 60)
    print("交叉熵 = 负对数似然")
    print("=" * 60)

    random.seed(42)
    n_samples = 1000
    n_classes = 3
    labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
    all_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

    ce_avg = negative_log_likelihood(labels, all_logits)
    nll_avg = -sum(
        math.log(softmax(lg)[lb])
        for lb, lg in zip(labels, all_logits)
    ) / n_samples

    print(f"  样本数:               {n_samples}")
    print(f"  交叉熵损失:          {ce_avg:.6f} nats")
    print(f"  负对数似然:          {nll_avg:.6f} nats")
    print(f"  差异:                {abs(ce_avg - nll_avg):.2e}")
    print(f"  它们相同。最小化 CE = 最大化似然。")

    print()
    print("=" * 60)
    print("互信息")
    print("=" * 60)

    independent = [[0.25, 0.25], [0.25, 0.25]]
    dependent = [[0.45, 0.05], [0.05, 0.45]]
    partial = [[0.3, 0.2], [0.1, 0.4]]

    print(f"  独立:     MI = {mutual_information(independent):.4f} bits")
    print(f"  相关:     MI = {mutual_information(dependent):.4f} bits")
    print(f"  部分:     MI = {mutual_information(partial):.4f} bits")

    print()
    print("=" * 60)
    print("比特 vs 纳特")
    print("=" * 60)

    fair_coin = [0.5, 0.5]
    print(f"  公平硬币熵:")
    print(f"    比特 (log2): {entropy(fair_coin, base=2):.4f}")
    print(f"    纳特 (ln):   {entropy(fair_coin, base=math.e):.4f}")
    print(f"    1 比特 = {1 / math.log2(math.e):.4f} 纳特")
    print(f"    1 纳特 = {math.log2(math.e):.4f} 比特")

    print()
    print("=" * 60)
    print("语言模型中的困惑度")
    print("=" * 60)

    random.seed(123)
    vocab_size = 50
    sequence_length = 100
    true_tokens = [random.randint(0, vocab_size - 1) for _ in range(sequence_length)]
    token_logits = [[random.gauss(0, 1) for _ in range(vocab_size)] for _ in range(sequence_length)]

    avg_ce = negative_log_likelihood(true_tokens, token_logits)
    ppl = perplexity(avg_ce)

    print(f"  词表大小:        {vocab_size}")
    print(f"  序列长度:         {sequence_length}")
    print(f"  平均 CE 损失:    {avg_ce:.4f} nats")
    print(f"  困惑度:          {ppl:.2f}")
    print(f"  随机基准:        {vocab_size:.2f}（均匀分布在词表上）")
    print(f"  如果困惑度 < 词表大小，模型比随机好。")

    print()
    print("=" * 60)
    print("条件熵和联合熵")
    print("=" * 60)

    joint_dep = [[0.45, 0.05], [0.05, 0.45]]
    joint_indep = [[0.25, 0.25], [0.25, 0.25]]

    print(f"\n  相关联合分布: {joint_dep}")
    print(f"    联合熵 H(X,Y):       {joint_entropy(joint_dep):.4f} bits")
    print(f"    条件 H(Y|X):         {conditional_entropy(joint_dep):.4f} bits")
    print(f"    互信息 I(X;Y):       {mutual_information(joint_dep):.4f} bits")

    hx_dep = entropy([sum(row) for row in joint_dep])
    print(f"    H(X):                {hx_dep:.4f} bits")
    print(f"    验证: H(X,Y) = H(X) + H(Y|X) = {hx_dep:.4f} + {conditional_entropy(joint_dep):.4f} = {hx_dep + conditional_entropy(joint_dep):.4f}")

    print(f"\n  独立联合分布: {joint_indep}")
    print(f"    联合熵 H(X,Y):       {joint_entropy(joint_indep):.4f} bits")
    print(f"    条件 H(Y|X):         {conditional_entropy(joint_indep):.4f} bits")
    print(f"    互信息 I(X;Y):       {mutual_information(joint_indep):.4f} bits")
    print("    独立时: H(Y|X) = H(Y) 且 I(X;Y) = 0")

    label_smoothing_demo()
    feature_selection_mi_demo()