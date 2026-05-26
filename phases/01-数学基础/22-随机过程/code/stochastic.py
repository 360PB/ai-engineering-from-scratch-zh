import numpy as np


# 一维随机游走
def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


# 二维随机游走
def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # 向右
    dx[directions == 1] = -1  # 向左
    dy[directions == 2] = 1   # 向上
    dy[directions == 3] = -1  # 向下
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y


# 马尔可夫链
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    # 一步转移
    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    # 模拟多条路径
    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    # 平稳分布（通过特征分解）
    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.clip(stationary, 0, None)
        total = stationary.sum()
        if total > 0:
            stationary = stationary / total
        return np.abs(stationary)

    # 经验分布
    def empirical_distribution(self, states):
        counts = np.zeros(self.n_states)
        for s in states:
            counts[s] += 1
        return counts / len(states)


# Langevin 动力学
def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)


# Metropolis-Hastings MCMC
def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    if n_samples < 1:
        raise ValueError("n_samples 至少为 1")
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate


# 前向扩散过程
def diffusion_forward(signal, n_steps, beta_start=0.0001, beta_end=0.02, seed=None):
    rng = np.random.RandomState(seed)
    betas = np.linspace(beta_start, beta_end, n_steps)
    trajectory = [signal.copy()]
    x = signal.copy()
    for t in range(n_steps):
        noise = rng.randn(*x.shape)
        x = np.sqrt(1 - betas[t]) * x + np.sqrt(betas[t]) * noise
        trajectory.append(x.copy())
    return np.array(trajectory), betas


def demo_random_walks():
    print("=" * 60)
    print("演示 1：一维随机游走")
    print("=" * 60)

    n_walks = 5
    n_steps = 1000
    print(f"\n{n_walks} 条游走，每条 {n_steps} 步：\n")

    final_positions = []
    for i in range(n_walks):
        walk = random_walk_1d(n_steps, seed=i)
        final_positions.append(walk[-1])
        print(f"  游走 {i+1}：最终位置 = {walk[-1]:+4d}，"
              f"最大值 = {walk.max():+4d}，最小值 = {walk.min():+4d}")

    print(f"\n理论：E[位置] = 0，std(位置) = sqrt({n_steps}) = {np.sqrt(n_steps):.1f}")

    n_many = 10000
    finals = []
    for i in range(n_many):
        walk = random_walk_1d(n_steps, seed=i)
        finals.append(walk[-1])
    finals = np.array(finals)
    print(f"\n{n_many} 条游走：均值 = {finals.mean():.2f}，"
          f"标准差 = {finals.std():.2f}（预期 {np.sqrt(n_steps):.2f}）")


def demo_markov_chain():
    print("\n" + "=" * 60)
    print("演示 2：天气马尔可夫链")
    print("=" * 60)

    P = [[0.7, 0.1, 0.2],
         [0.3, 0.4, 0.3],
         [0.4, 0.2, 0.4]]
    names = ["晴天", "雨天", "阴天"]
    mc = MarkovChain(P, state_names=names)

    pi = mc.stationary_distribution()
    print("\n平稳分布（解析解）：")
    for i, name in enumerate(names):
        print(f"  {name}：{pi[i]:.4f}")

    states = mc.simulate(start_state=0, n_steps=100000, seed=42)
    empirical = mc.empirical_distribution(states)
    print("\n经验分布（100000 步，从晴天开始）：")
    for i, name in enumerate(names):
        print(f"  {name}：{empirical[i]:.4f}")

    print("\n收敛检验：")
    for length in [100, 1000, 10000, 100000]:
        states = mc.simulate(start_state=1, n_steps=length, seed=42)
        emp = mc.empirical_distribution(states)
        error = np.abs(emp - pi).max()
        print(f"  {length:>7d} 步：最大误差 = {error:.4f}")

    short = mc.simulate(start_state=0, n_steps=20, seed=42)
    sequence = " -> ".join(names[s] for s in short[:15])
    print(f"\n样本轨迹：{sequence}...")


def demo_langevin():
    print("\n" + "=" * 60)
    print("演示 3：Langevin 动力学 -- 从高斯分布采样")
    print("=" * 60)

    target_mean = 3.0
    target_var = 2.0

    def grad_U(x):
        return (x - target_mean) / target_var

    trajectory = langevin_dynamics(
        grad_U=grad_U,
        x0=np.array([0.0]),
        dt=0.1,
        temperature=1.0,
        n_steps=50000,
        seed=42
    )

    samples = trajectory[5000:, 0]
    print(f"\n目标：均值 = {target_mean}，方差 = {target_var}")
    print(f"采样结果（5000 步预烧后 {len(samples)} 个样本）：")
    print(f"  均值：     {samples.mean():.4f}（预期 {target_mean}）")
    print(f"  方差：     {samples.var():.4f}（预期 {target_var}）")
    print(f"  标准差：   {samples.std():.4f}（预期 {np.sqrt(target_var):.4f}）")


def demo_metropolis_hastings():
    print("\n" + "=" * 60)
    print("演示 4：Metropolis-Hastings -- 双峰分布")
    print("=" * 60)

    def bimodal_log_prob(x):
        v = np.asarray(x).ravel()[0]
        log_p1 = -0.5 * (v - 3) ** 2
        log_p2 = -0.5 * (v + 3) ** 2
        return np.logaddexp(log_p1, log_p2) - np.log(2)

    samples, acc_rate = metropolis_hastings(
        target_log_prob=bimodal_log_prob,
        proposal_std=2.0,
        x0=np.array([0.0]),
        n_samples=100000,
        seed=42
    )

    samples_flat = samples[10000:, 0]
    print("\n双峰目标：N(-3,1) 和 N(+3,1) 的混合分布")
    print(f"接受率：{acc_rate:.2%}")
    print(f"样本均值：{samples_flat.mean():.4f}（预期约 0.0）")
    print(f"样本标准差：  {samples_flat.std():.4f}")

    left_mode = samples_flat[samples_flat < 0]
    right_mode = samples_flat[samples_flat >= 0]
    print(f"\n左峰（x < 0）：均值 = {left_mode.mean():.4f}，"
          f"计数 = {len(left_mode)}")
    print(f"右峰（x >= 0）：均值 = {right_mode.mean():.4f}，"
          f"计数 = {len(right_mode)}")
    print(f"各峰比例：{len(left_mode)/len(samples_flat):.2%} / "
          f"{len(right_mode)/len(samples_flat):.2%}（预期约 50/50）")

    print("\n提议标准差比较：")
    for std in [0.1, 0.5, 2.0, 5.0, 20.0]:
        _, rate = metropolis_hastings(bimodal_log_prob, std, np.array([0.0]), 10000, seed=42)
        print(f"  std = {std:5.1f}：接受率 = {rate:.2%}")


def demo_diffusion():
    print("\n" + "=" * 60)
    print("演示 5：前向扩散过程")
    print("=" * 60)

    n_points = 200
    t = np.linspace(0, 2 * np.pi, n_points)
    signal = np.sin(t) + 0.5 * np.sin(3 * t)

    trajectory, betas = diffusion_forward(
        signal, n_steps=100, beta_start=0.001, beta_end=0.05, seed=42
    )

    print(f"\n原始信号：sin(t) + 0.5*sin(3t)，{n_points} 个点")
    print(f"噪声调度：beta 从 {betas[0]:.4f} 到 {betas[-1]:.4f}")

    checkpoints = [0, 10, 25, 50, 75, 100]
    print("\n扩散步骤中信号退化情况：")
    print(f"{'步骤':>6s} | {'均值':>8s} | {'标准差':>8s} | {'信噪比（dB）':>10s} | {'相关性':>12s}")
    print("-" * 55)
    for step in checkpoints:
        x = trajectory[step]
        noise_power = np.mean((x - signal) ** 2)
        signal_power = np.mean(signal ** 2)
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        corr = np.corrcoef(signal, x)[0, 1]
        print(f"{step:>6d} | {x.mean():>8.4f} | {x.std():>8.4f} | "
              f"{snr:>10.2f} | {corr:>12.4f}")

    print("\n第 0 步：完美信号（相关性 = 1.0）")
    print("第 100 步：接近纯噪声（相关性接近 0）")
    print("这就是扩散模型的前向过程。")


if __name__ == "__main__":
    demo_random_walks()
    demo_markov_chain()
    demo_langevin()
    demo_metropolis_hastings()
    demo_diffusion()