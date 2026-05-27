# 游戏 RL——AlphaZero、MuZero 与 LLM 推理时代

> 1992：TD-Gammon 用纯 TD 击败人类冠军。2016：AlphaGo 击败李世石。2017：AlphaZero 从零开始横扫象棋、将棋和围棋。2024：DeepSeek-R1 证明同样的配方，用 GRPO 替换 PPO，在推理上有效。游戏是推动本 Phase 每个突破的基准。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 9 · 05（DQN），Phase 9 · 08（PPO），Phase 9 · 09（RLHF），Phase 9 · 10（MARL）
**时间：** 约 120 分钟

## 问题

游戏有 RL 想要的一切。干净奖励（输赢）。无限 episode（自对弈重置）。完美模拟（游戏*就是*模拟器）。离散或小连续动作空间。强制对抗鲁棒性的多智能体结构。

而游戏是每个主要 RL 突破的测试方式。TD-Gammon（西洋双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（StarCraft II，2019）。MuZero（学习模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——在文本上有效的游戏 RL 技术的最新演示。

本总结课通过一个统一视角——**自对弈 + 搜索 + 策略改进**——审视三个里程碑架构：AlphaZero、MuZero 和 GRPO。每一个都是前一个的推广；GRPO 尤其是将 AlphaZero 配方应用于 LLM 推理，token 作为动作，数学验证作为赢信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO：相同循环，不同环境](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # 与自己对弈
    policy_target = search.improved_policy(trajectory) # 搜索改进原始策略
    policy_net.update(policy_target, value_target)    # 在搜索输出上监督
```

**AlphaZero（2017）。** Silver 等人。给定一个已知规则的游戏（象棋、将棋、围棋）：

- 策略-值网络：一个塔 `f_θ(s) → (p, v)`。`p` 是合法动作上的先验。`v` 是预期游戏结果。
- 蒙特卡洛树搜索（MCTS）：每步，展开可能续着的树。使用 `(p, v)` 作为先验 + bootstrap。用 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自对弈：智能体对智能体下棋。在步 `t`，MCTS 访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。一个配方在每个游戏上几千万自对弈后掌握象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。移除规则已知的要求。

- 不是固定环境，而是学习一个*潜在动力学模型* `(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 值。
- MCTS 在*学到的潜在空间*中运行。相同搜索，相同训练循环。
- 在围棋、象棋、将棋*和* Atari 上有效——一个算法，无需规则知识。

**随机 MuZero（2022）。** 添加随机动力学和机会节点；扩展到西洋双陆棋类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 样本效率和确定性搜索的改进。

**GRPO（2024-2025）。** DeepSeek-R1 配方。应用到语言模型推理的相同 AlphaZero 形状循环：

- "游戏"：回答一个数学 / 编码 / 推理问题。"赢" = 验证器（测试用例通过、数字答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示 + 到此为止的回复。
- 无 critic（PPO 风格 V_φ）。相反，对每个提示，从策略采样 `G` 个补全。对每个计算奖励。使用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略的 KL 惩罚防止漂移（如 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型，无 critic，无 MCTS。组相对基线替换三者三个。在推理基准上以一小部分计算匹配或超过 PPO-RLHF 质量。

**R1 配方完整版。** DeepSeek-R1（DeepSeek 2025）是一篇论文中的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始。无 SFT。直接应用 GRPO，两个奖励成分：*准确性奖励*（基于规则——最终答案是否解析为正确数字 / 代码是否通过单元测试）和*格式奖励*（补全是否将其思维链包装在 `<think>…` 标签中）。经过数千步，平均回复长度从约 100 增长到约 10,000 token，数学基准分数攀升至接近 o1-preview 水平。模型从零开始学习推理。缺点：其思维链经常不可读，混语言，缺乏风格打磨。
- **R1。** 用四阶段管道修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 用几千个格式整洁的长 CoT 演示收集。监督微调基础模型。得到可读的起点。
  2. **推理导向 GRPO。** 应用 GRPO 加准确性+格式奖励加上*语言一致性*奖励以防止代码切换。
  3. **拒绝采样 + SFT 第二轮。** 从 RL 检查点采样约 600K 推理轨迹，只保留最终答案正确且 CoT 可读的，并与约 200K 非推理 SFT 示例（写作、QA、自我认知）结合。再次微调基础。
  4. **全谱 GRPO。** 又一轮 RL，覆盖推理（基于规则的奖励）和一般对齐（有用性/无害性基于偏好的奖励）。

结果以开放权重在 AIME 和 MATH-500 上与 o1 匹配，且足够小可以蒸馏。同篇论文还发布六个蒸馏密集模型（Qwen-1.5B 到 Llama-70B），通过在 R1 的推理轨迹上 SFT——学生无 RL。从强 RL 教师蒸馏在学生规模上始终优于从零 RL。

**为什么 GRPO 而非 PPO 用于推理。** DeepSeekMath 论文（2024 年 2 月）有三个原因：(1) 无值网络要训练，内存减半；(2) 组基线自然处理推理任务产生的稀疏端轨迹奖励；(3) 每提示归一化使优势跨难度差异很大的问题可比，而 PPO 的单一 critic 无法做到。

**无搜索 vs 基于搜索。** 游戏已分叉：

- *完美信息长视野游戏*（围棋、象棋）：仍是基于搜索的。AlphaZero / MuZero 主导。
- *LLM 推理*：生产中尚无 MCTS；GRPO 在完整 rollout 上，最佳 N 用于推理计算。过程奖励模型（PRM）暗示步级搜索可能加回。

## 构建

`code/main.py` 中的代码实现了**微型 GRPO**——一个多组样本的 bandit。算法与 LLM 上的相同；只有策略和环境更简单。它教授*损失*和*组相对优势*，这是 2024 年的创新。

### 步骤 1：一个微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

真实 GRPO 中验证器运行单元测试或检查数学等式。

### 步骤 2：策略：每个提示上 K 个答案 token 的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于在提示条件下 LLM 的最终层输出。

### 步骤 3：组采样和组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 惩罚：将 theta 拉向参考
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 技巧。无 critic。需要。"基线"是组均值，归一化用组标准差。

### 步骤 4：与 REINFORCE 基线（无值）比较

相同设置，相同计算，朴素 REINFORCE。GRPO 收敛更快更稳定。

### 步骤 5：观察熵和 KL

与 RLHF 相同的诊断：到参考的平均 KL、策略熵、回报随时间。一旦这些稳定，训练完成。

## 陷阱

- **通过验证器游戏的奖励黑客。** GRPO 继承 RLHF 的风险：如果验证器错误或可利用，LLM 会找到利用。健壮验证器（多个测试用例、形式证明）很重要。
- **组大小太小。** 组基线方差约为 `1/√G`。低于 `G = 4`，优势信号有噪声；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 补全有不同的对数概率。按 token 计数归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自对弈循环。** AlphaZero 风格训练可能在一般和游戏上陷入主导循环。通过多样化对手池（联盟学习，Lesson 10）缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略模仿搜索输出。如果策略网络太小无法表示搜索的分布，训练停滞。
- **计算下限。** MuZero / AlphaZero 需要大规模计算。单次消融经常是数百 GPU 小时。微型演示存在（例如，AlphaZero 玩 Connect Four）用于学习。
- **验证器覆盖率。** 通过错误解的单元测试强化 bug。设计捕捉边缘情况的验证器。

## 使用

2026 年游戏 RL 格局，按领域：

| 领域 | 主导方法 |
|--------|-----------------|
| 两人零和棋盘游戏（围棋、象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完美信息扑克 | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大规模多人策略（Dota、StarCraft） | PPO + 自对弈 + 联盟（OpenAI Five、AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开放复现） |
| LLM 对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好不是可验证的） |
| 机器人 | PPO + DR（不是游戏 RL，但使用相同策略梯度工具） |
| 组合问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

配方——自对弈、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO 是最年轻的实例；更多正在到来。

## 交付

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: 为给定领域设计游戏-RL 或推理-RL 训练管道（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定目标（完美信息游戏 / 不完美信息 / Atari / LLM 推理 / 组合），输出：

1. 环境契合度。已知规则？马尔可夫？随机？多智能体？决定 AlphaZero vs MuZero vs GRPO。
2. 搜索策略。MCTS（带学习先验的 PUCT）、Gumbel 采样、最佳 N、或无。
3. 自对弈计划。对称自对弈 / 联盟 / 离线数据 / 验证器生成。
4. 目标信号。游戏结果 / 验证器奖励 / 偏好 / 学到的模型。含鲁棒性计划。
5. 诊断。相对基线的胜率、ELO 曲线、验证器通过率、到参考的 KL。

拒绝在不完美信息游戏上用 AlphaZero（路由到 CFR）。拒绝无信任验证器的 GRPO。拒绝任何没有固定基线对手集的游戏 RL 管道（自对弈 ELO 是未校准的）。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO bandit。在 2 提示 × 4 答案 token 上训练。以 `G=8` 在 < 1,000 次更新内收敛。
2. **中等。** 插入 PPO（裁剪）和朴素 REINFORCE。在相同 bandit 上比较样本效率和回报方差到 GRPO。
3. **困难。** 扩展到长度-2 "推理链"：智能体发出两个 token，验证器奖励该对。测量 GRPO 如何处理两步序列上的信用分配。（提示：按*完整序列*计算组优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 人们怎么说 | 实际指什么 |
|------|-----------------|-----------------------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；带学习 `(p, v)` 先验的 UCB1/PUCT 选择。 |
| AlphaZero | "自对弈 + MCTS" | 策略-值网络训练来匹配 MCTS 访问和游戏结果。 |
| MuZero | "学习模型 AlphaZero" | 相同循环但在潜在空间通过学到的动力学。 |
| GRPO | "无 critic 的 PPO" | 组相对策略优化；带组均值基线 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)` — 在值估计和先验之间平衡。 |
| 自对弈 | "智能体 vs 过去的自己" | 零和标准；对称训练信号。 |
| 联盟学习 | "基于种群的自对弈" | 缓存过去 + 当前 + 探索者作为对手。处理策略循环。 |
| 验证器奖励 | "可验证 RL" | 奖励来自确定性检查器（测试通过、答案匹配）。 |
| 过程奖励 | "PRM" | 对每个推理步骤评分，不只是最终答案。 |

## 拓展阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整四阶段 R1 配方加 R1-Zero 消融。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + 大规模深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开启一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 用自定义奖励函数应用 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO 复现](https://github.com/QwenLM/Qwen2.5-Math) — 在多规模上 R1 配方的开放复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 自对弈、搜索和"设计奖励"的教材框架，R1 在 LLM 规模上实例化。