# 仿真到现实迁移

> 在模拟器中训练但在硬件上失效的策略，是一个记住了模拟器的策略。领域随机化、领域适应和系统辨识是让学习控制器跨越现实差距的三种工具。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 9 · 08（PPO），Phase 2 · 10（偏差/方差）
**时间：** 约 45 分钟

## 问题

训练真实机器人是缓慢的、危险的、昂贵的。一个双足机器人需要数百万个训练 episode 来学习走路；一个跌倒一次的真实双足机器人会破坏硬件。模拟给你无限的 resets、可重现的确定性、并行环境，没有物理损伤。

但模拟器是错的。轴承的摩擦比 MuJoCo 模型更多。相机有模拟器不包括的镜头畸变。电机有 99% 的模拟模型跳过的延迟、背隙和饱和。风、灰尘和可变照明破坏了在对无菌渲染上训练的策略。**现实差距**——sim 分布和 real 分布之间的系统差异——是部署 RL 用于机器人的核心问题。

你需要一个对 sim 到 real 分布偏移*稳健*的策略。三种历史方法：随机化模拟器（领域随机化）、用少量真实数据适应策略（领域适应 / 微调）、或识别真实系统参数并匹配它们（系统辨识）。2026 年主导配方将三者结合，配合大规模并行模拟（Isaac Sim、Isaac Lab、Mujoco MJX on GPU）。

## 概念

![三种 sim-to-real 机制：领域随机化、适应、系统辨识](../assets/sim-to-real.svg)

**领域随机化（DR）。** Tobin 等人 2017，Peng 等人 2018。训练期间，随机化每个可能与真实机器人不同的 sim 参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、照明、纹理、接触模型。策略学习"今天在哪个 sim 中"的条件分布，推广到完整跨度。如果真实机器人在训练包络内，策略有效。

- **优点：** 不需要真实数据。一个配方，多个机器人。
- **缺点：** 过度随机化训练产生"通用的"但过于谨慎的策略。太多噪声 ≈ 太多正则化。

**系统辨识（SI）。** 训练前用真实世界数据拟合模拟器参数。如果你能测量真实机器人上的臂关节摩擦，将其插入 sim。然后训练期望这些值的策略。需要访问真实系统但直接减少现实差距。

- **优点：** 精确、低噪声训练目标。
- **缺点：** 残留模型误差对策略不可见；小的未识别效应（例如，电机死区）仍然在部署时出问题。

**领域适应。** 在 sim 中训练，用少量真实数据微调。两种风格：

- **Real2Sim2Real：** 用真实 rollout 学习残差模拟器 `f(s, a, z) - f_sim(s, a)`，在修正后的 sim 中训练。用少量真实数据封闭差距。
- **观测适应：** 训练一个通过学到的特征提取器将真实 obs → sim 类 obs 的策略（例如，GAN 像素到像素）。控制器保持在 sim 中。

**特权学习 / 教师-学生。** Miki 等人 2022（ANYmal 四足）。训练一个在模拟中具有*特权信息*（地面真实摩擦、地形高度、IMU 漂移）的*教师*。蒸馏一个只看到真实传感器观测的*学生*。学生从历史中推断特权特征，对物理参数稳健。

**大规模并行模拟。** 2024–2026。Isaac Lab、Mujoco MJX、Brax 都在单个 GPU 上运行数千个并行机器人。带 4,096 个并行类人的 PPO 在几小时内收集数年的经验。随着训练分布加宽，"现实差距"缩小；当这 4,096 个环境中的每一个都有不同随机化参数时，DR 几乎免费。

**2026 年真实世界配方（四足行走示例）：**

1. 带领域随机化重力、摩擦、电机增益、有效载荷的大规模并行 sim。
2. 带特权信息（地形图、身体速度地面真实）的教师策略训练。
3. 学生策略从教师蒸馏，只用本体感觉（腿部关节编码器）。
4. 可选：通过真实 IMU 上的自编码器进行观测适应。
5. 部署。零样本在 10+ 环境上。如果失效，用安全约束 PPO 进行几分钟真实世界微调。

## 构建

本课的代码是对带*噪声*转移的 GridWorld 上领域随机化的极简演示。我们训练一个在"sim"中经历随机滑动概率的策略，并在训练中从未见过的滑动级别上评估"real"。形状直接映射到 MuJoCo 到硬件迁移。

### 步骤 1：参数化 sim

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模拟器暴露的参数。在真实机器人中可以是摩擦、质量、电机增益——任何在 sim 和 real 之间移动的东西。

### 步骤 2：用 DR 训练

每个 episode 开始时采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任何。大量 episode 做这个。

### 步骤 3：零样本评估"real"滑动

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支持内；`0.5` 和 `0.7` 在外。DR 训练的策略应在支持内保持接近最优，在支持外优雅降级。固定滑动训练的策略在训练滑动之外会脆弱。

### 步骤 4：与窄训练比较

训练第二个策略只用 `slip = 0.0`。在同一滑动扫描上评估。你应该看到一旦 real slip > 0 就灾难性下降。

## 陷阱

- **太多随机化。** 在 `slip ∈ [0, 0.9]` 上训练，你的策略如此风险厌恶从不去尝试最优路径。匹配*预期的*真实世界分布，不是"任何事情都可能发生"。
- **太少随机化。** 在薄切片上训练，策略根本无法泛化。使用自适应课程（自动领域随机化），随策略改善加宽分布。
- **错误参数空间随机化。** 随机化错误的东西（相机色调而真实差距是电机延迟）DR 没有帮助。先分析真实机器人。
- **特权信息泄露。** 教师使用全局状态做动作，而不只是观测，可能产生学生无法赶上的策略。确保教师的策略在给定观测历史时学生可以实现。
- **Sim-to-sim 迁移失败。** 如果你的策略对更难的 sim 变体不稳健，它对真实世界也不会稳健。部署前始终在留出的 sim 变体上测试。
- **无真实世界安全包络。** 一个在 sim 中有效且"在真实中有效"但没有低层安全护盾的策略仍然可能破坏硬件。在非学习控制器中添加速率限制、扭矩限制、关节限制。

## 使用

2026 年 sim-to-real 栈：

| 领域 | 栈 |
|--------|------|
| 腿部 locomotion（ANYmal、Spot、类人） | Isaac Lab + DR + 特权教师 / 学生 |
| 操作（灵巧手、抓取放置） | Isaac Lab + DR + DR-GAN 用于视觉 |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线适应 |
| 手指 / 手内操作 | OpenAI Dactyl（前所未有规模的 DR） |
| 工业臂 | MuJoCo-Warp + SI + 小量真实微调 |

对所有规模的控制，工作流是一致的：尽可能好地拟合 sim，随机化你不能拟合的，训练巨大策略，蒸馏，部署带安全护盾。

## 交付

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: 为给定机器人 + 任务规划 sim-to-real 迁移管道，涵盖 DR、SI 和安全。
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

给定机器人平台、任务和真实硬件时间，输出：

1. 现实差距清单。怀疑来源按预期影响排名（接触、感知、执行延迟、视觉）。
2. DR 参数。精确列表、范围、分布。每个范围相对于真实测量说明理由。
3. SI 步骤。要测量哪些参数；测量方法。
4. 教师/学生分割。教师使用哪些特权信息；学生使用哪些观测。
5. 安全包络。低层限制、紧急停止、备份控制器。

拒绝在没有 (a) 零样本 sim 变体测试、(b) 安全护盾、(c) 回滚计划的情况下部署。标记任何 DR 范围宽于测量真实变异性 3 倍的情况为可能过度随机化。
```

## 练习

1. **简单。** 在固定滑动 GridWorld（slip=0.0）上训练 Q-learning 智能体。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制回报 vs 滑动。
2. **中等。** 训练 DR Q-learning 智能体采样 `slip ~ Uniform[0, 0.3]`。评估相同扫描。在 slip=0.5（分布外）处 DR 获得了多少？
3. **困难。** 实现课程：从 slip=0.0 开始，每次策略达到 90% 最优时加宽 DR 范围。测量达到 slip=0.3 零样本的总环境步数 vs 固定 DR 基线。

## 关键术语

| 术语 | 人们怎么说 | 实际指什么 |
|------|-----------------|-----------------------|
| 现实差距 | "Sim 到 real 的差异" | 训练和部署物理/感知之间的分布偏移。 |
| 领域随机化（DR） | "跨随机 sim 训练" | 训练期间随机化 sim 参数以使策略泛化。 |
| 系统辨识（SI） | "测量真实并拟合 sim" | 估计真实物理参数；设置 sim 匹配。 |
| 领域适应 | "在真实数据上微调" | sim 训练后小量真实世界微调；可能适应 obs 或动态。 |
| 特权信息 | "教师的地面真实" | 只有 sim 有的信息；学生必须从 obs 历史推断。 |
| 教师/学生 | "蒸馏特权 → 可观测" | 教师用捷径训练；学生学习在无捷径时模仿。 |
| ADR | "自动领域随机化" | 随策略改善加宽 DR 范围的课程。 |
| Real2Sim | "用真实数据封闭差距" | 学习残差使 sim 模仿真实 rollout。 |

## 拓展阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始 DR 论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 动力学 DR，四足 locomotion。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，前所未有规模的 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的教师-学生。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 推动 2025–2026 部署的大规模并行 sim。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna 框架（用模型做规划 + rollout），这是现代 sim-to-real 管道的根基。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 带基准结果的 sim-to-real 方法分类。