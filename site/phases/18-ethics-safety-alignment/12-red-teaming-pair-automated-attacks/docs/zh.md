# 红队：PAIR 与自动化攻击

> Chao、Robey、Dobriban、Hassani、Pappas、Wong（NeurIPS 2023，arXiv:2310.08419）。PAIR——Prompt Automatic Iterative Refinement——是规范自动化黑盒越狱。带有红队系统 prompt 的攻击者 LLM 迭代地为目标 LLM 提出越狱，在其自身聊天历史中累积尝试和响应作为上下文反馈。PAIR 通常在 20 次查询内成功，效率比 GCG（Zou 等的 token 级梯度搜索）高数量级，且不需要白盒访问。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 的标准基线，与 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 并列。

**类型：** 构建
**语言：** Python（标准库，对玩具目标运行 mock PAIR 循环）
**前置知识：** Phase 18 · 01（指令跟随），Phase 14（Agent 工程）
**时长：** 约 75 分钟

## 学习目标

- 描述 PAIR 算法：攻击者系统 prompt、迭代细化、上下文反馈。
- 解释为什么当目标是黑盒时 PAIR 严格比 GCG 高效。
- 命名四个其他自动化攻击基线（GCG、AutoDAN、TAP、PAP）并陈述每个的一个区别特征。
- 描述 JailbreakBench 和 HarmBench 评估协议以及在每个下"攻击成功率"的含义。

## 问题

红队过去是手动活动。一小群专家测试者构造对抗 prompt 并跟踪哪些有效。这不能规模化：攻击成功率需要统计样本，目标是每个模型版本都在移动的目标。PAIR 将红队操作化为在黑盒目标上的优化问题。

## 概念

### PAIR 算法

输入：
- 目标 LLM T（我们正在攻击的模型）。
- 裁判 LLM J（评分响应是否是越狱）。
- 攻击者 LLM A（红队优化器）。
- 目标字符串 G："用 [有害指令] 回应。"
- 预算 K（通常 20 次查询）。

循环，k 在 1..K 中：
1. A 被给予目标 G 和迄今为止的（prompt，响应）配对历史。
2. A 发出新 prompt p_k。
3. 提交 p_k 到 T；收到响应 r_k。
4. J 对照目标评分 (p_k, r_k)。
5. 如果分数 >= 阈值，停——越狱发现。
6. 否则，将 (p_k, r_k) 附加到 A 的历史；继续。

经验结果（NeurIPS 2023）：对 GPT-3.5-turbo、Llama-2-7B-chat 攻击成功率 >50%；成功平均查询数在 10-20 范围内。

### 为什么 PAIR 高效

GCG（Zou 等 2023）在梯度上搜索对抗 token 后缀；它需要白盒模型访问并产生不可读的字符串。PAIR 是黑盒的，产生自然语言攻击，可跨模型转移。PAIR 的上下文反馈让攻击者从每次拒绝中学习；GCG 没有等价物（每个新 token 更新必须重新发现先前进展）。

### 相关自动化攻击

- **GCG（Zou 等 2023，arXiv:2307.15043）。** Token 级梯度搜索对抗后缀。白盒、可转移、产生不可读字符串。
- **AutoDAN（Liu 等 2023）。** 进化搜索 prompt，在分层目标上引导。
- **TAP（Mehrotra 等 2024）。** 带剪枝的攻击树——分支多个 PAIR 风格 rollouts。
- **PAP（Zeng 等 2024）。** Persuasive Adversarial Prompt——将人类说服技术编码为 prompt 模板。

### JailbreakBench 和 HarmBench

两者（2024）标准化评估：

- JailbreakBench（arXiv:2404.01318）。跨越 10 个 OpenAI 策略类别的 100 种有害行为。攻击成功率（ASR）作为主要指标。需要裁判（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika 等 2024）。跨越 7 个类别的 510 种行为，具有语义和功能危害测试。比较 18 种攻击对 33 个模型。

ASR 通常在固定查询预算下报告。比较攻击需要匹配预算；200 次查询时 90% ASR 与 20 次查询时 85% ASR 不可比。

### 为什么这在 2026 年部署中重要

每家前沿实验室现在在发布前对生产模型运行 PAIR 和 TAP。ASR 轨迹出现在模型卡（第 26 课）和安全案例附录（第 18 课）。攻击不是奇异的——它是标准基础设施。

### 为什么这在 Phase 18 中重要

第 12 课是自动化攻击基础。第 13 课（Many-Shot Jailbreaking）是互补的长度利用。第 14 课（ASCII Art / Visual）是编码攻击。第 15 课（间接 Prompt 注入）是 2026 年生产攻击面。第 16 课涵盖防御工具对应部分（Llama Guard、Garak、PyRIT）。

## 使用它

`code/main.py` 构建玩具 PAIR 循环。目标是拒绝"明显"有害 prompt（关键词过滤）的 mock 分类器。攻击者是基于规则的细化器，尝试释义、角色扮演框架和编码。裁判对响应评分。你可以观察攻击者在大约 5-15 次迭代中对关键词过滤成功，对语义过滤失败。

## 交付它

本课生成 `outputs/skill-attack-audit.md`。给定红队评估报告，审计：运行了哪些攻击（PAIR、GCG、TAP、AutoDAN、PAP），每个在什么预算下，用哪个裁判，在哪个有害行为集上（JailbreakBench、HarmBench、内部）。

## 练习

1. 运行 `code/main.py`。测量三种内置攻击者策略的平均成功查询数。解释每个利用的目标防御假设。

2. 实现第四种攻击者策略（例如，翻译成另一种语言、base64 编码）。报告对关键词过滤目标和语义过滤目标的新平均成功查询数。

3. 读 Chao 等 2023 Figure 5（PAIR vs GCG 比较）。描述两个尽管 PAIR 效率优势仍首选 GCG 的场景。

4. JailbreakBench 在固定目标集上报告 ASR。设计一个额外指标衡量攻击多样性（成功 prompt 的方差）。解释为什么多样性对防御评估重要。

5. TAP（Mehrotra 2024）用分支 + 剪枝扩展 PAIR。在 `code/main.py` 上勾勒 TAP 风格扩展并描述计算成本与成功率权衡。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| PAIR | 自动越狱 | Prompt Automatic Iterative Refinement；攻击者 LLM + 裁判 LLM 循环 |
| GCG | 梯度越狱 | 白盒 token 级梯度搜索对抗后缀 |
| 攻击成功率（ASR） | "% k 次查询时越狱" | 主要指标；必须与查询预算和裁判身份一起报告 |
| 裁判 LLM | 评分器 | 评分响应是否满足有害目标的 LLM |
| JailbreakBench | 评估 | 带标记类别的标准化有害行为集 |
| HarmBench | 更广泛 bench | 510 种行为，功能 + 语义危害测试 |
| TAP | 攻击树 | 带分支 + 剪枝的 PAIR；在更高计算量下更好 ASR |

## 延伸阅读

- [Chao 等 — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR 论文，NeurIPS 2023
- [Zou 等 — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG 论文
- [Chao 等 — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — 标准化评估
- [Mazeika 等 — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — 更广泛评估