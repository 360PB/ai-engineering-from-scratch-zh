# A/B 测试 LLM 功能 — GrowthBook、Statsig 和感觉问题

> 传统 A/B 测试不是为非决定性 LLM 构建的。关键区别：评估回答"模型能做这活吗？"A/B 测试回答"用户在乎吗？"两者都需要；凭感觉上线已过时。2026 年可测试内容：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs OSS；精度 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：聊天机器人 reward model 变体带来 +70% 对话长度和 +30% 留存；Nextdoor AI 主题行实验在 reward function 调优后带来 +1% CTR；Khan Academy Khanmigo 在延迟 vs 数学精度轴上迭代。平台分化：**Statsig**（2025 年 9 月被 OpenAI 以 $11 亿收购）——序贯测试、CUPED、一体化。**GrowthBook**——开源，仓库原生，Bayesian + Frequentist + Sequential 引擎，CUPED，SRM 检查，Benjamini-Hochberg + Bonferroni 校正。根据仓库-SQL 偏好和"被 OpenAI 收购"对组织是否有影响来选择。

**类型：** 精读
**语言：** Python（标准库，玩具级序贯测试模拟器）
**前置要求：** Phase 17 · 13（可观测性）、Phase 17 · 20（渐进式部署）
**时长：** 约 60 分钟

## 学习目标

- 区分评估（"模型能做这活吗"）和 A/B 测试（"用户在乎吗"）。
- 枚举三个可测试轴（提示词、模型、参数）并为每个选定指标。
- 解释 CUPED、序贯测试和 Benjamini-Hochberg 多重比较校正。
- 根据仓库-SQL 姿态和企业收购立场选择 Statsig 或 GrowthBook。

## 背景问题

你调优了系统提示词。感觉更好。上了线。转化率在噪声里变化。你怪指标。或者你上线了新模型，转化没动——模型退化还是变化太小检测不到？你不知道，因为上线时没做 A/B。

评估回答模型是否在标签集上能完成一项任务。它不回答用户是否偏好输出。只有受控在线实验回答，而且只有实验有足够功效、控制非决定性并校正多重比较时才能回答。

## 核心概念

### 评估 vs A/B 测试

**评估** — 离线，标签集，评判（评分标准或 LLM-as-judge 或人工）。回答："在这个固定分布上，输出正确/有用/安全吗？"

**A/B 测试** — 在线，真实用户，随机分组。回答："新变体移动了重要的用户级指标吗？"

两者都需要。评估在上线前捕获回归；A/B 确认产品影响后。

### 可测试的内容

1. **提示词工程** — 措辞、系统提示词结构、示例。指标：任务成功率、用户留存、每请求成本。
2. **模型选择** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：精度（任务）+ 每请求成本 + P99 延迟。多目标。
3. **生成参数** — temperature、top-p、max_tokens。指标：任务特定（输出多样性 vs 决定性）。

### CUPED — 方差缩减

Controlled-experiments Using Pre-Experiment Data。在比较后一期前，用前一期方差回归削减方差再比较。典型方差缩减：30-70%。有效样本量免费增加。

实现：Statsig 和 GrowthBook 都实现了。

### 序贯测试

经典 A/B 假设固定样本量。序贯测试（"偷看后决定"）控制在重复查看下的假阳性率。Always-valid 序贯程序（mSPRT、Howard 置信序列）让清晰获胜时提前停止。

### 多重比较校正

以 95% 置信跑 20 个 A/B 测试，按概率产生一个假阳性。Bonferroni 校正收紧了每个测试的 α；Benjamini-Hochberg 控制假发现率。GrowthBook 实现了两种。

### SRM — 样本比不匹配

分配哈希将用户随机分组到变体。如果 50/50 分组交付了 47/53，就是有东西坏了——SRM 检查标记它。两平台都实现了。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 $11 亿收购。托管 SaaS。
- 序贯测试、CUPED、held-out 人群。
- 一体化：特性标志 + 实验 + 可观测性。
- 最佳场景：团队已经想要打包产品，不在乎 OpenAI 所有权。

**GrowthBook**：
- 开源（MIT）；仓库原生（直接从 Snowflake/BigQuery/Redshift 读取）。
- 多引擎：Bayesian、Frequentist、Sequential。
- CUPED、SRM、Bonferroni、BH 校正。
- 自托管或托管云。
- 最佳场景：仓库-SQL 团队，数据团队控制指标层，想要 OSS。

### 非决定性使功效复杂化

同一提示词产生不同输出。传统功效计算假设 IID 观测。LLM 非决定性下有效样本量低于标称。作为安全裕度，将所需样本量乘以约 1.3-1.5 倍。

### 真实案例结果

- 聊天机器人 reward model 变体：+70% 对话长度，+30% 留存。
- Nextdoor 主题行：reward function 调优后 +1% CTR。
- Khan Academy Khanmigo：延迟 vs 数学精度的迭代权衡。

### 反模式：凭感觉上线

每个资深工程师都能举出一个"感觉更好"没做 A/B 就上线的功能例子。它们大多数悄悄回归了团队没注意到的产品指标。A/B 是迫使函数。

### 必须记住的数字

- Statsig 被 OpenAI 收购：$11 亿，2025 年 9 月。
- GrowthBook：开源 MIT；Bayesian + Frequentist + Sequential。
- CUPED 方差缩减：30-70%。
- LLM 非决定性 → 样本量缓冲加 30-50%。

## 用现成库

`code/main.py` 模拟带固定和序贯边界的序贯 A/B 测试。展示序贯测试如何允许提前停止。

## 产出

本课产出 `outputs/skill-ab-plan.md`。给定功能变更、工作负载、基线，选出平台、关卡和样本量。

## 练习

1. 运行 `code/main.py`。对于预期提升 5%、基线转化率 3%，80% 功效需要多少样本量？
2. 为医疗监管下的本地客户选 Statsig 还是 GrowthBook。
3. 设计一个测试 GPT-4 vs GPT-3.5 的 A/B，指标是每解决工单成本。主要指标、护栏指标、次要指标是什么？
4. 金丝雀通过但 A/B 显示转化率 -1.2%。你上线吗？写出升级标准。
5. 对前周期方差是后周期 60% 的数据应用 CUPED。计算有效样本量提升。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 评估 | "离线测试" | 在标签集上评估模型能力 |
| A/B 测试 | "实验" | 在用户上的实时随机比较 |
| CUPED | "方差缩减" | 用前周期数据回归削减方差 |
| 序贯测试 | "偷看 ok 测试" | 允许提前停止的 always-valid 程序 |
| 多重比较 | "族错误" | 跑很多测试会增加假阳性 |
| Bonferroni | "紧校正" | α 除以测试数 |
| Benjamini-Hochberg | "BH FDR" | 假发现率控制，不那么保守 |
| SRM | "坏分组" | 样本比不匹配；分配 bug |
| Statsig | "OpenAI 拥有" | 商业一体化，2025 年收购 |
| GrowthBook | "OSS 那个" | MIT 仓库原生平台 |
| mSPRT | "序贯概率比检验" | 经典序贯程序 |

## 扩展阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)