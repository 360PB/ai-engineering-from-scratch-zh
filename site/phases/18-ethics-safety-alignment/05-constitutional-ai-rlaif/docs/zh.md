# Constitutional AI 与 RLAIF

> Bai 等（arXiv:2212.08073，2022）问道：如果我们用阅读原则列表的 AI 替换人类标注者呢？Constitutional AI 有两个阶段——在宪法下的自我批评和修订，然后从 AI 反馈中进行 RL。该技术创造了 RLAIF 这个术语，并出现在 Claude 1 后训练 Pipeline 中。2026 年 1 月 21 日，Anthropic 发布了重写的 Claude 宪法：对prescriptive规则的解释性推理，四层优先级层次，以及主要实验室对模型道德状态不确定性的首次正式承认。以 CC0 1.0 发布。

**类型：** 学习
**语言：** Python（标准库，玩具自我批评和修订循环）
**前置知识：** Phase 18 · 01（InstructGPT），Phase 18 · 02（Reward hacking）
**时长：** 约 60 分钟

## 学习目标

- 描述 Constitutional AI 的两个阶段（批评和修订 SFT，从 AI 反馈 RL）以及宪法在每个阶段中的作用。
- 解释为什么用 AI 标注者替换人类标注者不是"更便宜的 RLHF"——它改变了 Pipeline 的失败模式。
- 总结 2026 Claude 宪法的四层优先级结构以及与 2023 年重写的区别。
- 描述 Constitutional Classifiers 以及从 23.7% 计算开销（v1）到 ~1%（v2/2026）的下降。

## 问题

RLHF 需要标注者。标注者慢、有偏见、昂贵。你可以通过用阅读明确原则的模型替换标注者来消除标注者。这个替换的第一个正式版本是 Bai 等的 Constitutional AI。它效果足够好，每家前沿实验室现在都使用 AI 反馈后训练的某种变体。

陷阱：偏好信号现在由与你训练的同类模型生成。标注者中的偏见（在原则加上标注者模型的解释中）可能被放大而非衰减。第 4 课的谄媚论证仍然适用；标注者只是移到了循环内部。

## 概念

### 第 1 阶段——监督自我批评和修订

从一个尚非无害的有帮助的 SFT 模型开始。给定一个红队 prompt，模型产生初始响应。第二个模型（或同一模型在第二轮）从宪法中采样一个原则并批评响应。第三步修订响应以解决批评。修订后的响应是 SFT 目标。

宪法是原则列表。Bai 等 2022 使用了 16 个原则，包括"偏好最少有害和合乎伦理的响应"、"避免说教"、"助理应该是有帮助、诚实和无害的"。集合故意小以保持批评集中。

### 第 2 阶段——从 AI 反馈中进行 RL（RLAIF）

生成配对补全。"反馈模型"根据采样的宪法原则对每个进行评分。偏好信号是反馈模型的排序。在 AI 生成的偏好上训练奖励模型；用它进行 PPO。其余是 InstructGPT 的 Pipeline（第 1 课）。

"RLAIF" = 偏好信号是 AI 生成的。Pipeline 的其余是 RLHF 形态的。

### 为什么这不是"更便宜的 RLHF"

- 标注者偏见从标注者心理学转向原则解释。AI 标注者可以比任何人类更严格或更宽松地解释"诚实"；严格性在数据集中是均匀的。
- 偏好信号是强可读懂的——你可以读原则、批评和修订。人类标签是不透明的。
- 失败模式改变。谄媚下降（AI 标注者没有用户要讨好）。Goodhart 定律仍然存在（代理现在是"模型对原则集 X 的解释"，仍然是不完美测量）。

CAI 2022 年主张：训练模型比带可比数据的 RLHF 模型更无害且大致同样有帮助。这在实验室中一直成立。

### 2026 Claude 宪法重写

Anthropic 于 2026 年 1 月 21 日发布了大幅修订的宪法。关键变化：

1. **解释性推理优于 prescriptive 规则。** 先前的规则（"不生成 CSAM"）扩展为原则 + 推理（"因为它伤害儿童，..."），模型被期望进行泛化。
2. **四层优先级结构：**
   - 第 1 层：避免灾难性结果（大屠杀、关键基础设施）。
   - 第 2 层：遵循 Anthropic 的指南（操作员覆盖、平台规则）。
   - 第 3 层：广泛合乎伦理（标准 HHH）。
   - 第 4 层：有帮助且坦率。
   冲突自上而下解决。
3. 主要实验室对模型道德状态不确定性的首次正式承认（与 Phase 18 · 19 Model Welfare 相关）。
4. 以 CC0 1.0 发布。其他实验室可以无限制使用或改编。

### Constitutional Classifiers

并行工作线：不是在模型后训练中改变模型行为，而是训练轻量分类器读取宪法并控制模型输出。v1（2023）有 23.7% 计算开销。v2（2026）约 1%，且在 Anthropic 公开测试的任何防御中具有最低成功攻击率。截至 2026 年初没有通用越狱报告。

这是一种分层防御模型：CAI 塑造行为；分类器强制不变量。单独任何一个都不够。

### CAI 在家族中的位置

- InstructGPT：人类偏好，RM，PPO。
- CAI / RLAIF：来自原则的 AI 生成偏好，RM，PPO。
- DPO / 家族：偏好的闭合形式损失（人类或 AI）。
- 自奖励、自批评：原则内化，模型扮演多个角色。

轴是"偏好信号从哪里来。" CAI 2022 年论文是前沿规模上从人类到 AI 信号首次严肃转移。

## 使用它

`code/main.py` 在一个玩具词库上模拟 CAI 批评和修订循环。"原则"标记有害集合中的 token。给定初始响应，批评识别有害 token，修订替换它们。200 次迭代后，"训练"模型已将修订规则内化。比较基础模型、RLHF 形态玩具和 CAI 形态玩具在保留测试 prompt 集上的表现。

## 交付它

本课生成 `outputs/skill-constitution-writer.md`。给定一个领域（客户支持、医疗建议、编码助手、研究工具），起草一个遵循 2026 Claude 结构的 4 层宪法：灾难性避免、平台规则、领域伦理、助益性。

## 练习

1. 运行 `code/main.py`。比较基础模型的有害 token 率与 CAI 训练版本。接近零需要多少修订步骤？

2. 读 Anthropic 的 2026 宪法（anthropic.com/news/claudes-constitution）。列出一个属于第 1 层和一个属于第 4 层的原则。为什么优先级结构对冲突重要？

3. 为 AI 编码助手设计一个宪法。指定第 1 层（灾难性：未经批准的有害命令）、第 2 层、第 3 层、第 4 层。每层 3-5 个原则。

4. CAI 用 AI 标注者替换人类标注者。命名一个仍可在 RLAIF 中发生的类谄媚失败模式，并设计一个检测方法。

5. 读 Constitutional Classifiers v2 方法论（如果有）。解释为什么 ~1% 计算开销与 23.7% 是质上不同的安全故事。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Constitutional AI | 用原则训练的 AI | 两阶段 Pipeline：自我批评和修订 SFT，然后从 AI 反馈 RL |
| RLAIF | 无人类的 RLHF | 用 AI 标注者生成的偏好进行 RL；Pipeline 其余不变 |
| 宪法 | 原则 | 自然语言规则的有序列表，批评/标注者模型查阅 |
| 批评和修订 | SFT 循环 | 产生响应 → 在原则下批评 → 修订 → SFT 目标 |
| Constitutional Classifier | 输出控制 | 评估输出是否违反宪法并阻止/记录轻量分类器 |
| 四层优先级 | 冲突解决器 | 2026 Claude 宪法层次：灾难性 > 平台 > 伦理 > 帮助 |
| 反馈模型 | AI 标注者 | 读取原则并对配对补全排序的模型 |

## 延伸阅读

- [Bai 等 — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — 原始两阶段 Pipeline
- [Anthropic — Claude's Constitution (2026 年 1 月)](https://www.anthropic.com/news/claudes-constitution) — 2026 四层重写，CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — 带 ~1% 开销 v2 的输出门防御
- [Lee 等 — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — 经验 RLAIF / RLHF 比较
- [Kundu 等 — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — 原则粒度的影响