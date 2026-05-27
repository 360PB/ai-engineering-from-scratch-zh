# 基准测试：SWE-bench 与 GAIA

> 三个基准测试锚定了 2026 年 Agent 评估。SWE-bench 测试代码修复。GAIA 测试通用工具使用。AgentBench 测试多环境推理。了解它们的组成、污染故事，以及它们不测量的东西。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** Phase 14 · 06（工具使用）
**时间：** 约 60 分钟

## 学习目标

- 说出 SWE-bench 的测试框架（FAIL_TO_PASS）并解释为何它以单元测试为门槛。
- 解释 SWE-bench Verified（OpenAI，500 题）为何存在，以及它去掉了什么。
- 描述 GAIA 的设计理念：人类简单，AI 困难；三档难度。
- 说出 AgentBench 的八个环境以及开源 LLM 的主要障碍。
- 总结 SWE-bench+ 污染发现及其影响。

## 问题背景

排行榜告诉你哪个模型在一个基准上赢了。它们不会告诉你：

- 该基准是否被污染（解决方案在训练数据中、测试泄露）。
- 该基准是否在测量你关心的东西（代码 vs 浏览 vs 通用）。
- 评估器是否稳健（AST 匹配、状态检查、人工审查）。

在引用数字之前，先了解三个锚定基准测试及其失败模式。

## 核心概念

### SWE-bench（Jimenez et al., ICLR 2024 oral）

- 来自 12 个流行 Python 仓库的 2,294 个真实 GitHub issue。
- Agent 获得：修复前 commit 的代码库 + 自然语言 issue 描述。
- Agent 产出：一个补丁。
- 评估器：应用补丁，运行仓库的测试套件。补丁必须翻转 FAIL_TO_PASS 测试（之前失败，现在通过），且不能破坏 PASS_TO_PASS 测试。

SWE-agent（Yang et al., 2024）在发布时通过强调 Agent-计算机接口（文件编辑命令、模型理解的搜索语法）达到 12.5%。

### SWE-bench Verified

OpenAI，2024 年 8 月。人类策划的 500 题子集。移除了模糊的 issue、不可靠的测试、修复不明确的任务。是"你的 Agent 是否能发出真实补丁？"的主要基准。

### 污染

- 94% 以上的 SWE-bench issue 先于大多数模型的截止日期。
- **SWE-bench+** 发现 32.67% 的成功补丁在 issue 文本中泄露了解决方案（模型在描述中看到了修复），31.08% 因测试覆盖薄弱而可疑。
- Verified 更干净，但不是零污染。

实践含义：在 SWE-bench 上得 50% 的模型在 SWE-bench+ 上可能只得 35%。如果声称 SWE-bench 性能，请同时报告两个。

### GAIA（Mialon et al., 2023 年 11 月）

- 466 个问题；300 个留在 huggingface.co/gaia-benchmark 的私人排行榜。
- 设计理念："对人类来说概念简单（92%），对 AI 困难（带插件的 GPT-4：15%）。"
- 测试推理、多模态、网络、工具使用。
- 三档难度；第三档需要跨模态的长工具链。

GAIA 是你用来衡量"通用能力"的基准。不要与代码专用基准混淆。

### AgentBench（Liu et al., ICLR 2024）

- 8 个环境，涵盖代码（Bash、DB、KG）、游戏（Alfworld、LTP）、网页（WebShop、Mind2Web）和开放式生成。
- 多轮，每分割约 4k–13k 轮。
- 主要发现：长期推理、决策和指令遵循是开源 LLM 追赶商业模型的障碍。

### 这些基准不测量的东西

- 真实运营成本（Token、系统时间）。
- 对抗条件下的安全行为。
- 你领域的性能（使用你自己的评估，第 30 课）。
- 尾部故障（基准测试取平均；生产运营商关心最差的 1%）。

### 基准测试会出错的地方

- **唯数字论。** SWE-bench 50% 不如 P50/P75/P95 成本 + 步数分布有意义。
- **污染性声称。** 报告 SWE-bench 而不提 Verified 或 SWE-bench+ 是误导性的。
- **基准作为开发目标。** 为基准优化会使产出偏离生产实用价值。

## 动手实现

`code/main.py` 实现了一个类 SWE-bench 的 toy 测试框架：

- 合成 bug-fix 任务（3 个）。
- 一个脚本化的"Agent"提出补丁。
- 一个测试运行器，检查 FAIL_TO_PASS（bug 已修复）和 PASS_TO_PASS（没有破坏）。
- 一个 GAIA 风格的问题分解深度难度分类器。

运行：

```
python3 code/main.py
```

输出展示了每个任务和每个难度的解决率，并将评估规则具象化。

## 用现成库

- **SWE-bench Verified** 用于代码 Agent。始终报告 Verified 分数。
- **GAIA** 用于通用 Agent。使用私人排行榜分割。
- **AgentBench** 用于多环境对比。
- **自定义评估**（第 30 课）用于你的产品实际形态。

## 产出

`outputs/skill-benchmark-harness.md` 为任意代码库-任务对构建 SWE-bench 风格测试框架，含 FAIL_TO_PASS / PASS_TO_PASS 门槛。

## 练习

1. 将玩具框架移植到你的一个真实仓库。为已知 bug 写 3 个 FAIL_TO_PASS 测试。
2. 添加步数指标。在你的 3 个任务上，每个解决方案耗费多少 Agent 步？
3. 读取 SWE-bench+ 论文。实现解决方案泄露检查（用正则匹配 issue 文本与 diff）。
4. 从公开分割下载一个 GAIA 问题。追踪 GPT-4 级 Agent 会做什么。它需要什么工具？
5. 读取 AgentBench 按环境分解。你的产品表面镜像哪个环境？该环境的"SOTA"是什么样？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SWE-bench | "代码 Agent 基准" | 2,294 个 GitHub issue；补丁必须翻转 FAIL_TO_PASS 测试 |
| SWE-bench Verified | "干净的 SWE-bench" | 500 题人类策划子集，OpenAI |
| FAIL_TO_PASS | "修复门槛" | 之前失败、补丁后必须通过的测试 |
| PASS_TO_PASS | "无退化门槛" | 之前通过的测试必须继续保持通过 |
| GAIA | "通用 Agent 基准" | 466 个人类简单 / AI 困难的多工具问题 |
| AgentBench | "多环境基准" | 8 个环境；长期多轮 |
| Contamination | "训练集泄露" | 基准任务出现在模型训练数据中 |
| SWE-bench+ | "污染审计" | 在成功的 SWE-bench 补丁中发现 32.67% 解决方案泄露 |

## 延伸阅读

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — 原始基准
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策划子集
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — 通用基准
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — 多环境套件