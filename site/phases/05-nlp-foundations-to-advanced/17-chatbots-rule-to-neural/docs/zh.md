# 聊天机器人——从规则到神经到 LLM Agent

> ELIZA 用模式匹配回复。DialogFlow 映射意图。GPT 从权重中回答。Claude 运行工具并验证。每个时代解决了前一个时代最糟糕的失败。

**类型：** 学习
**语言：** Python
**先修课程：** Phase 5 · 13（问答）、Phase 5 · 14（信息检索）
**耗时：** 约 75 分钟

## 问题

用户说"I want to change my flight." 系统必须弄清楚他们想要什么、缺少什么信息、如何获取以及如何完成操作。然后用户说"wait, what if I cancel instead?"，系统必须记住上下文、切换任务并保持状态。

对话对 ML 系统来说是困难的。输入是开放式的。输出必须跨越多轮保持连贯。系统可能需要在世界上行动（改签、扣款）。每个错误的步骤都对用户可见。

聊天机器人架构经历了四个范式的循环，每个都是因为前一个失败得太明显而被引入的。本课按顺序走遍它们。2026 年的生产图景是后两个的混合。

## 概念

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**基于规则（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并产生响应。意图分类器路由到预定义流程。槽填充状态机收集所需信息。在为其设计的狭窄范围内出色地工作。超出立即失败。仍然在不允许幻觉的安全关键领域（银行认证、航空公司预订）中使用。

**基于检索。** FAQ 风格系统。编码每对（话语、响应）。运行时，编码用户消息并检索最接近的存储响应。想 Zendesk 经典的"相关文章"功能。比规则更好地处理改写。不生成，所以没有幻觉。

**神经式（seq2seq）。** 在对话日志上训练的编码器-解码器。从头生成响应。流畅但容易产生通用输出（"I don't know"）和事实漂移。从不可靠地在话题上。Google、Facebook 和 Microsoft 在 2016-2019 年都有令人失望的聊天机器人的原因。

**LLM Agent。** 包裹在循环中的语言模型，计划、调用工具并验证结果。不是带有长提示的聊天机器人。是 Agent 循环：计划 → 调用工具 → 观察结果 → 决定下一步。RAG 式的检索首先接地防止幻觉。工具调用让它实际做事情。这是 2026 年的架构。

四个范式不是顺序替换。2026 年生产聊天机器人路由经过所有四个：对任何破坏性操作用基于规则，对 FAQ 用检索，对自然措辞用神经生成，对歧义的开放式查询用 LLM Agent。

## 构建

### 步骤 1：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行的 ELIZA。反射技巧（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年的经典心理治疗师演示。仍然有启发性。

### 步骤 2：基于检索（FAQ）

这个说明性代码片段需要 `pip install sentence_transformers`（会拉入 torch）。本课的 runnable `code/main.py` 使用 stdlib Jaccard 相似度代替，因此课程可以在没有外部依赖的情况下运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回 `None` 让系统升级。

### 步骤 3：神经生成（基线）

使用小型指令调优编码器-解码器（FLAN-T5）或微调对话模型。2026 年单独使用在生产中不可行（矛盾、话题漂移、事实胡说），但在混合系统内用于自然措辞。DialoGPT 风格的仅解码器模型需要明确的轮次分隔符和 EOS 处理来产生连贯回复；FLAN-T5 text2text 流水线开箱即用作为教学示例。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 步骤 4：LLM Agent 循环

2026 年生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

需要命名的三件事。工具是 LLM 可以调用的可调用函数。循环在 LLM 返回最终答案而非工具调用时终止。步预算防止歧义任务上的无限循环。

真实生产添加：检索优先接地（在每次 LLM 调用前注入相关文档）、护栏（破坏性操作未经确认拒绝）、可观测性（记录每步）和评估（自动检查 Agent 行为保持在规格内）。

### 步骤 5：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：对任何破坏性操作用确定性规则，对缓存 FAQ 用检索，对其他所有用 LLM Agent。这就是 2026 年客户支持系统上线的内容。

## 使用

2026 年技术栈：

| 用途 | 架构 |
|------|------|
| 预订、支付、认证 | 基于规则的状态机 + 槽填充 |
| 客户支持 FAQ | 在策划答案上检索 |
| 开放式帮助聊天 | 带 RAG + 工具调用的 LLM Agent |
| 内部工具 / IDE 助手 | 带工具调用的 LLM Agent（搜索、读取、写入） |
| 伴侣 / 角色聊天机器人 | 带角色系统提示的调优 LLM，在知识上检索 |

生产中始终使用混合路由。没有单一架构处理每个请求都好。路由层本身通常是一个小型意图分类器。

## 仍然上线的失败模式

- **自信捏造。** LLM Agent 声称完成了未做的操作。缓解：验证结果，记录工具调用，永远不要让 LLM 声称做了没有成功工具返回的事情。
- **提示注入。** 用户插入覆盖系统提示的文本。OWASP LLM 应用 2025 年 Top 10 排名第一。两种形式：直接注入（粘贴到聊天中）和间接注入（隐藏在 Agent 读取的文档、电子邮件或工具输出中）。

  攻击率因场景而异。在通用工具使用和编码基准中，跨前沿模型的测量成功率范围约为 0.5-8.5%。特定高风险设置（针对 AI 编码代理的自适应攻击、脆弱编排）达到约 84%。生产 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——Microsoft 365 Copilot 中的零点击数据泄露漏洞，由攻击者控制的电子邮件触发。

  缓解措施：在整个循环中将用户输入视为不可信的；工具调用前清理；将工具输出与主提示隔离；使用 Plan-Verify-Execute (PVE) 模式，Agent 先计划，然后验证每个操作后才执行（这阻止工具结果注入新的未计划操作）；破坏性操作需要用户确认；对工具范围应用最小权限。

  提示工程无法完全消除这种风险。需要外部运行时防御层（LLM Guard、允许列表验证、语义异常检测）。
- **范围蔓延。** Agent 因工具调用返回的切线相关信息而偏离任务。缓解：缩小工具合同；保持系统提示专注；添加离任务率评估。
- **无限循环。** Agent 不断调用同一工具。缓解：步预算、工具调用去重、LLM 判断"我们是否在取得进展"。
- **上下文窗口耗尽。** 长对话将最早的轮次推出上下文。缓解：汇总旧轮次，按相似度检索相关过去轮次，或使用长上下文模型。

## 交付

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: 为给定用例设计聊天机器人技术栈。
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

给定产品上下文（用户需求、合规约束、可用工具、数据量），输出：

1. 架构。基于规则、检索、神经、LLM Agent 或混合（指定哪些路径去哪里）。
2. LLM 选择（如适用）。命名模型家族（Claude、GPT-4、Llama-3.1、Mixtral）。与工具使用质量和成本匹配。
3. 接地策略。RAG 源、检索方法（见第 14 课）、工具合同。
4. 评估计划。任务成功率、工具调用正确率、离任务率、在留出对话上的幻觉率。

对于任何破坏性操作（付款、账户删除、数据修改）拒绝推荐纯 LLM Agent， without a structured confirmation flow. 如果 Agent 有写访问权限，拒绝跳过提示注入审计。
```

## 练习

1. **简单。** 用 10 个模式实现上面的基于规则的响应，用于咖啡店点单机器人。测试边缘情况：重复订单、修改、取消、意图不清楚。
2. **中等。** 构建混合 FAQ + LLM 回退。50 个 SaaS 产品的缓存 FAQ 条目，用在文档站点上检索进行 LLM 回退。在 100 个真实支持问题上测量拒绝率和准确率。
3. **困难。** 用三个工具（搜索、读取用户数据、发送邮件）实现上面的 Agent 循环。用 50 个测试场景运行评估，包括提示注入尝试。报告离任务率、失败任务率和任何注入成功率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Intent | 用户想要什么 | 分类标签（book_flight、reset_password）。路由到处理器。 |
| Slot | 一条信息 | Bot 需要的参数（日期、目的地）。槽填充是系列提问。 |
| RAG | 检索加生成 | 检索相关文档，然后让 LLM 的响应扎根其中。 |
| 工具调用 | 函数调用 | LLM 发出带名称 + 参数的结构化调用。运行时执行，返回结果。 |
| Agent 循环 | 计划、行动、验证 | 控制器，交织 LLM 调用和工具调用直到任务完成。 |
| 提示注入 | 用户攻击提示 | 试图覆盖系统提示的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) —— 原始基于规则的聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) —— Google 晚期的神经聊天机器人论文，就在 LLM Agent 接管之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) —— 命名 Agent 循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) —— 2024 年生产指导，在 2026 年仍然有效。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) —— 提示注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) —— 使提示注入成为首要安全担忧的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) —— 包括 Plan-Verify-Execute 和用户确认流程的实用编排层防御。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) —— 来自间接提示注入的规范零点击数据泄露 CVE。为什么写访问权限 Agent 需要运行时防御的参考案例。