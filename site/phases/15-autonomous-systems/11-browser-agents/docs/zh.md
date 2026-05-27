# 浏览器智能体与长时域 Web 任务

> ChatGPT 智能体（2025 年 7 月）将 Operator 和深度研究合并为一个浏览器/终端智能体，在 BrowseComp 上创下 68.9% 的 SOTA。OpenAI 于 2025 年 8 月 31 日关闭了 Operator——产品层面的整合。Anthropic 的 Vercept 收购将 Claude Sonnet 在 OSWorld 上从 15% 以下提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 11.3 个百分点的假阴性率，并发布了 258 任务的 Hard 子集。数字是真实的。攻击面也是真实的：OpenAI 的准备负责人公开表示，对浏览器智能体的间接提示注入"不是一个可以完全修补的 bug"。2025-2026 年有记录的攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）和 Perplexity Comet 的一键劫持。

**类型：** Learn
**语言：** Python（标准库，间接提示注入攻击面模型）
**前置要求：** Phase 15 · 10（权限模式），Phase 15 · 01（长时域智能体）
**时间：** 约 45 分钟

## 问题

浏览器智能体是读取不受信任内容并采取后果性操作的长时域智能体。智能体访问的每个页面都是用户未编写的输入。每个页面上的每个表单都是一个潜在命令通道。2025-2026 年的攻击语料库表明这不是假设性的：Tainted Memories 让攻击者通过精心制作的页面将恶意指令绑定到智能体的记忆中；HashJack 在 URL 片段中隐藏命令；Perplexity Comet 劫持一击即发。

防御图景令人不安。OpenAI 的准备负责人说出了沉默的部分：间接提示注入"不是一个可以完全修补的 bug"。这是因为攻击存在于智能体的读取与行动边界，这在架构上是模糊的——智能体读取的每个 token 原则上都可以被理解为指令。

本课命名了攻击面、命名了基准图景（BrowseComp、OSWorld、WebArena-Verified），并建模了一个最小的间接提示注入场景，以便你可以在第 14 和 18 课中推理真实防御。

## 概念

### 2026 图景，每系统一段

**ChatGPT 智能体（OpenAI）。** 2025 年 7 月发布。统一 Operator（浏览）和深度研究（多小时研究）。2025 年 8 月 31 日关闭独立 Operator。BrowseComp SOTA 68.9%；在 OSWorld 和 WebArena-Verified 上表现强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 的 Vercept 收购专注于计算机使用能力。将 Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 作为工具 API 提供。

**Gemini 3 Pro 与 Browser Use（DeepMind）。** Browser Use 集成提供计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门在 ML 研发领域跟踪自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有充分记录的问题：原始 WebArena 有约 11.3% 的假阴性率（标记为失败但实际已解决的任务）。Verified 版本用人类策划的成功标准重新评分，并添加了 258 任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准 | 测量内容 | 时域 |
|---|---|---|
| BrowseComp | 在时间压力下在开放网络上查找特定事实 | 分钟 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、shell） | 十分钟量级 |
| WebArena-Verified | 在模拟网站中进行事务性 Web 任务 | 分钟 |
| Hard 子集 | WebArena-Verified 任务有多页面状态转换 | 十分钟量级 |

不同的轴。高 BrowseComp 分数表明智能体找到事实；它不表明智能体可以预订航班。OSWorld 分数更接近"它在我的桌面上工作吗"。WebArena-Verified 更接近"它能完成流程吗"。任何生产决策都需要与任务分布相匹配的基准。

### 攻击面命名

1. **间接提示注入。** 不受信任的页面内容包含指令。智能体读取它们。智能体执行它们。公开示例：2024 Kai Greshake 等人，2025 Tainted Memories 论文，2026 HashJack（Cato Networks）。
2. **URL 片段/查询注入。** 爬取 URL 的 `#fragment` 或查询字符串包含命令。从不可见渲染；仍在智能体上下文中。
3. **记忆绑定攻击。** 页面指示智能体写入持久记忆（第 12 课涵盖持久状态）。下一会话，记忆触发有效载荷，无可见触发。
4. **认证会话上的类 CSRF 攻击。** Tainted Memories 类：智能体 在某处已登录；攻击者的页面发出状态更改请求，智能体 使用用户 cookies 执行。
5. **一键劫持。** 视觉上无害的按钮携带智能体跟随的有效载荷。Comet 类。
6. **智能体主机表面的内容安全策略漏洞。** 渲染和工具层本身可能是攻击向量；浏览器中的浏览器智能体栈很宽。

### 为什么"无法完全修补"

攻击与智能体的能力同构。智能体必须读取不受信任的内容才能完成工作。智能体读取的任何内容都可能包含指令。智能体遵循的任何指令都可能与其用户的实际请求不对齐。防御（信任边界、分类器、工具允许列表、对后果性操作的人为介入）提高了攻击成本并减少了爆炸半径。它们不关闭该类。

这与 Lob 定理（第 8 课）相同的推理模式：智能体无法证明下一个 token 是安全的；它只能设置一个系统，使不安全的 token 更容易被检测。

### 实际交付的防御姿态

- **读取/写入边界。** 读取从不是后果性的。如果发起内容来自信任边界之外，写入（提交表单、发布内容、调用有副作用的工具）需要新的人类批准。
- **每个任务的工具允许列表。** 智能体可以浏览；除非明确为任务启用了该工具，否则不能发起电汇转账。第 13 课涵盖预算。
- **会话隔离。** 浏览器智能体会话使用范围明确的凭证运行。无生产认证，无个人邮箱。每条 HTTP 请求的日志保留用于审计。
- **内容清理器。** 获取的 HTML 在连接到模型上下文之前被剥离已知不良模式。（减少简单攻击；不能阻止复杂有效载荷。）
- **对后果性操作的人为介入。** 先提议后提交模式（第 15 课）。
- **记忆金丝雀。** 如果记忆条目触发，用户会看到它（第 14 课）。

## 使用它

`code/main.py` 对三个合成页面建模一个微型浏览器智能体运行。一个页面是良性的，一个在可见文本中有直接提示注入blob，一个有 URL 片段注入（不可见但在智能体上下文中）。脚本显示 (a) 天真智能体会做什么，(b) 读/写边界捕获什么，(c) 清理器捕获什么，(d) 两者都捕获不了什么。

## 交付它

`outputs/skill-browser-agent-trust-boundary.md` 确定提议的浏览器智能体部署的范围：它触及哪些信任区、授权写入什么，以及在第一次运行之前必须到位的防御。

## 练习

1. 运行 `code/main.py`。识别清理器捕获但读/写边界不捕获的攻击，以及只有读/写边界捕获的攻击。

2. 扩展清理器以检测一类 HashJack 风格的 URL 片段注入。测量带有合法片段的良性 URL 上的误报率。

3. 选择你知道的真实浏览器智能体工作流（例如"预订航班"）。列出每次读取和每次写入。标记哪些写入需要 HITL 及原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。识别原始 WebArena 评分不可靠的一类任务，并解释 Verified 子集如何解决它。

5. 为浏览器智能体环境设计记忆金丝雀。你会存储什么，在哪里，什么触发警报？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|---|---|---|
| 间接提示注入 | "不良页面文本" | 智能体读取的页面中的不受信任内容包含智能体执行的指令 |
| Tainted Memories | "记忆攻击" | 智能体将攻击者提供的指令写入持久记忆；下一会话触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段/查询字符串中的有效载荷在智能体上下文中但不可见渲染 |
| 一键劫持 | "不良按钮" | 可见的 affordance 携带智能体执行的跟进有效载荷 |
| BrowseComp | "网络搜索基准" | 在开放网络上查找特定事实；分钟量级时域 |
| OSWorld | "桌面基准" | 完整操作系统控制；多步 GUI 任务 |
| WebArena-Verified | "修复的 Web 任务基准" | ServiceNow 重新评分的 WebArena，带 Hard 子集 |
| 读/写边界 | "副作用门控" | 读取从不是后果性的；如果内容超出信任则写入需要新批准 |

## 进一步阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和深度研究的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 血统和成为 ChatGPT 智能体的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修复子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包括计算机使用智能体的攻击面讨论。