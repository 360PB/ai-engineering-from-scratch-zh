# 群聊和说话者选择

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个智能体之间共享一个对话；选择器函数（LLM、round-robin 或自定义）在每轮选择下一个发言者。这是涌现多智能体对话的原型——智能体不在静态图中知道其角色，只对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中保留；AutoGen v0.4 将其重写为事件驱动 actor 模型。微软在 2026 年 2 月将 AutoGen 置于维护模式，并与 Semantic Kernel 合并为 Microsoft Agent Framework（RC 2026 年 2 月）。GroupChat 原语在两条track中都存活——学一次，到处使用。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** Phase 16 · 04（原语模型）
**时间：** 约60分钟

## 问题

静态图（LangGraph）在工作流已知时很棒。真实对话不是静态的：有时编码员问审查员，有时研究员，有时编写员。将每个可能的交接硬编码产生边爆炸。你要*智能体对共享池做出反应*，某个函数决定下一个发言者。

这正是 AutoGen GroupChat 做的。

## 概念

### 形状

```
              ┌─── 共享池 ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │（每个人都读所有消息）
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    智能体 A  智能体 B  智能体 C  智能体 D  选择器
                                           │
                                           ▼
                                  "下一个发言者 = C"
```

每个智能体看到每条消息。在每轮调用选择器函数选择下一个发言者。

### 三种选择器风格

**Round-robin。** 固定循环。确定性。可扩展线性增长于 N 但忽略上下文——编码员在话题是法律审查时仍获得轮次。

**LLM 选择。** 调用 LLM 读取最近池并返回最佳下一个发言者。上下文感知但慢：每轮加一个 LLM 调用。AutoGen 默认。

**自定义。** 具有你想要的任何逻辑的 Python 函数。典型：LLM 选择 + 后备规则（例如，"总是在编码员后给验证者轮次"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 拥有选择器。当智能体完成一轮时，管理者调用选择器，选择器返回下一个智能体。循环继续直到终止条件。

### 终止

三种常见模式：

- **最大轮数。** 硬上限总轮数。
- **"TERMINATE" token。** 智能体可以发出哨兵消息；出现时管理者停止。
- **目标达成检查。** 轻量级验证器每轮运行并在完成时停止聊天。

### AutoGen → AG2 分裂和 Microsoft Agent Framework 合并

2025 年初，微软开始围绕事件驱动 actor 模型对 AutoGen（v0.4）进行重大重写。社区将 AutoGen v0.2 的 GroupChat 语义 fork 为 AG2，保留早期采用者已整合的 API。

2026 年 2 月，微软宣布 AutoGen 将进入维护模式，事件驱动 actor 模型合并到 **Microsoft Agent Framework**（RC 2026 年 2 月，现与 Semantic Kernel 合并）。GroupChat 概念在两条 track 中存活；实现细节不同。AG2 是 v0.2 兼容代码的首选上游。

### 何时 GroupChat 适合

- **涌现对话。** 你不想预连线每个可能的下一个发言者。
- **角色混合任务。** 编码员问研究员，研究员问档案员，档案员回头问编码员。流不是 DAG。
- **探索性问题解决。** 想"头脑风暴会议"，不是"装配线"。

### 何时失败

- **严格确定性。** LLM 选择器可能不一致。相同提示，不同运行，不同下一个发言者。
- **谄媚级联。** 智能体 deference 到最确信的声音。Counter-prompt 明确。
- **上下文膨胀。** 每个智能体读取每条消息；10 轮后上下文巨大。使用投影（Lesson 15）限制视图。
- **热门发言者。** 一个智能体因选择器倾向其专业而主导对话。引入说话者平衡作为选择器特征。

### 群聊 vs 主管

相同原语，不同默认值：

- 主管：一个智能体规划，他人执行。选择器是"问规划器下一步做什么"。
- 群聊：所有智能体对等；选择器是共享池上的函数。

两者都使用来自 Lesson 04 的四个原语。群聊默认为 LLM 选择编排和完整池共享状态。

## 构建

`code/main.py` 从零实现 GroupChat 在标准库中。三个智能体（编码员、审查员、管理者），round-robin 和 LLM 选择变体，以及 TERMINATE token 终止。

演示打印对话记录和两种变体的选择器决策跟踪。

运行：

```
python3 code/main.py
```

## 使用

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat 选择器——round-robin vs LLM 选择 vs 自定义，以及什么选择器输入（最近消息、智能体专业、轮次计数）使用。

## 交付

清单：

- **最大轮次上限。** 永远。典型任务 10-20。
- **说话者平衡指标。** 跟踪每智能体轮次；在不平衡超过阈值时警报。
- **终止 token。** `TERMINATE` 或专用验证器智能体。
- **投影或作用域内存。** 约 10 条消息后，考虑给每个智能体仅作用域视图防止上下文膨胀。
- **选择器日志。** 对于 LLM 选择变体，日志选择器的输入和选择。不然无法调试。

## 练习

1. 运行 `code/main.py`。比较 round-robin 和 LLM 选择下的对话。每个变体下哪个智能体主导？
2. 在选择器中添加"每智能体最大发言次数"规则。如何影响记录？
3. 实现目标达成终止：审查员返回"approved"时停止。它在轮上限前触发的频率？
4. 阅读 AutoGen 稳定文档中 GroupChat（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2）并比较其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 添加了什么具体属性（吞吐量、容错、可组合性）？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| GroupChat | "智能体在一个聊天室" | 共享消息池 + 选择器函数。AutoGen / AG2 原语。 |
| 说话者选择 | "下一个发言者" | 选择下一个智能体的函数。Round-robin、LLM 选择或自定义。 |
| GroupChatManager | "会议主持人" | AutoGen 拥有选择器并循环轮次的组件。 |
| ConversableAgent | "基础智能体" | AutoGen 基类；可发送和接收消息的智能体。 |
| 终止 token | ""停止"词" | 结束聊天的哨兵字符串（通常 `TERMINATE`）。 |
| 热门发言者 | "一个智能体主导" | 选择器不断选取同一智能体的失败模式。 |
| 上下文膨胀 | "池增长无界" | 每个智能体读取每条之前消息；上下文随轮次增长。 |
| 投影 | "作用域视图" | 角色特定共享池视图防止上下文膨胀。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html)——参考实现
- [AG2 仓库](https://github.com/ag2ai/ag2)——社区 AutoGen v0.2 延续
- [Microsoft Agent Framework 文档](https://microsoft.github.io/agent-framework/)——合并的继承者，RC 2026 年 2 月
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/)——事件驱动 actor 模型重写详情
