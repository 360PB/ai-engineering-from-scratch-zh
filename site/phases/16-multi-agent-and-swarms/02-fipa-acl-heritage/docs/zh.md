# FIPA-ACL 遗产与言语行为理论

> 在 MCP 之前，在 A2A 之前，就有 FIPA-ACL。2000 年，IEEE Foundation for Intelligent Physical Agents 批准了一个包含二十种述行语、两种内容语言和一组交互协议（合同网、订阅/通知、请求-当）的智能体通信语言。它从业界消失是因为本体论开销对于 web 来说太重了，但 LLM 多智能体系统的复兴正在悄悄重新实现同样的想法，只是没有形式语义：JSON 契约取代了述行语，自然语言取代了本体论。本课认真阅读 FIPA-ACL，这样你就能看清 2026 年协议决策中哪些是重新发明、哪些是创新，以及当前这波浪潮将重新发现哪些2000年代已经解决的问题。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** Phase 16 · 01（为什么要用多智能体）
**时间：** 约60分钟

## 问题

2026 年的智能体协议领域很热闹：MCP 用于工具、A2A 用于智能体、ACP 用于企业审计、ANP 用于去中心化信任、NLIP 用于自然语言内容，加上 CA-MCP 和二十几个研究提案。每个规范都声称自己是基础性的。

诚实的解读是：它们大多数都在重新发现一个非常具体的二十年前的决策树。奥斯汀（1962）和塞尔（1969）的言语行为理论给出了"话语即行为"。KQML（1993）将其转化为有线协议。FIPA-ACL（2000年批准）产生了参考标准化：二十种述行语、SL0/SL1 内容语言、合同网和订阅-通知的交互协议。JADE 和 JACK 是 Java 参考平台。这股热潮在 2010 年左右消退，因为本体论开销太重，web 赢了。

当你看 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的是 FIPA 决策的一个更宽松的、JSON 原生的重新实现。了解这段历史告诉你两件事：哪些新"创新"实际上是重新发明，以及哪些旧的失败模式新规范会重新发现。

## 概念

### 言语行为，一段话说清

奥斯汀注意到有些句子不描述世界——它们改变世界。"我承诺。""我请求。""我宣布。"他称之为述行话语。塞尔将其形式化为五类：断言、指令、承诺、表达、宣告。KQML（Finin 等，1993）使这对于软件智能体可操作：一条消息是述行语（动作）加上内容（动作是关于什么的）。FIPA-ACL 清理了 KQML 的漏洞并围绕二十种述行语进行了标准化。

### 二十种 FIPA 述行语（部分列表）

| 述行语 | 意图 |
|---|---|
| `inform` | "我告诉你 P 为真" |
| `request` | "我请你做 X" |
| `query-if` | "P 为真吗？" |
| `query-ref` | "X 的值是什么？" |
| `propose` | "我提议我们做 X" |
| `accept-proposal` | "我接受该提议" |
| `reject-proposal` | "我拒绝该提议" |
| `agree` | "我同意做 X" |
| `refuse` | "我拒绝做 X" |
| `confirm` | "我确认 P 为真" |
| `disconfirm` | "我否认 P" |
| `not-understood` | "你的消息无法解析" |
| `cfp` | "就 X 征集提案" |
| `subscribe` | "当 X 变化时通知我" |
| `cancel` | "取消进行中的 X" |
| `failure` | "我尝试 X 但失败了" |

完整列表在 `fipa00037.pdf`（FIPA ACL 消息结构）中。重点不是记住它——重点是这些每一种都对应一个 LLM 协议最终会重新添加的原语。

### 经典 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段携带协议信封；一个字段（`content`）携带有效载荷。其余字段正是你每次将重试、线程和本体论追加到 JSON 协议时重新发明的东西。

### 两个遗留平台

**JADE**（Java Agent DEvelopment framework，1999–2020年代）是最常用的 FIPA 兼容运行时。智能体扩展基类，交换 ACL 消息，在容器内运行，并使用"行为"协调。交互协议库包含合同网、订阅-通知、请求-当和提议-接受。

**JACK**（商业，Agent Oriented Software）在 FIPA 消息之上强调 BDI（信念-欲望-意图）推理。更形式化，采用较少。

两者都在 web 技术栈蚕食多智能体用例后衰落。MCP 和 A2A 是 2026 年的运行时"容器"。

### FIPA 衰落的原因

- **本体论开销。** FIPA 要求共享本体来解析 `content`。就本体达成一致是一个长达数年的标准化过程。Web 直接使用 HTTP + JSON。
- **没人用的形式语义。** SL（语义语言）给出了严格的真值条件，但大多数生产系统使用自由形式内容并忽略形式化。
- **工具锁定。** JADE 仅限 Java；JACK 是商业产品。多语言团队绕过了两者。
- **互联网赢得了技术栈。** REST、然后 JSON-RPC、然后 gRPC 取代了 ACL 的传输层。

### LLM 复兴是 FIPA 精简版

将 FIPA `request` 与 MCP `tools/call` 进行比较：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

相同的信封，不同的语法。都携带：谁发给谁、意图、有效载荷、关联 ID。两者都不是另一者的革命——它们是同一设计的不同权衡。

2025 年 Liu 等人的调查（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP"，arXiv:2505.02279）明确了这谱系：MCP 对应工具使用言语行为，A2A 对应对等言语行为，ACP 对应审计跟踪言语行为，ANP 对应去中心化身份扩展。新规范是 ACL 后裔，具有 JSON 语法和更宽松的语义。

### 权衡，坦白说

**FIPA 给了你什么而现代规范放弃了：**

- 形式语义——你可以证明 `inform` 意味着发送者相信内容。
- 述行语的规范目录——你不必重新争论"我们应该有一个 `cancel` 吗？"
- 数十年的交互协议模式——合同网、订阅-通知、提议-接受——具有已知的正确性属性。

**现代规范给了你什么而 FIPA 没有：**

- JSON 原生有效载荷，兼容每个现代工具。
- 自然语言内容，LLM 可以在没有手工编码本体的情况下解释。
- Web 技术栈传输（HTTP、SSE、WebSocket）。
- 通过自描述文档（MCP `listTools`、A2A Agent Card）进行能力发现。

更宽松的意图语义以换取更容易的实现。这正是权衡所在。

### 值得移植的交互协议

FIPA 发布了约 15 个交互协议。有三个值得引入 LLM 多智能体系统：

1. **合同网协议（CNP）。** 管理者发布 `cfp`（征集提案）；投标者用 `propose` 消息回应；管理者接受/拒绝。 这是规范的任务市场模式（Phase 16 · 16 谈判）。
2. **订阅/通知。** 订阅者发送 `subscribe`；发布者在主题变化时发送 `inform`。这是 2026 年每个事件总线的模式。
3. **请求-当。** "当条件 Y 成立时执行 X。" 带有前置条件的延迟动作。2026 年的类比是持久工作流引擎中的延迟任务（Phase 16 · 22 生产扩展）。

每个都可以干净地映射到现代消息队列、HTTP + 轮询或 SSE 流。

### 放弃本体论时会发生什么

没有共享本体，智能体从自然语言内容推断含义。2026 年有记录的失败模式是**语义漂移**：两个智能体用同一个词（`"customer"`）表示微妙不同的概念，接收方智能体根据错误的解释采取行动，没有模式验证器在解析时捕获它。FIPA 的本体论要求会在解析时拒绝该消息。

不用完整本体论的缓解措施：

- JSON Schema `content`——在传输层拒绝结构错误。
- 类型化产物（A2A）——拒绝错误的模态。
- 信封中明确的述行语——即使内容是自然语言，意图也是明确的。

### 2026 年规范，映射到言语行为遗产

| 现代规范 | FIPA 类比 | 保留什么 | 放弃什么 |
|---|---|---|---|
| MCP `tools/call` | `request` | 明确意图、关联 ID | 形式语义、本体论 |
| MCP `resources/read` | `query-ref` | 明确意图、关联 ID | 形式语义 |
| A2A 任务生命周期 | 合同网 + 请求-当 | 异步生命周期、状态转换 | 形式完整性保证 |
| A2A 流事件 | 订阅/通知 | 异步推送 | 类型化谓词订阅 |
| CA-MCP 共享上下文 | 黑板（Hayes-Roth 1985） | 多写者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | 模式 |

从上往下读，模式是：保留结构原语，放弃形式化，让 LLM 掩盖歧义。

## 构建

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 翻译器。它编码和解码经典的 ACL 信封，并展示每个 MCP / A2A 消息形状如何归约为相同的七个字段。演示：

- 将五个 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价物。
- 运行一个玩具合同网协商，一个管理者和三个投标者，使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal`。

运行：

```
python3 code/main.py
```

输出是并排跟踪，显示每条现代消息的 2026 JSON 形式和 FIPA-ACL 形式，然后是合同网竞标的往返。相同的协议原语在往返中存活；只有语法不同。

## 使用

`outputs/skill-fipa-mapper.md` 是一个技能，读取任何智能体协议规范并生成 FIPA-ACL 映射。在采用新协议之前使用它来回答："这是真正新的，还是带 JSON 语法的 `inform`？"

## 交付

不要把 FIPA-ACL 带回来。把它的清单带回来：

- 每个消息的意图原语（述行语）是什么？
- 有没有请求-响应和取消的关联 ID？
- 有没有明确的内容语言（JSON-RPC、纯文本、结构化类型化产物）？
- 交互协议是否是第一类的，还是你从头重新实现合同网？
- 当两个智能体对内容含义产生分歧时会发生什么（语义漂移）？

在任何新协议投入生产之前，为这五个问题编写文档。

## 练习

1. 运行 `code/main.py`。观察往返编码。识别 FIPA 述行语对应 `tools/call`、`resources/read` 和 A2A 任务创建中的哪一个。
2. 用 `cancel` 述行语扩展合同网演示，让管理者在投标中途撤回任务。`cancel` 解决了重试单独不能解决的哪个故障情况？
3. 阅读 FIPA ACL 消息结构（http://www.fipa.org/specs/fipa00037/）第 4.1-4.3 节。选择一个本课未涵盖的述行语，并描述其现代 JSON-RPC 类比。
4. 阅读 Liu 等人，arXiv:2505.02279。对于 MCP、A2A、ACP、ANP，列出它们保留和放弃的 FIPA 述行语族。
5. 为你系统中 `request` 述行语的 `content` 字段设计一个最小的 JSON Schema。这个 Schema 给了你什么纯自然语言没有的，它又付出了什么代价？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 言语行为（Speech act） | "做某事的话语" | 奥斯汀/塞尔：话语作为行为。ACL 的理论父级。 |
| FIPA | "那个旧的 XML 东西" | IEEE Foundation for Intelligent Physical Agents。2000年标准化了 ACL。 |
| ACL | "智能体通信语言" | FIPA 的信封格式：述行语 + 内容 + 元数据。 |
| 述行语（Performative） | "动词" | 消息的意图类：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | "FIPA 的前身" | 知识查询和操作语言（1993）。更简单，范围更窄。 |
| 本体论（Ontology） | "共享词汇表" | 定义内容语言所谈论概念的正式定义。 |
| SL0 / SL1 | "FIPA 内容语言" | 语义语言 0 级和 1 级——形式内容语言系列。 |
| 合同网（Contract Net） | "任务市场" | 管理者发布 cfp；投标者提议；管理者接受。规范的交互协议。 |
| 交互协议（Interaction protocol） | "消息模式" | 具有已知正确性的述行语序列：请求-当、订阅-通知等。 |

## 延伸阅读

- [Liu 等——智能体互操作性协议调查：MCP、ACP、A2A、ANP](https://arxiv.org/html/2505.02279v1)——2025 年规范调查，将现代规范与 FIPA 遗产连接
- [FIPA ACL 消息结构规范（fipa00037）](http://www.fipa.org/specs/fipa00037/)——2000 年信封格式
- [FIPA 通信行为库规范（fipa00037）](http://www.fipa.org/specs/fipa00037/)——完整述行语目录
- [MCP 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)——`request`/`query-ref` 的现代工具使用等价物
- [A2A 规范](https://a2a-protocol.org/latest/specification/)——合同网和订阅-通知的现代对等版
