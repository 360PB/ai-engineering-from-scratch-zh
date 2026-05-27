# 并行工具调用与工具流式传输

> 三个独立的天气查询串行化需要三次往返。并行运行它们，总时间缩短到最慢单次调用的时间。现在每个前沿提供商都在单轮中发出多个工具调用。收益是真实的；管道是微妙的。本课走完两个部分：并行扇出和流式参数重组，重点是 id 关联陷阱。

**类型：** 构建
**语言：** Python（标准库、线程池 + 流式 harness）
**前置要求：** Phase 13 · 02（函数调用深度解析）
**时间：** 约 75 分钟

## 学习目标

- 解释为什么 `parallel_tool_calls: true` 存在，何时禁用它。
- 在并行扇出期间将流式参数块关联到正确的工具调用 id。
- 将部分 `arguments` 字符串重组为完整 JSON，不过早解析。
- 运行三城市天气基准测试，展示串行 vs 并行延迟。

## 问题

没有并行调用，回答"班加罗尔、东京和苏黎世的天气如何"的 Agent 这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> 运行执行器，用结果回复
LLM -> call get_weather(Tokyo)
host -> 运行执行器，用结果回复
LLM -> call get_weather(Zurich)
host -> 运行执行器，用结果回复
LLM -> 最终文本答案
```

三次 LLM 往返，每次还要加上执行器延迟。大约是理想挂钟时间的 4 倍。

使用并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> 并发运行所有三个执行器，用三个结果回复
LLM -> 最终文本答案
```

一次 LLM 往返。执行器时间是三个中最长的，不是总和。OpenAI、Anthropic 和 Gemini 的生产基准测试显示，扇出工作负载的挂钟时间减少 60% 到 70%。

代价是关联复杂性。当三个调用乱序完成时，你的结果必须携带匹配的 `tool_call_id`，以便模型可以对齐它们。当结果流式传输时，你必须将部分参数片段组装成完整 JSON 才能执行。Gemini 3 在 part 中添加了唯一 id，以解决一个真实世界问题：两个并行调用同一个工具时无法区分。

## 概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false`（Claude 3.5 及以上默认）并行。设为 `true` 串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型决定。

当工具有排序依赖（先 `create_file` 再 `write_file`）、一个调用的输出告知另一个的输入、或速率限制器无法处理扇出时，禁用并行。

### Id 关联

模型发出的每个调用都有 `id`。宿主机返回的每个结果必须包含相同的 id。没有这个，结果是模糊的。

- **OpenAI。** 每个工具角色消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，这对同名并行调用会出问题）。

### 并发运行调用

宿主机在各自线程、协程或远程 worker 上运行每个调用的执行器。最简单的 harness 使用线程池；生产环境使用 asyncio 的 `asyncio.gather` 或结构化并发。完成顺序不可预测——id 是标识符。

一个常见 bug：按调用列表顺序回复结果而非完成顺序。这通常能工作，因为模型只关心 `tool_call_id`，但如果结果被丢弃或重复，乱序提交会使调试更困难。偏好按完成顺序回复，并附上显式 id。

### 流式工具调用

当模型流式传输时，`arguments` 是分块到达的。三个并行调用的三组独立块在一根线路上交错。你需要每个 id 一个累加器。

按提供商的形状：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（调用列表中的位置）。你按 index 累加，在 id 首次出现时读取，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件为 `message_start`，然后每个类型为 `tool_use` 的块有一个 `content_block_start`（包含 id、name、空 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）发出带 `functionCallId` 的块，使调用可以干净地交错。Gemini 3 之前，流式传输一次返回一个完整调用。

### 部分 JSON 和过早解析陷阱

在 `arguments` 完整之前不能解析。诸如 `{"city": "Bengal` 的部分 JSON 是无效的，会抛出异常。正确的门控是提供商的结束调用信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`，或 Gemini 的流结束事件。只有在那时才尝试 `json.loads`。更稳健的方法使用增量 JSON 解析器，在结构完成时产生事件；OpenAI 的流式传输指南建议将此用于显示实时"思考"指示器的 UX。大括号计数作为完整性测试是不可靠的（引号内的括号或转义内容会导致误报），应仅用作非正式调试启发式。

### 乱序完成

```
call_A: 快 API，先返回
call_B: 慢 API，第二返回
call_C: 中等 API，第三返回
```

宿主机回复仍需引用 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复中的顺序对 OpenAI 或 Anthropic 的正确性不重要。Gemini 接受任何顺序，只要 id 匹配。

### 基准测试：串行 vs 并行

`code/main.py` 中的 harness 用 400、600 和 800 毫秒延迟模拟三个执行器。串行运行总计 1800 毫秒。并行运行总计 max(400, 600, 800) = 800 毫秒。差异是常数而非比例，所以节省量随工具数量增长。

现实世界的注意事项：并行调用对下游 API 造成压力。对速率受限服务的 10 路扇出将会失败。Phase 13 · 17 涵盖网关级背压；重试语义计划在后续 phase 中添加。

### 流式扇出挂钟时间

如果模型本身流式传输，你可以在一个调用的参数完成时就开始执行，而不是等待所有调用最终确定。这是 OpenAI 记录但并非所有 SDK 都公开的优化。本课的 harness 这样做：只要模拟流产生完整的参数对象，宿主机就启动该调用。

## 使用它

`code/main.py` 有两部分。第一部分用 `concurrent.futures.ThreadPoolExecutor` 串行和并行运行三个模拟天气调用，并打印挂钟时间。第二部分重放一个假流式响应——三个并行调用的 `arguments` 块在一根流上交错——并用 `StreamAccumulator` 按 id 重组它们。无需 LLM，无需网络，只需重组逻辑。

要注意的点：

- 串行计时器耗时 1.8 秒。并行计时器在相同假延迟下耗时 0.8 秒。
- 累加器通过按 id 缓冲并仅在每个调用的 JSON 完整时才解析来处理乱序到达的块。
- 执行器在一个 id 的参数确定后立即启动，而不是等所有流结束。

## 发布它

本课生成 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该 Skill 审查哪些工具可以安全并行化、哪些有排序依赖、哪些会压垮下游速率限制——返回带有每个工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行对串行的比率大约是 `max/sum`（真实运行因线程调度、序列化和 harness 开销而与理想值略有偏差）。在什么延迟分布下并行不再重要？

2. 扩展累加器以处理"调用在流中途被取消"的情况，方法是丢弃其缓冲区并发出 `cancelled` 事件。哪个提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。你应该在小幅收益，因为异步上下文切换成本较低，但仅在执行器做真实 I/O 时。

4. 选择两个不应该并行化的工具（例如 `create_file` 然后 `write_file`）。在注册表中添加 `ordering_dependency` 图，并基于该图对并行扇出门控。这是依赖感知调度的最小机制，将在后续 agent 工程 phase 中形式化。

5. 阅读 OpenAI 的并行函数调用部分和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行性的一个真实世界工具类型。（提示：对同一资源的重大变更。）

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|----------|
| Parallel tool calls（并行工具调用） | "单轮扇出" | 模型在一个 assistant 消息中发出多个工具调用 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多调用发出 |
| `disable_parallel_tool_use` | "Anthropic 的反向" | 退出标志；默认并行启用 |
| Tool call id（工具调用 id） | "关联句柄" | 结果消息必须回显的每次调用标识符 |
| Accumulator（累加器） | "流缓冲区" | 每个 id 的部分 `arguments` 块字符串缓冲区 |
| Out-of-order completion（乱序完成） | "最快者先" | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| Dependency graph（依赖图） | "排序约束" | 输出馈入其他工具输入的工具；不能并行化 |
| Parse-early trap（过早解析陷阱） | "JSON.parse 爆炸" | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | "Gemini 3 功能" | 带每个调用唯一 id 的流式参数块 |
| Completion-order reply（完成顺序回复） | "不等所有" | 按 id 键控，收到结果即回复 |

## 延伸阅读

- [OpenAI — 并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和退出标志
- [Anthropic — 工具使用：实现工具使用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和结果批处理
- [Google — Gemini 函数调用并行部分](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 的 id 关联并行调用
- [OpenAI — 带工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流的分块参数重组
- [Anthropic — 流式消息](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`