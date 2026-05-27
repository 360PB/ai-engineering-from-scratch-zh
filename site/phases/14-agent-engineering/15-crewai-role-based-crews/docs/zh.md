# CrewAI：基于角色的团队与 Flows

> CrewAI 是 2026 年基于角色的多 Agent 框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自主角色协作）和 Flows（事件驱动、确定性）。文档毫不客气地说："任何生产就绪的应用，从 Flow 开始。"

**类型：** 学习 + 动手实现
**语言：** Python（标准库）
**前置知识：** Phase 14 · 12（工作流模式）、Phase 14 · 14（Actor 模型）
**时间：** 约 75 分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）各自的职责。
- 区分 Sequential、Hierarchical 和计划中的 Consensus process；根据工作负载选择。
- 区分 Crews（自主角色驱动）和 Flows（事件驱动确定性），并解释文档对生产的建议。
- 用 `@tool` 装饰器和 `BaseTool` 子类接入工具；理解结构化输出与自由文本的区别。
- 说出 CrewAI 的四种记忆类型以及各自的适用场景。
- 用标准库实现一个三 Agent 团队（研究员、写手、编辑）产出简报。
- 识别 CrewAI 三种常见失败模式：提示词膨胀、管理者 LLM 税、脆弱的交接。

## 问题背景

采用多 Agent 框架的团队总是撞上同一堵墙。"自主协作"在演示里听起来很棒。然后客户提了一个 bug，你需要确定性回放。或者财务问：这个 LLM 路由团队每次运行要花多少钱。或者 oncall 需要知道凌晨 3 点是哪个 Agent 卡住了。

自由形式的 LLM 路由团队无法干净利落地回答这些问题。纯 DAG 可以回答所有这些问题，但失去了头脑风暴 Agent 需要的探索性形态。

CrewAI 的分工诚实面对了这个权衡。Crews 用于协作性、角色驱动、探索性的工作。Flows 用于事件驱动、代码所有、可审计的生产场景。同一框架，两种形态，按界面选择。

## 核心概念

### 四个原语

CrewAI 的表面很小。记住这个，其余都是配置。

- **Agent。** `role + goal + backstory + tools + (可选) llm`。backstory 是承力结构。它塑造语气、判断力、Agent 停止的时机。工具是 Agent 可以调用的函数（详见下文）。
- **Task。** `description + expected_output + agent + (可选) context + (可选) output_pydantic`。一个可复用的工作单元。`expected_output` 是契约。`context` 列出上游任务的输出，这些输出会传入当前任务。`output_pydantic` 强制输出为特定结构。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory`、`verbose`、`manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。决定运行的形态。

Agent 不直接看到彼此。Task 引用 Agent。 Crew 编排 Task 的顺序。Process 决定谁选择下一个任务。这就是整个心智模型。

> **已针对** CrewAI 0.86（2026-05）验证。新版本可能重命名或合并 process 类型；依赖特定形态时请查阅 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** Task 按声明顺序执行。第 N 个 Task 的输出作为 `context` 传入第 N+1 个 Task。成本最低，最可预测。用于顺序固定的场景。
- **Hierarchical。** 一个管理者 Agent（独立的 LLM 调用）在专家之间路由。CrewAI 从你的 `manager_llm` 配置或默认配置孵化管理者。管理者每轮选择下一个专家任务，可以拒绝或重新路由。用于有四个或更多专家且顺序真正依赖前面输出时。
- **Consensus。** 计划中，尚未在公开 API 中实现。文档保留这个名字用于未来基于投票的流程。不要依赖它。

Hierarchical 在每个专家调用之上额外加了一轮管理者的 LLM 调用。五步运行时 Token 成本可能翻三倍。只有在确实需要路由时才付这个代价。

### Crews vs Flows

这是文档在 2026 年开篇就强调的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时决定形态。适用于：研究、头脑风暴、初稿、路径本身是答案一部分的工作。难以回放，难以测试，原型成本低。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记在另一个步骤发出该 topic 时触发的步骤。每步是普通 Python（可以在内部调用 Crew）。适用于：生产。可观测。可测试。确定性。

文档 2026 年对生产的建议：从 Flow 开始。当自主性值得其成本时，在 Flow 步骤内部调用 `Crew.kickoff()` 引入 Crew。Flow 给你审计边界，Crew 给你探索能力。组合使用，不必选择。

### 工具接入

三种方式给 Agent 配备工具。选择最简单的那个。

1. **`@tool` 装饰器。** 纯函数变成工具。签名即 schema；文档字符串是 LLM 看到的描述。适用于一次性辅助工具。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，含显式参数 schema、异步支持、重试。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 提供第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一行导入即可接入。

结构化输出使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 根据模型验证 LLM 响应，或强制转换或重试。配合紧凑的 `expected_output` 字符串使用。自由文本输出适合草稿；结构化输出是下游 Flow 可以消费的形态。

### 记忆钩子

CrewAI 开箱提供四种记忆类型。它们可以组合：一个 Crew 可以同时启用全部四种。

> **已针对** CrewAI 0.86（2026-05）验证。最近版本将所有记忆路由到统一 `Memory` 系统包装这四个存储。下面的概念模型仍然成立，但在新版本中公开类表面可能塌缩为单一 `Memory` 入口；请查阅 [CrewAI memory docs](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **短期记忆。** 单次运行内的对话缓冲。运行结束时清除。
- **长期记忆。** 跨运行持久化。存储在向量数据库中（默认 Chroma，可替换）。根据与当前任务的相似度检索。
- **实体记忆。** 按实体的 Facts。"客户 X 在企业版计划上。"按实体键控，非相似度。跨运行持久化。
- **上下文记忆。** 组装时检索。在 Agent 需要它的时刻拉取相关记忆，而非预加载。

在 Crew 上用 `memory=True` 或按类型配置启用。后端是你配置的嵌入提供商（默认 OpenAI，可替换为本地）。记忆是 CrewAI 比更薄的框架更有价值的地方之一；纯 LangGraph 需要你自己接入每一个。

### CrewAI 适用场景

- 三到六个有命名角色的 Agent，协作工作流。起草、审查、规划、头脑风暴。
- 路由中 LLM 对下一步的判断本身是价值的一部分时（Hierarchical）。
- 团队读 `role + goal + backstory` 比读图定义更舒服的地方。

### CrewAI 不适用场景

- 有严格顺序的确定性 DAG。使用 LangGraph（第 13 课）。图结构是正确的抽象；CrewAI 的角色描述是摩擦。
- 亚秒级延迟预算。Hierarchical 增加往返。即使是 Sequential 也要序列化包含 backstory 和前面输出的提示词。
- 单 Agent 循环。跳过框架；Agent 循环（第 1 课）加工具注册表更短。

第 17 课（Agent 框架取舍）用矩阵形式整理了这些。简而言之：CrewAI 位于"协作角色驱动"那个角落。

### 依赖形态

独立于 LangChain。Python 3.10 至 3.13。使用 `uv`。Star 数量：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 快照）。记录了 AWS Bedrock 集成；厂商基准测试报告在 QA 工作负载上比 LangGraph 有明显加速，但方法论（数据集、硬件、评估指标）未公布，因此将框架-厂商数字视为方向性参考。

### 这个模式会出问题的地方

- **Backstory 导致提示词膨胀。** 每个 Agent 2000 词的 backstory，五 Agent 团队在第一次工具调用之前就烧光了上下文预算。backstory 控制在 200 词以内。在 Agent 之间复用措辞，不要重复同一种风格五次。
- **管理者 LLM Token 税。** Hierarchical process 在每个专家调用之前加了一个管理者 LLM 调用。五 Task 团队变成六次 LLM 调用而非五次，而且管理者调用携带完整的任务列表加前面输出。除非路由真的依赖输出，否则切到 Sequential。
- **脆弱的交接。** 第 N 个 Task 的 `expected_output` 是"一份大纲"。第 N+1 个 Task 读取它作为 `context` 并尝试解析三个部分。LLM 生成了四个。下游 Agent 即兴发挥。用第 N 个 Task 上的 `output_pydantic` 修复，使第 N+1 个 Task 读到类型化对象而非自由文本。
- **Crew 即生产。** 将自由形式的 Crew 发送到生产环境，不加 Flow 包装。输出变异性高；无法回放；oncall 无法对比坏运行和好运行的差异。用 Flow 包装。

## 动手实现

`code/main.py` 实现了两种形态的标准库版本，加一个三 Agent 团队。

形态：

- `Agent`、`Task` 数据类，与 CrewAI 的表面一致。
- `SequentialCrew.kickoff(inputs)` 按声明顺序执行任务，线程化输出作为 `context`。
- `HierarchicalCrew.kickoff(topic)` 添加一个管理者 Agent 每轮选择下一个专家，遇到"done"时停止。
- `Flow`，含 `@start` 和 `@listen(topic)` 装饰器、一个小事件循环和执行跟踪。
- `tool(name)` 装饰器镜像 CrewAI 的 `@tool` 形态。
- `Memory`，含 `short_term`、`long_term`、`entity` 存储；模拟相似度使用 numpy。
- 模拟 LLM 响应是用 role 加输入前缀键控的硬编码字符串。无网络。确定性。

具体演示：研究员、写手、编辑团队产出"agent 工程 2026"简报。研究员拉取（模拟）来源。写手起草。编辑精简。同一团队通过 Flow 运行，展示确定性形态。

运行：

```bash
python3 code/main.py
```

执行跟踪覆盖：`context` 线程化输出的顺序团队、管理者选择下一步的层级团队（研究员、写手、编辑，然后"done"）、通过显式 topic（`researched`、`drafted`、`edited`）运行相同三步的 Flow、`@tool` 路由的工具调用、以及跨两次 kickoff 持久化的长期记忆。

Crew 执行跟踪是流动的；管理者原则上可以重新排序。Flow 执行跟踪是固定的。这个选择就是本课要讲的内容。

## 用现成库

- **CrewAI Flow** 用于生产。即使 Flow 只有一步调用 `Crew.kickoff()`。Flow 给出了审计边界。
- **CrewAI Crew（Sequential）** 用于顺序清晰、需协作的工作，尤其是初稿和审查循环。
- **CrewAI Crew（Hierarchical）** 当路由依赖输出且有四个或更多专家时。
- **LangGraph**（第 13 课）用于显式状态机、持久化恢复、严格排序。
- **AutoGen v0.4**（第 14 课）用于 Actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先的产品，含交接和护栏。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品，含子 Agent 和会话存储。

## 产出

`outputs/skill-crew-or-flow.md` 为任务选择 Crew vs Flow 并脚手架最小实现。硬性拒绝：无 backstory 的 Crew、无显式 topic 的 Flow、少于三个专家的 Hierarchical。

## 陷阱

- **Backstory 沦为调味料。** 它影响输出。为每个 Agent 测试三个变体；变异性是真实存在的。选一个，冻结它。
- **跳过 `expected_output`。** 没有每个 Task 的契约，下游 Task 只能拾起 LLM 生成的任何东西。团队运行了；审计失败了。
- **记忆常开。** 长期记忆每次运行都写入。向量数据库增长。检索变得有噪声。只对事实具有持久性的任务限定写入范围。
- **管理者提示词漂移。** Hierarchical 的管理者提示词是隐含的。如果路由变得奇怪，开启 verbose 模式并读取它。
- **Crew 中的工具副作用。** Crew 可能比预期更频繁地调用工具。POST、DELETE、支付属于 Flow 步骤，永远不要放在 Crew 工具中。

## 练习

1. 将 Sequential crew 转换为 Flow。统计可变性的接触点减少了多少。标注可读性下降的地方。
2. 给团队添加实体记忆：关于客户的事实跨 kickoff 持久化。验证检索拉取正确的实体。
3. 实现一个 Hierarchical process，其中管理者拒绝在写手输出少于三段之前路由给编辑。追踪重试。
4. 接入一个 `BaseTool` 子类实现（模拟的）网络搜索。对比执行跟踪形态与 `@tool` 装饰器版本。
5. 在编辑任务上添加 `output_pydantic=Brief`，其中 `Brief` 含 `title`、`summary`、`sections`。让写手任务输出一次格式错误的 JSON；验证 CrewAI 在跟踪中的重试行为。
6. 读取 CrewAI 文档介绍。将玩具迁移到真实的 `crewai` API。标准库版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse（第 24 课）接入一次真实运行。你在标准库版本中遗漏了哪些跟踪？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Agent | "角色" | Role + goal + backstory + tools |
| Task | "工作单元" | Description + expected output + assignee + 可选结构化输出 |
| Crew | "Agent 团队" | Agents + Tasks + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus（计划中） |
| Flow | "确定性工作流" | 事件驱动、代码所有、可测试 |
| Backstory | "角色提示词" | 塑造 Agent 的语气和判断力 |
| `@tool` | "函数工具" | 将函数装饰成 Agent 可调用的工具 |
| `BaseTool` | "类工具" | 基于类的工具，含参数 schema、重试、异步支持 |
| Entity memory | "按实体的事实" | 作用域限定为客户 / 账户 / 问题维度的记忆 |
| Long-term memory | "跨运行记忆" | 向量后端记忆，跨 kickoff 持久化 |
| Contextual memory | "即时检索" | 在 Agent 需要它的时刻拉取的记忆 |
| Manager LLM | "路由器 Agent" | Hierarchical process 中额外调用 LLM 选择下一个任务 |
| `expected_output` | "任务契约" | 告诉 Agent（和审计）返回形状的字符串 |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：概念与推荐的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：事件驱动形态，`@start`，`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`，`BaseTool`，内置工具包
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：多 Agent 何时有帮助，何时没有
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案