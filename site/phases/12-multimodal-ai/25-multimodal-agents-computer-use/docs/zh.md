# 多模态 Agent 与计算机使用（Capstone）

> 2026 年前沿产品是一个多模态 agent，读取截图、点击按钮、导航网页 UI、填写表单、端到端完成工作流。SeeClick 和 CogAgent（2024）证明了 GUI grounding 原语。Ferret-UI 添加了移动端。ChartAgent 引入图表视觉工具使用。VisualWebArena 和 AgentVista（2026）是前沿追赶的基准——即使 Gemini 3 Pro 和 Claude Opus 4.7 在 AgentVista 困难任务上得分约 30%。本 capstone 汇聚 Phase 12 的每条线索：感知（高分辨率 VLM）、推理（带工具使用的 LLM）、接地（坐标输出）、长期记忆和评估。

**类型：** Capstone
**语言：** Python（标准库，动作 schema + agent 循环骨架）
**前置知识：** Phase 12 · 05（LLaVA），Phase 12 · 09（Qwen-VL JSON），Phase 14（Agent Engineering）
**时间：** 约 240 分钟

## 学习目标

- 设计多模态 agent 循环：感知 → 推理 → 行动 → 观察 → 重复。
- 构建 VLM 可以作为 JSON 发出的 GUI grounding 输出 schema（点击坐标、输入文本、滚动、拖动）。
- 比较仅截图 agent 与可访问性树 agent 与混合 agent。
- 在小型 VisualWebArena 片上设置多模态 agent 基准评估。

## 问题背景

预订网站工作流："帮我找 4 月 15 日去东京的航班，靠过道 800 美元以下，并预订。"

多模态 agent 需要：

1. 拍浏览器截图。
2. 将截图 + URL + 目标解析为计划。
3. 发出结构化动作：点击（在 x,y）、输入"Tokyo"（在元素 E）、向下滚动、选择（单选按钮）。
4. 将动作应用于浏览器。
5. 观察新状态（下一截图）。
6. 重复直到任务完成。

每步都是多模态 VLM 调用。VLM 输出必须是可解析的 JSON。错误在步骤间累积，所以恢复很重要。

## 核心概念

### GUI grounding——原语

GUI grounding 是：给定截图和自然语言指令，输出点击的 (x, y) 坐标（或其他动作）。

SeeClick（arXiv:2401.10935）是首个规模化开源结果：在合成 + 真实 GUI 数据上微调 VLM，输出坐标作为纯文本 token。可行。

CogAgent（arXiv:2312.08914）添加 1120x1120 高分辨率编码用于密集 UI。分数：网页导航约 84%。

Ferret-UI（arXiv:2404.05719）专注于移动 UI，与 iOS 可访问性数据集成。

输出格式通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 帮助恢复：如果截图间坐标漂移，语义提示让系统重新接地。

### 动作 schema

典型动作 schema 有 6-10 种动作类型：

- `click`：(x, y)
- `type`：(text, x?, y?)
- `scroll`：(direction, amount)
- `drag`：(x0, y0, x1, y1)
- `select`：(option_index)
- `hover`：(x, y)
- `navigate`：(url)
- `wait`：(ms)
- `done`：(success, explanation)

Agent 每步发出一个动作。浏览器包装器执行并返回新状态。

### 仅截图 vs 可访问性树

两种输入模式：

- 仅截图：完整图像，无结构信息。最通用；适用于任何应用。
- 可访问性树：结构化 DOM / iOS 可访问性信息。接地可靠得多；在树可用的地方工作。
- 混合：两者兼用，以树作为原子动作的可靠接地器，截图用于语义上下文。

生产 agent 尽可能使用混合。浏览器自动化（Selenium + 可访问性）始终有树；桌面应用有时有。

### 长期记忆

20 步工作流生成 20 张截图。VLM 的上下文快速填满。三种压缩策略：

- 摘要链：每 5 步后，总结发生了什么，丢弃旧截图。
- 跳帧：保留第一张、最后一张和每第 3 张截图。
- 工具记录日志：执行动作，保留已做事项的文本日志；不要重新查看旧截图。

Claude 的 computer-use API 使用日志模式。更简单，更可靠。

### 视觉工具使用

ChartAgent（arXiv:2510.04514）为图表理解引入视觉工具使用：裁剪、放大、OCR、调用外部检测。Agent 可以输出"裁剪到区域 (100, 200, 300, 400) 然后调用 OCR"作为工具调用。工具返回文本；VLM 继续推理。

这个模式泛化：Set-of-Mark 提示、区域注释和外部检测工具都适合相同的"输出工具调用，接收结构化响应"schema。

### 2026 年基准

- ScreenSpot-Pro。在约 1k 网页截图上的 GUI grounding。开源 SOTA Qwen2.5-VL-72B ~85%。前沿 ~90%。
- VisualWebArena。端到端网页任务（购物、论坛、分类广告）。开源 SOTA ~20%。Gemini 3 Pro ~27%。
- AgentVista（arXiv:2602.23166）。2026 年最难基准。12 个领域的现实工作流。前沿模型得分 27-40%；开源模型 10-20%。
- WebArena / WebShop。更老的基准；被前沿饱和。

### 为什么仍然困难

Agent 性能瓶颈：

1. 细粒度视觉接地。"点击小 X" 在移动分辨率下经常失败。
2. 长期规划。10 个动作后，agent 偏离目标。
3. 错误恢复。当点击失败（按钮错误）时，检测 + 恢复很少在训练数据中。
4. 跨页面上下文。在标签页之间跳转或长表单丢失状态。

研究方向：记忆架构、显式重规划、多模态验证（截图匹配动作成功）。

### Capstone 构建

Capstone 任务：构建计算机使用 agent：

1. 读取预订网站模拟页面的 HTML + 截图。
2. 规划多步序列：搜索 → 选择 → 填写表单 → 提交。
3. 发出匹配动作 schema 的 JSON 动作。
4. 在固定的 10 任务片上评估。

课程提供脚手架代码，易于扩展到真实浏览器。

## 使用方法

`code/main.py` 是 capstone 脚手架：

- 动作 schema JSON 定义（10 个动作）。
- 模拟浏览器状态作为字典。
- Agent 循环骨架：接收状态，发出动作，应用，循环。
- 10 任务迷你基准（合成页面）用于衡量端到端成功率。
- 动作失败时的错误恢复钩子。

## 输出作品

本节生成 `outputs/skill-multimodal-agent-designer.md`。给定计算机使用产品（领域、动作集、评估目标），设计完整 agent 循环、记忆策略、接地模式和预期基准分数。

## 练习

1. 用 `screenshot_region` 工具（裁剪 + 放大）扩展动作 schema。什么任务受益？

2. 阅读 AgentVista（arXiv:2602.23166）。描述最难的任务类别，为什么前沿模型仍然失败。

3. 长期记忆压缩：设计一个摘要链，最多保留 4 张实时截图，任意数量记录日志。

4. 构建错误恢复钩子：当动作失败（按钮未找到）时，agent 下一步做什么？

5. 在 10 个网页任务上比较仅截图 Claude 4.7 与混合截图 + 可访问性树 Qwen2.5-VL。哪个在什么任务上赢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| GUI grounding | "点击坐标" | 模型在截图上输出指令目标的 (x,y) |
| Action schema | "工具定义" | 有效动作（click、type、scroll、drag）的 JSON 描述 |
| Accessibility tree | "结构化 DOM" | 来自浏览器/iOS API 的机器可读 UI 层级 |
| Hybrid agent | "截图 + 树" | 同时使用图像和结构化信息；比任一单独更可靠 |
| Visual tool use | "放大/裁剪/检测" | Agent 在计划中途调用外部视觉工具（OCR、检测） |
| Summary-chain | "记忆压缩" | 定期文本摘要替换长截图历史 |
| VisualWebArena | "E2E 网页基准" | 2024 年端到端网页任务基准 |
| AgentVista | "2026 年困难基准" | 12 个领域现实工作流；即使 Gemini 3 Pro 得分约 30% |

## 延伸阅读

- [Cheng 等 — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong 等 — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You 等 — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh 等 — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)