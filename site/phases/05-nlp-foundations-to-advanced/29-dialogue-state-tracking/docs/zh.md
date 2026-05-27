# 对话状态追踪

> "我要找一家便宜的餐厅，在北部……其实改成中档吧……再加一个意大利菜。"三轮对话，三次状态更新。DST 保持 slot-value 字典同步，确保预订不会出错。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 5 · 17（聊天机器人）、Phase 5 · 20（结构化输出）
**时长：** 约 75 分钟

## 问题

在任务导向对话系统中，用户的目标编码为一组 slot-value 对：`{cuisine: italian, area: north, price: moderate}`。每轮用户输入都可能添加、更改或删除一个 slot。系统必须读取整个对话并正确输出当前状态。

一个 slot 填错，系统就会订错餐厅、排错航班或刷错卡。DST 是用户所说的内容和后端执行之间的枢纽。

2026 年尽管有 LLM，DST 仍然重要的原因：

- 合规敏感领域（银行、医疗、机票预订）需要确定性的 slot 值，而不是自由形式生成。
- 工具调用 agent 在调用 API 前仍然需要 slot 解析。
- 多轮纠错比看起来更难："其实不要了，改成周四。"

现代管道：经典 DST 概念 + LLM 抽取器 + 结构化输出护栏。

## 概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** schema 定义域（餐厅、酒店、出租车）及其 slot（菜系、区域、价格、用餐人数）。每个 slot 可以为空、用封闭集合中的值填充（price: {cheap, moderate, expensive}），或自由形式值（name: "The Copper Kettle"）。

**两种 DST 公式。**

- **分类。** 对每个 (slot, candidate_value) 对，预测是/否。适用于封闭词表 slot。2020 年前的标准。
- **生成。** 给定对话，自由文本生成 slot 值。适用于开放词表 slot。当代默认。

**指标。** 联合目标准确率（Joint Goal Accuracy，JGA）——每一轮所有 slot 都正确的比例。全有或全无。MultiWOZ 2.4 排行榜在 2026 年约达 83% 顶级水平。

**架构。**

1. **基于规则（slot regex + 关键词）。** 窄领域的强基线。可调试。
2. **TripPy / BERT-DST。** BERT 编码的基于复制的生成。LLM 前标准。
3. **LDST（LLaMA + LoRA）。** 带域-slot 提示的指令微调 LLM。在 MultiWOZ 2.4 上达到 ChatGPT 级质量。
4. **无本体论（2024–26）。** 跳过 schema；直接生成 slot 名称和值。处理开放域。
5. **提示 + 结构化输出（2024–26）。** 带 Pydantic schema + 约束解码的 LLM。5 行代码，生产可用。

### 经典失败模式

- **跨轮共指。** "保留第一个选项。"需要确定是哪个选项。
- **覆盖 vs 追加。** 用户说"加一个意大利菜"。是替换 cuisine 还是追加？
- **隐式确认。** "好的"——是接受还是拒绝了预订提议？
- **纠错。** "其实改成 7 点。"必须更新 time 而不清理其他 slot。
- **指向前一个系统话语的共指。** "是的，那个。"哪个"那个"？

## 构建

### 步骤 1：基于规则的 slot 抽取器

参见 `code/main.py`。正则表达式 + 同义词字典覆盖窄领域 70% 的规范表述：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

超出规范词汇表就脆弱。适用于确定性 slot 确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三条不变式：

- 永不重置用户没有触碰的 slot。
- 显式否定（"不要 cuisine 了"）必须清理。
- 用户纠错（"其实……"）必须覆盖，不追加。

### 步骤 3：LLM 驱动的 DST + 结构化输出

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证有效状态对象。无正则、无 schema 不匹配、无幻觉 slot。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统把所有 slot 都做对的轮次占多少比例？MultiWOZ 2.4，2026 年顶级系统：80–83%。你的域内系统在窄词汇上应该超过这个，否则 LLM 基线就超过了你。

### 步骤 5：处理纠错

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到纠错时，覆盖最后更新的 slot 而不是追加。没有 LLM 帮助很难做对。现代模式：始终让 LLM 从历史中重新生成整个状态，而不是增量更新——这自然地处理了纠错。

## 陷阱

- **全历史重新生成成本。** LLM 每轮重新生成状态，总 token 消耗为 O(n²)。限制历史长度或汇总旧轮次。
- **Schema 漂移。** 事后添加新 slot 破坏旧训练数据。对 schema 打版本。
- **大小写敏感性。** "Italian" vs "italian" vs "ITALIAN"——处处规范化。
- **隐式继承。** 如果用户之前指定了"4 人用餐"，新时间请求不应清理人数。始终传递完整历史。
- **自由形式 vs 封闭集合。** 名称、时间和地址需要自由形式 slot；菜系和区域是封闭集合。在 schema 中混合使用。

## 使用

2026 年技术栈：

| 场景 | 方法 |
|------|------|
| 窄领域（一个或两个 intent） | 基于规则 + regex |
| 宽领域，有标注数据 | LDST（LLaMA + LoRA on MultiWOZ 风格数据） |
| 宽领域，无标注，生产可用 | LLM + Instructor + Pydantic schema |
| 语音 / 口语 | ASR + 规范化器 + LLM-DST |
| 多域预订流程 | 带每域 Pydantic 模型的 Schema-guided LLM |
| 合规敏感 | 基于规则为主，LLM 为辅 + 确认流程 |

## 上线

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: 设计对话状态追踪器——schema、抽取器、更新策略、评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

给定用例（领域、语言、词汇开放度、合规需求），输出：

1. Schema。域列表，每域 slot，每个 slot 开放 vs 封闭词汇表。
2. 抽取器。基于规则 / seq2seq / LLM-with-Pydantic。理由。
3. 更新策略。重新生成整个状态 / 增量；纠错处理；否定处理。
4. 评估。在留出对话集上的 JGA、slot 级精确率/召回率、最难 slot 的混淆矩阵。
5. 确认流程。何时显式要求用户确认（破坏性操作、低置信度抽取）。

合规敏感 slot 拒绝纯 LLM DST，必须有基于规则的二次检查。拒绝无法在用户纠错时回滚 slot 的任何 DST。标记没有版本标签的 schema。
```

## 练习

1. **简单。** 在 `code/main.py` 中构建基于规则的状态追踪器，3 个 slot（cuisine、area、price）。在 10 个人工构造的对话上测试。测量 JGA。
2. **中等。** 在同一数据集上使用 Instructor + Pydantic + 小型 LLM。比较 JGA。检查最难的几轮。
3. **困难。** 两者都实现并路由：基于规则为主，置信度 <2 slot 时 LLM 兜底。测量组合 JGA 和每轮推理成本。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|---------|
| DST | 对话状态追踪 | 在对话轮次之间维护 slot-value 字典。 |
| Slot | 用户意图单元 | 后端需要的命名参数（cuisine、date）。 |
| Domain | 任务区域 | 餐厅、酒店、出租车——slot 的集合。 |
| JGA | 联合目标准确率 | 每一轮所有 slot 都正确的比例。全有或全无。 |
| MultiWOZ | 基准测试 | 多域 WOZ 数据集；标准 DST 评估。 |
| Ontology-free DST | 无 schema | 直接生成 slot 名称和值，无固定列表。 |
| Correction | "其实……" | 覆盖之前已填充 slot 的轮次。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 标准基准测试。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 指令微调用于 DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的 DST 主力。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM 无监督 TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 标准 DST 结果。