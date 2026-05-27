# Capstone 17 — 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 在2026年都以规模交付了自适应多模态辅导。共同形态是苏格拉底策略（从不直接给出答案）、每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音 + 文本 + 拍照数学输入、课程图谱检索、间隔重复调度，以及适龄内容的硬安全过滤器。毕业项目是交付一个学科特定导师（K-12代数或入门Python），在10名学习者上运行两周功效研究，并通过内容安全审计。

**类型：** 毕业项目
**语言：** Python（后端，学习者模型），TypeScript（Web 应用），SQL（通过 Postgres + Neo4j 做课程图谱）
**前置知识：** Phase 5（NLP）、Phase 6（语音）、Phase 11（LLM工程）、Phase 12（多模态）、Phase 14（Agent工程）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时长：** 30小时

## 问题

自适应辅导曾是教育科技研究的一个细分领域。到2026年它成了消费品。Khanmigo 部署到大多数美国学区。Duolingo Max 达到数千万月活。Google 的 LearnLM / Gemini for Education 为 Google Classroom 中的辅导提供支持。Quizlet Q-Chat 放在抽认卡旁边。Synthesis Tutor 因面向好奇孩子的导师而走红。共同要素：多模态输入（打字、说话、拍照方程）、苏格拉底教学法（先问后解释）、每次交互后更新的学习者模型，以及严格的适龄安全。

你将为特定人群构建其中一个。测量标准是真正的功效研究：两周内10名学习者的前测和后测分数。语音循环必须感觉自然（capstone 03 子栈）。记忆必须尊重隐私。安全过滤器必须通过 K-12 的 COPPA 感知红队。

## 核心概念

四个组件。**导师策略**是苏格拉底循环：当学习者要答案时，策略问一个引导性问题；当他们答对时，进展到下一概念；当他们卡住时，提供脚手架式提示。**学习者模型**是贝叶斯知识追踪（或简单变体），每次交互后更新每个课程节点的掌握概率。**课程图谱**是带先修边的概念的 Neo4j；策略走查图谱来选择下一概念。**记忆**是情景 + 语义存储（agentmemory 风格），保存过去交互、错误和偏好。

UX 是多模态的。打字输入用于文本答案。语音输入通过 LiveKit + Whisper（复用 capstone 03）。拍照输入用于数学问题通过 dots.ocr 或 PaliGemma 2。语音输出通过 Cartesia Sonic-2。安全使用 Llama Guard 4 加适龄过滤器（阻止成人内容、暴力、自残）和 COPPA 感知记忆保留策略。

功效研究是交付物。10名学习者，前测和后测，两周。报告学习收益差值和置信区间。与非自适应基线（在无导师策略的情况下线性传递相同内容）对比。

## 架构

```
学习者设备
  |
  +-- 文本         -> Web 应用
  +-- 语音        -> LiveKit Agents（ASR + TTS）
  +-- 拍照数学   -> dots.ocr / PaliGemma 2
       |
       v
  导师策略（LangGraph）
       - 苏格拉底决策头
       - 下一概念选择器（课程图谱走查）
       - 提示脚手架
       - 掌握度更新
       |
       v
  学习者模型（BKT / 项目反应理论）
       - 每个概念的掌握概率
       - 间隔重复调度器（SM-2 或 FSRS）
       |
       v
  记忆（agentmemory 风格）
       - 情景的：每次交互
       - 语义的：已学错误、偏好
       - 保留策略：COPPA / GDPR 感知
       |
       v
  课程图谱（Neo4j）
       - 先修边
       - OER 内容附加
       |
       v
  安全：
    Llama Guard 4 + 适龄过滤器
    记忆访问受学习者 ID 范围保护
```

## 技术栈

- 学科选择：K-12代数或入门 Python（选一个深入）
- 导师策略：基于 Claude Sonnet 4.7 的 LangGraph（带 prompt caching）
- 学习者模型：贝叶斯知识追踪（经典）或 FSRS 用于间隔
- 课程图谱：概念 + 先修边 + OER 内容的 Neo4j
- 记忆：agentmemory 风格持久向量 + 情景 + 语义存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用 capstone 03 子栈）
- 拍照数学：dots.ocr 或 PaliGemma 2 用于方程识别
- 安全：Llama Guard 4 + 自定义适龄过滤器
- 评估：Bloom 级问题生成、前/后测工具、功效研究工具

## 动手实现

1. **课程图谱。** 构建 50-150 个概念节点的 Neo4j（如，从"数轴"到"二次公式"的 K-12 代数），带先修边。每个节点附加 OER 内容（Open Textbook、OpenStax）。

2. **学习者模型。** 用先验初始化贝叶斯知识追踪：猜、错漏、学习率。每次交互后更新每概念掌握度。按学习者持久化。

3. **导师策略。** LangGraph 含节点：`read_signal`（学习者答案正确 / 部分 / 卡住？）、`select_concept`（走查课程图谱，选择最高优先级概念）、`scaffold`（苏格拉底提示）、`update_mastery`。

4. **记忆。** 每次交互写入情景存储。错误和偏好晋升到语义记忆。COPPA 感知保留策略：1年后自动删除，家长可访问。

5. **语音路径。** LiveKit Agents worker 连接到导师策略。ASR 通过 Whisper-v3-turbo。TTS 通过 Cartesia Sonic-2。支持插话（复用 capstone 03 机制）。

6. **拍照数学路径。** 上传或拍摄图像；运行 dots.ocr 或 PaliGemma 2 识别方程；作为结构化输入给导师。

7. **安全。** 每个模型输出通过 Llama Guard 4 + 适龄过滤器（阻止自残、成人内容、暴力）。记忆访问按学习者 ID 范围；家长访问面用于删除。

8. **功效研究。** 10名学习者，前测（标准化30题基线），两周导师交互（每周3次），后测。与相同内容的10名学习者非自适应基线组对比。

9. **每周进度报告。** 按学习者，自动生成 PDF 摘要：已探索主题、掌握度轨迹、推荐下一步。

## 用现成库

```
学习者：我不懂为什么 3x + 6 = 12 意味着 x = 2
[信号]   卡住了
[概念]   '变量分离'（先修：加减法与等式）
[脚手架] "为了开始，你会从两边减去哪个数？"
学习者：6
[信号]   正确
[掌握度] 加减法与等式：0.62 -> 0.77
[概念]   继续'变量分离'
[脚手架] "好。那 3x / 3 等于什么？"
```

## 产出

`outputs/skill-ai-tutor.md` 是交付物。一个学科特定自适应导师，带多模态输入、学习者模型、记忆、安全，以及有测量功效。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | 学习收益差值 | 10名学习者两周研究的前/后测差值 |
| 20 | 苏格拉底保真度 | 抄本样本上的评分标准得分 |
| 20 | 多模态 UX | 语音 + 拍照 + 文本端到端一致性 |
| 20 | 安全 + 隐私态势 | Llama Guard 4 通过率 + COPPA 感知保留 |
| 15 | 课程广度和图谱质量 | 概念覆盖率 + 先修图谱一致性 |
| **100** | | |

## 练习

1. 有无自适应学习者模型运行功效研究（随机概念顺序）。报告差值。期待自适应胜出，但差值大小才是有趣的数字。

2. 添加多模态探测：同一概念问题以文本、语音和拍照方式传递。测量学习者是否在他们偏好的模态下收敛更快。

3. 构建家长仪表盘：已练习主题、掌握度轨迹、即将到来概念、安全事件（任何防护栏命中）。符合 COPPA。

4. 添加语言切换模式：导师接受西班牙语输入并用西班牙语教学。测量 X-Guard 覆盖率。

5. 压力测试记忆隐私：验证即使通过语音片段重新摄取攻击，学习者 A 也看不到学习者 B 的数据。记录尝试的访问并告警。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Socratic policy | "问，不要倾倒" | 导师问引导性问题而非给答案 |
| Bayesian knowledge tracing | "BKT" | 经典学习者模型方程，每概念掌握概率 |
| FSRS | "自由间隔重复调度器" | 2024年间隔重复调度器，比 SM-2 更好 |
| Curriculum graph | "概念 DAG" | 带先修边的概念 Neo4j |
| Episodic memory | "每次交互日志" | 为后续检索存储的每次交互 |
| Semantic memory | "已学模式存储" | 从情景晋升的压缩错误和偏好 |
| COPPA | "儿童隐私法" | 限制13岁以下儿童数据收集的美国法律 |

## 扩展阅读

- [Khanmigo（可汗学院）](https://www.khanmigo.ai) — 参考消费级 K-12 导师
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参考语言学习导师
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 备选参考
- [Synthesis Tutor](https://www.synthesis.com) — 创业公司参考
- [FSRS 算法](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器
- [贝叶斯知识追踪](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型经典
- [LiveKit Agents](https://github.com/livekit/agents) — 语音栈