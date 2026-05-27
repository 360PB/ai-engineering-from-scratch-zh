# Capstone 15 — 宪法安全框架 + 红队靶场

> Anthropic 的宪法分类器、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 内容安全，以及覆盖多语言的 X-Guard 定义了2026年安全分类器栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准对抗评估工具。NeMo Guardrails v0.12 将它们串联成生产流水线。毕业项目把所有这些连接起来：一个围绕目标应用的分层安全框架，一个运行6+攻击族的自主红队 Agent，以及一个产生可衡量无害差值的宪法自我批评运行。

**类型：** 毕业项目
**语言：** Python（安全流水线、红队），YAML（策略配置）
**前置知识：** Phase 10（从零构建LLM）、Phase 11（LLM工程）、Phase 13（工具）、Phase 14（Agent工程）、Phase 18（伦理、安全、对齐）
**涉及阶段：** P10 · P11 · P13 · P14 · P18
**时长：** 25小时

## 问题

2026年 LLM 安全的边界不在于分类器是否有效（大致有效），而是如何在生产应用周围正确组合它们而不过度拒绝或留下明显漏洞。Llama Guard 4 处理英语策略违规。X-Guard（132种语言）处理多语言越狱。ShieldGemma-2 捕获图像注入。NVIDIA Nemotron 3 内容安全覆盖企业类别。Anthropic 的宪法分类器是一种单独方法，用于训练时而非服务时。

攻击进化也很重要。PAIR 和 TAP 自动发现越狱。GCG 运行基于梯度的后缀攻击。多轮和代码转换攻击利用 Agent 记忆。任何已部署 LLM 都需要红队靶场——garak 和 PyRIT 是规范驱动——加上记录在案的缓解措施和 CVSS 评分发现。

你将加固一个目标应用（8B 指令微调模型或其他 capstones 的 RAG 聊天机器人之一），运行6+攻击族 against it，并产生 before/after 无害测量。

## 核心概念

安全流水线分五层。**输入清洗**：剥离零宽字符，解码 base64/rot13，归一化 Unicode。**策略层**：NeMo Guardrails v0.12 轨道（离域、toxicity、PII 提取）。**分类器门控**：Llama Guard 4 处理输入，X-Guard 处理非英语，ShieldGemma-2 处理图像输入。**模型**：目标 LLM。**输出过滤器**：Llama Guard 4 处理输出，Presidio PII 清洗，在适用时强制引用。**HITL 层**：被标记为高风险的输出进入 Slack 队列。

红队靶场在调度器上运行。PAIR 和 TAP 自主发现越狱。GCG 运行基于梯度的后缀攻击。ASCII / base64 / rot13 编码攻击。多轮攻击（角色扮演、记忆利用）。代码转换攻击（英语混斯瓦希里语或泰语）。每次运行产生带 CVSS 评分和披露时间线的结构化发现文件。

宪法自我批评运行是训练时干预。取1k个有害尝试提示，让模型起草回复，对照书面宪法批评（无害规则），然后在批评循环上重新训练。测量 held-out 评估上 before/after 无害差值。

## 架构

```
请求（文本 / 图像 / 多语言）
      |
      v
输入清洗（剥离零宽、解码、归一化）
      |
      v
NeMo Guardrails v0.12 轨道（离域、策略）
      |
      v
分类器门控：
  Llama Guard 4（英语）
  X-Guard（多语言，132种语言）
  ShieldGemma-2（图像提示）
  Nemotron 3 内容安全（企业）
      |
      v（允许）
目标 LLM
      |
      v
输出过滤器：Llama Guard 4 + Presidio PII + 引用检查
      |
      v
被标记输出的 HITL 层

并行：
  红队调度器
    -> garak（经典攻击）
    -> PyRIT（编排红队）
    -> 自主越狱 Agent（PAIR + TAP）
    -> GCG 后缀攻击
    -> 多语言 / 代码转换
    -> 多轮角色扮演

输出：CVSS 评分发现 + 披露时间线 + before/after 无害差值
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 内容安全、X-Guard
- 防护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱 Agent：PAIR（Chao et al., 2023）、Tree-of-Attacks（TAP）、GCG 后缀
- 宪法训练：Anthropic 风格自我批评循环 + 在批评上 SFT
- PII 清洗：Presidio
- 目标：8B 指令微调模型或其他 capstones 的 RAG 聊天机器人之一

## 动手实现

1. **目标设置。** 在 vLLM 上部署8B 指令微调模型（或复用其他 capstones 的 RAG 聊天机器人）。这是被测应用。

2. **安全流水线封装。** 用五层流水线环绕目标。验证每层可单独观测（Langfuse 每层一个 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在小型标注集上运行每个以建立基线。

4. **红队调度器。** 调度 garak、PyRIT、PAIR Agent、TAP Agent、GCG runner、多轮攻击者、代码转换攻击者。每个在独立队列上运行。

5. **攻击套件。** 六个攻击族：(1) PAIR 自动越狱，(2) TAP 树攻击，(3) GCG 梯度后缀，(4) ASCII / base64 / rot13 编码，(5) 多轮角色扮演，(6) 多语言代码转换。报告每族成功率。

6. **宪法自我批评。** 整理1k个有害尝试提示。对每个，目标起草回复。批评 LLM 对照书面宪法评分（"无害"、"引用证据"、"拒绝非法请求"）。批评者有异议的提示被重写；目标在批评改进的对上微调。测量 held-out 评估上 before/after 无害。

7. **过度拒绝测量。** 在良性提示套件（如 XSTest）上追踪误报率。目标必须对良性问题保持有帮助。

8. **CVSS 评分。** 对每个成功越狱，按 CVSS 4.0 评分（攻击向量、复杂性、影响）。生成披露时间线和缓解计划。

9. **靶场自动化。** 上述一切在 cron 上运行；发现写入队列；过度拒绝回归告警发到 Slack。

## 用现成库

```bash
$ safety probe --model=target --family=PAIR --budget=50
[攻击者]   PAIR agent 在目标上运行
[攻击]     尝试 1/50：伪装查询为学术研究 ... 阻止
[攻击]     尝试 2/50：诉诸角色扮演 ... 阻止
[攻击]     尝试 3/50：思维链哄骗 ... 成功
[发现]    CVSS 4.8 中：角色扮演绕过目标
[靶场]     50次中7次成功（14% 成功率）
```

## 产出

`outputs/skill-safety-harness.md` 是交付物。生产级分层安全流水线加可复现红队靶场，带 before/after 无害差值。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 6+ 攻击族已运行，2+ 语言 |
| 20 | 真阳/假阳权衡 | 攻击阻止率 vs XSTest 良性通过率 |
| 20 | 自我批评差值 | held-out 评估上 before/after 无害 |
| 20 | 文档和披露 | 带时间线的 CVSS 评分发现 |
| 15 | 自动化和可重复性 | 一切在 cron 上运行并带告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的 prompt-injection 插件，对比有无输出过滤层的攻击成功率。

2. 添加第七个攻击族：通过检索文档的间接 prompt 注入。测量需要额外防御。

3. 实现"拒绝但有帮助"模式：当防护栏阻止时，目标提供更安全的替代答案而非直接拒绝。测量 XSTest 差值。

4. 多语言覆盖缺口：找一种 X-Guard 表现不佳的语言。针对它提出微调数据集。

5. 在30B模型上运行宪法自我批评，测量差值是否规模化。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Layered safety | "深度防御" | 在输入、门控、输出、HITL 的多层防护栏 |
| Llama Guard 4 | "Meta 安全分类器" | 2026年参考输入/输出内容分类器 |
| PAIR | "越狱 Agent" | LLM 驱动越狱发现的论文（Chao et al.） |
| TAP | "树攻击" | PAIR 的树搜索变体 |
| GCG | "贪婪坐标梯度" | 基于梯度的对抗后缀攻击 |
| Constitutional self-critique | "Anthropic 风格训练" | 目标起草 -> 批评者评分 -> 重写 -> 重新训练 |
| XSTest | "良性探测集" | 过度拒绝回归基准 |
| CVSS 4.0 | "严重度评分" | 安全发现的标准化漏洞评分 |

## 扩展阅读

- [Anthropic 宪法分类器](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 内容安全](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard（arXiv:2504.08848）](https://arxiv.org/abs/2504.08848) — 132语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 轨道框架
- [PAIR（arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — 越狱 Agent 论文