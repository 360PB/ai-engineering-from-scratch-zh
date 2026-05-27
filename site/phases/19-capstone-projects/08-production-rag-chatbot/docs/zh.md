# Capstone 08 — 垂类监管领域生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在2026年都跑着相同的生产形态。用 docling 或 Unstructured 摄取，ColPali 处理视觉内容。混合检索。用 bge-reranker-v2-gemma 重排。用 Claude Sonnet 4.7 合成，prompt caching 命中率达 60-80%。Llama Guard 4 和 NeMo Guardrails 做防护。用 Langfuse 和 Phoenix 监控。用 RAGAS 在200题黄金集上评分。在一个受监管领域（法律、临床、保险）构建一个——毕业项目是让机器人通过黄金集、红队和漂移仪表盘。

**类型：** 毕业项目
**语言：** Python（流水线 + API），TypeScript（聊天 UI）
**前置知识：** Phase 5（NLP）、Phase 7（Transformers）、Phase 11（LLM工程）、Phase 12（多模态）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18
**时长：** 30小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险单）是2026年最广泛落地的生产形态，因为 ROI 明确、风险具体。Harvey（Allen & Overy）为法律领域构建了它。Mendable 发版开发者文档版。Glean 覆盖企业搜索。模式是：高保真摄取、混合检索加重排、带引用强制和 prompt caching 的合成、多层安全防护、持续漂移监控。

难点不在模型。在于司法管辖感知合规（HIPAA、GDPR、SOC2）、引用级审计能力、成本控制（prompt caching 在高命中时享 60-90% 折扣）、通过 RAGAS 幻觉检测，以及源文档更新但索引未跟上的漂移检测。毕业项目要求你在200题黄金集和红队套件旁边把全部这些交付。

## 核心概念

流水线分两边。**摄取**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的内容；chunk 带摘要、标签和基于角色的访问标签。向量进 pgvector + pgvectorscale（5000万向量以内）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 处理记忆和多轮；每个查询运行混合检索，用 bge-reranker-v2-gemma-2b 重排，用 Claude Sonnet 4.7（prompt 缓存）合成，输出过 Llama Guard 4 和 NeMo Guardrails，发回带引用的回复。

评估栈有四层。**黄金集**（200条带引用的标注 Q/A）用于正确性。**红队**（越狱、PII 提取尝试、离域问题）用于安全。**RAGAS** 用于faithfulness / answer relevance / context precision 每轮自动评估。**漂移仪表盘**（Arize Phoenix）每周监控检索质量和幻觉分数。

Prompt caching 是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示 + 检索上下文。在 60-80% 命中时，每查询成本降低 3-5 倍。流水线必须为稳定前缀（系统提示 + 重排后的上下文优先）设计，以达到高缓存命中率。

## 架构

```
文档（合同、方案、政策）
      |
      v
docling / Unstructured 解析 + ColPali 处理视觉
      |
      v
chunk + 摘要 + 角色标签 + 司法管辖标签
      |
      v
pgvector + pgvectorscale  +  BM25（Tantivy）
      |
查询 + 角色 + 司法管辖
      |
      v
LangGraph 对话 Agent
   +--- 检索（混合）
   +--- 按角色 + 司法管辖过滤
   +--- 重排（bge-reranker-v2-gemma-2b 或 Voyage rerank-2）
   +--- 合成（Claude Sonnet 4.7，prompt 缓存）
   +--- 防护（Llama Guard 4 + NeMo Guardrails + Presidio 输出 PII 清洗）
   +--- 引用 + 返回
      |
      v
评估:
  RAGAS faithfulness / answer_relevance / context_precision（在线）
  Langfuse 标注队列（抽样）
  Arize Phoenix 漂移（每周）
  红队套件（发布前）
```

## 技术栈

- 摄取：Unstructured.io 或 docling 解析结构化文档；ColPali 处理视觉丰富 PDF
- 向量库：5000万向量以内用 pgvector + pgvectorscale；否则用 Qdrant Cloud
- 稀疏：Tantivy BM25，带字段权重
- 编排：LlamaIndex Workflows（摄取）+ LangGraph（对话）
- 重排器：bge-reranker-v2-gemma-2b 自托管 或 Voyage rerank-2 托管
- LLM：Claude Sonnet 4.7 带 prompt caching；备选 Llama 3.3 70B 自托管
- 评估：RAGAS 0.2 在线，DeepEval 用于幻觉和越狱套件
- 可观测性：Langfuse 自托管带标注队列；Arize Phoenix 用于漂移
- 防护栏：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 清洗
- 合规：chunk 上的基于角色访问标签；GDPR/HIPAA 司法管辖标签

## 动手实现

1. **摄取。** 用 Unstructured 或 docling 解析语料库（严肃构建用 1000-10000 份文档）。对扫描/视觉密集页面走 ColPali。产出带摘要、角色标签、司法管辖标签的 chunk。

2. **索引。** 密集嵌入（Voyage-3 或 Nomic-embed-v2）到 pgvector + pgvectorscale。Tantivy 的 BM25 侧索引。角色和司法管辖过滤器作为 payload。

3. **混合检索。** 先按角色+司法管辖过滤；然后并行密集 + BM25；RRF 合并；top-20 进重排；top-5 进合成。

4. **带 prompt caching 的合成。** 系统提示 + 静态策略放在缓存头部；重排后的上下文作为缓存扩展；用户问题作为未缓存后缀。目标是稳态下 60-80% 缓存命中率。

5. **防护栏。** Llama Guard 4 处理输入；NeMo Guardrails 轨道阻止离域问题或策略禁止话题；Presidio 清洗输出中的意外 PII；引用强制后过滤器。

6. **黄金集。** 200个 Q/A 对，由领域专家标注（答案、引用的来源）。在精确引用匹配、答案正确性、faithfulness（RAGAS）上对 Agent 评分。

7. **红队。** 50个对抗提示：越狱（PAIR、TAP）、PII 泄露尝试、离域、跨司法管辖泄露。按通过/失败和严重程度评分。

8. **漂移仪表盘。** Arize Phoenix 每周跟踪检索质量（nDCG、引用 faithfulness）。下降 5% 触发告警。

9. **成本报告。** Langfuse：prompt-cache 命中率、每查询 token 数、各阶段 $/query 明细。

## 用现成库

```bash
$ chat --role=analyst --jurisdiction=GDPR
> 在我们的合同下，EU 用户档案的数据保留义务是什么？
[检索]    混合 top-20，过滤到 GDPR + analyst-role
[重排]    保留 top-5
[合成]    claude-sonnet-4.7，缓存命中 74%，0.8s
答案:
  合同（2024-03-11 主服务协议第12.4条）
  要求在 GDPR 第17条规定的终止后30天内删除 EU 用户档案。
  DPA 修正案（DPA-v2.1，第5条）将"受限"类别数据的删除期限延长至14天。
  引用: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 产出

`outputs/skill-production-rag.md` 描述交付物。带合规标签的垂类监管聊天机器人，通过评分标准，有实时漂移监控。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | 黄金集（200 Q/A）在线得分 |
| 20 | 引用正确性 | 可验证源锚点的答案比例 |
| 20 | 防护栏覆盖率 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | Prompt-cache 命中率、p95 延迟、$/query |
| 15 | 漂移监控仪表盘 | Phoenix 实时仪表盘，含每周检索质量趋势 |
| **100** | | |

## 练习

1. 在不同司法管辖下构建第二个语料库切片（如 HIPAA 与 GDPR 并行）。用20题跨司法管辖探测演示角色+司法管辖过滤防止跨泄露。

2. 测量一周生产流量的 prompt-cache 命中率。识别哪些查询破坏了缓存前缀。重构。

3. 添加带 10k-token 摘要缓冲区的多轮记忆。测量随对话增长 faithfulness 是否下降。

4. 把 Claude Sonnet 4.7 换成 Llama 3.3 70B 自托管。测量 $/query 和 faithfulness 差值。

5. 添加"不确定"模式：如果 top 重排分数低于阈值，Agent 说"我没有确信的引用"而非回答。测量降低虚假置信度的效果。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Prompt caching | "缓存系统 + 上下文" | Claude/OpenAI 特性：命中的缓存前缀 token 折扣 60-90% |
| RAGAS | "RAG 评估器" | faithfulness、answer relevance、context precision 自动评分 |
| Golden set | "标注评估" | 200+ 专家标注 Q/A 带引用；基准真值 |
| Jurisdiction tag | "合规标签" | 附加到 chunk 的 GDPR/HIPAA/SOC2 范围；检索过滤器强制执行 |
| Citation faithfulness | "有据答案率" | 有可检索源跨度支持的声明比例 |
| Drift | "检索质量衰减" | nDCG 或引用分数的每周变化；告警阈值 5% |
| Red team | "对抗评估" | 发布前越狱、PII 提取、离域探测 |

## 扩展阅读

- [Harvey AI](https://www.harvey.ai) — 法律生产栈参考
- [Glean 企业搜索](https://www.glean.com) — 企业规模 RAG 参考
- [Mendable 文档](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄取
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 文档](https://docs.ragas.io/) — 权威 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移可观测性参考
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略轨道框架