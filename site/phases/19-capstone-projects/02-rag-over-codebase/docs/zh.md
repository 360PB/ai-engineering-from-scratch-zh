# 顶点课 02 — 代码库 RAG（跨仓库语义搜索）

> 到 2026 年，每个正经工程组织都在运行一个理解语义的内部代码搜索——不只是字符串匹配。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业图谱、Aider 的 repomap、Pinterest 的内部 MCP，形态如出一辙。 ingesting 多个仓库，用 tree-sitter 解析，按函数/类级别切块，混合搜索，重排序，带引用回答。本顶点课要求你构建一个能处理 2M 行代码（跨 10 个仓库）并在每次 git push 时增量重建索引的系统。

**类型：** 顶点课
**语言：** Python（ingestion），TypeScript（API + UI）
**前置要求：** Phase 5（NLP 基础）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 17（基础设施）
**涉及的 Phase：** P5 · P7 · P11 · P13 · P17
**时间：** 30 小时

## 问题

到 2026 年，每个前沿编码 Agent 都内置了代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude 的 1M-token 上下文有帮助，但不能消除排序检索的需求。在生成代码上、单体仓库重复上、很少被导入的符号长尾上，naive 余弦相似度搜索会破坏结果。生产级答案是：AST 感知的块（dense + BM25）混合搜索 + 重排序器，背后是符号引用图。

你通过索引真实舰队来学习——不是一个教程仓库——并测量 MRR@10、引用准确性和增量新鲜度。失败模式是基础设施层面的：100k 文件的单体仓库、涉及半数文件的推送、需要跨四个仓库才能正确回答的查询。

## 概念

AST 感知的 ingestion 管道用 tree-sitter 解析每个文件，提取函数和类节点，按节点边界切块（而非固定 token 窗口）。每个块获得三个表示：dense 向量（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项、短自然语言摘要。摘要增加了第三个可检索模态——用户问"X 是如何授权的"，摘要提到"authz"，即使代码里只有 `check_permission`。

检索是混合的。查询同时触发 dense 和 BM25 搜索，合并 top-k，把并集交给 cross-encoder 重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表送入长上下文综合器（Claude Sonnet 4.7 搭配 prompt caching，或 Llama 3.3 70B 自托管），附带指令要求每个声明都按文件和行范围标注引用。无引用的答案被后置过滤器拒绝。

增量新鲜度是基础设施问题。Git push 触发 diff：哪些文件变了，哪些符号变了。只重新 embedding 受影响的块。受影响的跨文件符号边（import、方法调用）重新计算。索引在每次提交时不重处理 2M 行代码，保持一致。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter 搭配 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- Dense embedding：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 备选
- 稀疏索引：Tantivy（Rust）搭配 BM25F，字段加权（符号名 vs 正文权重不同）
- 向量 DB：Qdrant 1.12 搭配混合搜索；或 pgvector + pgvectorscale（团队规模 < 50M 向量时）
- 块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，prompt cached
- 重排序器：Cohere rerank-3 或 bge-reranker-v2-gemma-2b 自托管
- 编排：LlamaIndex Workflows（ingestion）、LangGraph（查询 Agent）
- 综合器：Claude Sonnet 4.7（1M 上下文）搭配 prompt caching
- 符号图：Neo4j（托管）或 kuzu（嵌入式），用于 import 和 call 边
- 可观测：Langfuse spans，每步检索和综合都有 span

## 构建步骤

1. **Ingestion 爬虫。** 在每个 push 钩子上迭代 git 历史。收集变更的文件。对每个文件用 tree-sitter 解析，提取函数和类节点及其完整源码跨度。发出块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **块摘要器。** 将块批量送入 Haiku 4.5 调用（系统前缀 prompt cached）。Prompt："用一句话总结此函数，命名其公开契约和副作用。"摘要与块一起存储。

3. **Embedding 池。** 两个并行队列：dense（Voyage-code-3 批量 128）和摘要（同一模型，但在摘要字符串上）。向量写入 Qdrant，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4，符号正文权重 1，摘要权重 2。实现"按名称找函数"和"按功能找函数"两类查询。

5. **符号图。** 对每个块记录边：import（此文件使用来自 repo Z 的符号 Y）、call（此函数调用类 C 上的方法 M）、继承。存储在 kuzu 中。查询时用于跨仓库边界扩展检索。

6. **查询 Agent。** LangGraph 含三个节点。`retrieve` 并行触发 dense 和 BM25，按 (repo, path, symbol) 去重。`rerank` 用 cross-encoder 对 top-50 打分，保留 top-10。`synth` 用重排序后的块作为上下文调用 Claude Sonnet 4.7，缓存系统 prompt，要求 file:line 引用。

7. **引用强制。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明都被标记为重新提问或丢弃。只向用户返回有引用的答案。

8. **增量重建索引。** 每次 webhook，计算符号级 diff。只重新 embedding 文本变更的块。重新计算 import 变更的块的符号边。测量：50 文件的推送在 2M-LOC 舰队上 60 秒内完成重建。

9. **评测。** 用金标准 file:line 答案标注 100 道跨仓库问题。测量 MRR@10、nDCG@10、引用准确性（可验证锚点声明比例）、p50/p99 延迟。

## 使用示例

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付标准

交付物 `outputs/skill-codebase-rag.md`。给定一个仓库语料，它启动 ingestion 管道、混合索引和查询 Agent，为任何跨仓库问题返回带引用的答案。评分标准如下：

| 权重 | 指标 | 测量方式 |
|:-:|---|---|
| 25 | 检索质量 | 100 题 holdout 上的 MRR@10 和 nDCG@10 |
| 20 | 引用准确性 | 答案声明中具有可验证 file:line 锚点的比例 |
| 20 | 延迟和规模 | 索引规模下 10k QPS 的 p95 查询延迟 |
| 20 | 增量索引正确性 | 从 git push 到可搜索的时间（50 文件提交）|
| 15 | UX 和答案格式 | 引用可点击性、片段预览、后续交互便利性 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 换成 nomic-embed-code 自托管。测量 MRR@10 差值。报告开启重排序后差距是否缩小。

2. 向语料中注入 20% 生成代码（LLM 生成的样板），重新评测。观察检索污染。向 payload 添加"generated"标志并降低这类命中的权重。

3. 在你的语料规模上对比 Qdrant 混合搜索与 pgvector + pgvectorscale。报告批量大小为 1 时的 p99。

4. 添加基于采样的漂移检测：每周重新运行 100 题评测。MRR@10 下降 > 5% 时告警。

5. 扩展到跨语言符号解析：Python 函数通过 gRPC 调用 Go 服务。用符号图链接两者。

## 关键术语

| 术语 | 别人怎么称呼 | 实际含义 |
|------|-----------------|------------------------|
| AST-aware chunking | "函数级切分" | 按 tree-sitter 节点边界切代码，而非固定 token 窗口 |
| Hybrid search | "Dense + sparse" | BM25 和向量搜索并行执行，合并 top-k，重排序 |
| Cross-encoder rerank | "二阶段排序" | 模型对每个 (query, candidate) 对一起评分，比余弦更准 |
| Prompt caching | "缓存系统 prompt" | 2026 Claude/OpenAI 功能，对重复前缀 Token 折扣最高达 90% |
| Symbol graph | "代码图" | 跨文件和跨仓库的 import、call、继承边 |
| Citation faithfulness | "grounded answer rate" | 用户可以通过点击锚点并阅读引用跨度来验证的声明比例 |
| Incremental re-index | "push-to-search time" | 从 git push 到变更的符号可查询的墙上时钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本顶点课的参考深度解析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排序的仓库视图
- [Augment Code 企业图谱](https://www.augmentcode.com) — 商业符号图 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码 embedding](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder 参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考