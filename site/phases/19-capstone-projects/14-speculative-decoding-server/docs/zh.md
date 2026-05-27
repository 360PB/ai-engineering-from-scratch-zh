# Capstone 14 — 投机解码推理服务器

> vLLM 0.7 中的 EAGLE-3 在真实流量上达到 2.5-3 倍吞吐。P-EAGLE（AWS 2026）将并行推测推得更远。SGLang 的 SpecForge 大规模训练草案头。Red Hat 的 Speculators hub 发布了针对常见开源模型的对齐草案。TensorRT-LLM 在 NVIDIA 上将投机解码设为第一等。2026年生产服务技术栈是带 EAGLE 系列草案的 vLLM 或 SGLang、FP8 或 INT4 量化、队列等待的 HPA。毕业项目是用 2.5 倍+ 基线吞吐服务两个开源模型，并附完整的尾延迟报告。

**类型：** 毕业项目
**语言：** Python（服务），C++ / CUDA（内核检查），YAML（配置）
**前置知识：** Phase 3（深度学习）、Phase 7（Transformers）、Phase 10（从零构建LLM）、Phase 17（基础设施）
**涉及阶段：** P3 · P7 · P10 · P17
**时长：** 30小时

## 问题

投机解码在2026年成为商品。EAGLE-3 草案头在目标模型隐藏状态上训练，预测前方 N 个 token；目标模型一次通过验证。60-80% 的接受率转化为 2-3 倍端到端吞吐。vLLM 0.7 原生集成。SGLang + SpecForge 给你训练流水线。Red Hat 的 Speculators 发布 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的对齐草案。

功力在服务运营，不在模型。接受率随流量分布漂移（ShareGPT vs 代码 vs 领域数据）。拒绝时的尾延迟比不用推测时更差——你必须报告多个 batch size 下的 p99，而不只是稳态 tokens/s。$/1M tokens 对比 Anthropic / OpenAI API 是可信度杠杆。

## 核心概念

投机解码有两层。**草案**模型（EAGLE-3 头、ngram 或更小的目标对齐模型）每步提出 k 个候选 token。**目标**模型一次通过验证所有 k 个；任何被接受的前缀替换贪婪路径。接受率取决于草案-目标对齐度和输入分布。

EAGLE-3 在大多数流量上击败 ngram 草案。P-EAGLE 运行并行推测以获得更深的草案树。权衡：拒绝时 P99 延迟更高，因为验证通过更大。服务配置必须报告 batch-size 分桶延迟以揭露这一点。

部署在 Kubernetes 上。vLLM 0.7 每 GPU 或张量并行分片运行一个副本。HPA 跟踪队列等待而非 CPU 扩展。FP8（Marlin）和 INT4（AWQ）量化将 GPU 内存控制在 H100 / H200 范围内。端到端报告：吞吐、接受率、batch 1/8/32 的 p50/p99、$/1M tokens。

## 架构

```
请求入口
    |
    v
vLLM 服务器（0.7）或 SGLang（0.4）
    |
    +-- 草案：EAGLE-3 头 | P-EAGLE 并行 | ngram 后备
    +-- 目标：Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     量化 FP8-Marlin 或 INT4-AWQ
    |
    v
验证通过：批量 k 个草案 token 通过目标
    |
    v（接受前缀；拒绝后缀重新采样）
    v
token 流回客户端
    |
    v
Prometheus 指标：吞吐、接受率、队列等待、延迟 p50/p99
    |
    v
HPA 跟踪队列等待指标
```

## 技术栈

- 服务：vLLM 0.7 或 SGLang 0.4
- 投机方法：EAGLE-3 草案头、P-EAGLE 并行投机、ngram 后备
- 草案训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA device plugin；队列等待指标上 HPA
- 评估：ShareGPT、MT-Bench-v2、GSM8K、HumanEval 用于领域分布接受率测量
- 参考：TensorRT-LLM 投机解码作为供应商基线

## 动手实现

1. **目标模型准备。** 选 Llama 3.3 70B。通过 Marlin 量化为 FP8。在 1xH100 上（或 2x 张量并行）通过 vLLM 0.7 部署。

2. **草案来源。** 从 Red Hat Speculators 拉取对齐的 EAGLE-3 草案头（或通过 SpecForge 训练一个）。加载到 vLLM 的投机解码配置。

3. **基线数字。** 推测前：batch 1/8/32 下 tokens/s、p50/p99 延迟、GPU 利用率。发布。

4. **启用 EAGLE-3。** 切换配置；重跑相同基准。报告加速比、接受率、p99 尾延迟差值。

5. **P-EAGLE。** 启用并行投机；测量更深的草案树 vs 串行 EAGLE-3。报告 P-EAGLE 帮助 vs 损害的拐点。

6. **领域流量。** 用 ShareGPT vs HumanEval vs 领域特定流量在相同服务器上运行。测量每分布的接受率。识别草案漂移的时机。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同流水线。草案更棘手（MoE 路由噪声）。报告。

8. **K8s HPA。** 在 K8s 下部署，HPA 跟踪 `queue_wait_ms`。演示负载 tripled 时自动扩展。

9. **成本对比。** 在相同评估上计算 $/1M tokens vs Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4。发布。

## 用现成库

```bash
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[服务]     vLLM 0.7，Llama 3.3 70B FP8，EAGLE-3 活跃
[解码]    bs=8，accepted_tokens_per_step=3.2，acceptance_rate=0.76
[延迟]    首个 token 42ms，完整响应 980ms（620 tokens）
[成本]    持续吞吐下 $0.34 / 1M 输出 token
```

## 产出

`outputs/skill-inference-server.md` 描述交付物。带投机解码的测量服务栈，完整基准报告，和 K8s 部署。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | vs 基线可衡量加速 | 两个模型上匹配质量的 2.5x+ 吞吐 |
| 20 | 真实流量接受率 | 每分布接受率报告 |
| 20 | P99 尾延迟纪律 | batch 1/8/32 有/无推测的 p99 |
| 20 | 运维 | K8s 部署、队列等待上 HPA、滚动发布顺畅 |
| 15 | 报告和方法论 | 清晰解释什么变了及原因 |
| **100** | | |

## 练习

1. 测量草案落后目标一个版本时的接受率退化（如 Llama 3.3 -> 3.4 漂移）。构建监控告警。

2. 实现 ngram 后备：如果 EAGLE-3 接受率低于阈值，切换到 ngram 草案。报告可靠性提升。

3. 运行控制 MoE 实验：相同 Qwen3-Coder-30B，有/无路由噪声注入。测量草案接受率敏感性。

4. 扩展到 H200（141 GB）。报告每副本模型大小余量增加，以及是否可以服务未量化的 Llama 3.3 70B。

5. 在同一 H100 硬件上基准 TensorRT-LLM 投机解码。报告它在哪胜 vLLM。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Draft model | "推测器" | 提出 N 个 token 供目标验证的小模型 |
| EAGLE-3 | "2026 草案架构" | 在目标隐藏状态上训练的草案头；约 75% 接受率 |
| P-EAGLE | "并行推测" | 在一次目标通过中验证的草案分支树 |
| Acceptance rate | "命中率" | 无需重新采样的草案 token 比例 |
| Quantization | "FP8 / INT4" | 低精度权重以在 GPU 内存中装下更大模型 |
| Queue wait | "HPA 指标" | 请求在推理开始前在待处理队列中等待的时间 |
| Speculators hub | "对齐草案" | 常见开源模型 EAGLE 草案的 Red Hat Neural Magic hub |

## 扩展阅读

- [vLLM EAGLE 和 P-EAGLE 文档](https://docs.vllm.ai) — 参考服务栈
- [P-EAGLE（AWS 2026）](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行投机解码论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草案头训练流水线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草案 hub
- [TensorRT-LLM 投机解码](https://nvidia.github.io/TensorRT-LLM/) — 供应商替代
- [Fireworks.ai 服务架构](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 论文（arXiv:2503.01840）](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM 仓库](https://github.com/vllm-project/vllm) — 代码和基准