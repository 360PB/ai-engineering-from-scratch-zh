# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场已不再是 GPU 时间租赁。它分叉为定制硅片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先 marketplace（Replicate、DeepInfra）。Fireworks 于 2026 年 5 月 1 日将 GPU 租赁提价 1 美元/小时，40B+ 估值、日均 10T+ token 处理量说明卷量驱动的模式是可行的。Baseten 于 2026 年 1 月以 50 亿美元估值完成 3 亿美元 E 轮融资。竞争定位规律很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业品质，Modal 优化 Python 原生 DX，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本课提供一张可供创始人直接使用的矩阵。

**类型：** 精读
**语言：** Python（标准库，玩具级每请求经济性比较器）
**前置要求：** Phase 17 · 01（托管 LLM 平台）、Phase 17 · 04（vLLM 推理内部原理）
**时长：** 约 60 分钟

## 学习目标

- 说出三个市场细分（定制硅片、GPU 平台、API 优先），并将每个供应商映射到对应细分。
- 解释为什么"按 token" API 定价模型趋近于推理引擎的成本曲线，而非硬件的成本曲线。
- 跨至少三家供应商计算每请求有效成本，并解释什么时候按分钟计费（Baseten、Modal）优于按 token 计费。
- 识别给定工作负载的正确默认平台（突发 serverless、稳定高吞吐、微调变体、多模态）。

## 背景问题

你评估了托管超大规模云平台后，决定需要一个更专注、更快速的提供商——Fireworks 主打延迟，Together 主打广度，Baseten 用来跑微调自定义模型。现在你有了六个真实选项，但定价页面对不齐。Fireworks 展示每百万 token 美元价；Baseten 展示每分钟美元；Modal 展示每秒美元；Replicate 展示每预测美元。不对工作负载建模就无法直接对比。

更糟的是，每张定价页面背后的商业模式不同。Fireworks 运行自己的定制引擎（FireAttention）在共享 GPU 上；按 token 费率反映他们的利用率曲线。Baseten 提供 Truss 加专属 GPU；按分钟计费反映独占性。Modal 是真正的 Python serverless——按秒计费，冷启动亚秒级。相同输出（一个 LLM 响应），三种不同成本函数。

本课为六家建模，告诉你各自在什么场景下胜出。

## 核心概念

### 三个细分市场

**定制硅片** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。通常比同等模型 GPU 集群快 5-10 倍。每 token 定价更高（Groq 在 2025 年底对 Llama-70B 约 0.99 美元/百万 token），但对延迟敏感场景无可匹敌。Groq 是语音代理和实时翻译的生产选择。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（H100、H200、2026 年还有 B200）有时也用 AMD。是"原始 GPU 租赁"（RunPod、Lambda）和"超大规模云托管服务"（Bedrock）之间的经济层。

**API 优先 marketplace** — Replicate、DeepInfra、OpenRouter、Fal。目录广泛，按预测或按秒计费，强调接入首通时间。

### Fireworks — 延迟优化的 GPU 平台

- FireAttention 引擎（定制）；宣传比同等配置 vLLM 延迟低 4 倍。
- 批量档约 serverless 价格的 50%，适合非交互式工作负载。
- 微调模型与基础模型同价——相比那些对 LoRA 收取溢价的供应商，这是一个真正的差异化点。
- 2026 年中：将按需 GPU 租赁提价 1 美元/小时，2026 年 5 月 1 日生效。规模可谈批量价。
- 财务信号：40 亿美元估值，日均 10T+ token。

### Together — 广度优化

- 200+ 模型，包括上游发布后几天内的开源发布。
- 比 Replicate 在同等 LLM 模型上便宜 50-70%——"AI 原生云"的定位就是卷量和目录。
- 推理 + 微调 + 训练，一个 API 全搞定。

### Baseten — 企业品质优化

- Truss 框架：一个 manifest 打包模型依赖、密钥和服务配置。
- GPU 范围从 T4 到 B200。按分钟计费，冷启动缓解合理。
- SOC 2 Type II，HIPAA 就绪。常见的金融科技和医疗选择。
- 2026 年 1 月 E 轮，50 亿美元估值（3 亿美元，来自 CapitalG、IVP、NVIDIA）。

### Modal — Python 原生优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数，一行命令部署。
- 按秒计费。预热冷启动 2-4 秒；小模型 <1 秒。
- 2025 年 8700 万美元 B 轮，11 亿美元估值。独立调查中开发者体验得分最高。

### Replicate — 多模态广度

- 按预测计费。是图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS 插件）。
- LLM 每 token 费率竞争力较弱，但在多模态多样性上胜出。

### Anyscale — Ray 原生

- 基于 Ray 构建；RayTurbo 是 Anyscale 专有推理引擎（与 vLLM 在 Ray 集群上竞争）。
- 最适合推理步骤是大图谱中一个节点的分布式 Python 工作负载。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token vs 按分钟——各自何时胜出

按 token 适合延迟不敏感、突发性强的工作负载——只用多少付多少。按分钟适合利用率高且可预测的工作负载——一旦 GPU 跑满，就比按 token 便宜。

粗略规律：对于持续利用率约 30% 的专属 GPU 以上，按分钟（Baseten、Modal）开始优于按 token（Fireworks、Together）。低于该值，按 token 胜出因为你不用为空闲付钱。

### 定制引擎是真正的护城河

每个自称超越 vLLM 和 SGLang 的平台都声称有定制引擎。FireAttention、RayTurbo、Baseten 的推理栈。定制引擎的声称是营销话术——说实话，vLLM + SGLang 代表了生产级开源推理约 80% 的份额，平台层的差异化实际上是 DX、归因和 SLA。

### 记忆数字

- Fireworks GPU 租赁：2026 年 5 月 1 日起涨价 1 美元/小时。
- Fireworks 声称：比同等配置 vLLM 延迟低 4 倍。
- Together：比 Replicate 在 LLM 上便宜 50-70%。
- Baseten 估值：50 亿美元（E 轮，2026 年 1 月，3 亿美元）。
- Modal 估值：11 亿美元（B 轮，2025 年）。
- 持续利用率约 30% 以上，按分钟开始优于按 token。

## 运用它

`code/main.py` 在综合负载下比较六家供应商，跨定价模型。报告每天成本和有效每百万 token 成本。运行它来找按 token 和按分钟的盈亏平衡点。

## 交付它

本课产出 `outputs/skill-inference-platform-picker.md`。给定负载画像、SLA 和预算，选择主推理平台并命名备选。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下 Baseten（按分钟）能在一块 H100 的 70B 模型上击败 Fireworks（按 token）？自己推导交叉点并与经验法则比较。
2. 你的产品同时提供图像生成、聊天和语音转文字。为每种模态选择平台，并命名统一它们的网关模式。
3. Fireworks 将你的主力模型提价 1 美元/小时。如果 40% 流量转入批量档（5 折），建模混合成本影响。
4. 一个受监管客户要求 SOC 2 Type II + HIPAA + 专属 GPU。哪三个平台可行，哪个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks serverless、Together 按需、Baseten 专属和 Replicate API 上每 1000 次预测的成本。每日 10 次预测时哪个最便宜？每日 10,000 次呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 定制硅片 | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU——针对 decode 优化 |
| FireAttention | "Fireworks 引擎" | 定制注意力 kernel；宣传比 vLLM 延迟低 4 倍 |
| Truss | "Baseten 格式" | 模型打包 manifest；依赖 + 密钥 + 服务配置 |
| 按 token | "API 定价" | 按消耗的 token 计费；不为空闲付钱 |
| 按分钟 | "专属定价" | 按墙上时钟 GPU 时间计费；高利用率时胜出 |
| 按预测 | "Replicate 定价" | 每次模型调用计费；常见于图像/视频 |
| RayTurbo | "Anyscale 引擎" | Ray 上的专有推理；与 Ray 集群上的 vLLM 竞争 |
| 批量档 | "5 折" | 非交互队列，按折扣价；Fireworks 和 OpenAI 均有 |
| 微调同价 | "Fireworks LoRA" | 以基础模型费率计收 LoRA 服务的请求（差异化点） |

## 延伸阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按 token 费率、批量档、GPU 租赁。
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业档。
- [Modal 定价](https://modal.com/pricing) — 按秒 GPU 费率及免费档。
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录和按 token 费率。
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价。
- [Northflank — Fireworks AI 替代品](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 对比评估。
- [Infrabase — AI 推理 API 供应商 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商全景。