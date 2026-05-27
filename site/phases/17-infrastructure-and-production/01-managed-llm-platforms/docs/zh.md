# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大超大规模云厂商，三种截然不同的策略。AWS Bedrock 是一个模型 marketplace——Claude、Llama、Titan、Stability、Cohere 通过一个 API 统一提供。Azure OpenAI 是独家 OpenAI 合作，外加用于专属容量的预置吞吐量单位（PTU）。Vertex AI 以 Gemini 为核心，拥有最好的长上下文和多模态支持。2026 年，Artificial Analysis 评测 Azure OpenAI 在 Llama 3.1 405B 等效规模上 P50 TTFT 约 50 ms， Bedrock 约 75 ms——PTU 解释了差距，因为专属容量优于共享按需资源。决策规则不是"哪个最快"，而是"哪个模型目录和 FinOps 归因面与我的产品匹配"。本课教你带着权衡清单做选择，而不是凭感觉。

**类型：** 精读
**语言：** Python（标准库，玩具级成本与延迟比较器）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具与协议）
**时长：** 约 60 分钟

## 学习目标

- 说出三种平台策略（marketplace vs 独家合作 vs Gemini 优先），并能将每种策略匹配到对应的产品用例。
- 解释 Azure OpenAI 中的预置吞吐量单位（PTU）为你买到了什么，以及为什么在 405B 规模上按需使用 Bedrock 通常慢约 25 ms。
- 绘制三种平台各自的 FinOps 归因面（Bedrock Application Inference Profiles vs Vertex 按项目分配 vs Azure 范围 + PTU 预留）。
- 写出"双供应商最低原则"策略，并解释为什么 2026 年单一供应商锁定是最昂贵的错误。

## 背景问题

你为产品选型了 Claude 3.7 Sonnet。现在需要提供推理服务。你可以直连 Anthropic API，也可以通过 AWS Bedrock 调用，或者通过网关。三种方案的复杂度递增：直接 API 最简单；Bedrock 增加了 BAAs、VPC 端点、IAM 和 CloudWatch 归因；网关则增加了跨供应商的故障转移、统一计费和速率限制。

更深层的问题是目录。如果你需要在同一款产品里同时用到 Claude、Llama 和 Gemini，就无法从一家供应商买到——除非同时找 Bedrock 加 Vertex 加 Azure OpenAI。三家超大规模云厂商并不等价——每家都押注了不同的谁来掌控模型层。

本课将三种赌注、延迟差距、FinOps 差距和锁定风险一一映射。

## 核心概念

### 三种策略

**AWS Bedrock** — marketplace。Claude（Anthropic）、Llama（Meta）、Titan（AWS 一方）、Stability（图像）、Cohere（嵌入向量）、Mistral，外加图像和嵌入子目录。一个 API、一套 IAM 面、一个 CloudWatch 导出。Bedrock 的赌注是客户更看重灵活性而非单一模型。

**Azure OpenAI** — 独家合作。提供 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及 Azure 数据中心内 OpenAI 模型的微调。"Azure OpenAI Service" 目录中没有非 OpenAI 模型——那些走 Azure AI Foundry（独立产品）。Azure 的赌注是 OpenAI 保持前沿地位，且客户需要对这段特定关系的企业级管控。

**Vertex AI** — Gemini 优先，万物其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，外加 Model Garden（第三方）。Vertex 的赌注是多模态长上下文——100 万 token 的 Gemini 上下文是真正的差异化点。

### 大规模下的延迟差距

Artificial Analysis 运行持续基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI P50 TTFT 约 50 ms；Bedrock 约 75 ms。差距并非 AWS 的失败——而是容量模式差异。Azure 销售 PTU（预置吞吐量单位），为你的租户预留 GPU 容量。Bedrock 对应的配置（预置吞吐量）存在，但单价约 21 美元/小时/单元，大多数客户仍留在共享按需档。

按需共享容量与所有其他客户的流量竞争。专属容量则不会。如果你的产品 SLA 要求 P99 TTFT < 100 ms，你要么买 Azure PTU，要么买 Bedrock 预置吞吐量，要么接受默认的高方差。

### 预置吞吐量经济学

Azure PTU：一块预留的推理算力。相比按需最高可节省 70%。按小时固定计费，无论流量如何——空闲时也要付费。盈亏平衡点通常在 40-60% 的持续利用率。

Bedrock 预置吞吐量：每单元 21-50 美元/小时，取决于模型和区域。类似逻辑——盈亏平衡点在峰值利用率的约一半。需要月度承诺。

Vertex 预置容量按 Gemini SKU 销售；定价因模型和区域而异，且公开程度较低。

### FinOps 面——真正的差异化因素

**Bedrock Application Inference Profiles** 是 marketplace 中最干净的归因方案。用 `team`、`product`、`feature` 标记一个 profile；将所有模型调用路由通过它；CloudWatch 按 profile 分项输出成本，无需后期处理。2025 年新增，至今仍是超大规模云厂商原生方案中最细粒度的。

**Vertex** 归因采用按团队一个 GCP 项目加标签铺满所有资源的方式。结合 BigQuery 计费导出 + DataStudio 汇总，工作量更大，但 BigQuery 让你对成本数据做任意 SQL 查询。

**Azure** 依赖订阅/资源组范围加标签，PTU 预留作为一等成本对象。标签从资源组继承，而非从请求继承，因此每请求归因需要 Application Insights 自定义指标，或在网关上添加标头。

规律：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 不加 instrument 就最不透明。

### 锁定是 2026 年的风险

单一超大规模云厂商承诺在一种模型主导时没有问题。2026 年前沿模型每月都在移动——Q1 是 Claude 3.7，Q2 是 Gemini 2.5，Q3 又是 GPT-5。锁定一家平台意味着被另外三分之二的前沿模型拒之门外。

成熟团队的普遍做法：任何产品关键 LLM 调用路径的双供应商最低原则。Bedrock 加 Azure OpenAI 是常见组合——一家提供 Claude，另一家提供 GPT，中间加网关做故障转移，成本增加微乎其微（因为网关选择最优路由），但在宕机期间（如 2025 年 1 月 Azure OpenAI 事故、AWS us-east-1 故障）可用性提升是决定性的。

### 数据驻留、BAAs 和受监管行业

Bedrock：多数地区有 BAAs；VPC 端点；安全护栏。常见的金融科技默认值。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业受监管场景的默认值。
Vertex：HIPAA、GDPR、按地区数据驻留；Google Cloud 合规体系。

三家都满足基本合规要求。差异在于数据保留策略、日志处理方式，以及滥用监控是否读取你的流量（多数默认选择加入；企业可退出）。

### 记忆数字

- Azure OpenAI 在 Llama 3.1 405B 等效规模上的 P50 TTFT：约 50 ms（用 PTU）。
- Bedrock 按需 P50 TTFT：约 75 ms。
- Bedrock 预置吞吐量：每单元 21-50 美元/小时。
- Azure PTU 盈亏平衡：约 40-60% 持续利用率。
- 高利用率下 PTU 相比按需节省：最高 70%。

## 运用它

`code/main.py` 在综合负载下比较三种平台——对按需 vs PTU 经济性、TTFT 方差和成本归因保真度建模。运行它来查看 PTU 在哪里值得、marketplace 的模型广度如何弥补 TTFT 差距。

## 交付它

本课产出 `outputs/skill-managed-platform-picker.md`。给定负载画像（所需模型、TTFT SLA、日间容量、合规要求），推荐主平台、备用平台和 FinOps instrument 方案。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下 Azure PTU 能在 70B 类模型上击败按需？计算盈亏平衡并与宣传的 40-60% 区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计双供应商部署——哪个模型放哪个超大规模云厂商，网关放在哪里，故障转移策略是什么？
3. 一位受监管的医疗客户要求 BAAs、美国东部数据驻留和 P99 TTFT <100ms。选一个平台并用三个具体功能来论证。
4. 你发现 Bedrock 账单本月增长了 4 倍，但流量没有变化。在没有 Application Inference Profiles 的情况下如何追查元凶？有 profiles 的话需要多久？
5. 阅读 Azure OpenAI 和 Bedrock 定价页面。对于每月 1 亿 token 的 Claude 工作负载，直接 Anthropic API、Bedrock 按需或 Bedrock 预置吞吐量，哪个更便宜？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Bedrock | "AWS LLM 服务" | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型 marketplace |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心内的独家 OpenAI 模型，带企业管控 |
| Vertex AI | "Google 的 LLM" | Gemini 优先平台，Model Garden 提供第三方模型 |
| PTU | "专属容量" | 预置吞吐量单位——预留推理 GPU，按小时计价 |
| Application Inference Profile | "Bedrock 标记" | 带标签的按产品成本/用量 profile，CloudWatch 原生 |
| Model Garden | "Vertex 目录" | Vertex AI 第三方模型分区，独立于 Gemini |
| 双供应商最低原则 | "LLM 冗余" | 每个关键 LLM 路径至少跨两个超大规模云厂商运行的策略 |
| BAA | "HIPAA 手续" | 业务关联协议；PHI 所必需；三家均提供 |
| 滥用监控 | "日志监控者" | 提供商端对提示词/输出的安全扫描；企业可退出 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) — 权威费率表和预置吞吐量定价。
- [Azure OpenAI Service 定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济性和费率表。
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 分层和 Model Garden 附加费。
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/) — 跨供应商持续延迟和吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归因机制逐项对比。