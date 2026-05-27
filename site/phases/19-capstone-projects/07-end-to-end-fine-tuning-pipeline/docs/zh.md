# Capstone 07 — 端到端微调流水线（从数据到 SFT 到 DPO 到服务）

> 一个在你的数据上训练的 8B 模型，用你的偏好做 DPO 对齐，量化，带投机解码，服务于可量化的 $/1M tokens。2026年的开源技术栈是 Axolotl v0.8、TRL 0.15、Unsloth 用于迭代、GPTQ/AWQ/GGUF 量化、vLLM 0.7 配 EAGLE-3 服务。毕业项目是从零跑通完整流水线——YAML 进，端点出——并在 2026 Model Openness Framework 下发布模型卡。

**类型：** 毕业项目
**语言：** Python（流水线），YAML（配置），Bash（脚本）
**前置知识：** Phase 2（机器学习）、Phase 3（深度学习）、Phase 7（Transformers）、Phase 10（从零构建LLM）、Phase 11（LLM工程）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时长：** 35小时

## 问题

2026年每个正经 AI 团队都有一套随时可用的微调流水线。不是因为要发一个前沿基座模型，而是因为下游适配——领域 SFT、针对标注偏好的 DPO、为投机解码提炼草案、用 EAGLE-3 服务——才是有可衡量收益的地方。Axolotl v0.8 处理多卡 SFT 配置。TRL 0.15 处理 DPO 和 GRPO。Unsloth 让单卡迭代飞快。vLLM 0.7 配 EAGLE-3 在不损失质量的情况下将解码吞吐推高 2-3 倍。工具链是现成的；功力在 YAML、数据卫生和评估纪律。

你将把 8B 基座（Llama 3.3、Qwen3 或 Gemma 3）通过 SFT 然后 DPO 在任务特定数据上跑一遍，量化用于服务，并在 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 上测量收益。你将在 2026 Model Openness Framework 下生成模型卡。要点是可复现性——一条命令从零重新运行整个流水线。

## 核心概念

流水线分五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 清洗、针对公开基准污染的切分卫生检查。**SFT**：Axolotl YAML，8xH100 上 ZeRO-3，余弦调度，序列打包，2-3 epochs。**DPO 或 GRPO**：TRL 配置，1个 epoch，偏好对可以是人工标注或模型判断的，调整 beta。**量化**：GPTQ + AWQ + GGUF，灵活部署。**服务**：vLLM 0.7 配 EAGLE-3 投机头（或者 SGLang 配 SpecForge），K8s 部署，队列等待上 HPA。

消融实验是交付物：SFT-only vs SFT+DPO vs SFT+GRPO 在三个任务特定基准上的对比。服务指标：batch 1/8/32 下 tokens/s、EAGLE-3 接受率、$/1M tokens。安全评估：Llama Guard 4 通过率。模型卡：偏差评估、可复现性种子、数据许可。

## 架构

```
原始数据（HF datasets + 内部）
    |
    v
Datatrove 去重 + Nemotron-CC 质量过滤 + PII 清洗
    |
    v
切分卫生（MMLU-Pro 污染检查）
    |
    v
Axolotl SFT 配置（YAML）  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO 配置       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF 量化
    |
    v
vLLM 0.7 + EAGLE-3 投机解码
    |
    v
K8s 部署，队列等待上 HPA
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
模型卡（2026 MOF）+ 安全评估（Llama Guard 4）
```

## 技术栈

- 数据：Datatrove 去重，Nemotron-CC 分类器质量过滤，Presidio PII 清洗
- 基座：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，ZeRO-3，Flash Attention 3，序列打包
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单卡迭代
- 量化：GPTQ（Marlin）、AWQ、llama.cpp 的 GGUF
- 服务：vLLM 0.7 配 EAGLE-3 投机解码（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin，队列等待指标上 HPA
- 可观测性：W&B 用于训练，Langfuse 用于推理

## 动手实现

1. **数据流水线。** 在原始语料上运行 Datatrove 去重。应用 Nemotron-CC 风格质量分类器。Presidio 清洗 PII。用显式种子写 train/val 切分。

2. **污染检查。** 对每个验证切分，计算与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集的 MinHash 相似度。拒绝任何重叠。

3. **Axolotl SFT。** YAML 含 ZeRO-3、FA3、序列打包。8xH100 上 2-3 epochs。日志到 W&B。

4. **TRL DPO / GRPO。** 取 SFT 检查点，在偏好对上跑一个 epoch DPO（或者 GRPO 在数学/代码上用可验证奖励）。调 beta 超参。

5. **量化。** 产出三个量化版本：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M for llama.cpp。记录大小和标称吞吐。

6. **带投机解码的服务。** vLLM 0.7 配置，通过 Red Hat Speculators 训练的 EAGLE-3 草案头。在 batch 1/8/32 下测量接受率和尾延迟。报告与同 eval 下 Anthropic / OpenAI 的 $/1M tokens 对比。

7. **评估矩阵。** 在基座、SFT-only、SFT+DPO、SFT+GRPO 上跑 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成表格。

8. **安全评估。** Llama Guard 4 在 dev 集上的通过率。ShieldGemma-2 输出过滤器。

9. **模型卡。** MOF 2026 模板：数据、训练、评估、安全、许可，含 YAML 和 commit SHA 的可复现性部分。

## 用现成库

```bash
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[数据]   300k 去重后，12k 过滤后，280k 接受（seed=7）
[SFT]    3 epochs，8xH100，6h12m，val loss 1.42 -> 1.03
[DPO]    1 epoch，beta=0.08，4xH100，1h40m
[量化]   GPTQ-INT4 4.6 GB，AWQ-INT4 4.8 GB，GGUF-Q4_K_M 5.1 GB
[服务]   vLLM 0.7，EAGLE-3 接受率 0.74，p99 126ms @ bs=8
[评估]   MMLU-Pro +3.2，MT-Bench-v2 +0.41，RewardBench-2 +0.08
[模型卡] model-card.md 按 2026 MOF 生成
```

## 产出

`outputs/skill-finetuning-pipeline.md` 描述交付物。一条命令把数据跑过 SFT 跑过 DPO 跑过量化跑过服务跑过评估，输出模型卡和服务端点。

| 权重 | 指标 | 衡量方式 |
|:-:|---|---|
| 25 | vs 基座评估差值 | 在目标任务上的可衡量收益（MMLU-Pro、MT-Bench-v2、任务特定） |
| 20 | 流水线可复现性 | 一条命令用相同种子从零重新运行 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查通过 |
| 20 | 服务效率 | bs=1/8/32 下 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一任务特定基准上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种偏好方法胜出以及胜出多少。

2. 把 Llama 3.3 8B 换成 Qwen3 14B。在匹配质量下测量 $/1M tokens。

3. 在领域数据 vs 通用 ShareGPT 上测量 EAGLE-3 接受率。报告差值及对延迟预算的影响。

4. 注入 1% 污染（把 MMLU-Pro 答案泄露到训练数据）并重新运行评估。观察 MMLU-Pro 准确率异常跳高。构建一个能捕获此事的污染检查 CI 门控。

5. 添加 LoRA SFT 作为全量微调的替代方案。在 10 倍内存降低下测量质量差距。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | "SFT 训练器" | 统一的 YAML 驱动 SFT/DPO/蒸馏训练器 |
| TRL | "偏好调优器" | Hugging Face 的 DPO、GRPO、PPO 库 |
| GRPO | "组相对策略优化" | DeepSeek R1 的 RL 配方，用可验证奖励 |
| EAGLE-3 | "投机解码草案" | 预测前方 N 个 token 的草案头；vLLM 用目标模型验证 |
| MOF | "模型开放框架" | 2026年按数据、代码、许可对模型发布评级的标准 |
| Contamination check | "切分卫生" | 基于 MinHash 检测测试集泄露到训练集 |
| Acceptance rate | "EAGLE / MTP 指标" | 目标模型接受的草案 token 比例 |

## 扩展阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — SFT / DPO 参考训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单卡迭代参考
- [DeepSeek R1 论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 备选投机解码训练器
- [Model Openness Framework 2026](https://isocpp.org/) — 开放发布评分标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 权威评估运行器