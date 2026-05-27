# 专家混合（MoE）

> 密集 70B transformer 为每个 token 激活每个参数。671B MoE 每个 token 仅激活 37B 且在每个基准上胜出。稀疏是本十年最重要的扩展思路。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 第 5 课（完整 Transformer）、Phase 7 第 7 课（GPT）
**时长：** ~45 分钟

## 问题

密集 transformer 推理时的 FLOPs 等于其参数计数（乘以 2 作为前向传递）。放大密集模型，每个 token 都要支付全额账单。到 2024 年前沿遇到了计算墙：要有意义地更智能，你需要指数级更多的 FLOPs 每个 token。

专家混合打破这个链接。将每个 FFN 替换为 `E` 个独立专家 + 一个路由器，每个 token 选择 `k` 个专家。总参数 = `E × FFN_size`。每个 token 激活参数 = `k × FFN_size`。2026 年典型配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年前沿几乎全是 MoE：DeepSeek-V3（671B 总计 / 37B 激活）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前 10 名开源模型全是 MoE。

## 概念

![MoE 层：路由器为每个 token 选择 k 个专家](../assets/moe.svg)

### FFN 交换

密集 transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 每个 token 选 k 个专家
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家是一个独立 FFN（通常是 SwiGLU）。路由器是一个线性层。每个 token 选择自己的 `k` 个专家，并获得它们输出的门控混合。

### 负载均衡问题

如果路由器将 90% 的 token 放入专家 3，其他专家会饿死。尝试了三种修复：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。添加与专家使用方差成比例的惩罚。有效，但添加超参数和第二个梯度信号。
2. **专家容量 + token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 跳过该层。损害质量。
3. **无辅助损失均衡**（DeepSeek-V3）。添加学习的每专家偏置，转移路由器的 top-k 选择。偏置在训练损失外更新。无主目标上的惩罚。2024 年的大解锁。

DeepSeek-V3 的方法：每训练步后，对每个专家检查其使用率是否高于或低于目标。用 `±γ` 调整偏置。选择使用 `scores + bias`。用于门控的专家概率是原始 `scores` 不变。将路由与表达解耦。

### 共享专家

DeepSeek-V2/V3 还将专家分为*共享*和*路由*。每个 token 通过所有共享专家。路由专家通过 top-k 选择。共享专家捕获共同知识；路由专家专门化。V3 运行 1 个共享专家加 256 个路由专家的 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家与完整 FFN 一样宽。`E` 小（8–64），`k` 小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（1/8 FFN 大小）。`E` 大（256+），`k` 较大（8+）。总参数相同，但组合扩展更快。`C(256, 8) = 400 万亿` 每个 token 可能的"专家"。质量上升，延迟保持平坦。

### 成本曲线

每 token、每层：

| 配置 | 每 token 激活参数 | 总参数 |
|------|------------------|--------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（密集） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3 在几乎每个基准上击败 Llama 3 70B（密集），同时**每个 token 执行更少激活 FLOPs**。更多参数 = 更多知识。更多激活 FLOPs = 每 token 更多计算。MoE 将它们解耦。

### 陷阱：内存

所有专家都在 GPU 上，无论哪些激活。每个 671B 模型在 fp16 权重下需要 ~1.3 TB VRAM。前沿 MoE 部署需要专家并行——将专家分片到 GPU，跨网络路由 token。延迟由 all-to-all 通信主导，而非矩阵乘法。

## 构建

见 `code/main.py`。纯标准库的紧凑 MoE 层，包含：

- `n_experts=8` SwiGLU-ish 专家（每个说明一个线性，用于说明）
- top-k=2 路由
- softmax 归一化门控权重
- 通过每专家偏置的无辅助损失均衡

### 第一步：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 在原始分数的已选专家上 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这是 DeepSeek-V3 技巧——偏置纠正负载不平衡而不引导模型预测。

### 第二步：100 个 token 通过路由器

跟踪哪些专家多久激活一次。没有偏置，使用偏向斜。有偏置更新循环（对过度使用的专家 `-γ`，对未充分使用的 `+γ`），使用在几次迭代内收敛到均匀分布。

### 第三步：参数计数比较

打印 MoE 配置的"密集等效"。DeepSeek-V3 形状：256 个路由 + 1 个共享，8 个激活，d_model=7168。总参数计数令人目眩。激活计数是密集 Llama 3 70B 的七分之一。

## 使用

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE 路由。SGLang 有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选 MoE：**
- 你想以前沿质量降低每个 token 推理成本。
- 你有 VRAM / 专家并行基础设施。
- 你工作负载是 token 密集型（聊天、代码）而非上下文密集型（长文档）。

**何时不选 MoE：**
- 边缘部署——你为任何激活 FLOP 支付完整存储。
- 延迟关键的单用户服务——专家路由增加开销。
- 小型模型（<7B）—— MoE 的质量优势仅在计算阈值以上出现（约 6B 激活参数）。

## 交付

见 `outputs/skill-moe-configurator.md`。该 skill 根据参数预算、训练 token 和部署目标为新 MoE 选择 E、k 和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失偏置更新如何在 50 次迭代中平均化专家使用。
2. **中等。** 用基于哈希的路由器替换学习的路由器（确定性，无学习）。比较质量和均衡。为什么学习的路由器更好？
3. **困难。** 实现 GRPO 风格的"rollout 匹配路由"（DeepSeek-V3.2 技巧）：记录推理期间哪些专家激活，强制梯度计算期间相同的路由。在玩具策略梯度设置上测量效果。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------|---------|
| 专家 | "众多 FFN 之一" | 独立前馈网络；参数专用于 FFN 计算的稀疏切片。 |
| 路由器 | "门" | 一个微型线性层，为每个 token 对每个专家评分；top-k 选择。 |
| Top-k 路由 | "每个 token k 个激活专家" | 每个 token 的 FFN 计算通过恰好 k 个专家，按门加权。 |
| 辅助损失 | "负载均衡惩罚" | 惩罚偏向专家使用的额外损失项。 |
| 无辅助损失 | "DeepSeek-V3 的技巧" | 通过路由器选择上的每专家偏置均衡；无额外梯度。 |
| 共享专家 | "始终开启" | 每个 token 都通过的额外专家；捕获共同知识。 |
| 专家并行 | "按专家分片" | 将不同专家分发到不同 GPU；跨网络路由 token。 |
| 稀疏性 | "激活参数 < 总参数" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 约 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 这个想法。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失 MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置的均衡论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课路由器使用的细粒度 + 共享专家分割。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原创共享专家论文。