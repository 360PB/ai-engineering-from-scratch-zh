# 音频-语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年音频-语言模型对语音 + 环境声音 + 音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 持平。Audio Flamingo Next 在 LongAudioBench 上超越 Gemini 2.5 Pro。开放与封闭的差距基本消除——除了多音频任务，所有人接近随机。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 04（ASR）、Phase 12 · 03（视觉-语言模型）、Phase 7 · 10（音频 Transformer）
**时间：** 约 45 分钟

## 问题

你有 5 秒音频：狗叫，某人喊"停！"，然后静音。有用的问题跨越多个轴：

- **转录。** "说了什么？" — ASR 领域。
- **语义推理。** "这个人有危险吗？" — 需要对狗叫 + 喊叫 + 静音的联合理解。
- **音乐推理。** "哪些乐器演奏旋律？"
- **长音频检索。** "90 分钟讲座中，讲师在何处解释了梯度下降？"

一个模型用同一 prompt 回答所有这些问题是**音频-语言模型**（LALM / ALM）。与纯 ASR 的区别：LALM 输出自由形式的自然语言答案，而不仅是转录。

## 概念

![音频-语言模型：音频编码器 + 投影器 + LLM 解码器](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 年 LALM 都具有相同的骨架：

1. **音频编码器。** Whisper 编码器 · BEATs · CLAP · WavLM · 或每个模型的自定义编码器。
2. **投影器。** 将音频编码器特征桥接到 LLM 的 token 嵌入空间的线性或 MLP。
3. **LLM。** Llama / Qwen / Gemma 解码器。接受交错的文本 + 音频 token；生成文本。

训练：

- **阶段 1。** 冻结编码器 + LLM；仅在 ASR / 字幕数据上训练投影器。
- **阶段 2。** 完整 / LoRA 微调指令-following 音频任务（QA、推理、音乐理解）。
- **阶段 3（可选）。** 语音 in / 语音 out 添加一个语音解码器。Qwen2.5-Omni 和 AF3-Chat 这样做。

### 2026 年模型图

| 模型 | 主干 | 音频编码器 | 输出模态 | 访问 |
|------|------|-----------|----------|------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准现实检验（2026）

**MMAU-Pro。** 1800 QA 对，覆盖语音/声音/音乐/混合。多音频子集包括。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频列对所有人来说都是致命的。** 4 选项选择题的随机机会 = 25%；大多数模型约在 25% 左右。LALM 仍然难以比较两个片段。

### 2026 年 LALM 有用的地方

- **合规审计呼叫中心录音。** "代理是否提到了要求的披露？"
- **无障碍。** 向聋人用户描述声音事件（不仅是转录）。
- **内容审核。** 检测暴力语言 + 威胁性语气 + 背景上下文。
- **播客/会议章节化。** 语义摘要，而不仅是说话人轮次。
- **音乐目录分析。** "找到所有有 B 段转调的曲目。"

### 还不行的（目前）

- 细粒度音乐理论（和弦级以下）。
- 长对话上的说话人归属推理（超过 10 分钟退化）。
- 多音频比较（22–26% 仅略高于随机）。
- 实时流推理（大多数是离线批量推理）。

## 构建

### 步骤 1：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步骤 2：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就这样。投影器通常是 1-3 个线性层。在 ASR 对（音频 → 转录）上训练它是阶段 1 借口任务。

### 步骤 3：在 MMAU / LongAudioBench 上基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分别报告每类（语音/声音/音乐/多音频）。聚合数字掩盖了模型失败的地方。

## 使用

| 任务 | 2026 选择 |
|------|-----------|
| 自由形式音频 QA（开放） | Qwen2.5-Omni-7B |
| 最佳开放长音频 | Audio Flamingo Next |
| 最佳闭源 | Gemini 2.5 Pro |
| 语音 in / 语音 out 代理 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | Gemini 2.5 Pro via API，带 RAG over 你的策略文档 |

## 坑

- **过度信任多音频。** 如果你的任务需要"哪个片段有 X"，随机机会水平的性能是真实的。
- **长音频退化。** 超过 10 分钟，大多数模型的说话人归属会崩溃。先做日志（Lesson 6），再总结。
- **静音上的幻觉。** 与 Whisper 风格问题相同，由使用 Whisper 编码器的 LALM 继承。VAD 门控。
- **基准 cherry-picking。** 供应商博客文章突出最佳案例类别。自己去跑 MMAU-Pro 多音频子集。

## 发货

保存为 `outputs/skill-alm-picker.md`。为给定音频理解任务选择 LALM + 基准子集 + 输出模态（文本 vs 语音）。

## 练习

1. **简单。** 运行 `code/main.py` 查看玩具投影器模式 + 假 LALM 路由（audio-embedding, text-tokens）→ 输出 tokens。
2. **中等。** 在 100 个 MMAU-Pro 语音项目上评分 Qwen2.5-Omni-7B。与论文报告数字比较。
3. **困难。** 构建一个最小音频-字幕基线：BEATs 编码器 + 2 层投影器 + 冻结 Llama-3.2-1B。仅在 AudioCaps 上微调投影器。在 Clotho-AQA 上与 SALMONN 比较。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------|--------|
| LALM | 音频 ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| 投影器 | 适配器 | 小 MLP 将音频特征映射到 LLM 嵌入空间。 |
| MMAU | 基准 | 跨语音、声音、音乐的 10k 音频-QA 对。 |
| MMAU-Pro | 更难 MMAU | 1800 多音频/推理重问题。 |
| LongAudioBench | 长格式评估 | 带语义查询的多分钟片段。 |
| 语音 in / 语音 out | 语音原生 | 模型摄入语音并发出语音，不经过文本兜底。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音 in-语音 out。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开放长音频领先者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。
- [MMAU-Pro 排行榜](https://mmaubenchmark.github.io/) — 2026 年实时排名。