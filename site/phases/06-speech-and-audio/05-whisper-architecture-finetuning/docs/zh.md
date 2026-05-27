# Whisper — 架构与微调

> Whisper 是一个30秒窗口的 Transformer encoder-decoder，在 680k 小时多语言弱监督音频-文本对上训练。一个架构，多个任务，跨99种语言鲁棒。2026年的参考 ASR。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 6 · 04（ASR），Phase 5 · 10（注意力），Phase 7 · 05（完整 Transformer）
**时长：** 约75分钟

## 问题

Whisper，由 OpenAI 于2022年9月发布，是第一个作为商品发货的 ASR 模型：粘贴音频，得到文本，99种语言，鲁棒抗噪，笔记本可跑。到2024年 OpenAI 发了 Large-v3 和 Turbo 变体；到2026年，Whisper 成为播客转录、语音助手、YouTube 字幕等一切的默认基线。

但 Whisper 不是能永远当黑盒用的流水线。领域偏移会杀死它——技术术语、说话人口音、专有名词、短片段、静默。你需要知道：

1. 它内部实际是什么。
2. 如何正确处理分块、流式或长音频。
3. 何时微调以及如何微调。

## 核心概念

**架构。** 标准 Transformer encoder-decoder。

- 输入：30秒对数 mel 频谱图，80 mels，10ms 跳 → 3000 帧。短于30秒的片段补零，长于30秒的片段分块。
- Encoder：conv下采样（步幅2）+ `N` 个 Transformer 块。对于 Large-v3：32层，1280维，20头。
- Decoder：`N` 个带因果自注意 + 交叉注意到 encoder 输出的 Transformer 块。与 encoder 同大小。
- 输出：51,865-token 词汇表上的 BPE token。

Large-v3 有 1.55B 参数。Turbo 用4层 decoder（来自32层），延迟降低8×，WER 损失<1%。

**提示格式。** Whisper 是一个由 decoder 中特殊 token 引导的多任务模型：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签；强制翻译 vs 转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 从任何语言输入翻译为英语输出，或逐字转录。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

提示是一个模型做多任务的入口。换 `<|en|>` 为 `<|fr|>` 它就转录法语。

**30秒窗口。** 一切锁定在30秒。长片段需要分块；短片段补零。窗口不原生流式——这是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数 mel 归一化。** `(log_mel - mean) / std`，统计量来自 Whisper 自己的训练语料。你*必须*用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），不是 `librosa.feature.melspectrogram`。

## 微调

标准工作流：

1. 收集10–100小时目标领域音频与对齐转录。
2. 用 `generate_with_loss` 回调运行 `transformers.Seq2SeqTrainer`。
3. 参数高效：在注意力层的 `q_proj`、`k_proj`、`v_proj` 上用 LoRA，GPU 内存降低4×，WER 损失<0.3%。
4. 如果少于10小时，冻结 encoder。只调 decoder。
5. 用 Whisper 自己的 tokenizer 和提示格式；永换 tokenizer。

## 动手实现

### 步骤 1：开箱即用 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防止失控重复
)
```

### 步骤 2：分块长音频

```python
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

### 步骤 3：LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
```

## 陷阱

- **静默时的幻觉文本。** Whisper 训练的标题包括"Thanks for watching!"、"Subscribe!"。总是先 VAD 门控。
- **`condition_on_previous_text` 级联。** 一个幻觉污染后续窗口。除非需要跨块流畅性，设为 `False`。
- **短片段补零。** 补零到30秒的2秒片段会在尾部静默中幻觉。设置 `pad=False` 或 VAD 门控。
- **错误的 mel 统计。** 用 librosa 的 mels 而非 Whisper 的产生接近随机的输出。用 `whisper.audio.log_mel_spectrogram`。

## 产出

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper 微调或推理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它标记 Whisper 风格提示，计算解码形状预算，打印10分钟片段的块调度。
2. **中等。** 安装 `faster-whisper`，转录10分钟播客，与人工转录比较 WER。尝试 `language="auto"` vs 强制 `language="en"`。
3. **困难。** 用 HF `datasets`，选 Whisper 表现差的语言（如乌尔都语），在2小时上微调 Medium 加 LoRA 2个 epoch，报告 WER 差值。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|-----------------|-----------------------|
| 30秒窗口 | Whisper 的限制 | 硬输入上限；长音频需分块。 |
| SOT | 转录开始 | `<|startoftranscript|>` 启动 decoder 提示。 |
| 时间戳 token | 时序对齐 | 每 0.02s 偏移是51k词汇表中的一个特殊 token。 |
| Turbo | 快速变体 | 4层 decoder，8×更快，<1% WER 回归。 |
| WhisperX | 长格式包装 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效调优 | 在注意层加低秩适配器；训练约0.3%的参数。 |
| Hallucination | 静默失败 | Whisper 从噪声/静默中产生流畅英语。 |

## 扩展阅读

- [Radford et al. (2022). Whisper 论文](https://arxiv.org/abs/2212.04356) — 原始架构和训练配方。
- [OpenAI (2024). Whisper Large-v3-turbo 发布](https://github.com/openai/whisper/discussions/2363) — 4层 decoder，8×加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长格式、词对齐、说话人分离。
- [Systran — faster-whisper 仓库](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 后端，4×更快。
- [HuggingFace — Whisper 微调教程](https://huggingface.co/blog/fine-tune-whisper) — 标准 LoRA / 全量 FT 演练。