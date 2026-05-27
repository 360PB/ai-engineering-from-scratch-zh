# OCR 与文档理解

> OCR 是三阶段流水线——检测文本框、识别字符、排列布局。每个现代 OCR 系统都重新排序或合并这些阶段。

**类型:** Learn + Use
**语言:** Python
**前置要求:** Phase 4 Lesson 06 (检测), Phase 7 Lesson 02 (自注意力)
**时长:** 约 45 分钟

## 学习目标

- 梳理经典 OCR 流水线（检测 -> 识别 -> 布局）和现代端到端替代方案（Donut、Qwen-VL-OCR）
- 实现 CTC（连接时序分类）损失用于序列到序列 OCR 训练
- 使用 PaddleOCR 或 EasyOCR 进行产品级文档解析，无需训练
- 区分 OCR、布局解析和文档理解——并根据任务选择正确工具

## 问题背景

充满文字的图像无处不在：收据、发票、身份证、扫描书籍、表单、白板、标识、截图。从中提取结构化数据——不只是字符，而是"这是总金额"——是最高价值的应用视觉问题之一。

该领域分为三个技能层次：

1. **纯 OCR**：把像素变成文字。
2. **布局解析**：将 OCR 输出按区域分组（标题、正文、表格、页眉）。
3. **文档理解**：从布局中提取结构化字段（"invoice_total = $42.50"）。

每个层次都有经典方法和现代方法，从"我想要图像中的文字"到"我需要这张收据的总金额"，差距比大多数团队意识到的要大。

## 核心概念

### 经典流水线

```mermaid
flowchart LR
    IMG["图像"] --> DET["文本检测<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["词/行<br/>边界框"]
    BOX --> CROP["裁剪每个区域"]
    CROP --> REC["识别<br/>(CRNN + CTC)"]
    REC --> TXT["文本串"]
    TXT --> LAY["布局<br/>排序"]
    LAY --> OUT["阅读顺序文本"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测** 产出每行或每词的四边形。
- **识别** 将每个区域裁剪为固定高度，CNN + BiLSTM + CTC 产出字符序列。
- **布局** 重建阅读顺序（拉丁文从左到右、从上到下；阿拉伯文、日文不同）。

### CTC 一段话

OCR 识别从固定长度特征图产出变长序列。CTC（Graves et al., 2006）让你无需字符级对齐就能训练。模型在每个时间步输出（词表 + blank）的分布；CTC 损失对所有能归约为目标文本的对齐（合并重复、去掉 blank）边缘化。

```
原始输出: "h h h _ _ e e l l _ l l o _ _"
合并重复 + 去掉 blank 后: "hello"
```

CTC 是 2015 年 CRNN 起作用的原因，也是 2026 年大多数产品 OCR 模型仍在用的训练方式。

### 现代端到端模型

- **Donut**（Kim et al., 2022）—— ViT 编码器 + 文本解码器；读入图像直接输出 JSON。无文本检测器，无布局模块。
- **TrOCR** —— ViT + Transformer 解码器，用于行级 OCR。
- **Qwen-VL-OCR / InternVL** —— 完整视觉-语言模型，在 OCR 任务上微调；2026 年复杂文档最佳准确率。
- **PaddleOCR** —— 经典 DB + CRNN 流水线，打包成熟；仍是开源扛把子。

端到端模型需要更多数据和算力，但避免了多阶段流水线中的错误累积。

### 布局解析

对结构化文档，运行布局检测器（LayoutLMv3、DocLayNet）标注每个区域：Title、Paragraph、Figure、Table、Footnote。阅读顺序就是"按布局顺序遍历区域，拼接"。

对表单，使用**键值提取**模型（视觉丰富文档用 Donut，纯扫描用 LayoutLMv3）。它们接受图像 + 检测到的文本 + 位置，预测结构化键值对。

### 评估指标

- **字符错误率（CER）** —— Levenshtein 距离 / 参考长度。越低越好。生产目标：干净扫描 < 2%。
- **词错误率（WER）** —— 词级同上。
- **结构字段 F1** —— 键值任务；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON 编辑距离** —— 端到端文档解析；Donut 论文引入归一化树编辑距离。

## 动手实现

### 步骤 1：CTC 损失 + 贪婪解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) 词表（含 blank 在 index 0）上的 log-softmax
    targets:        (N, S) int 目标（无 blank）
    input_lengths:  (N,) 每样本的时间步数
    target_lengths: (N,) 每样本的目标长度
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: index 序列列表（已去 blank，已合并重复）
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在有 CuDNN 时使用高效实现。贪婪解码器比 Beam Search 简单，通常只差 1% CER。

### 步骤 2：微型 CRNN 识别器

CNN + BiLSTM 的行级 OCR 最小实现。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入（CNN 将高度池化为 1）。宽度是 CTC 的时间维度。

### 步骤 3：合成 OCR 数据

生成黑底白字数字串做端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实 OCR 数据集增加字体、噪声、旋转、模糊和颜色。流水线完全相同。

### 步骤 4：训练草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个简单合成数据上，200 步内损失应从约 3 降到约 0.2。

## 用现成库

三条生产路径：

- **PaddleOCR** —— 成熟、快速、多语言。一行调用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** —— Python 原生、多语言、PyTorch 主干。
- **Tesseract** —— 经典；在模型困难的旧扫描文档上仍有用了。

端到端文档解析用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

有可重复结构的收据、发票、表单，微调 Donut。需要任意文档或有推理的 OCR，用 Qwen-VL-OCR 这个 2026 年当前默认值。

## 产出

本课产出：

- `outputs/prompt-ocr-stack-picker.md` —— 给定文档类型、语言和结构，选择 Tesseract / PaddleOCR / Donut / VLM-OCR 的 prompt。
- `outputs/skill-ctc-decoder.md` —— 从零写贪婪和 Beam Search CTC 解码器的 skill，含长度归一化。

## 练习

1. **(简单)** 在 5 位随机数字串上训练 TinyCRNN 500 步。报告在留出集上的 CER。
2. **(中等)** 将贪婪解码替换为 Beam Search（beam_width=5）。报告 CER 变化。Beam Search 在哪些输入上胜出？
3. **(困难)** 在 20 张收据上用 PaddleOCR，提取行项目，对 `{item_name, price}` 对手标真值计算 F1。

## 关键术语

| 英文 | 中文 | 实际含义 |
|------|------|---------|
| OCR | OCR | 将图像区域转为字符序列 |
| CTC | 连接时序分类 | 无需逐时间步标签的序列模型训练损失；边缘化所有对齐方式 |
| CRNN | CRNN | 经典 OCR 模型：卷积特征提取器 + BiLSTM + CTC；2015 年基准，生产仍在用 |
| Donut | Donut | 端到端 OCR：ViT 编码器 + 文本解码器；图像直出 JSON |
| Layout parsing | 布局解析 | 在文档中检测并标注 Title/Table/Figure/Paragraph 区域 |
| Reading order | 阅读顺序 | 将识别出的区域排序成句子；拉丁文简单，混合布局复杂 |
| CER / WER | 字符/词错误率 | 以 Levenshtein 距离 / 参考长度计算，在字符或词粒度上 |
| VLM-OCR | VLM-OCR | 为 OCR 任务训练或 prompting 的视觉-语言模型；复杂文档当前 SOTA |

## 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) —— CNN+RNN+CTC 原始架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— 原始 CTC 论文；算法思想密度极大
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) —— 无 OCR 文档理解 Transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) —— 开源产品级 OCR 栈