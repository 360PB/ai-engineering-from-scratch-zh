---
name: prompt-tensor-debugger
description: Step-by-step debugging prompt for tensor shape errors in deep learning code
phase: 1
lesson: 12
---

我的深度学习代码中有张量形状错误。请帮我修复它。

**错误信息：** [在此粘贴错误]

**我的张量形状：**
- [名称]: [形状]
- [名称]: [形状]

**我尝试执行的操作：** [描述它]

---

调试时，遵循此确切流程：

**第一步：识别操作类型。**
是什么操作产生了错误？将其映射到以下之一：
- 矩阵乘法 / Linear 层（内维度必须匹配）
- 广播（从右对齐，每个维度必须相等或为 1）
- 连接（除 cat 维度外所有维度匹配）
- 卷积（期望特定秩和通道位置）
- Reshape（总元素数必须保持）
- 注意力（检查 batch、heads、seq_len、head_dim）

**第二步：写出形状契约。**
对于识别的操作，显式写出期望的形状：
```
matmul(A, B): A 是 (..., m, k), B 是 (..., k, n) -> (..., m, n)
broadcast(A, B): 从右对齐，每对必须 (相等) 或 (一个为 1)
cat([A, B], dim=d): 除 dim d 外所有维度匹配
Linear(in_f, out_f): 输入最后维度必须等于 in_f
Conv2d(in_c, out_c, k): 输入必须为 (B, in_c, H, W)
```

**第三步：找到不匹配。**
将实际形状与契约进行比较。识别违反规则的精确维度。

**第四步：选择最小修复。**
从下表中选择：

| 症状 | 修复 |
|---|---|
| 缺少 batch 维度 | `.unsqueeze(0)` |
| 缺少通道维度 | `.unsqueeze(1)` |
| 多余的大小为 1 的维度 | `.squeeze(dim)` |
| matmul 内维度错误 | `.transpose(-1, -2)` 或检查权重形状 |
| 需要 NCHW 但得到 NHWC | `.permute(0, 2, 3, 1)` |
| 需要 NHWC 但得到 NCHW | `.permute(0, 3, 1, 2)` |
| 为 linear 展平空间维度 | `.flatten(1)` 或 `.reshape(B, -1)` |
| 分割头: (B,T,D) 到 (B,H,T,D/H) | `.reshape(B, T, H, D//H).transpose(1, 2)` |
| 合并头: (B,H,T,D/H) 到 (B,T,D) | `.transpose(1, 2).reshape(B, T, H*(D//H))` |
| 非连续张量使用 .view() | `.contiguous().view(...)` 或用 `.reshape(...)` |

**第五步：验证修复。**
显示每一步后的形状。确认在任何 reshape 期间总元素数保持。确认操作的形状契约现在得到满足。

**第六步：检查静默 bug。**
即使形状匹配，也要验证：
- 广播是否在预期轴上发生（而非意外发生）
- 归约是否在正确的维度上求和
- batch 维度（dim 0）是否在整个前向传播中保持
- 当维度顺序重要时使用了 transpose + reshape（而非仅 reshape）

将你的响应格式化为：
```
操作： [失败的操作]
期望： [形状契约]
实际： [提供了什么形状]
不匹配： [哪个维度，为什么]
修复： [确切的代码]
结果： [修复后的形状]
```