# StyleGAN

> 大多数生成器将 `z` 同时注入每一层。StyleGAN 将其分开：首先将 `z` 映射到中间 `w`，然后通过 AdaIN 在每个分辨率级别*注入* `w`。这单一改变解开了潜在空间，使逼真人脸在七年内成为已解决的问题。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 03（GAN），Phase 4 · 08（归一化），Phase 3 · 07（CNN）
**时间：** 约 45 分钟

## 问题

DCGAN 通过一堆转置卷积将 `z` 映射到图像。问题是：`z` 同时控制一切——姿态、光照、身份、背景——纠缠在一起。沿着 `z` 的一个轴移动，所有四个都改变。你不能要求模型"同一个人，不同姿态"，因为表示不支持那种因式分解。

Karras 等人（2019，NVIDIA）提出：不要将 `z` 直接馈送到卷积层。馈入一个学习的常量 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP 将 `z ∈ Z → w ∈ W`。通过*自适应实例归一化*（AdaIN）在每个分辨率注入 `w`：归一化每个卷积特征图，然后按 `w` 的仿射投影缩放和移位。为每个特征图添加逐层噪声以控制随机细节（皮肤毛孔、发丝）。

结果：`W` 空间对"高级风格"（姿态、身份）和"细粒度风格"（光照、颜色）有近似正交的轴。你可以通过对低分辨率级别使用图像 A 的 `w`，对高分辨率级别使用图像 B 的 `w` 来交换两个图像之间的风格。这解锁了编辑、跨域风格化，以及整个"StyleGAN 反演"研究线。

## 核心概念

![StyleGAN：映射网络 + AdaIN + 逐层噪声](../assets/stylegan.svg)

**映射网络。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制为高斯分布——它学习数据自适应的形状。

**合成网络。** 从一个学习的常量 `4×4×512` 开始。每个分辨率块：`上采样 → 卷积 → AdaIN(w_i) → 噪声 → 卷积 → AdaIN(w_i) → 噪声`。分辨率翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。按每个特征图归一化，然后重新风格化。这里的"风格"是特征图的一阶和二阶统计量。

**逐层噪声。** 添加到每个特征图的单通道高斯噪声，按学习的每通道因子缩放。控制随机细节而不影响全局结构。

**截断技巧。** 在推理时，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是许多样本上 `w` 的均值。`ψ < 1` 以多样性换取质量。几乎每个 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新 |
|------|------|------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进式增长。 |
| StyleGAN2 | 2020 | 权重解调替换 AdaIN（修复液滴伪影）；skip/residual 架构；路径长度正则化。 |
| StyleGAN3 | 2021 | 无别名卷积 + 等变核；消除纹理粘附在像素网格上。 |
| StyleGAN-XL | 2022 | 类别条件，1024²，ImageNet。 |
| R3GAN | 2024 | 以更强正则化重新命名；用 20 倍更少参数在 FFHQ-1024 上缩小与扩散的差距。 |

在 2026 年，StyleGAN3 仍然是（a）高 FPS 窄领域逼真感的默认选择，（b）少样本领域适应（用 100 张图像在新数据集上训练，冻结映射），（c）基于反演的编辑（找到重建真实照片的 `w`，然后编辑该 `w`）的默认选择。对于开放领域文生图，它不是工具——扩散才是。

## 构建

`code/main.py` 在 1-D 中实现了一个玩具"style-GAN lite"：一个映射 MLP、一个合成函数（接收学习的常量向量并用 `w` 衍生的缩放/偏置调制它）和逐层噪声。它表明通过仿射调制注入 `w` 匹配或优于将 `z` 连接到生成器输入。

### 第 1 步：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 第 2 步：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每个特征图的缩放和偏置通过线性投影来自 `w`。

### 第 3 步：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每个通道的 sigma 是可学习的。

## 陷阱

- **液滴伪影。** StyleGAN 1 在特征图中产生了斑驳的液滴，因为 AdaIN 将均值清零。StyleGAN 2 的权重解调通过缩放卷积权重而不是激活来修复它。
- **纹理粘附。** StyleGAN 1 和 2 的纹理跟随像素坐标，而不是对象坐标（插值时可见）。StyleGAN 3 的无别名卷积用加窗 sinc 滤波器修复了这个问题。
- **模式覆盖。** 截断 `ψ < 0.7` 看起来干净但从狭窄的锥中采样；如果需要多样性，使用 `ψ = 1.0`。
- **反演有损。** 将真实照片反演到 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。多次迭代后结果会漂移。

## 使用

| 用例 | 方法 |
|------|------|
| 逼真人脸（动漫、产品、窄领域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片编辑人脸 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 换脸 / 重演 | StyleGAN + 编码器 + 混合 |
| Avatar pipeline | 带 ADA 的 StyleGAN3 用于少数据微调 |
| 从少量图像进行领域适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 不要——使用扩散 |

对于答案是"一个人的照片"的产品级演示，StyleGAN 在推理成本（单次前向传播，4090 上 <10ms）和相同质量栏下锐度方面击败扩散。

## 发布

保存为 `outputs/skill-stylegan-inversion.md`。Skill 接收一张真实照片并输出：反演方法（e4e / ReStyle / HyperStyle）、预期潜在损失、编辑预算（在伪影出现之前可以在 `W` 中移动多远）、以及已知良好编辑方向列表（年龄、表情、姿态）。

## 练习

1. **简单。** 用 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较固定潜在 vs 扰动潜在时输出的分布范围。
2. **中等。** 实现混合正则化：对于一个训练 batch，计算 `w_a`、`w_b`，并将 `w_a` 应用于合成的前半部分，`w_b` 应用于后半部分。解码器是否学会了 disentangled 风格？
3. **困难。** 加载预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在带标签样本上训练 SVM 找到控制"微笑"的 `w` 方向；报告在身份漂移之前可以推多远。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 映射网络 | "MLP" | `f: Z → W`，8 层，将潜在几何与数据统计解耦。 |
| W 空间 | "风格空间" | 映射网络的输出；近似 disentangled。 |
| AdaIN | "自适应实例归一化" | 归一化特征图，然后按 `w` 投影缩放 + 移位。 |
| 截断技巧 | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 以多样性换取质量。 |
| 路径长度正则化 | "PL reg" | 惩罚 `w` 单位变化时图像的大变化；使 `W` 更平滑。 |
| 权重解调 | "StyleGAN2 修复" | 归一化卷积权重而不是激活；消灭液滴伪影。 |
| 无别名 | "StyleGAN3 的技巧" | 加窗 sinc 滤波器；消除纹理粘附在像素网格上。 |
| 反演 | "为真实图像找到 w" | 优化或编码 `x → w` 使得 `G(w) ≈ x`。 |

## 生产笔记：为什么 StyleGAN 在 2026 年仍在部署

StyleGAN3 在 4090 上生成 1024² FFHQ 人脸在 10 ms 以内——`num_steps = 1`，无 VAE 解码，无交叉注意力传递。在生产术语中，这是任何图像生成器的最低延迟。50 步 SDXL + VAE 解码 pipeline 在相同分辨率下约 3 秒。**300 倍差距**，对于窄领域产品（头像服务、身份证文档 pipeline、库存人脸生成），这在 TCO 上胜出。

两个操作后果：

- **无调度器，无批处理器。** 静态 batch 在目标占用率下是最优的。连续批处理（对 LLM 和扩散必不可少）对 GAN 零收益，因为每个请求花费相同的 FLOPs。
- **截断 `ψ` 是安全旋钮。** `ψ < 0.7` 从映射网络范围的狭窄锥中采样。这是服务层控制样本方差的唯一杠杆。在高峰负载时降低 `ψ`，为高级用户提高 `ψ`。

## 进一步阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代最小 GAN 配方。