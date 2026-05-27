# 图像生成 — GANs

> GAN 是两个神经网络之间的一场固定博弈。一个负责画画，一个负责批评。它们共同变强，直到画作骗过评审。

**类型:** Build
**语言:** Python
**前置要求:** Phase 4 Lesson 03 (CNNs), Phase 3 Lesson 06 (优化器), Phase 3 Lesson 07 (正则化)
**时长:** 约 75 分钟

## 学习目标

- 解释生成器和判别器之间的最小最大博弈，以及为什么均衡对应于 p_model = p_data
- 在 PyTorch 中实现 DCGAN，并在 60 行以内让它生成连贯的 32x32 合成图像
- 使用三个标准技巧稳定 GAN 训练：非饱和损失、谱归一化、TTUR（双时间尺度更新规则）
- 读取训练曲线，区分健康收敛与模式崩溃、振荡和判别器完全胜出

## 问题背景

分类教网络将图像映射到标签。生成则将问题反转：从同一分布中采样新的图像。没有"正确"输出可以 diff；只有一个你想模拟的分布。

标准损失函数（MSE、交叉熵）无法衡量"这个样本是否来自真实分布"。最小化逐像素误差产生模糊的平均值，而不是逼真的样本。突破来自于学习损失：训练第二个网络，其工作是区分真假，并用其判断来推动生成器。

GANs（Goodfellow et al., 2014）定义了这个框架。到 2018 年，StyleGAN 生成了 1024x1024 的人脸，与照片无法区分。此后扩散模型在质量和可控性方面夺得了王座，但使扩散模型变得实用的每一个技巧——归一化选择、潜空间、特征损失——都是在 GANs 上首先理解的。

## 核心概念

### 两个网络

```mermaid
flowchart LR
    Z["z ~ N(0, I)<br/>噪声"] --> G["生成器<br/>转置卷积"]
    G --> FAKE["假图像"]
    REAL["真图像"] --> D["判别器<br/>卷积分类器"]
    FAKE --> D
    D --> OUT["P(真)"]

    style G fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

**生成器** G 接受一个噪声向量 `z` 并输出一张图像。**判别器** D 接受一张图像并输出一个标量：该图像为真的概率。

### 这场博弈

G 想要 D 出错。D 想要正确。从形式上看：

```
min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

从右向左读：D 在真图像（`log D(real)`）和假图像（`log (1 - D(fake))`）上最大化准确率。G 最小化 D 对假图像的准确率——它希望 `D(G(z))` 很高。

Goodfellow 证明了这个最小最大问题有一个全局均衡，其中 `p_G = p_data`，D 处处输出 0.5，生成分布与真实分布之间的 Jensen-Shannon 散度为零。难点在于如何达到均衡。

### 非饱和损失

上述形式在数值上不稳定。在训练早期，`D(G(z))` 对每个假图像都接近零，因此 `log(1 - D(G(z)))` 对 G 的梯度接近于零。修复方法：翻转 G 的损失。

```
L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
L_G = -E_z[log D(G(z))]                          # 非饱和版本
```

现在当 `D(G(z))` 接近零时，G 的损失很大且其梯度信息丰富。每个现代 GAN 都使用这个变体。

### DCGAN 架构规则

Radford、Metz、Chintala（2015）将多年的失败实验提炼成五条规则，使 GAN 训练变得稳定：

1. 用步幅卷积替换池化（两个网络都用）。
2. 在生成器和判别器中都使用批归一化，但 G 的输出层和 D 的输入层除外。
3. 在更深的架构中移除全连接层。
4. G 在所有层使用 ReLU，输出层使用 tanh（输出范围 [-1, 1]）。
5. D 在所有层使用 LeakyReLU（negative_slope=0.2）。

每个现代基于卷积的 GAN（StyleGAN、BigGAN、GigaGAN）仍然从这些规则出发，然后一次替换一个组件。

### 失败模式及其特征

```mermaid
flowchart LR
    M1["模式崩溃<br/>G 产生一组<br/>窄输出"] --> S1["D 损失低，<br/>G 损失振荡，<br/>样本多样性下降"]
    M2["梯度消失<br/>D 完全胜出"] --> S2["D 准确率 ~100%，<br/>G 损失巨大且静止"]
    M3["振荡<br/>G 和 D 不断<br/>交替胜出"] --> S3["两个损失都<br/>剧烈波动无下降趋势"]

    style M1 fill:#fecaca,stroke:#dc2626
    style M2 fill:#fecaca,stroke:#dc2626
    style M3 fill:#fecaca,stroke:#dc2626
```

- **模式崩溃**：G 找到一张能骗过 D 的图像，然后只生成那一张。修复：添加小批量判别、谱归一化或标签条件化。
- **判别器胜出**：D 变得太强太快，G 的梯度消失。修复：更小的 D、更低的 D 学习率，或对真实标签应用标签平滑。
- **振荡**：两个网络不断交替胜出，从未接近均衡。修复：TTUR（D 比 G 学得快 2-4 倍），或切换到 Wasserstein 损失。

### 评估

GANs 没有 ground truth，你怎么知道它们在工作？

- **样本检查**——每个 epoch 结束时直接看 64 个样本。这是必须的。
- **FID（Fréchet Inception Distance）**——真实集和生成集的 Inception-v3 特征分布之间的度量。越低越好。社区标准。
- **Inception Score**——更老，更脆弱；优先用 FID。
- **生成模型的精确率/召回率**——分别衡量质量（精确率）和覆盖度（召回率）。比单独用 FID 信息更丰富。

对于小型合成数据运行，样本检查就够了。

## 构建过程

### 步骤 1：生成器

一个小型 DCGAN 生成器，接受 64 维噪声，生成 32x32 图像。

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

四个转置卷积，每个都用 `kernel_size=4, stride=2, padding=1`，因此空间尺寸干净地翻倍。输出激活通过 tanh 映射到 [-1, 1]。

### 步骤 2：判别器

生成器的镜像。LeakyReLU，步幅卷积，以标量 logit 结束。

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 2, feat * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x).view(-1)
```

最后一个卷积将 `4x4` 特征图缩减为 `1x1`。每张图像输出一个标量；在损失计算时才应用 sigmoid。

### 步骤 3：训练步

交替：每批次先更新 D 一次，再更新 G 一次。

```python
import torch.nn.functional as F

def train_step(G, D, real, z, opt_g, opt_d, device):
    real = real.to(device)
    bs = real.size(0)

    # D 步
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
    loss_d.backward()
    opt_d.step()

    # G 步
    opt_g.zero_grad()
    d_fake = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()
```

D 步中的 `G(z).detach()` 至关重要：你不希望梯度在 G 的更新期间流入 G。忘记这一点是经典的新手错误。

### 步骤 4：在合成形状数据集上的完整训练循环

```python
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def synthetic_images(num=2000, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((num, 3, size, size), dtype=np.float32) - 1.0
    for i in range(num):
        r = rng.uniform(6, 12)
        cx, cy = rng.uniform(r, size - r, size=2)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.5, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = synthetic_images()
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

G = Generator(z_dim=64, img_channels=3, feat=32).to(device)
D = Discriminator(img_channels=3, feat=32).to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for (batch,) in loader:
        z = torch.randn(batch.size(0), 64, device=device)
        ld, lg = train_step(G, D, batch, z, opt_g, opt_d, device)
    print(f"epoch {epoch}  D {ld:.3f}  G {lg:.3f}")
```

`Adam(lr=2e-4, betas=(0.5, 0.999))` 是 DCGAN 默认配置——低的 beta1 防止动量项将对抗博弈稳定得太快。

### 步骤 5：采样

```python
@torch.no_grad()
def sample(G, n=16, z_dim=64, device="cpu"):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1)
```

采样前一定要切换到评估模式。对于 DCGAN 这很重要，因为用的是批归一化的运行统计量而非当前批的统计量。

### 步骤 6：谱归一化

替换判别器中 BN 的直接替代方案，保证网络是 1-Lipschitz。修复大多数"D 赢得太狠"的失败。

```python
from torch.nn.utils import spectral_norm

def build_sn_discriminator(img_channels=3, feat=64):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(img_channels, feat, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat, feat * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 4, 1, 4, 1, 0)),
    )
```

将 `Discriminator` 替换为 `build_sn_discriminator()`，通常就不需要 TTUR 技巧了。谱归一化是你可以应用的最简单的单一鲁棒性升级。

## 应用

对于严肃的生成任务，使用预训练权重或转向扩散。两个标准库：

- `torch_fidelity` 在你的生成器上计算 FID / IS，无需编写自定义评估代码。
- `pytorch-gan-zoo`（legacy）和 `StudioGAN` 提供了经过测试的 DCGAN、WGAN-GP、SN-GAN、StyleGAN 和 BigGAN 实现。

在 2026 年，GANs 仍然是以下任务的最佳选择：实时图像生成（延迟 <10 ms）、风格迁移、需要精确控制的图像到图像翻译（Pix2Pix、CycleGAN）。扩散模型在逼真度和文本条件化方面胜出。

## 交付物

本课产出：

- `outputs/prompt-gan-training-triage.md`——一个提示词，读取训练曲线描述，选择失败模式（模式崩溃、D 胜出、振荡），并给出单一推荐的修复方法。
- `outputs/skill-dcgan-scaffold.md`——一个技能，从 `z_dim`、目标 `image_size` 和 `num_channels` 编写 DCGAN 脚手架，包含训练循环和样本保存器。

## 练习

1. **(简单)** 在合成圆数据集上训练上面的 DCGAN，并在每个 epoch 结束时保存 16 个样本的网格。到了第几个 epoch，生成的圆明显是圆的？
2. **(中等)** 将判别器的批归一化替换为谱归一化。并行训练两个版本。哪个收敛更快？哪个在三个随机种子下方差更低？
3. **(困难)** 实现条件 DCGAN：将类别标签输入 G 和 D（在 G 中将 one-hot 与噪声拼接，在 D 中拼接类别嵌入通道）。在 lesson 7 的合成"圆 vs 方"数据集上训练，并通过使用特定标签采样来展示类别条件化有效。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 生成器 (G) | "画东西的网络" | 将噪声映射到图像；训练目标是骗过判别器 |
| 判别器 (D) | "评审" | 二分类器；训练目标是区分真实图像和生成图像 |
| 最小最大 | "这场博弈" | 在对抗损失上对 G 取最小、对 D 取最大；均衡时 p_G = p_data |
| 非饱和损失 | "数值上合理的版本" | G 的损失是 -log(D(G(z))) 而非 log(1 - D(G(z)))，避免训练早期的梯度消失 |
| 模式崩溃 | "生成器只生成一样东西" | G 只产生数据分布的一个小子集；用 SN、小批量判别或更大批次修复 |
| TTUR | "两个学习率" | D 比 G 学得更快，通常快 2-4 倍；稳定训练 |
| 谱归一化 | "1-Lipschitz 层" | 限制每个层的 Lipschitz 常数的权重归一化；阻止 D 变得任意陡峭 |
| FID | "Fréchet Inception Distance" | 真实集和生成集的 Inception-v3 特征分布之间的距离；标准评估指标 |

## 延伸阅读

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)——开创一切的论文
- [DCGAN (Radford, Metz, Chintala, 2015)](https://arxiv.org/abs/1511.06434)——使 GAN 可训练的架构规则
- [Spectral Normalization for GANs (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957)——最单一、有用的稳定化技巧
- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423)——SOTA GAN；读起来像过去十年所有技巧的集大成之作