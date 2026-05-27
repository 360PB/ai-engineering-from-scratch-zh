# 3D 视觉 — 点云与 NeRF

> 3D 视觉有两种形态。点云是传感器的原始输出。NeRF 是学习得到的体素场。两者都回答"空间中有什么，在哪里"。

**类型:** Learn + Build
**语言:** Python
**前置要求:** Phase 4 Lesson 03 (CNNs), Phase 1 Lesson 12 (张量运算)
**时长:** 约 45 分钟

## 学习目标

- 区分显式（点云、网格、体素）和隐式（符号距离场、NeRF）3D 表示，以及各自适用的场景
- 理解 PointNet 的对称函数技巧，使神经网络对无序点集具有排列不变性
- 追踪 NeRF 前向传播：光线投射、体绘制、位置编码、MLP 密度+颜色头
- 使用 `nerfstudio` 或 `instant-ngp` 从少量带位姿的图像中进行预训练 3D 重建

## 问题背景

相机产生 2D 图像。激光雷达产生一组无排序的 3D 点。从运动恢复结构（SfM）管线产生稀疏的 3D 关键点云。NeRF 从少量带位姿的图像中重建完整 3D 场景。这些都是"视觉"，但没有一个看起来像 CNN 想要的那种密集张量。

3D 视觉之所以重要，是因为几乎所有高价值机器人任务都在 3D 空间中进行：抓取、障碍物规避、导航、增强现实遮挡、3D 内容捕获。只懂 2D 图像的视觉工程师被这个领域增长最快的部分（AR/VR 内容、机器人技术、自动驾驶堆栈、用于房地产或建筑的基于 NeRF 的 3D 重建）拒之门外。

两种表示因不同原因各自主导。点云是传感器免费给你的。NeRF 及其后继者（3D 高斯溅射、神经 SDF）是让神经网络学习场景时得到的东西。

## 核心概念

### 点云

点云是 R^3 中 N 个点的无序集合，每个点可选带有特征（颜色、强度、法向量）。

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

没有网格，没有连通性。两个特性使神经网络处理起来困难：

- **排列不变性**——输出不能依赖于点的顺序。
- **可变 N**——单一模型必须处理不同大小的点云。

PointNet（Qi et al., 2017）用一个想法同时解决了这两个问题：对每个点应用共享 MLP，然后用对称函数（最大池化）聚合。结果是一个固定大小的向量，与顺序无关。

```
f(P) = max_{p in P} MLP(p)
```

这就是 PointNet 的全部核心。更深的变体（PointNet++、Point Transformer）添加了分层采样和局部聚合，但对称函数技巧没有改变。

### PointNet 架构

```mermaid
flowchart LR
    PTS["N 个点<br/>(x, y, z)"] --> MLP1["共享 MLP<br/>(64, 64)"]
    MLP1 --> MLP2["共享 MLP<br/>(64, 128, 1024)"]
    MLP2 --> MAX["最大池化<br/>(对称)"]
    MAX --> FEAT["全局特征<br/>(1024,)"]
    FEAT --> FC["MLP 分类器"]
    FC --> CLS["类别 logits"]

    style MLP1 fill:#dbeafe,stroke:#2563eb
    style MAX fill:#fef3c7,stroke:#d97706
    style CLS fill:#dcfce7,stroke:#16a34a
```

"共享 MLP"意味着相同的 MLP 在每个点上独立运行。为了效率，实现为对点维度的 1x1 卷积。

### 神经辐射场（NeRF）

NeRF（Mildenhall et al., 2020）提出了一个问题："能否从 N 张照片重建 3D 场景？"并用一个作为场景的神经网络来回答。该网络将 `(x, y, z, viewing_direction)` 映射到 `(density, colour)`。渲染新视角是通过该网络的光线投射循环。

```
NeRF MLP:  (x, y, z, theta, phi) -> (sigma, r, g, b)

渲染新视角的像素 (u, v)：
  1. 从相机穿过像素 (u, v) 投射一条光线
  2. 在距离 t_1, t_2, ..., t_N 处沿光线采样点
  3. 在每个点查询 MLP
  4. 按 (1 - exp(-sigma * dt)) 加权合成颜色
  5. 和就是渲染的像素颜色
```

损失比较渲染像素与训练照片中真实像素。沿渲染步骤反向传播更新 MLP。没有 3D ground truth，没有显式几何——场景存储在 MLP 权重中。

### NeRF 中的位置编码

对 `(x, y, z)` 的普通 MLP 无法表示高频细节，因为 MLP 在频谱上偏向低频。NeRF 通过在 MLP 之前将每个坐标编码为傅里叶特征向量来修复这个问题：

```
gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), sin(2^1 pi p), cos(2^1 pi p), ...)
```

最多 L=10 个频率级别。这与 Transformer 用于位置的技巧相同，在扩散时间条件化中再次出现（Lesson 10）。没有它，NeRF 看起来会很模糊。

### 体绘制

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i

T_i  = exp(- sum_{j<i} sigma_j * delta_j)
delta_i = t_{i+1} - t_i
```

`T_i` 是透射率——有多少光到达点 i 仍然存活。`(1 - exp(-sigma_i * delta_i))` 是点 i 处的不透明度。`c_i` 是颜色。最终像素是沿光线的加权和。

### 什么取代了 NeRF

纯 NeRF 训练慢（小时级），渲染慢（每张图像秒级）。之后的演进：

- **Instant-NGP**（2022）——用哈希网格编码替换 MLP 的位置输入；秒级训练。
- **Mip-NeRF 360**——处理无界场景和抗锯齿。
- **3D 高斯溅射**（2023）——用数百万个 3D 高斯替换体素场；分钟级训练，实时渲染。当前的生产默认。

2026 年几乎每个真实的 NeRF 产品实际上都是 3D 高斯溅射。但心智模型仍然是 NeRF。

### 数据集和基准测试

- **ShapeNet**——点云格式的 3D CAD 模型分类和分割。
- **ScanNet**——室内真实扫描用于分割。
- **KITTI**——用于自动驾驶的户外激光雷达点云。
- **NeRF Synthetic / Blended MVS**——用于视图合成的带位姿图像数据集。
- **Mip-NeRF 360 数据集**——无界真实场景。

## 构建过程

### 步骤 1：PointNet 分类器

```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (N, 3, num_points) — 转置后用于 Conv1d
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=-1)[0]       # (N, 1024)
        return self.head(x)

pts = torch.randn(4, 3, 1024)
net = PointNet(num_classes=10)
print(f"output: {net(pts).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

约 1.6M 参数。每点云运行在 1,024 个点上。

### 步骤 2：位置编码

```python
def positional_encoding(x, L=10):
    """
    x: (..., D) -> (..., D * 2 * L)
    """
    freqs = 2.0 ** torch.arange(L, dtype=x.dtype, device=x.device)
    args = x.unsqueeze(-1) * freqs * 3.141592653589793
    sinc = torch.cat([args.sin(), args.cos()], dim=-1)
    return sinc.reshape(*x.shape[:-1], -1)

x = torch.randn(5, 3)
y = positional_encoding(x, L=10)
print(f"input:  {x.shape}")
print(f"encoded: {y.shape}     # (5, 60)")
```

乘以 `2^l * pi` 得到逐步更高的频率。

### 步骤 3：小型 NeRF MLP

```python
class TinyNeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden=128):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        pos_dim = 3 * 2 * L_pos
        dir_dim = 3 * 2 * L_dir
        self.trunk = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
        )
        self.sigma = nn.Linear(hidden, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        x_enc = positional_encoding(x, self.L_pos)
        d_enc = positional_encoding(d, self.L_dir)
        h = self.trunk(x_enc)
        sigma = torch.relu(self.sigma(h)).squeeze(-1)
        rgb = self.color(torch.cat([h, d_enc], dim=-1))
        return sigma, rgb

nerf = TinyNeRF()
x = torch.randn(128, 3)
d = torch.randn(128, 3)
s, c = nerf(x, d)
print(f"sigma: {s.shape}   rgb: {c.shape}")
```

与原始 NeRF（有两个深度为 8 的 MLP 主干）相比是微型的。足以展示架构。

### 步骤 4：沿光线的体绘制

```python
def volumetric_render(sigma, rgb, t_vals):
    """
    sigma: (..., N_samples)
    rgb:   (..., N_samples, 3)
    t_vals: (N_samples,) 沿光线的距离
    """
    delta = torch.cat([t_vals[1:] - t_vals[:-1], torch.full_like(t_vals[:1], 1e10)])
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans
    rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    return rendered, depth, weights


N = 64
t_vals = torch.linspace(2.0, 6.0, N)
sigma = torch.rand(N) * 0.5
rgb = torch.rand(N, 3)
rendered, depth, weights = volumetric_render(sigma, rgb, t_vals)
print(f"rendered colour: {rendered.tolist()}")
print(f"depth:           {depth.item():.2f}")
```

一条光线，64 个采样点，合成一个 RGB 像素和深度。

## 应用

真实工作：

- `nerfstudio`（Tancik et al.）——当前 NeRF / Instant-NGP / 高斯溅射的参考库。命令行加网页查看器。
- `pytorch3d`（Meta）——可微分渲染、点云工具、网格操作。
- `open3d`——点云处理、配准、可视化。

部署方面，3D 高斯溅射在很大程度上取代了纯 NeRF，因为渲染速度快 100 倍。重建质量相当。

## 交付物

本课产出：

- `outputs/prompt-3d-task-router.md`——一个提示词，基于任务和输入数据路由到正确的 3D 表示（点云、网格、体素、NeRF、高斯溅射）。
- `outputs/skill-point-cloud-loader.md`——一个技能，为 .ply / .pcd / .xyz 文件编写 PyTorch `Dataset`，包含正确的归一化、居中和点采样。

## 练习

1. **(简单)** 证明 PointNet 是排列不变的：运行相同点云两次，一次打乱点顺序。验证输出在浮点误差范围内相同。
2. **(中等)** 实现一个最小光线生成函数：给定相机内参和位姿，为 H x W 图像的每个像素生成光线起点和方向。
3. **(困难)** 在彩色立方体的渲染视图合成数据集（通过可微分渲染或简单光线追踪器生成）上训练 TinyNeRF。报告 epoch 1、10、100 时的渲染损失。模型在第几个 epoch 产生可识别的视图？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 点云 | "激光雷达的 3D 点" | 每个点的 (x, y, z) + 可选特征的无序集合 |
| PointNet | "点云上的第一个神经网络" | 每个点上的共享 MLP + 对称（最大）池化；通过构造实现排列不变性 |
| NeRF | "作为场景的 MLP" | 将 (x, y, z, dir) 映射到 (密度, 颜色) 的网络；通过光线投射渲染 |
| 位置编码 | "傅里叶特征" | 将每个坐标编码为多频率的 sin/cos，以克服 MLP 的低频偏差 |
| 体绘制 | "光线积分" | 使用透射率和 alpha 沿光线将采样合成到单个像素 |
| Instant-NGP | "哈希网格 NeRF" | 用多分辨率哈希网格替换 NeRF 的坐标 MLP；快 100-1000 倍 |
| 3D 高斯溅射 | "数百万个高斯" | 场景 = 3D 高斯集合；实时渲染，分钟级训练 |
| SDF | "符号距离场" | 返回到最近表面的带符号距离的函数；另一种隐式表示 |

## 延伸阅读

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593)——排列不变分类器
- [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934)——使从照片进行 3D 重建成为神经网络问题的论文
- [Instant-NGP (Müller et al., 2022)](https://arxiv.org/abs/2201.05989)——哈希网格，快 1000 倍
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079)——在生产中取代 NeRF 的架构