# 从零实现 3D 高斯溅射

> 一个场景是数百万个 3D 高斯的集合。每个高斯具有位置、朝向、尺度、透明度和随视角变化的颜色。栅格化它们，反向传播梯度，搞定。

**类型:** 动手实现
**语言:** Python
**前置要求:** Phase 4 Lesson 13 (3D 视觉与 NeRF)、Phase 1 Lesson 12 (张量运算)、Phase 4 Lesson 10 (扩散模型基础，可选)
**时间:** ~90 分钟

## 学习目标

- 解释为什么 3D 高斯溅射在 2026 年取代 NeRF 成为逼真 3D 重建的生产默认方案
- 说出每个高斯的 6 个参数（位置、旋转四元数、尺度、透明度、球谐颜色、可选特征）以及每个参数占用多少浮点数
- 从零实现 2D 高斯溅射光栅器（使用 alpha 合成），然后说明 3D 情况如何投影到相同的循环
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建场景，并导出为 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat` schema

## 问题

NeRF 将场景存储为 MLP 的权重。每个渲染像素都沿射线进行数百次 MLP 查询。训练需要数小时，渲染需要秒，且权重无法编辑——如果想移动场景中的椅子，必须重新训练。

3D 高斯溅射（Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023）彻底改变了这一切。场景是一组显式的 3D 高斯。渲染是 GPU 光栅化，速度超过 100 fps。训练仅需分钟。编辑是直接的：平移部分高斯即移动了椅子。截至 2026 年，Khronos Group 已批准 Gaussian splats 的 glTF 扩展，OpenUSD 26.03 包含 Gaussian splat schema，Zillow 和 Apartments.com 用它们渲染房产，且大多数 3D 重建新论文都是对核心 3DGS 思想的变体。

心智模型简单，但数学部分较复杂——大多数介绍直接从光栅化开始而跳过投影和球谐函数。本课构建完整内容——先做 2D 版本，再扩展到 3D。

## 概念

### 一个高斯携带什么

一个 3D 高斯是空间中的一个参数化斑点，具有以下属性：

```
position         mu         (3,)    世界坐标系中的中心点
rotation         q          (4,)    编码朝向的单位四元数
scale            s          (3,)    每轴的对数尺度（渲染时取指数）
opacity          alpha      (1,)    sigmoid 后的透明度 [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   视角相关颜色
```

旋转 + 尺度构建 3x3 协方差矩阵：`Sigma = R S S^T R^T`。这就是高斯在 3D 中的形状。球谐函数让颜色随视角变化——镜面高光、微妙的光泽、视角相关的发光——无需存储逐视角纹理。SH 度为 3 时，每个颜色通道有 16 个系数，仅颜色部分就有 48 个浮点数。

一个场景通常有 1-5 百万个高斯。每个存储约 60 个浮点数（3 + 4 + 3 + 1 + 48 + 其他）。即一个 500 万高斯场景约 240 MB——远小于带逐点纹理的等价点云，比高分辨率重渲染的 NeRF MLP 权重小一个数量级。

### 光栅化，而非光线行进

```mermaid
flowchart LR
    SCENE["数百万个 3D 高斯<br/>(位置、旋转、尺度、<br/>透明度、SH 颜色)"] --> PROJ["投影到 2D<br/>(相机外参 + 内参)"]
    PROJ --> TILES["分配到瓦片<br/>(16x16 屏幕空间)"]
    TILES --> SORT["每瓦片按深度排序"]
    SORT --> ALPHA["Alpha 合成<br/>(从前到后)"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五个步骤，全部 GPU 友好。每个像素不需要 MLP 查询。一块 RTX 3080 Ti 以 147 fps 渲染 600 万个 splats。

### 投影步骤

世界坐标中位置 `mu`、3D 协方差 `Sigma` 的 3D 高斯，投影到屏幕位置 `mu'`、2D 协方差 `Sigma'` 的 2D 高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 视图变换（相机旋转 + 平移）
J = mu' 处透视投影的雅可比矩阵
```

2D 高斯的足迹是 `Sigma'` 特征向量定义的椭圆。椭圆内每个像素接收高斯的贡献，加权系数为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### Alpha 合成规则

覆盖一个像素的高斯按从后到前排序（或等效地从前到后使用反转型公式）。颜色使用与 1980 年代以来每个半透明光栅器相同的方程合成：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       到 i 处的透射率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局部贡献
c_i = eval_SH(SH_i, view_direction)    视角相关颜色
```

这就是 **NeRF 体渲染的同一公式**，只是现在在显式稀疏高斯集上而非沿射线密集采样。正是这个恒等式让渲染质量与 NeRF 匹配——两者都在积分相同的辐射场方程。

### 为什么可微分

每个步骤——投影、瓦片分配、alpha 合成、SH 求值——对高斯参数都是可微的。给定真值图像，计算渲染像素损失，通过光栅器反向传播，用梯度下降更新所有 `(mu, q, s, alpha, c_lm)`。经过约 30,000 次迭代，高斯找到正确的位置、尺度和颜色。

### 致密化与剪枝

固定的高斯集无法覆盖复杂场景。训练包括两个自适应机制：

- **克隆**：当高斯的梯度幅度高但尺度小时，在其当前位置克隆一个高斯——重构在这里需要更多细节。
- **分裂**：当大尺度高斯的梯度高时，分裂为两个较小的——一个大高斯太平滑，无法拟合该区域。
- **剪枝**：剪掉透明度低于阈值的高斯——它们没有贡献。

致密化每 N 次迭代运行一次。场景通常从约 10 万初始高斯（SfM 点播种）增长到训练结束时的 1-5 百万。

### 球谐函数一句话

视角相关颜色是单位球上的函数 `c(direction)`。球谐函数是球面的傅里叶基。截断到度 `L` 得到每通道 `(L+1)^2` 个基函数。求新视角的颜色是学习到的 SH 系数与视角处基函数的点积。0 度 = 一个系数 = 常数颜色。3 度 = 16 个系数 = 足以捕捉 Lambertian 明暗、镜面和轻度反射。SD Gaussian Splatting 论文默认使用 3 度。

### 2026 年生产技术栈

```
1. 采集          智能手机 / DJI 无人机 / 手持扫描仪
2. SfM / MVS     COLMAP 或 GLOMAP 推导相机位姿 + 稀疏点
3. 训练 3DGS     nerfstudio / gsplat / inria 官方 / PostShot (~10-30 分钟，RTX 4090)
4. 编辑          SuperSplat / SplatForge（清理漂浮点、分割）
5. 导出          .ply -> glTF KHR_gaussian_splatting 或 .usd (OpenUSD 26.03)
6. 查看          Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 和生成式变体

- **4D 高斯溅射**——高斯是时间的函数；用于体视频（Superman 2026、A$AP Rocky 的 "Helicopter"）。
- **生成式 splats**——文本到 splat 模型（World Labs 的 Marble）可生成整个场景。
- **3D 高斯无损变换**——NVIDIA NuRec 的自动驾驶仿真变体。

## 动手实现

### 步骤 1：一个 2D 高斯

先构建 2D 光栅器。3D 情况在投影后归结为它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      中心点
    covs:   (G, 2, 2)   协方差矩阵
    points: (H, W, 2)   像素坐标
    returns: (G, H, W)  每个高斯在每个像素处的密度
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 对每个（高斯，像素）对执行二次型 `diff^T Sigma^-1 diff`。

### 步骤 2：2D 溅射光栅器

从前到后 alpha 合成。2D 中深度无意义，因此用学习的逐高斯标量来排序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     在 [0, 1] 范围内
    depths:    (G,)     用于排序的逐高斯标量
    image_size: (H, W)
    returns:   (H, W, 3) 渲染图像
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

不快速——真实实现使用基于瓦片的 CUDA 内核——但数学正确且完全可微。

### 步骤 3：可训练的 2D splat 场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 2D 中单角
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，在渲染时通过正确的激活函数映射。这是每个 3DGS 实现的标准模式。

### 步骤 4：拟合 2D 高斯到目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 红圆
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # 蓝方块
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过 200 步，64 个高斯落入两个形状。这就是全部原理——对显式几何基元做梯度下降。

### 步骤 5：从 2D 到 3D

3D 扩展保持相同循环。增加的要素：

1. 逐高斯旋转是四元数而非单角。
2. 协方差是 `R S S^T R^T`，其中 `R` 由四元数构建，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 使用相机外参和 `mu` 处透视投影的雅可比。
4. 颜色变为球谐展开；按视角方向求值。
5. 深度排序按实际相机空间 z 而非学习的标量。

每个生产实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）在 GPU 上用基于瓦片的 CUDA 内核做完全相同的事。

### 步骤 6：球谐函数求值

SH 基到底 3 度有 16 项每通道。求值：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   最后一个维度是 RGB 通道
    dirs:      (..., 3)       单位向量
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 项略去；完整 16 系数版本在代码文件中
    return result
```

学习到的 `sh_coeffs` 存储该高斯"每个方向对应的颜色"。渲染时按当前视角方向求值，得到 3 维 RGB 向量。

## 用现成库

对于真实 3DGS 工作，使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。在 RTX 4090 上运行一个典型场景需要 10-30 分钟。

2026 年重要的导出选项：

- `.ply` — 原始高斯点云（可移植，文件最大）。
- `.splat` — PlayCanvas / SuperSplat 量化格式。
- glTF `KHR_gaussian_splatting` — Khronos 标准，可跨查看器和引擎移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD 原生，用于 NVIDIA Omniverse 和 Vision Pro 流水线。

对于 4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 用时变均值和透明度扩展相同机制。

## 产出

本课产出：

- `outputs/prompt-3dgs-capture-planner.md` — 一个 prompt，为给定场景类型规划采集会话（照片数量、相机路径、光照）。
- `outputs/skill-3dgs-export-router.md` — 一个 skill，根据下游查看器或引擎选择正确的导出格式（`.ply` / `.splat` / glTF / USD）。

## 练习

1. **（简单）** 在不同的合成图像上运行上述 2D splat 训练器。在 `[16, 64, 256]` 中改变 `num_splats`，绘制每个的 MSE vs 步数曲线。找出收益递减点。
2. **（中等）** 扩展 2D 光栅器以支持依赖标量"视角"的逐高斯 RGB 颜色，通过 2 度谐波。训练一对目标图像，验证模型重建两者。
3. **（困难）** 克隆 `nerfstudio` 并在任何场景（桌子、植物、脸、房间）的 20 张照片上训练 `splatfacto`。导出为 glTF `KHR_gaussian_splatting`，在查看器（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）中打开。报告训练时间、高斯数量和渲染 fps。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|------|----------|
| 3DGS | "高斯 splats" | 显式场景表示，由数百万个 3D 高斯组成，每个具有位置、旋转、尺度、透明度、SH 颜色 |
| 协方差 | "高斯形状" | `Sigma = R S S^T R^T`；一个高斯的朝向和各向异性尺度 |
| Alpha 合成 | "从后到前混合" | 与 NeRF 体渲染相同的公式，但现在在显式稀疏集上 |
| 致密化 | "克隆与分裂" | 在重建欠拟合处自适应添加新高斯 |
| 剪枝 | "删除低透明度" | 移除训练中透明度崩溃到接近零的高斯 |
| 球谐函数 | "视角相关颜色" | 球面上的傅里叶基；以视角方向的函数存储颜色 |
| Splatfacto | "nerfstudio 的 3DGS" | 2026 年训练 3DGS 最简途径 |
| `KHR_gaussian_splatting` | "glTF 标准" | Khronos 2026 扩展，使 3DGS 可跨查看器和引擎移植 |

## 扩展阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 生产级 CUDA 光栅器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 参考训练方案
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026 年可移植格式
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` schema
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 行业概览