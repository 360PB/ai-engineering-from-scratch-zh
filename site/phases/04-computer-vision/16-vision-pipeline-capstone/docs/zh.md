# 构建完整视觉流水线 — 毕业项目

> 生产视觉系统是由数据契约串联的模型和规则链。组件已在阶段内；毕业项目是将它们端到端串联起来。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 4 第 01-15 课
**时长：** 约120分钟

## 学习目标

- 设计一个检测物体、分类并输出结构化 JSON 的生产视觉流水线——每个失败路径都有处理
- 将检测器（Mask R-CNN 或 YOLO）、分类器（ConvNeXt-Tiny）和数据契约（Pydantic）插入一个服务
- 基准测试端到端流水线，识别第一个瓶颈（通常是预处理，然后是检测器）
- 发布一个最小 FastAPI 服务，接收图像上传，运行流水线，返回带分类的检测结果

## 问题

单个视觉模型有用；视觉产品是它们的链。零售货架审计是检测器加产品分类器加价格 OCR 流水线。自动驾驶是2D 检测器加3D 检测器加分割器加跟踪器加规划器。医疗预筛查是分割器加区域分类器加临床医生 UI。

串联这些链是区分 ML 原型和产品的部分。每个模型间的接口都是一个新的 bug 滋生点。每个坐标变换、每个归一化、每个 mask 调整大小都是静默失败候选。流水线强度等于其最弱接口。

这个毕业项目设置最小可行流水线：检测 + 分类 + 结构化输出 + 服务层。Phase 4 中其他所有内容都可以插入这个骨架：把 Mask R-CNN 换成 YOLOv8，添加 OCR 头，添加分割分支，添加跟踪器。架构稳定；组件可插拔。

## 核心概念

### 流水线

Seven stages. The two model stages are expensive; the five other stages are where the bugs live.

### 带 Pydantic 的数据契约

每个模型边界变成类型化对象。这将静默失败变成大声失败。

### 延迟去哪了

三个事实在几乎所有视觉流水线中成立：

1. **预处理通常是最大的单一块。** 解码 JPEG、转换色彩空间、缩放——这些是 CPU 绑定的，容易被遗忘。
2. **检测器主导 GPU 时间。** 70-90% 的 GPU 时间在检测前向传播中。
3. **后处理（NMS、RLE 编码/解码）在 GPU 上便宜，在 CPU 上贵。** 总是用实际目标来分析。

了解分布是将优化变成优先级列表的关键。

### 失败模式

- **空检测** — 返回空列表，不要崩溃。记录日志。
- **越界框** — 裁剪前夹紧到图像大小。
- **过小裁剪** — 跳过对小于分类器最小输入的框的分类。
- **损坏上传** — 400 响应带特定错误码，不是 500。
- **模型加载失败** — 在服务启动时失败，不在第一次请求时。

生产流水线处理每个这些，不用隐藏失败的通用 `try/except`。每个失败都有一个命名码和响应。

### 批处理

生产服务服务多个客户端。跨请求批处理检测和分类将吞吐相乘。权衡：等待批次填充的额外延迟。典型设置：收集请求最多20ms，批在一起，处理，分配响应。`torchserve` 和 `triton` 原生做这个；对于可预测负载的小服务，自定义微批处理器。

## 动手实现

### 步骤 1：数据契约

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)，绝对像素
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float
```

五秒代码在严肃流水线上节省一小时的调试。

### 步骤 2：最小流水线类

```python
import time
import numpy as np
import torch
from PIL import Image

class VisionPipeline:
    def __init__(self, detector, classifier, class_names,
                 device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop = min_crop

    def preprocess(self, image):
        """
        image: PIL.Image 或 np.ndarray (H, W, 3) uint8
        返回: CHW float tensor 在设备上
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))

    def run(self, image, image_id="anonymous"):
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        det = self.detect(tensor)

        crops = []
        detections = []
        valid_indices = []
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(cls),
            ))
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        class_preds = self.classify(crops)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.class_names[cls_id],
                score=float(cls_score),
            ))

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )
```

每个接口都是类型化的。每个失败路径都有具体的处理决策。

### 步骤 3：接一个检测器和一个分类器

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

# 使用 ImageNet 预训练权重，无需训练即可获得真实流水线
detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
classifier = convnext_tiny(weights="DEFAULT")
class_names = [f"imagenet_class_{i}" for i in range(1000)]

pipe = VisionPipeline(detector, classifier, class_names)

# 用合成图像冒烟测试
test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
result = pipe.run(test_image, image_id="demo")
print(result.model_dump_json(indent=2)[:500])
```

### 步骤 4：FastAPI 服务

```python
from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO

app = FastAPI()
pipe = None  # 启动时初始化

@app.on_event("startup")
def load():
    global pipe
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    classifier = convnext_tiny(weights="DEFAULT").eval()
    pipe = VisionPipeline(detector, classifier, class_names=[f"c{i}" for i in range(1000)])

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    result = pipe.run(img, image_id=file.filename or "upload")
    return result.model_dump()
```

用 `uvicorn main:app --host 0.0.0.0 --port 8000` 运行。用 `curl -F 'file=@dog.jpg' http://localhost:8000/detect` 测试。

### 步骤 5：流水线基准测试

```python
import time

def benchmark(pipe, num_runs=20, image_size=(400, 600)):
    img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
    pipe.run(img)  # 预热

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tensor = pipe.preprocess(img)
        t1 = time.perf_counter()
        det = pipe.detect(tensor)
        t2 = time.perf_counter()
        crops = []
        for box in det["boxes"]:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
                crop = tensor[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                )[0]
                crops.append(crop)
        pipe.classify(crops)
        t3 = time.perf_counter()
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["classify"].append((t3 - t2) * 1000)
        stages["total"].append((t3 - t0) * 1000)

    for stage, times in stages.items():
        times.sort()
        print(f"{stage:12s}  p50={times[len(times)//2]:7.1f} ms  p95={times[int(len(times)*0.95)]:7.1f} ms")
```

CPU 上的典型输出：预处理约3ms，检测300-500ms，分类20-40ms，总计350-550ms。GPU 上，检测是20-40ms，预处理+分类在相对意义上开始更重要。

## 用现成库

生产模板收敛到相同结构，加：

- **模型版本控制** — 总是在响应中记录模型名和权重哈希。
- **每请求 Trace ID** — 记录每个阶段每个请求的时序，以便将慢响应与阶段关联。
- **后备路径** — 如果分类器超时，返回不带分类的检测结果，而非整个请求失败。
- **安全过滤器** — NSFW / PII 过滤器在分类后、响应离开服务前运行。
- **批处理端点** — `/detect_batch` 接受图像 URL 列表用于批量处理。

对于生产服务，`torchserve`、`Triton Inference Server` 和 `BentoML` 开箱即用地处理批处理、版本控制、指标和健康检查。直接运行 `FastAPI` 对原型和小规模产品来说没问题。

## 产出

本课产出：

- `outputs/prompt-vision-service-shape-reviewer.md` — 一个 prompt，审查视觉服务代码的契约/响应形状违规，并命名第一个破坏性 bug。
- `outputs/skill-pipeline-budget-planner.md` — 一个 skill，给定目标延迟和吞吐，为每个流水线阶段分配时间预算，并标记哪个阶段将首先超出预算。

## 练习

1. **（简单）** 在任何开放数据集的10张图像上运行流水线。报告每阶段平均时间和每张图像检测数量分布。
2. **（中等）** 给 `Detection` 添加 mask 输出字段并编码为 RLE。验证即使对于10个物体的图像 JSON 保持在1MB以下。
3. **（困难）** 在分类器前添加微批处理器：收集 crops 最多10ms，一次 GPU 调用分类全部，返回每请求结果。在每秒5个并发请求下测量吞吐增益和增加的延迟。

## 关键术语

| 术语 | 行话 | 实际含义 |
|------|----------------|----------------------|
| Pipeline | "系统" | 预处理、推理、后处理的有序链，每对之间有类型化接口 |
| Data contract | "模式" | 每个阶段输入和输出遵守的 Pydantic / dataclass 定义；在边界捕获集成 bug |
| Preprocessing | "模型之前" | 解码、色彩转换、缩放、归一化；通常是最大 CPU 时间消耗 |
| Postprocessing | "模型之后" | NMS、mask 调整大小、阈值、RLE 编码；GPU 上便宜，CPU 上贵 |
| Microbatcher | "收集后转发" | 聚合器，等待固定窗口收集多个请求，运行单次批处理前向传播 |
| Trace ID | "请求 id" | 每请求标识符，在每个阶段记录，以便端到端追踪慢请求 |
| Failure code | "命名错误" | 每个失败类的特定错误码而非通用 500；启用客户端重试逻辑 |
| Health check | "就绪探针" | 报告服务是否能应答的廉价端点；负载均衡器依赖这个 |

## 扩展阅读

- [Full Stack Deep Learning — Deploying Models](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/) — 生产 ML 部署的权威概述
- [BentoML 文档](https://docs.bentoml.com) — 带批处理、版本控制和指标的服务框架
- [torchserve 文档](https://pytorch.org/serve/) — PyTorch 官方服务库
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) — 带批处理和多模型支持的高吞吐服务