# JAX 入门

> PyTorch 改变张量。TensorFlow 构建图。JAX 编译纯函数。最后一种改变了你对深度学习的思考方式。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 03 Lessons 01-10，基本 NumPy
**时间：** 约 90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的命令式变化与 JAX 的函数式编译模型之间的关键设计差异
- 应用 JIT 编译和 vmap 向量化来加速训练循环，与朴素 Python 比较
- 在 JAX 中训练一个简单网络，并将显式状态管理与 PyTorch 的面向对象方法对比

## 问题

你知道如何在 PyTorch 中构建神经网络。你定义 `nn.Module`，调用 `.backward()`，执行优化器步。它有效。数百万人使用它。

但 PyTorch 在其 DNA 中有一个约束：它急切地追踪操作，一次一个，在 Python 中。每个 `tensor + tensor` 是单独的内核启动。每次训练步重新解释相同的 Python 代码。在你需要跨 2,048 个 TPU 训练 5400 亿参数模型之前，这一切都很好。然后开销杀死你。

Google DeepMind 用 JAX 训练 Gemini。Anthropic 用 JAX 训练 Claude。这些不是小操作——它们是地球上最大的神经网络训练运行。他们选择 JAX，因为它将你的训练循环视为可编译的程序，而不是 Python 调用序列。

JAX 是带三个超能力的 NumPy：自动微分、JIT 编译到 XLA、自动向量化。你写一个处理单个样本的函数。JAX 给你一个处理批次、计算梯度、编译到机器码、跨多个设备运行的函数。所有这些都无需更改原始函数。

## 概念

### JAX 哲学

JAX 是一个函数式框架。没有类，没有可变状态，没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| 有状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 急切执行 | 通过 XLA 的 JIT 编译 |
| 手动循环 `for x in batch:` | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变 `model.parameters()` | 不可变 pytree 数组 |

这不是风格偏好。这是编译器约束。JIT 编译需要纯函数——相同输入总是产生相同输出，无副作用。这种约束是实现 100 倍加速的原因。

### jax.numpy：熟悉的外表

JAX 在加速器上重新实现 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名。相同的广播规则。相同的切片语义。但数组在 GPU/TPU 上，每个操作都可被编译器追踪。

一个关键区别：JAX 数组是不可变的。没有 `a[0] = 5`。取而代之：`a = a.at[0].set(5)`。这初感别扭，但一周后就明白了——不可变性是使 `grad`、`jit` 和 `vmap` 等变换可组合的原因。

### jax.grad：函数式自动微分

PyTorch 将梯度附加到张量（`.grad`）。JAX 将梯度附加到函数。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 取一个函数并返回计算梯度的新函数。无需 `.backward()` 调用。没有存储在张量上的计算图。梯度只是另一个你可以调用、组合或 JIT 编译的函数。

这可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。Jacobians。Hessians。全通过组合 `grad`。PyTorch 也可以做这个（`torch.autograd.functional.hessian`），但它是附加上去的。在 JAX 中，它是基础。

约束：`grad` 只对纯函数有效。内部无 print 语句（它们在追踪期间运行，非执行时）。无外部状态修改。无随机数生成，除非有显式 key 管理。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

在首次调用时，JAX 追踪函数——它记录发生哪些操作，而不执行它们。然后它将追踪结果交给 XLA（加速线性代数），Google 为 TPU 和 GPU 开发的编译器。XLA 融合操作，消除冗余内存拷贝，并生成优化机器码。

后续调用完全跳过 Python。编译的代码在加速器上以 C++ 速度运行。

JIT 有帮助的场景：
- 训练步（相同计算重复数千次）
- 推理（相同模型，不同输入）
- 任何以相似形状输入多次调用的函数

JIT 有害的场景：
- 依赖值的 Python 控制流（`if x > 0` 其中 x 是追踪数组）
- 一次性计算（编译开销超过运行时）
- 调试（追踪隐藏实际执行）

控制流限制是真实的。`jax.lax.cond` 替代 `if/else`。`jax.lax.scan` 替代 `for` 循环。这些不是可选的——它们是编译的代价。

### vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将其提升为处理批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 表示：不批处理 `params`（共享），批处理 x 的第 0 维。无需手动 `for` 循环。无需重塑。无需批处理维度线程。JAX 算出批维度并向量化整个计算。

这不只是语法糖。`vmap` 生成融合向量化代码，比 Python 循环快 10-100 倍。它与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度。一行。在 PyTorch 中不用 hack 几乎不可能做到。

### pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 将函数复制到所有可用设备（GPU/TPU）并分割批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 跨设备同步梯度。

Google 用 `pmap`（及其后继者 `shard_map`）跨数千个 TPU v5e 芯片训练 Gemini。编程模型：写单设备版本，用 `pmap` 包装，完成。

### Pytrees：通用数据结构

JAX 操作"pytrees"——列表、元组、字典和数组的嵌套组合。你的模型参数是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换——`grad`、`jit`、`vmap`——知道如何遍历 pytrees。`jax.tree.map(f, tree)` 将 f 应用到每个叶子。这就是优化器同时更新所有参数的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法。没有参数注册。树结构就是模型。

### 函数式 vs 面向对象

PyTorch 在对象内存储状态：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params 被传入。什么都不存储。什么都不改变。这使每个函数可测试、可组合、可编译。这也意味着你自己管理 params——或使用像 Flax 或 Equinox 这样的库。

### JAX 生态

JAX 给你原语。库给你人体工程学：

| 库 | 角色 | 风格 |
|--------|------|------|
| **Flax**（Google） | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 Pytree，Python 风格 |
| **Optax**（DeepMind） | 优化器 + LR 调度 | 可组合梯度变换 |
| **Orbax**（Google） | 检查点 | 保存/恢复 pytrees |
| **CLU**（Google） | 指标 + 日志 | 训练循环工具 |

Optax 是标准优化器库。它将梯度变换（Adam、SGD、裁剪）与参数更新分离，使其 trivial 组合：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时用 JAX vs PyTorch

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU 支持 | 一级（Google 构建了二者） | 社区维护（torch_xla） |
| GPU 支持 | 好（通过 XLA 的 CUDA） | 最好（原生 CUDA） |
| 调试 | 难（追踪 + 编译） | 简单（命令式，逐行） |
| 生态 | 研究导向（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（无处不在） |
| 大规模训练 | 优越（XLA、pmap、mesh） | 好（FSDP、DeepSpeed） |
| 原型速度 | 较慢（函数式开销） | 较快（可变即走） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 谁用 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

诚实的答案：除非有特定原因使用 JAX，否则用 PyTorch。那些原因是——TPU 访问、需要逐样本梯度、大规模多设备训练、或在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每个随机操作需要一个显式 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

这初感讨厌。但它保证跨设备和编译的可复现性——PyTorch 的 `torch.manual_seed` 在多 GPU 设置中无法保证的属性。

## 构建

### 第 1 步：设置和数据

我们用 JAX 和 Optax 在 MNIST 上训练 3 层 MLP。784 输入，两个隐藏层各 256 和 128 神经元，10 个输出类。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第 2 步：初始化参数

无类。只返回一个 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手动 He 初始化。三个 PRNG key 从一个种子分割。每个权重是嵌套字典中的不可变数组。

### 第 3 步：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数进，预测出。无 `self`，无存储状态。`loss_fn` 从头计算交叉熵——softmax、log、负均值。

### 第 4 步：JIT 编译的训练步

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 一次返回 loss 值和梯度。`@jax.jit` 装饰器将两个函数都编译到 XLA。首次调用后，每次训练步都不接触 Python。

### 第 5 步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch。约 97% 测试准确率。第一个 epoch 慢（JIT 编译）。第 2-10 个 epoch 快速。

注意缺失的部分：无 `.zero_grad()`，无 `.backward()`，无 `.step()`。整个更新是一个组合函数调用。梯度计算、Adam 变换、参数应用——全部在 `train_step` 内部。

## 使用

### Flax：Google 标准

Flax 是最常见的 JAX 神经网络库。它添加回 `nn.Module`，但带显式状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

与 PyTorch 相同的结构，但 `params` 与模型分离。`model.init()` 创建 params。`model.apply(params, x)` 运行前向传播。模型对象无状态。

### Equinox：Python 风格替代

Equinox（by Patrick Kidger）将模型表示为 pytrees：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身是一个 pytree。无需 `.apply()`。参数只是模型的叶子。这更接近 JAX 的思维方式。

### Optax：可组合优化器

Optax 将梯度变换与更新分离：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——全部作为变换链组合。每个变换看到梯度、修改它、传递给下一个。没有单体优化器类。

## 发布

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 首次 JIT 调用慢（编译）。在基准测试前预热。
- 避免在 JIT 内对 JAX 数组使用 Python 循环。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内有效。常规 `print()` 无效。
- 用 `jax.profiler` 或 TensorBoard 分析。XLA 编译可以隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU 内存。设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 禁用。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` -- 一个用于选择正确 JAX 优化器配置的提示
- `outputs/skill-jax-patterns.md` -- 一个涵盖 JAX 函数式模式的技能

## 练习

1. 在 MLP 中添加 dropout。在 JAX 中，dropout 需要一个 PRNG key——在 forward pass 中线程化一个 key，并为每个 dropout 层分割它。比较有和无 dropout 的测试准确率。

2. 使用 `jax.vmap` 计算 32 个 MNIST 图像批次的逐样本梯度。计算每个样本的梯度范数。哪些样本有最大梯度，为什么？

3. 用通用 `mlp_forward(params, x)` 替换手动 forward 函数，适用于任意层数。使用 `jax.tree.leaves` 自动确定深度。

4. 基准测试有和无 `@jax.jit` 的训练步。计时 100 步。在你的硬件上加速多大？首次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。有和无裁剪训练。绘制训练中的梯度范数以查看效果。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| XLA | "使 JAX 变快的东西" | 加速线性代数——一个编译器，融合操作并从计算图生成优化的 GPU/TPU 内核 |
| JIT | "即时编译" | JAX 在首次调用时追踪函数，编译到 XLA，然后在后续调用中运行编译版本 |
| 纯函数（Pure function） | "无副作用" | 输出仅依赖输入的函数——无全局状态、无修改、无显式 key 的随机性 |
| vmap | "自动批处理" | 将处理一个样本的函数转换为处理批次的函数，无需重写 |
| pmap | "自动并行" | 将函数复制到多个设备并分割输入批次 |
| Pytree | "嵌套数组字典" | JAX 可以遍历和变换的列表、元组、字典和数组的任何嵌套结构 |
| 追踪（Tracing） | "记录计算" | JAX 用抽象值执行函数以构建计算图，而不计算真实结果 |
| 函数式自动微分 | "函数的梯度" | 通过变换函数而非通过附加到张量的梯度存储来计算导数 |
| Optax | "JAX 的优化器库" | 可组合的梯度变换库——Adam、SGD、裁剪、调度——链在一起 |
| Flax | "JAX 的 nn.Module" | Google 的 JAX 神经网络库，在保持状态显式的同时添加层抽象 |

## 延伸阅读

- JAX 文档：https://jax.readthedocs.io/ -- 官方文档，包含关于 grad、jit 和 vmap 的优秀教程
- "JAX：Python+NumPy 程序的可组合变换"（Bradbury 等，2018）-- 解释设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/ -- Google 的 JAX 神经网络库
- Patrick Kidger，"Equinox：通过可调用 PyTrees 和过滤变换在 JAX 中构建神经网络"（2021）-- Flax 的 Python 风格替代方案
- DeepMind，"Optax：可组合梯度变换和优化" -- 标准优化器库
- "你不懂 JAX"（Colin Raffel，2020）-- 从 T5 作者之一开始的 JAX 陷阱和模式实用指南