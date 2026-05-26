# AI 工程从零开始 — 中文翻译项目

## 项目概览

这是 [rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch) 的中文翻译版。

- **原文**: 435 节课，20 个阶段，~320 小时，Python/TypeScript/Rust/Julia
- **目标**: 将课程文档、代码注释、网站 UI 全面中文化
- **进度**: 第 1 阶段（数学基础）01~04 课已完成 + 网站首页翻译
- **上游仓库**: https://github.com/rohitg00/ai-engineering-from-scratch
- **中文仓库**: https://gitee.com/qianchilang/ai-engineering-from-scratch-zh（私有）

## 目录结构

```
zh/
├── SKILL.md                          # 本文件
├── 课程目录.md                        # 小白友好的课程导览
├── 17个综合大项目.md                  # 17 个 Capstone 详解
├── phases/                           # 课程正文（翻译目标）
│   └── 01-数学基础/
│       ├── 01-线性代数直觉/
│       ├── 02-向量矩阵与运算/
│       ├── 03-矩阵变换与特征值/
│       ├── 04-机器学习中的微积分/
│       └── ... (待翻译)
└── site/                             # 官方网站前端
    ├── index.html                    # 首页（已翻译导航/UI）
    ├── app.js                        # 交互逻辑
    ├── lesson.html                   # 课程页
    ├── catalog.html / glossary.html  # 索引/术语表
    └── ...
```

## 翻译规范

### 命名规则

| 原文 | 中文命名 | 示例 |
|------|----------|------|
| Phase X | 第 X 阶段 | `01-数学基础` |
| Lesson X | 第 X 课 | `01-线性代数直觉` |
| docs/en.md | docs/zh.md | 课程文档 |
| outputs/*.md | outputs/*-zh.md | 产物（保留原文件名 + `-zh` 后缀） |
| code/*.py | 不变 | 代码文件保持原名，注释改为中文 |

### 术语对照表（必须一致）

| 英文术语 | 中文翻译 | 说明 |
|----------|----------|------|
| Vector | 向量 | 非"矢量" |
| Matrix | 矩阵 | |
| Dot product | 点积 / 内积 | 两者均可 |
| Gradient | 梯度 | |
| Derivative | 导数 | |
| Partial derivative | 偏导数 | |
| Eigenvalue / Eigenvector | 特征值 / 特征向量 | |
| Determinant | 行列式 | |
| Transpose | 转置 | |
| Neural network | 神经网络 | |
| Loss function | 损失函数 | |
| Backpropagation | 反向传播 | |
| Forward pass | 前向传播 | |
| Activation function | 激活函数 | |
| Embedding | 嵌入 / 词嵌入 | 根据上下文 |
| Attention | 注意力 | |
| Transformer | Transformer | 不翻译 |
| Phase | 阶段 | |
| Lesson | 课 / 课程 | |
| Build It | 动手实现 | |
| Use It | 用现成库 | |
| Ship It | 产出 | |

### 文档翻译要点

1. **MOTTO（一句话核心）**: 保留原句风格，译为简洁有力的中文
2. **六拍子结构**: 保持 Build It → Use It → Ship It 的结构，分别译为"动手实现""用现成库""产出"
3. **代码块**: 代码中的变量名、函数名不翻译，但输出字符串和注释改为中文
4. **mermaid 图**: 图中的英文标签需要翻译
5. **术语表**: 每个课程末尾的 Key Terms 表格必须翻译
6. **练习题**: 保留原题号，题目内容翻译为中文

### 代码注释规范

```python
# 中文注释说明这段代码在做什么
# 不要逐字翻译英文注释，而是用中文重新表达

def numerical_derivative(f, x, h=1e-7):
    """数值导数：用中心差分近似 f'(x)。
    公式：(f(x+h) - f(x-h)) / (2h)
    h=1e-7 对 float64 效果最佳。
    """
    return (f(x + h) - f(x - h)) / (2 * h)
```

### 网站翻译要点

- `index.html`: 导航、按钮、标签、状态文字
- `app.js`: 模态框交互文字、确认对话框
- `cmdpalette.js`: 搜索面板占位符和快捷键标签
- `data.js`: 不直接翻译，等课程文档翻译完用 `build.js` 重新生成

## 工作流程

### 翻译新课程

1. 读取原文：`ai-engineering-from-scratch/phases/XX-阶段名/XX-课程名/docs/en.md`
2. 创建目录：`zh/phases/XX-中文阶段名/XX-中文课程名/{code,docs,outputs,assets}`
3. 翻译文档：写入 `docs/zh.md`
4. 翻译代码：复制原文 `code/` 文件，添加中文注释和输出
5. 翻译产物：写入 `outputs/*-zh.md`
6. 提交：`git add . && git commit -m "translate: 第X阶段第X课 — 课程名"`
7. 推送：`git push`

### 更新网站

1. 翻译 `site/*.html` 和 `site/*.js` 中的用户可见文字
2. 用 `sed` 批量替换常见词汇（导航、按钮等）
3. 单独处理不唯一的字符串
4. 提交推送

### 推送命令

```bash
cd zh
git add .
git commit -m "translate: 课程名"
git push origin master
```

## 关键文件速查

| 用途 | 路径 |
|------|------|
| 原文课程库 | `ai-engineering-from-scratch/phases/` |
| 中文课程库 | `zh/phases/` |
| 课程目录（中文） | `zh/课程目录.md` |
| 综合项目（中文） | `zh/17个综合大项目.md` |
| 网站首页 | `zh/site/index.html` |
| 网站交互 | `zh/site/app.js` |
| 搜索面板 | `zh/site/cmdpalette.js` |
| 课程页 | `zh/site/lesson.html` |
| 构建脚本 | `zh/site/build.js` |
| 课程数据 | `zh/site/data.js` |

## 翻译进度

### 第 1 阶段 — 数学基础（22 课）

| 课号 | 名称 | 状态 |
|------|------|------|
| 01 | 线性代数直觉 | ✅ 完成 |
| 02 | 向量矩阵与运算 | ✅ 完成 |
| 03 | 矩阵变换与特征值 | ✅ 完成 |
| 04 | 机器学习中的微积分 | ✅ 完成 |
| 05 | 链式法则与自动求导 | ⬜ 待翻译 |
| 06~22 | ... | ⬜ 待翻译 |

### 网站翻译

| 页面 | 状态 |
|------|------|
| index.html | ✅ 导航/UI 已翻译 |
| app.js | ✅ 交互文字已翻译 |
| cmdpalette.js | ✅ 搜索面板已翻译 |
| lesson.html | ✅ 基础导航已翻译 |
| catalog.html | ✅ 基础导航已翻译 |
| glossary.html | ✅ 基础导航已翻译 |
| prereqs.html | ✅ 基础导航已翻译 |
| data.js | ⬜ 需重新生成 |
