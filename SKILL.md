# AI 工程从零开始 — 中文翻译项目

## 项目概览

这是 [rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch) 的中文翻译版。

- **原文**: 435 节课，20 个阶段，~320 小时，Python/TypeScript/Rust/Julia
- **目标**: 将课程文档、代码注释、网站 UI 全面中文化
- **进度**: Phase 0 ✅ (12课), Phase 1 ✅ (22课), Phase 2 ✅ (18课), Phase 3-19 待翻译
- **已翻译目录**: `phases/` (Phase 0-1 完整), `site/phases/` (Phase 0-1 部分)
- **对照翻译**: 英文原版已放在本仓库 `original/` 目录下，无需额外克隆
- **中文仓库**: https://gitee.com/qianchilang/ai-engineering-from-scratch-zh（私有）

## 目录结构

```
zh/
├── SKILL.md                          # 本文件（翻译规范 + 进度）
├── README.md                          # Gitee 首页展示的中文介绍
├── README_BUILD.md                    # build.js 读取的课程目录（含中文课名）
├── ROADMAP.md                         # 课程状态追踪（含中文课名）
├── glossary/terms.md                  # 83 个 AI 术语中文翻译
├── 17个综合大项目.md                  # 17 个 Capstone 详解
├── original/                          # ← 英文原版（对照翻译用）
│   ├── phases/00~19/                  # 全部 435 课原文
│   ├── site/                          # 英文网站
│   ├── README.md
│   └── ...
├── phases/                           # ← 中文翻译（目标）
│   ├── 00-setup-and-tooling/          # Phase 0（12课全部完成）
│   ├── 01-数学基础/                    # Phase 1（22课全部完成）
│   ├── 02-机器学习基础/                # Phase 2
│   ├── 03-深度学习核心/                # Phase 3
│   ├── 04-计算机视觉/                  # Phase 4
│   ├── 05-NLP基础到进阶/               # Phase 5
│   ├── 06-语音与音频/                  # Phase 6
│   ├── 07-Transformers深度解析/       # Phase 7
│   ├── 08-生成式AI/                    # Phase 8
│   ├── 09-强化学习/                    # Phase 9
│   ├── 10-从零构建LLM/                # Phase 10
│   ├── 11-LLM工程/                     # Phase 11
│   ├── 12-多模态AI/                    # Phase 12
│   ├── 13-工具与协议/                  # Phase 13
│   ├── 14-Agent工程/                   # Phase 14
│   ├── 15-自主系统/                    # Phase 15
│   ├── 16-多Agent与Swarm/              # Phase 16
│   ├── 17-基础设施与生产部署/          # Phase 17
│   ├── 18-伦理安全与对齐/              # Phase 18
│   └── 19-毕业项目/                    # Phase 19
└── site/                             # ← 中文网站（目标）
    ├── index.html                     # 首页（已翻译导航/UI）
    ├── lesson.html                    # 课程页（支持加载 docs/zh.md）
    ├── catalog.html                   # 课程索引
    ├── glossary.html                  # 术语表
    ├── prereqs.html                   # 路线图
    ├── data.js                        # 课程数据（手动维护中文课名）
    ├── build.js                       # 构建脚本（已改 Gitee + zh.md）
    └── ...
```

## 翻译规范

### 命名规则

| 原文 | 中文命名 | 示例 |
|------|----------|------|
| Phase X | 第 X 阶段 | `01-数学基础` |
| Lesson X | 第 X 课 | `01-线性代数直觉` |
| docs/en.md | docs/zh.md | 课程文档 |
| outputs/*.md | outputs/*.md | 产物（保持原文件名，内容翻译为中文） |
| code/*.py | 不变 | 代码文件保持原名，注释改为中文 |

### ⚠️ 重要：路径命名规则

- `phases/` 下：**中文目录名**（如 `01-数学基础/01-线性代数直觉`）
- `site/phases/` 下：**英文目录名**（如 `01-math-foundations/01-linear-algebra-intuition`）
- **原因**：`data.js` 和 `lesson.html` 中的 `path` 字段使用英文路径，保持一致性
- **同步命令**：`cp -r phases/01-数学基础/01-线性代数直觉 site/phases/01-math-foundations/01-linear-algebra-intuition/`

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
- `data.js`: **手动维护**，不再用 `build.js` 自动生成。翻译完新课程后，直接修改 data.js 中对应的课程名
- `build.js`: 已配置为读取 `README_BUILD.md` + `docs/zh.md` + Gitee 链接，但**当前不使用**（手动维护 data.js 更可控）
- `lesson.html`: 已改为用**相对路径** `./phases/{path}/docs/zh.md` 加载本地课程文件（避免 Gitee raw 跨域问题）

## 工作流程

### 翻译新课程（4步）

1. **翻译课程文档**
   - 读取原文：`original/phases/XX-阶段名/XX-课程名/docs/en.md`
   - 创建目录：`phases/XX-中文阶段名/XX-中文课程名/{code,docs,outputs,assets}`
   - 写入 `docs/zh.md`
   - 复制 `code/` 文件，注释改为中文
   - 写入 `outputs/*-zh.md`

2. **同步到 site/phases/（EdgeOne Pages 实际部署目录）**
   - `cp -r phases/XX-中文阶段名/XX-中文课程名 site/phases/XX-英文阶段名/XX-英文课程名/`
   - **必须保持英文路径**（data.js 和 lesson.html 用英文路径匹配）

3. **更新 data.js 课程名**
   - 找到对应课程的 `name` 字段，改成中文
   - 例如：`"name": "Dev Environment"` → `"name": "开发环境"`

4. **⚠️ 提交前必须检查 data.js 语法**
   ```bash
   # Windows
   node -c site/data.js

   # 或用 Node.js 测试
   node -e "try { eval(require('fs').readFileSync('site/data.js','utf8')); console.log('data.js OK') } catch(e) { console.error('SYNTAX ERROR:', e.message); process.exit(1) }"
   ```
   **常见错误**：
   - 中文引号 `""` → 改为 `「」`
   - URL 缺少引号 `"url": "phases/...` → 检查是否完整闭合引号

5. **提交推送**
   ```bash
   git add .
   git commit -m "translate: Phase X 第Y课 — 课程名"
   git push origin master
   ```

### 更新网站

1. 翻译 `site/*.html` 和 `site/*.js` 中的用户可见文字
2. `lesson.html` 加载逻辑已改（相对路径 `./phases/...`），一般无需改动
3. 提交推送

### 推送命令

```bash
git add .
git commit -m "translate: 课程名"
git push origin master
```

## 关键文件速查

| 用途 | 路径 |
|------|------|
| 原文课程库 | `original/phases/` |
| 中文课程库 | `phases/` |
| 部署课程库 | `site/phases/`（EdgeOne Pages 实际部署） |
| Gitee 首页 | `README.md` |
| 构建用课程目录 | `README_BUILD.md`（build.js 读取） |
| 课程状态追踪 | `ROADMAP.md` |
| 术语表 | `glossary/terms.md` |
| 综合项目 | `17个综合大项目.md` |
| 网站首页 | `site/index.html` |
| 课程页 | `site/lesson.html` |
| 课程索引 | `site/catalog.html` |
| 术语表页 | `site/glossary.html` |
| 路线图页 | `site/prereqs.html` |
| 课程数据 | `site/data.js`（手动维护） |
| 构建脚本 | `site/build.js`（备用） |

## 翻译进度

### Phase 0 — 环境搭建与工具（12 课）✅ 全部完成

| 课号 | 英文路径 | 中文名 | 状态 |
|------|----------|--------|------|
| 01 | 01-dev-environment | 开发环境 | ✅ |
| 02 | 02-git-and-collaboration | Git 与协作 | ✅ |
| 03 | 03-gpu-setup-and-cloud | GPU 配置与云端 | ✅ |
| 04 | 04-apis-and-keys | API 与密钥 | ✅ |
| 05 | 05-jupyter-notebooks | Jupyter 笔记本 | ✅ |
| 06 | 06-python-environments | Python 环境 | ✅ |
| 07 | 07-docker-for-ai | Docker 与 AI | ✅ |
| 08 | 08-editor-setup | 编辑器配置 | ✅ |
| 09 | 09-data-management | 数据管理 | ✅ |
| 10 | 10-terminal-and-shell | 终端与 Shell | ✅ |
| 11 | 11-linux-for-ai | AI 中的 Linux | ✅ |
| 12 | 12-debugging-and-profiling | 调试与性能分析 | ✅ |

### Phase 1 — 数学基础（22 课）✅ 全部完成

| 课号 | 中文名 | 状态 |
|------|--------|------|
| 01 | 线性代数直觉 | ✅ |
| 02 | 向量矩阵与运算 | ✅ |
| 03 | 矩阵变换与特征值 | ✅ |
| 04 | 机器学习中的微积分 | ✅ |
| 05 | 链式法则与自动求导 | ✅ |
| 06 | 概率与分布 | ✅ |
| 07 | 贝叶斯定理 | ✅ |
| 08 | 优化：梯度下降家族 | ✅ |
| 09 | 信息论：熵与KL散度 | ✅ |
| 10 | 降维：PCA、t-SNE、UMAP | ✅ |
| 11 | 奇异值分解 | ✅ |
| 12 | 张量运算 | ✅ |
| 13 | 数值稳定性 | ✅ |
| 14 | 范数与距离 | ✅ |
| 15 | 机器学习统计 | ✅ |
| 16 | 采样方法 | ✅ |
| 17 | 线性系统 | ✅ |
| 18 | 凸优化 | ✅ |
| 19 | AI 中的复数 | ✅ |
| 20 | 傅里叶变换 | ✅ |
| 21 | 图论与机器学习 | ✅ |
| 22 | 随机过程 | ✅ |

### Phase 2 — 机器学习基础（18 课）✅ 全部完成

| 课号 | 中文名 | 状态 |
|------|--------|------|
| 01 | 什么是机器学习 | ✅ |
| 02 | 从零实现线性回归 | ✅ |
| 03 | 逻辑回归与分类 | ✅ |
| 04 | 决策树与随机森林 | ✅ |
| 05 | 支持向量机 | ✅ |
| 06 | KNN 与距离度量 | ✅ |
| 07 | 无监督学习：K-Means、DBSCAN | ✅ |
| 08 | 特征工程与选择 | ✅ |
| 09 | 模型评估：指标、交叉验证 | ✅ |
| 10 | 偏差、方差与学习曲线 | ✅ |
| 11 | 集成方法：Boosting、Bagging、Stacking | ✅ |
| 12 | 超参数调优 | ✅ |
| 13 | ML 流水线与实验追踪 | ✅ |
| 14 | 朴素贝叶斯 | ✅ |
| 15 | 时间序列基础 | ✅ |
| 16 | 异常检测 | ✅ |
| 17 | 处理不平衡数据 | ✅ |
| 18 | 特征选择 | ✅ |

### Phase 3-19 — 待翻译

Phase 3 至 Phase 19 共 383 课未翻译。课程名已翻译为中文存储在 `site/data.js` 中。

### 网站翻译

| 页面 | 状态 | 说明 |
|------|------|------|
| index.html | ✅ | 导航/UI 已翻译 |
| lesson.html | ✅ | 加载 docs/zh.md，课程名中文显示 |
| catalog.html | ✅ | 课程索引，链接指向 Gitee |
| glossary.html | ✅ | 83 个术语全部中文化 |
| prereqs.html | ✅ | 路线图已翻译 |
| data.js | ✅ | **手动维护**，Phase 0-19 全部阶段名和课程名已翻译 |
| build.js | ✅ | 已改 Gitee + zh.md（备用）|
