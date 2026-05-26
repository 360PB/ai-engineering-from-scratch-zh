# 特征工程与选择

> 一个好特征值一千个数据点。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 1（ML统计学、线性代数）、Phase 2 第1-7课
**时间：** 约90分钟

## 学习目标

- 实现数值变换（标准化、最小最大缩放、对数变换、分箱），并解释每种方法的适用场景
- 为分类特征构建独热编码、标签编码和目标编码，并识别目标编码中的数据泄露风险
- 从零构建 TF-IDF 向量化器，并解释为什么它比原始词频更适合文本分类
- 应用基于过滤器的特征选择（方差阈值、相关性、互信息）来降维

## 问题

你有一个数据集。你选了一个算法。你训练它。结果很一般。你试了一个更花哨的算法。还是一般。你花了一周调超参数。改进甚微。

然后有人把原始数据转换成更好的特征，一个简单的逻辑回归就超过了你精心调参的梯度提升集成模型。

这种情况屡见不鲜。在传统 ML 中，数据的表示比算法的选择更重要。一个带有"平方英尺"和"卧室数量"的房价模型，无论学习器多复杂，都会超过用"地址原始字符串"作为输入的模型。算法只能用你给它的东西来处理。

特征工程是将原始数据转换为使模型更容易找到模式的表示的过程。特征选择是扔掉添加噪声而不添加信号的特征的过程。两者加在一起，是传统 ML 中杠杆效应最高的活动。

## 概念

### 特征流水线

```mermaid
flowchart LR
    A[原始数据] --> B[处理缺失值]
    B --> C[数值变换]
    B --> D[分类编码]
    B --> E[文本特征]
    C --> F[特征交互]
    D --> F
    E --> F
    F --> G[特征选择]
    G --> H[模型可用数据]
```

### 数值特征

原始数字很少是模型就绪的。常用变换：

**缩放：** 将特征放到相同范围，使基于距离的算法（K-Means、KNN、SVM）对所有特征一视同仁。最小最大缩放映射到 [0, 1]。标准化（z-score）映射到均值=0，标准差=1。

**对数变换：** 压缩右偏分布（收入、人口、词频）。将乘法关系转换为加法关系。

**分箱：** 将连续值转换为类别。当特征与目标的关系是非线性但呈阶梯式时很有用（例如年龄段）。

**多项式特征：** 创建 x^2、x^3、x1*x2 项。让线性模型能够捕捉非线性关系，代价是更多特征。

### 分类特征

模型需要数字。类别需要编码。

**独热编码：** 为每个类别创建一个二进制列。"颜色 = 红/蓝/绿"变成三列：is_red、is_blue、is_green。适合低基数特征，但类别多时会爆炸。

**标签编码：** 将每个类别映射到整数：红=0，蓝=1，绿=2。引入虚假的排序（模型可能认为绿 > 蓝 > 红）。只适合基于树模型的单独值分裂。

**目标编码：** 用该类别目标变量的均值替换每个类别。强大但危险：数据泄露风险很高。必须仅在训练数据上计算，然后应用于测试数据。

### 文本特征

**词频向量器：** 统计每个词在文档中出现的次数。"the cat sat on the mat" 变成 {the: 2, cat: 1, sat: 1, on: 1, mat: 1}。

**TF-IDF：** 词频-逆文档频率。用词在文档间有多独特来加权。类似 "the" 这样的常见词得到低权重。稀有的、有区分度的词得到高权重。

```
TF(word, doc) = 词在文档中出现次数 / 文档总词数
IDF(word) = log(总文档数 / 包含该词的文档数)
TF-IDF = TF * IDF
```

### 缺失值

真实数据有洞。策略：

- **删除行：** 仅在缺失数据稀少且随机时使用
- **均值/中位数填充：** 简单，保留分布形状（中位数对离群点更鲁棒）
- **众数填充：** 用于分类特征
- **指示列：** 填充前添加一个二进制列"was_this_missing"。数据是否缺失本身可能就有信息
- **前向/后向填充：** 用于时间序列数据

### 特征交互

有时候关系在组合中。"身高"和"体重"单独不如"BMI = 体重 / 身高^2"有预测力。特征交互扩展了特征空间，所以要用领域知识来选择正确的交互。

### 特征选择

更多特征并非总是更好。不相关特征添加噪声、增加训练时间，并可能导致过拟合。

**过滤方法（模型前）：**
- 相关性：删除相互高度相关的特征（冗余）
- 互信息：衡量知道一个特征能在多大程度上减少对目标的不确定性
- 方差阈值：删除几乎不变的特征

**包装方法（基于模型）：**
- L1 正则化（Lasso）：将不相关特征的权重精确推到零
- 递归特征消除：训练，删除最不重要的特征，重复

**为什么选择重要：** 拥有10个好特征的模型通常优于拥有10个好特征加90个噪声特征的模型。噪声特征给了模型在不会泛化的训练数据模式上过拟合的机会。

## 构建

### 第1步：从零实现数值变换

```python
import math


def min_max_scale(values):
    # 最小最大缩放：将值缩放到 [0, 1] 范围
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def standardize(values):
    # 标准化（z-score）：减去均值，除以标准差
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    return [(v - mean) / std for v in values]


def log_transform(values):
    # 对数变换：压缩右偏分布
    return [math.log(v + 1) for v in values]


def bin_values(values, n_bins=5):
    # 分箱：将连续值转换为离散类别
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / n_bins
    if bin_width == 0:
        return [0] * len(values)
    result = []
    for v in values:
        bin_idx = int((v - min_val) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)
        result.append(bin_idx)
    return result


def polynomial_features(row, degree=2):
    # 多项式特征：创建 x^2、x1*x2 等交互项
    n = len(row)
    result = list(row)
    if degree >= 2:
        for i in range(n):
            result.append(row[i] ** 2)
        for i in range(n):
            for j in range(i + 1, n):
                result.append(row[i] * row[j])
    return result
```

### 第2步：从零实现分类编码

```python
def one_hot_encode(values):
    # 独热编码：为每个类别创建一列二进制值
    categories = sorted(set(values))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    n_cats = len(categories)

    encoded = []
    for v in values:
        row = [0] * n_cats
        row[cat_to_idx[v]] = 1
        encoded.append(row)

    return encoded, categories


def label_encode(values):
    # 标签编码：将类别映射到整数（注意：引入虚假排序）
    categories = sorted(set(values))
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    return [cat_to_int[v] for v in values], cat_to_int


def target_encode(feature_values, target_values, smoothing=10):
    # 目标编码：用类别目标均值替换类别（注意：有数据泄露风险）
    global_mean = sum(target_values) / len(target_values)

    category_stats = {}
    for feat, target in zip(feature_values, target_values):
        if feat not in category_stats:
            category_stats[feat] = {"sum": 0.0, "count": 0}
        category_stats[feat]["sum"] += target
        category_stats[feat]["count"] += 1

    encoding = {}
    for cat, stats in category_stats.items():
        cat_mean = stats["sum"] / stats["count"]
        weight = stats["count"] / (stats["count"] + smoothing)
        encoding[cat] = weight * cat_mean + (1 - weight) * global_mean

    return [encoding[v] for v in feature_values], encoding
```

### 第3步：从零实现文本特征

```python
def count_vectorize(documents):
    # 词频向量器：统计每个词在每个文档中的出现次数
    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    vectors = []
    for doc in documents:
        vec = [0] * len(vocab)
        for word in doc.lower().split():
            vec[vocab[word]] += 1
        vectors.append(vec)

    return vectors, vocab


def tfidf(documents):
    # TF-IDF：词频乘以逆文档频率
    n_docs = len(documents)

    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    # 计算文档频率（每个词出现在多少文档中）
    doc_freq = {}
    for doc in documents:
        seen = set()
        for word in doc.lower().split():
            if word not in seen:
                doc_freq[word] = doc_freq.get(word, 0) + 1
                seen.add(word)

    vectors = []
    for doc in documents:
        words = doc.lower().split()
        word_count = len(words)
        tf_map = {}
        for word in words:
            tf_map[word] = tf_map.get(word, 0) + 1

        vec = [0.0] * len(vocab)
        for word, count in tf_map.items():
            tf = count / word_count
            idf = math.log(n_docs / doc_freq[word])
            vec[vocab[word]] = tf * idf
        vectors.append(vec)

    return vectors, vocab
```

### 第4步：从零实现缺失值填充

```python
def impute_mean(values):
    # 均值填充：用均值替换缺失值
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values), 0.0
    mean = sum(present) / len(present)
    return [v if v is not None else mean for v in values], mean


def impute_median(values):
    # 中位数填充：用中位数替换缺失值（对离群点更鲁棒）
    present = sorted(v for v in values if v is not None)
    if not present:
        return [0.0] * len(values), 0.0
    n = len(present)
    if n % 2 == 0:
        median = (present[n // 2 - 1] + present[n // 2]) / 2
    else:
        median = present[n // 2]
    return [v if v is not None else median for v in values], median


def impute_mode(values):
    # 众数填充：用最常见值替换缺失值（用于分类特征）
    present = [v for v in values if v is not None]
    if not present:
        return values, None
    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    mode = max(counts, key=counts.get)
    return [v if v is not None else mode for v in values], mode


def add_missing_indicator(values):
    # 添加缺失值指示列：标记哪些值是缺失的
    return [0 if v is not None else 1 for v in values]
```

### 第5步：从零实现特征选择

```python
def correlation(x, y):
    # 皮尔逊相关系数：衡量两个变量之间的线性相关性
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def mutual_information(feature, target, n_bins=10):
    # 互信息：衡量知道一个变量能在多大程度上减少对另一个的不确定性
    feat_min = min(feature)
    feat_max = max(feature)
    bin_width = (feat_max - feat_min) / n_bins if feat_max != feat_min else 1.0
    feat_binned = [
        min(int((f - feat_min) / bin_width), n_bins - 1) for f in feature
    ]

    n = len(feature)
    target_classes = sorted(set(target))

    feat_bins = sorted(set(feat_binned))
    p_feat = {}
    for b in feat_bins:
        p_feat[b] = feat_binned.count(b) / n

    p_target = {}
    for t in target_classes:
        p_target[t] = target.count(t) / n

    mi = 0.0
    for b in feat_bins:
        for t in target_classes:
            joint_count = sum(
                1 for fb, tv in zip(feat_binned, target) if fb == b and tv == t
            )
            p_joint = joint_count / n
            if p_joint > 0:
                mi += p_joint * math.log(p_joint / (p_feat[b] * p_target[t]))

    return mi


def variance_threshold(features, threshold=0.01):
    # 方差阈值：删除几乎不变化的特征
    n_features = len(features[0])
    n_samples = len(features)
    selected = []

    for j in range(n_features):
        col = [features[i][j] for i in range(n_samples)]
        mean = sum(col) / n_samples
        var = sum((v - mean) ** 2 for v in col) / n_samples
        if var >= threshold:
            selected.append(j)

    return selected


def remove_correlated(features, threshold=0.9):
    # 删除高度相关的特征（冗余特征）
    n_features = len(features[0])
    n_samples = len(features)

    to_remove = set()
    for i in range(n_features):
        if i in to_remove:
            continue
        col_i = [features[r][i] for r in range(n_samples)]
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            col_j = [features[r][j] for r in range(n_samples)]
            corr = abs(correlation(col_i, col_j))
            if corr >= threshold:
                to_remove.add(j)

    return [i for i in range(n_features) if i not in to_remove]
```

### 第6步：完整流水线和演示

```python
import random


def make_housing_data(n=200, seed=42):
    # 生成模拟房价数据（包含数值、分类和缺失值）
    random.seed(seed)
    data = []
    for _ in range(n):
        sqft = random.uniform(500, 5000)
        bedrooms = random.choice([1, 2, 3, 4, 5])
        age = random.uniform(0, 50)
        neighborhood = random.choice(["downtown", "suburbs", "rural"])
        has_pool = random.choice([True, False])

        # 模拟缺失值（约5-8%的缺失率）
        sqft_with_missing = sqft if random.random() > 0.05 else None
        age_with_missing = age if random.random() > 0.08 else None

        # 生成目标（房价）
        price = (
            50 * sqft
            + 20000 * bedrooms
            - 1000 * age
            + (50000 if neighborhood == "downtown" else 10000 if neighborhood == "suburbs" else 0)
            + (15000 if has_pool else 0)
            + random.gauss(0, 20000)
        )

        data.append({
            "sqft": sqft_with_missing,
            "bedrooms": bedrooms,
            "age": age_with_missing,
            "neighborhood": neighborhood,
            "has_pool": has_pool,
            "price": price,
        })
    return data


if __name__ == "__main__":
    data = make_housing_data(200)

    print("=== 原始数据样例 ===")
    for row in data[:3]:
        print(f"  {row}")

    sqft_raw = [d["sqft"] for d in data]
    age_raw = [d["age"] for d in data]
    prices = [d["price"] for d in data]

    print("\n=== 缺失值处理 ===")
    sqft_missing = sum(1 for v in sqft_raw if v is None)
    age_missing = sum(1 for v in age_raw if v is None)
    print(f"  平方英尺缺失：{sqft_missing}/{len(sqft_raw)}")
    print(f"  房龄缺失：{age_missing}/{len(age_raw)}")

    sqft_indicator = add_missing_indicator(sqft_raw)
    age_indicator = add_missing_indicator(age_raw)
    sqft_imputed, sqft_fill = impute_median(sqft_raw)
    age_imputed, age_fill = impute_mean(age_raw)
    print(f"  平方英尺用中位数填充：{sqft_fill:.0f}")
    print(f"  房龄用均值填充：{age_fill:.1f}")

    print("\n=== 数值变换 ===")
    sqft_scaled = standardize(sqft_imputed)
    age_scaled = min_max_scale(age_imputed)
    sqft_log = log_transform(sqft_imputed)
    age_binned = bin_values(age_imputed, n_bins=5)
    print(f"  平方英尺标准化：均值={sum(sqft_scaled)/len(sqft_scaled):.4f}, 标准差={math.sqrt(sum(v**2 for v in sqft_scaled)/len(sqft_scaled)):.4f}")
    print(f"  房龄最小最大：[{min(age_scaled):.2f}, {max(age_scaled):.2f}]")
    print(f"  房龄分箱：{sorted(set(age_binned))}")

    print("\n=== 分类编码 ===")
    neighborhoods = [d["neighborhood"] for d in data]

    ohe, ohe_cats = one_hot_encode(neighborhoods)
    print(f"  独热类别：{ohe_cats}")
    print(f"  样例编码：{neighborhoods[0]} -> {ohe[0]}")

    le, le_map = label_encode(neighborhoods)
    print(f"  标签编码映射：{le_map}")

    te, te_map = target_encode(neighborhoods, prices, smoothing=10)
    print(f"  目标编码：{(dict((k, round(v)) for k, v in te_map.items()))}")

    print("\n=== 文本特征 ===")
    descriptions = [
        "large modern house with pool",
        "small cozy cottage near downtown",
        "spacious family home with large yard",
        "modern apartment downtown with view",
        "rustic cabin in rural area",
    ]
    cv, cv_vocab = count_vectorize(descriptions)
    print(f"  词汇表大小：{len(cv_vocab)}")
    print(f"  文档0非零特征数：{sum(1 for v in cv[0] if v > 0)}")

    tf, tf_vocab = tfidf(descriptions)
    print(f"  TF-IDF 词汇表大小：{len(tf_vocab)}")
    top_words = sorted(tf_vocab.keys(), key=lambda w: tf[0][tf_vocab[w]], reverse=True)[:3]
    print(f"  文档0 Top TF-IDF 词：{top_words}")

    print("\n=== 多项式特征 ===")
    sample_row = [sqft_scaled[0], age_scaled[0]]
    poly = polynomial_features(sample_row, degree=2)
    print(f"  输入：{[round(v, 4) for v in sample_row]}")
    print(f"  多项式：{[round(v, 4) for v in poly]}")
    print(f"  特征：[x1, x2, x1^2, x2^2, x1*x2]")

    print("\n=== 特征选择 ===")
    feature_matrix = [
        [sqft_scaled[i], age_scaled[i], float(sqft_indicator[i]), float(age_indicator[i])]
        + ohe[i]
        for i in range(len(data))
    ]

    print(f"  总特征数：{len(feature_matrix[0])}")

    surviving_var = variance_threshold(feature_matrix, threshold=0.01)
    print(f"  方差阈值（0.01）后保留：{len(surviving_var)} 个特征")

    surviving_corr = remove_correlated(feature_matrix, threshold=0.9)
    print(f"  相关性过滤（0.9）后保留：{len(surviving_corr)} 个特征")

    binary_prices = [1 if p > sum(prices) / len(prices) else 0 for p in prices]
    print("\n  与目标的互信息：")
    feature_names = ["sqft", "age", "sqft_missing", "age_missing"] + [f"neigh_{c}" for c in ohe_cats]
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        mi = mutual_information(col, binary_prices, n_bins=10)
        print(f"    {feature_names[j]}：MI={mi:.4f}")

    print("\n  与价格的相关性：")
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        corr = correlation(col, prices)
        print(f"    {feature_names[j]}：r={corr:.4f}")
```

## 使用

用 scikit-learn，这些变换是可组合的流水线：

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["sqft", "age"]),
    ("cat", categorical_pipe, ["neighborhood"]),
])
```

从零实现的版本精确展示了每个变换内部发生了什么。库版本添加了边界情况处理、稀疏矩阵支持和流水线组合，但数学原理是相同的。

## 交付

本课产出：
- `outputs/prompt-feature-engineer.md` - 一个系统化地从原始数据工程化特征的提示词

## 练习

1. 添加鲁棒缩放（使用中位数和四分位距代替均值和标准差）到数值变换中。在有极端离群点的数据上将其与标准缩放进行比较。

2. 实现留一目标编码：对每行，计算不包括该行自身目标值的目标均值。展示这如何减少与朴素目标编码相比的过拟合。

3. 构建自动化特征选择流水线，结合方差阈值、相关性过滤和互信息排序。将其应用于房价数据集，并比较使用全部特征 vs 精选特征时的模型性能（使用简单线性回归）。

## 关键术语

| 术语 | 说法 | 实际含义 |
|------|------|---------|
| 特征工程 | "制作新列" | 将原始数据转换为暴露模式给模型的表示 |
| 标准化 | "使其正态化" | 减去均值并除以标准差，使特征均值为0、标准差为1 |
| 独热编码 | "制作虚拟变量" | 每类别创建一列二进制值，每行恰好有一列为1 |
| 目标编码 | "用答案来编码" | 用该类别的平均目标值替换每个类别，加平滑以防止过拟合 |
| TF-IDF | "花哨的词计数" | 词频乘以逆文档频率：按词在语料库中的独特程度加权 |
| 填充 | "填补空白" | 用估计值（均值、中位数、众数或模型预测）替换缺失值 |
| 特征选择 | "扔掉坏列" | 删除添加噪声或冗余的特征，只保留对目标有信号的 |
| 互信息 | "一件事告诉你多少关于另一件事" | 通过观察变量 X 获得的对变量 Y 不确定性减少程度的度量 |
| 数据泄露 | "意外作弊" | 在训练期间使用了预测时不会有的信息，导致结果虚假乐观 |

## 扩展阅读

- [Feature Engineering and Selection (Max Kuhn & Kjell Johnson)](http://www.feat.engineering/) - 涵盖特征工程全貌的免费在线书籍
- [scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html) - 所有标准变换的实用参考
- [Target Encoding Done Right (Micci-Barreca, 2001)](https://dl.acm.org/doi/10.1145/507533.507538) - 带平滑的目标编码原始论文