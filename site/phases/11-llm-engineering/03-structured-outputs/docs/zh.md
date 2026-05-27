# Structured Outputs: JSON, Schema Validation, Constrained Decoding

> 你的 LLM 返回一个字符串。你的应用需要 JSON。这个差距让更多生产系统崩溃，比任何模型幻觉都多。结构化输出是自然语言和类型数据之间的桥梁。做好它，你的 LLM 成为一个可靠的 API。做不好，你凌晨 3 点用 regex 解析自由文本。

**类型:** 构建
**语言:** Python
**前置知识:** Phase 10，Lessons 01-05（从头学 LLM）
**时间:** 约 90 分钟
**相关:** Phase 5 · 20（Structured Outputs & Constrained Decoding）涵盖解码器级别理论（FSM/CFG logit 处理器、Outlines、XGrammar）。本课聚焦生产 SDK 表面（OpenAI `response_format`、Anthropic tool use、Instructor）——如果你想了解 API 以下发生了什么，先读 Phase 5 · 20。

## 学习目标

- 使用 OpenAI 和 Anthropic API 参数实现 JSON 模式和 schema 约束输出
- 构建一个拒绝格式错误的 LLM 输出并用错误反馈重试的 Pydantic 验证层
- 解释 constrained decoding 如何在 token 级别强制有效 JSON 而不需要后处理
- 设计鲁棒的提取 prompt，将非结构化文本可靠转换为类型化数据结构

## 问题

你问一个 LLM："从这段文本中提取产品名称、价格和库存状态。"它回复：

```
产品是 Sony WH-1000XM5 耳机，价格 $348.00，目前有货。
```

这是一个完全正确的答案。但对你的应用完全没用。你的库存系统需要 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要具有特定键、特定类型和特定值约束的 JSON 对象。你不需要一个句子。

Naive 解决方案：在 prompt 中加"用 JSON 响应"。这 90% 的时间有效。另外 10% 的时间模型用 markdown 代码块包装 JSON，或添加"这是 JSON："这样的前导，或产生语法无效的 JSON（因为提前关闭了括号）。你的 JSON 解析器崩溃。你的 pipeline 断开。你加了 try/except 和重试循环。重试有时产生不同数据。现在你在解析问题之上有了一致性问题。

这不是 prompt engineering 问题，是解码问题。模型从左到右生成 token。在每个位置，它从 100K+ 选项的词汇表中挑选最可能的下一个 token。大多数选项在任意给定位置都会产生无效 JSON。如果模型刚发出 `{"price":`，下一个 token 必须是数字、引号（字符串）、`null`、`true`、`false` 或负号。任何其他都会产生无效 JSON。没有约束，模型可能挑选一个完全合理的英语单词，但语法上是灾难性的错误。

## 概念

### 结构化输出谱系

有四个级别的结构化输出控制，可靠性依次增强。

```mermaid
graph LR
    subgraph Spectrum["结构化输出谱系"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% 有效"] --> B["JSON Mode\n保证有效 JSON\n无 schema 保证"]
        B --> C["Schema Mode\nJSON + 匹配 schema\n保证合规"]
        C --> D["Constrained Decoding\nToken 级别强制\n100% 合规"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**Prompt-based**（"用有效 JSON 响应"）：无强制。模型通常遵守但有时不。可靠性：~90%。失败模式：markdown 围栏、前导文本、截断输出、错误结构。

**JSON mode**：API 保证输出是有效 JSON。OpenAI 的 `response_format: { type: "json_object" }` 启用这个。输出将无解析错误地解析。但可能不匹配你的预期 schema——额外的键、错误的类型、缺失的字段。

**Schema mode**：API 获取一个 JSON Schema 并保证输出匹配它。2026 年每个主要供应商都原生支持：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也作为 `tool_choice="required"`）、Anthropic 的 tool use 加 `input_schema` 和 Gemini 的 `response_schema` 加 `response_mime_type: "application/json"`。输出有你指定的确切键、类型和约束。

**Constrained decoding**：在生成期间每个 token 位置，解码器将所有会产生无效输出的 token 屏蔽掉。如果 schema 要求数字而模型即将发出一个字母，该 token 的概率被设为零。模型只能产生导致有效输出的 token。这是 OpenAI 的 structured output mode 和 Outlines、Guidance 等库在底层实现的。

### JSON Schema：契约语言

JSON Schema 是你告诉模型（或验证层）输出必须是什么形状的方式。每个主要结构化输出系统都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

此 schema 说：输出必须是一个带有字符串 `product`、非负数 `price`、布尔 `in_stock` 和可选字符串数组 `categories` 的对象。任何不匹配的输出都会被拒绝。

Schema 处理困难情况：嵌套对象、带类型项的数组、枚举（将字符串约束为特定值）、模式匹配（字符串上的 regex）和组合器（oneOf、anyOf、allOf，用于多态输出）。

### Pydantic 模式

在 Python 中，你不手工写 JSON Schema。你定义一个 Pydantic 模型，它为你生成 schema。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这产生与上面相同的 JSON Schema。Instructor 库（和 OpenAI 的 SDK）直接接受 Pydantic 模型：传递模型类，得到验证实例。如果 LLM 输出不匹配，Instructor 自动重试。

### Function Calling / Tool Use

同一问题的替代接口。不是要求模型直接产生 JSON，而是定义带类型参数的"工具"（函数）。模型输出带有结构化参数的工具调用。OpenAI 称之为"function calling"。Anthropic 称之为"tool use"。结果是相同的：结构化数据。

```mermaid
graph TD
    subgraph ToolUse["Tool Use 流程"]
        U["用户：从这段评论文本\n提取产品信息"] --> M["模型处理输入"]
        M --> TC["工具调用：\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["根据函数 schema\n验证"]
        V --> R["结构化结果：\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当你有 10 个不同的提取 schema 且模型必须根据输入选择正确的一个时，tool use 是首选。它同时给你 schema 选择和结构化输出。

### 常见失败模式

即使有 schema 强制，结构化输出也会以微妙的方式失败。

**幻觉值**：输出匹配 schema 但包含虚构数据。模型输出 `{"price": 299.99}` 而文本说是 $348。Schema 验证无法捕获这个——类型正确，值错误。

**枚举混淆**：你将字段约束为 `["in_stock", "out_of_stock", "preorder"]`。模型输出 `"available"`——语义正确，但不在允许集合中。好的 constrained decoding 防止这个。Prompt-based 方法做不到。

**嵌套对象深度**：深度嵌套的 schema（4+ 层）产生更多错误。每层嵌套都是模型可能丢失结构的另一个地方。

**数组长度**：模型可能产生太多或太少的数组项。Schema 支持 `minItems` 和 `maxItems` 但不是所有供应商都在解码级别强制它们。

**可选字段省略**：模型省略技术上可选但对你的用例语义上重要的字段。在 schema 中将它们设为 required——即使数据有时缺失——强制模型显式产生 `null`。

## 构建

### 第 1 步：JSON Schema 验证器

从零构建一个验证器，检查 Python 对象是否匹配 JSON Schema。这是运行在输出端验证合规性的。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### 第 2 步：Pydantic 风格 Model 到 Schema

构建一个最小类到 schema 转换器。定义一个 Python 类并自动生成其 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### 第 3 步：Constrained Token 过滤器

模拟 constrained decoding。给定部分 JSON 字符串和 schema，确定当前位置哪些 token 类别有效。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### 第 4 步：提取 Pipeline

将所有内容组合成一个提取 pipeline：定义 schema，模拟 LLM 产生结构化输出，验证输出，并处理重试。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### 第 5 步：运行完整 Pipeline

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 使用

### OpenAI Structured Outputs

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的 structured output mode 在内部使用 constrained decoding。模型生成的每个 token 都保证产生匹配 Pydantic schema 的输出。不需要重试。不需要验证。约束被烘入解码过程。

### Anthropic Tool Use

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic 通过 tool use 实现结构化输出。模型发出带有匹配 input_schema 的结构化参数的工具调用。相同结果，不同 API 表面。

### Instructor 库

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor 包装任何 LLM 客户端并添加带验证的自动重试。如果第一次尝试验证失败，它将错误作为上下文发送回模型并要求修复输出。这适用于任何供应商，而不仅仅是 OpenAI。

## 交付

本课产生 `outputs/prompt-structured-extractor.md` —— 一个可复用的 prompt 模板，给定 schema 定义从任何文本中提取结构化数据。输入 JSON Schema 和非结构化文本，得到验证后的 JSON。

还产生 `outputs/skill-structured-outputs.md` —— 一个决策框架，用于根据你的供应商、可靠性要求和 schema 复杂度选择正确的结构化输出策略。

## 练习

1. 扩展 schema 验证器支持 `oneOf`（数据必须恰好匹配几个 schema 之一）。这处理多态输出——例如，一个字段可以是不同形状的 `Product` 或 `Service` 对象。

2. 构建一个"schema diff"工具，比较两个 schema 并识别破坏性变更（删除 required 字段、改变类型）vs 非破坏性变更（添加可选字段、放宽约束）。这对于生产中的版本化提取 schema 至关重要。

3. 实现一个更现实的 constrained decoding 模拟器。给定一个 JSON Schema 和 100 个 token 的词汇表（字母、数字、标点、关键词），逐步逐步生成，在每个位置屏蔽无效 token。测量每个步骤中有多少百分比的词汇表是有效的。

4. 构建一个提取 eval 套件。创建 50 个带有手工标记 JSON 输出的产品描述。在所有 50 个上运行你的提取 pipeline 并测量精确匹配、字段级准确性和类型合规性。识别哪些字段最难正确提取。

5. 在你的提取 pipeline 中添加"置信度分数"。对于每个提取字段，根据 token 概率估计模型有多自信（或通过运行提取 3 次并测量一致性）。标记低置信度字段供人工审核。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| JSON mode | "返回 JSON" | API 标志，保证语法有效的 JSON 输出，但不强制任何特定 schema |
| Structured output | "类型化 JSON" | 匹配特定 JSON Schema 的输出，具有正确的键、类型和约束 |
| Constrained decoding | "引导生成" | 在每个 token 位置，屏蔽会产生无效输出的 token——保证 100% schema 合规 |
| JSON Schema | "JSON 模板" | 用于描述 JSON 数据结构、类型和约束的声明式语言（被 OpenAPI、JSON Forms 等使用）|
| Pydantic | "Python 数据类+" | Python 库，用类型验证定义数据模型，被 FastAPI 和 Instructor 用于生成 JSON Schema |
| Function calling | "Tool use" | LLM 输出结构化函数调用（名称 + 类型参数）而不是自由文本——OpenAI 和 Anthropic 都支持 |
| Instructor | "LLM 的 Pydantic" | Python 库，包装 LLM 客户端返回验证的 Pydantic 实例，带验证失败自动重试 |
| Token masking | "过滤词汇表" | 在生成期间将特定 token 概率设为零，使模型无法产生它们 |
| Schema compliance | "匹配形状" | 输出具有每个 required 字段、正确类型、值在约束内，且无额外不允许字段 |
| Retry loop | "重试直到成功" | 将验证错误发送回模型并要求它修复输出——Instructor 自动执行此操作，可配置最大重试次数 |

## 扩展阅读

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) —— OpenAI API 中基于 JSON Schema 的 constrained decoding 官方文档
- [Willard & Louf, 2023 -- "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702) —— Outlines 论文，描述如何将 JSON Schema 编译为有限状态机以进行 token 级别约束
- [Instructor documentation](https://python.useinstructor.com/) —— 从任何 LLM 获取结构化输出的标准库，带 Pydantic 验证和重试
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use) —— Claude 如何通过带 JSON Schema input_schema 的 tool use 实现结构化输出
- [JSON Schema specification](https://json-schema.org/) —— 每个主要结构化输出系统使用的 schema 语言完整规范
- [Outlines library](https://github.com/outlines-dev/outlines) —— 使用 regex 和 JSON Schema 编译为有限状态机的开源 constrained generation
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100) —— 当前最先进的 grammar engine；下推自动机编译，约 100 ns/token 的 token 屏蔽。
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094) —— LMQL 论文，将 constrained decoding 框架为带类型和值约束的查询语言。
- [Microsoft Guidance (framework docs)](https://github.com/guidance-ai/guidance) —— 模板驱动的 constrained generation；vendor-agnostic 补充 Outlines 和 XGrammar。