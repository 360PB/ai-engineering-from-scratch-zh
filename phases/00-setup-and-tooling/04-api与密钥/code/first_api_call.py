import os
import json
import urllib.request


def call_with_sdk():
    try:
        import anthropic
    except ImportError:
        print("安装 SDK: pip install anthropic")
        return

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
    )
    print(f"SDK 响应: {response.content[0].text}")
    print(f"Token 使用量: {response.usage.input_tokens} 输入, {response.usage.output_tokens} 输出")


def call_raw_http():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("请先设置 ANTHROPIC_API_KEY 环境变量")
        return

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What is a neural network in one sentence?"}],
    }).encode()

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"原始 HTTP 响应: {result['content'][0]['text']}")
        print(f"Token 使用量: {result['usage']['input_tokens']} 输入, {result['usage']['output_tokens']} 输出")


if __name__ == "__main__":
    print("=== API 调用 ===\n")
    print("1. 使用 SDK:")
    call_with_sdk()
    print("\n2. 使用原始 HTTP:")
    call_raw_http()
