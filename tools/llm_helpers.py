#!/usr/bin/env python3
"""
统一LLM调用接口，支持多种后端。

用法:
    from llm_helpers import call_llm_json, call_llm_text
    result = call_llm_json("prompt", model="gpt-4o-mini")
"""
import os
import json

# 尝试导入OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  警告: openai 包未安装，请运行: pip install openai")


def _make_client() -> OpenAI:
    """Create an OpenAI-compatible client from environment configuration."""
    base_url = (
        os.environ.get("LLM_API_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://localhost:4141/v1"
    )
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "dummy"
    )

    try:
        import httpx

        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(trust_env=False),
        )
    except ImportError:
        return OpenAI(base_url=base_url, api_key=api_key)


def call_llm_json(prompt: str, model: str = "gpt-4.1", max_tokens: int = None) -> dict:
    """
    调用LLM并返回JSON格式结果。

    Args:
        prompt: 提示词
        model: 模型名称（gpt-4.1, gpt-4, etc）
        max_tokens: 最大token数（None = 使用模型默认值）

    Returns:
        解析后的JSON dict

    Raises:
        ValueError: 如果LLM输出不是有效JSON
        RuntimeError: 如果API调用失败
    """
    if not HAS_OPENAI:
        raise RuntimeError("openai 包未安装")

    client = _make_client()

    try:
        # 构建请求参数
        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        # 只在明确指定时才设置 max_tokens
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        resp = client.chat.completions.create(**create_kwargs)

        txt = resp.choices[0].message.content

        # ⭐ 去除可能的 markdown 代码块包装
        # 某些模型即使在 JSON mode 下也可能返回 ```json ... ```
        txt = txt.strip()
        if txt.startswith("```"):
            # 去除开头的 ```json 或 ```
            lines = txt.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            # 去除结尾的 ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            txt = '\n'.join(lines)

        return json.loads(txt)

    except json.JSONDecodeError as e:
        raise ValueError(f"LLM输出不是有效JSON: {txt[:500]}...") from e
    except Exception as e:
        raise RuntimeError(f"API调用失败: {e}") from e


def call_llm_text(prompt: str, model: str = "gpt-4.1", max_tokens: int = 1000) -> str:
    """
    调用LLM并返回纯文本结果（不要求JSON格式）。

    Args:
        prompt: 提示词
        model: 模型名称
        max_tokens: 最大token数

    Returns:
        LLM输出的文本
    """
    if not HAS_OPENAI:
        raise RuntimeError("openai 包未安装")

    client = _make_client()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"API调用失败: {e}") from e


# 测试函数
if __name__ == "__main__":
    print("测试 call_llm_json...")
    result = call_llm_json('输出JSON: {"test": "hello"}', model="gpt-4.1")
    print(f"✓ 结果: {result}")

    print("\n测试 call_llm_text...")
    text = call_llm_text("说'你好'", model="gpt-4.1")
    print(f"✓ 结果: {text}")
