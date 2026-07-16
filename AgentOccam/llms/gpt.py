import openai
from openai import OpenAI, AzureOpenAI
import time
import numpy as np
from PIL import Image
import base64
import io
import requests
import os
import httpx

# Token 计数器 - 全局累计
_token_stats = {
    "total_input_tokens": 0,
    "call_count": 0,
}

def get_token_stats():
    """获取当前 token 统计"""
    return _token_stats.copy()

def reset_token_stats():
    """重置 token 统计（每个任务开始时调用）"""
    _token_stats["total_input_tokens"] = 0
    _token_stats["call_count"] = 0

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """计算文本的 token 数量"""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # 如果模型不支持，使用默认编码
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 如果没有安装 tiktoken，或 tokenizer 缓存/下载失败，使用简单估算
        return len(text) // 4

def count_messages_tokens(messages: list, model: str = "gpt-4") -> int:
    """计算消息列表的 token 数量"""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 3  # 每条消息的固定开销
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            if isinstance(message, dict):
                for key, value in message.items():
                    if key == "content":
                        if isinstance(value, str):
                            num_tokens += len(encoding.encode(value))
                        elif isinstance(value, list):
                            # 处理多模态内容
                            for item in value:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        num_tokens += len(encoding.encode(item.get("text", "")))
                                    elif item.get("type") == "image_url":
                                        # 图像大约 85 tokens (低分辨率估计)
                                        num_tokens += 85
                    elif key == "role":
                        num_tokens += len(encoding.encode(value))
                    elif key == "name":
                        num_tokens += len(encoding.encode(value))
                        num_tokens += tokens_per_name

        num_tokens += 3  # 消息结束标记
        return num_tokens
    except Exception:
        # 简单估算；tokenizer 缓存/下载失败不能中断评测
        total_text = ""
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    total_text += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            total_text += item.get("text", "")
        return len(total_text) // 4

# 官方 OpenAI 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)

# 第三方 API 配置
LLM_API_KEY = os.environ.get("LLM_API_KEY", OPENAI_API_KEY)
LLM_API_BASE_URL = os.environ.get("LLM_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")


def _configured_seed():
    value = os.environ.get("LLM_SEED")
    return int(value) if value not in (None, "") else None


def _seed_kwargs():
    seed = _configured_seed()
    return {"seed": seed} if seed is not None else {}

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {OPENAI_API_KEY}"
}
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

def _make_openai_client(base_url=None):
    """Create an OpenAI-compatible client without inheriting broken shell proxies."""
    kwargs = {
        "api_key": LLM_API_KEY,
        "http_client": httpx.Client(trust_env=False),
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)

def _post_without_env_proxy(url, headers, json_payload):
    with requests.Session() as session:
        session.trust_env = False
        return session.post(url, headers=headers, json=json_payload)

def call_gpt(prompt, model_id="gpt-3.5-turbo", system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=0.1):
    num_attempts = 0

    # 计算 input tokens
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_tokens = count_messages_tokens(messages, model_id)
    _token_stats["total_input_tokens"] += input_tokens
    _token_stats["call_count"] += 1

    while True:
        if num_attempts >= 10:
            raise ValueError("OpenAI request failed.")
        try:
            # 支持自定义 API 端点
            if LLM_API_BASE_URL:
                client = _make_openai_client(base_url=LLM_API_BASE_URL)
            else:
                client = _make_openai_client()

            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                **_seed_kwargs(),
            )

            # 兼容不同的响应格式
            try:
                return response.choices[0].message.content.strip()
            except (AttributeError, KeyError):
                # 处理某些代理返回的非标准格式
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    choice = response.choices[0]
                    if hasattr(choice, 'message'):
                        return choice.message.get('content', '').strip() if isinstance(choice.message, dict) else choice.message.content.strip()
                    elif isinstance(choice, dict) and 'message' in choice:
                        return choice['message'].get('content', '').strip()
                raise
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1

def arrange_message_for_gpt(item_list):
    def image_path_to_bytes(file_path):
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
        return image_bytes
    combined_item_list = []
    previous_item_is_text = False
    text_buffer = ""
    for item in item_list:
        if item[0] == "image":
            if len(text_buffer) > 0:
                combined_item_list.append(("text", text_buffer))
                text_buffer = ""
            combined_item_list.append(item)
            previous_item_is_text = False
        else:
            if previous_item_is_text:
                text_buffer += item[1]
            else:
                text_buffer = item[1]
            previous_item_is_text = True
    if item_list[-1][0] != "image" and len(text_buffer) > 0:
        combined_item_list.append(("text", text_buffer))
    content = []
    for item in combined_item_list:
        item_type = item[0]
        if item_type == "text":
            content.append({
                "type": "text",
                "text": item[1]
            })
        elif item_type == "image":
            if isinstance(item[1], str):
                image_bytes = image_path_to_bytes(item[1])
                image_data = base64.b64encode(image_bytes).decode("utf-8")
            elif isinstance(item[1], np.ndarray):
                image = Image.fromarray(item[1]).convert("RGB")
                width, height = image.size
                image = image.resize((int(0.5*width), int(0.5*height)), Image.LANCZOS)
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='JPEG')
                image_bytes = image_bytes.getvalue()
                image_data = base64.b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                },
            })
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages

def call_gpt_with_messages(
    messages,
    model_id="gpt-3.5-turbo",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    temperature=0.1,
):
    # 计算 input tokens
    full_messages = messages if messages[0]["role"] == "system" else [{"role": "system", "content": system_prompt}] + messages
    input_tokens = count_messages_tokens(full_messages, model_id)
    _token_stats["total_input_tokens"] += input_tokens
    _token_stats["call_count"] += 1

    # 支持自定义 API 端点
    if LLM_API_BASE_URL:
        client = _make_openai_client(base_url=LLM_API_BASE_URL)
    elif AZURE_ENDPOINT:
        client = AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=OPENAI_API_KEY, api_version="2024-02-15-preview")
    else:
        client = _make_openai_client()

    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("OpenAI request failed.")
        try:
            if any("image" in c["type"] for m in messages for c in m["content"]):
                payload = {
                    "model": "gpt-4-turbo",
                    "messages": messages,
                    "temperature": temperature,
                    **_seed_kwargs(),
                }

                # 支持自定义 API 端点处理图像
                if LLM_API_BASE_URL:
                    api_url = f"{LLM_API_BASE_URL.rstrip('/')}/chat/completions"
                    headers_with_key = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {LLM_API_KEY}"
                    }
                else:
                    api_url = "https://api.openai.com/v1/chat/completions"
                    headers_with_key = headers

                response = _post_without_env_proxy(api_url, headers_with_key, payload)
                return response.json()["choices"][0]["message"].get("content", "").strip()
            else:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages if messages[0]["role"] == "system" else [{"role": "system", "content": system_prompt}] + messages,
                    temperature=temperature,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    **_seed_kwargs(),
                )
                # 兼容不同的响应格式
                try:
                    return response.choices[0].message.content.strip()
                except (AttributeError, KeyError):
                    # 处理某些代理返回的非标准格式
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        choice = response.choices[0]
                        if hasattr(choice, 'message'):
                            return choice.message.get('content', '').strip() if isinstance(choice.message, dict) else choice.message.content.strip()
                        elif isinstance(choice, dict) and 'message' in choice:
                            return choice['message'].get('content', '').strip()
                    raise
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        
if __name__ == "__main__":
    prompt = '''CURRENT OBSERVATION:
RootWebArea [2634] 'My Account'
	link [3987] 'My Account'
	link [3985] 'My Wish List'
	link [3989] 'Sign Out'
	text 'Welcome to One Stop Market'
	link [3800] 'Skip to Content'
	link [3809] 'store logo'
	link [3996] 'My Cart'
	combobox [4190] 'Search' [required: False]
	link [4914] 'Advanced Search'
	button [4193] 'Search' [disabled: True]
	tablist [3699]
		tabpanel
			menu "[3394] 'Beauty & Personal Care'; [3459] 'Sports & Outdoors'; [3469] 'Clothing, Shoes & Jewelry'; [3483] 'Home & Kitchen'; [3520] 'Office Products'; [3528] 'Tools & Home Improvement'; [3533] 'Health & Household'; [3539] 'Patio, Lawn & Garden'; [3544] 'Electronics'; [3605] 'Cell Phones & Accessories'; [3620] 'Video Games'; [3633] 'Grocery & Gourmet Food'"
	main
		heading 'My Account'
		text 'Contact Information'
		text 'Emma Lopez'
		text 'emma.lopezgmail.com'
		link [3863] 'Change Password'
		text 'Newsletters'
		text "You aren't subscribed to our newsletter."
		link [3877] 'Manage Addresses'
		text 'Default Billing Address'
		group [3885]
			text 'Emma Lopez'
			text '101 S San Mateo Dr'
			text 'San Mateo, California, 94010'
			text 'United States'
			text 'T:'
			link [3895] '6505551212'
		text 'Default Shipping Address'
		group [3902]
			text 'Emma Lopez'
			text '101 S San Mateo Dr'
			text 'San Mateo, California, 94010'
			text 'United States'
			text 'T:'
			link [3912] '6505551212'
		link [3918] 'View All'
		table 'Recent Orders'
			row '| Order | Date | Ship To | Order Total | Status | Action |'
			row '| --- | --- | --- | --- | --- | --- |'
			row "| 000000170 | 5/17/23 | Emma Lopez | 365.42 | Canceled | View OrderReorder\tlink [4110] 'View Order'\tlink [4111] 'Reorder' |"
			row "| 000000189 | 5/2/23 | Emma Lopez | 754.99 | Pending | View OrderReorder\tlink [4122] 'View Order'\tlink [4123] 'Reorder' |"
			row "| 000000188 | 5/2/23 | Emma Lopez | 2,004.99 | Pending | View OrderReorder\tlink [4134] 'View Order'\tlink [4135] 'Reorder' |"
			row "| 000000187 | 5/2/23 | Emma Lopez | 1,004.99 | Pending | View OrderReorder\tlink [4146] 'View Order'\tlink [4147] 'Reorder' |"
			row "| 000000180 | 3/11/23 | Emma Lopez | 65.32 | Complete | View OrderReorder\tlink [4158] 'View Order'\tlink [4159] 'Reorder' |"
		link [4165] 'My Orders'
		link [4166] 'My Downloadable Products'
		link [4167] 'My Wish List'
		link [4169] 'Address Book'
		link [4170] 'Account Information'
		link [4171] 'Stored Payment Methods'
		link [4173] 'My Product Reviews'
		link [4174] 'Newsletter Subscriptions'
		heading 'Compare Products'
		text 'You have no items to compare.'
		heading 'My Wish List'
		text 'You have no items in your wish list.'
	contentinfo
		textbox [4177] 'Sign Up for Our Newsletter:' [required: False]
		button [4072] 'Subscribe'
		link [4073] 'Privacy and Cookie Policy'
		link [4074] 'Search Terms'
		link [4075] 'Advanced Search'
		link [4076] 'Contact Us'
		text 'Copyright 2013-present Magento, Inc. All rights reserved.'
		text 'Help Us Keep Magento Healthy'
		link [3984] 'Report All Bugs'
Today is 6/12/2023. Base on the aforementioned webpage, tell me how many fulfilled orders I have over the past month, and the total amount of money I spent over the past month.'''
    print(call_gpt(prompt=prompt, model_id="gpt-4-turbo"))
