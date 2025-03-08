from typing import Dict, List, Union
import json

import requests
import streamlit as st

from prompts import medical_chat_prompt_en, medical_chat_prompt_zh, chat_prompt_en, chat_prompt_zh
from utils import has_chinese_char


# 设置远程服务器的地址
REMOTE_SERVER_URL: str = (
    "http://localhost:8000/v1/chat/completions"  # 替换为你的服务器地址和端口
)

# Streamlit 页面配置
st.set_page_config(page_title="医学问题助手（powered by VLLM Remote Inference）", layout="wide")
st.title("医学问题小助手")

# 输入框
user_input: str = st.text_area(
    "请输入您的问题", height=200, placeholder="在这里输入您的问题或文本..."
)
prompt: str
if has_chinese_char(user_input):
    prompt = medical_chat_prompt_zh.format(user_input, "")
else:
    prompt = medical_chat_prompt_en.format(user_input, "")

# 按钮
if st.button("发送"):
    if user_input.strip():
        # 构造请求数据
        request_data = {
            "model": "../models/DeepSeek-R1-Distill-Qwen-32B",
            "messages": [{"role": "user", "content": prompt}],
        }

        # 发送 POST 请求到远程服务器
        try:
            response = requests.post(REMOTE_SERVER_URL, json=request_data)
            response.raise_for_status()  # 检查请求是否成功
            result: Dict = response.json()
            print(result)
            # 修正内容提取路径
            content: str = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "亲，好像有问题，请重新试一下呢~")
            )
            content = content.split("</think>")[1]
            if not content:
                content = "亲，好像有问题，请重新试一下呢~"
            
            # 显示结果
            st.subheader("小助手建议")
            st.write(content)

        except requests.exceptions.RequestException as e:
            st.error(f"请求失败: {e}")
    else:
        st.warning("请输入一个有效的提示！")

# 添加一些说明
st.markdown(
    """
### 使用说明
- 输入您的问题后，点击"发送"按钮。
- 医学小助手的回答结果将显示在下方。
- 如果遇到问题，请检查远程服务器是否正常运行。
"""
)
