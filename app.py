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
prompt: str
messages_history: List[str] = []

# Streamlit 页面配置
st.set_page_config(page_title="医学问题助手（powered by VLLM Remote Inference）", layout="wide")
st.title("医学问题小助手")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个人工智能助手"}]

# 显示聊天历史，不显示第一条系统消息
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 上传文件
uploaded_file = st.file_uploader("请选择您的文件", type=["txt", "csv", "pdf"], help="支持文本文件、CSV文件或PDF文件")
upload_button = st.button("上传文件")
if upload_button and uploaded_file:
    st.header("已上传文件")
    if uploaded_file.name.endswith(".txt") or uploaded_file.name.endswith(".csv"):
        # file_content = uploaded_file.read().decode("utf-8")  # 读取文本或CSV文件
        pass
    elif uploaded_file.name.endswith(".pdf"):
        # st.write("PDF文件无法直接显示内容，请使用支持PDF的工具查看。")
        pass
    else:
        st.write("暂不支持该文件类型，无法处理。")
    st.write("文件名：", uploaded_file.name)
elif upload_button and not uploaded_file:
    st.warning("请先选择至少一个文件再点击上传文件按钮")

# 清空聊天历史
clear_button = st.button("清空聊天")
if clear_button:
    st.session_state.messages = [{"role": "system", "content": "你是一个人工智能助手"}]
    st.rerun()

if user_input := st.chat_input("请输入您的问题"):
    # 用户输入
    with st.chat_message("user"):
        st.markdown(user_input)
    if has_chinese_char(user_input):
        prompt = medical_chat_prompt_zh.format(user_input, "")
    else:
        prompt = medical_chat_prompt_en.format(user_input, "")
    st.session_state.messages.append({"role": "user", "content": user_input})  # 添加用户消息添加到历史消息列表

    # 调用 vllm 服务并流式显示模型（助手）响应
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # 用户消息占位符
        generated_text = ""

        # 构造请求数据
        request_data = {
            "model": "./models/DeepSeek-R1-Distill-Qwen-32B",
            "messages": st.session_state.messages[:-1] + [prompt],  # 不会改变st.session_state.messages的内容
            "stream": True
        }

        # 发送流式请求
        response = requests.post(VLLM_API_URL, json=request_data, stream=True)

        # 逐行接收流式响应
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                try:
                    chunk = json.loads(decoded_line)
                    if "content" in chunk.get("choices", [{}])[0].get("delta", {}):
                        new_text = chunk["choices"][0]["delta"]["content"]
                        generated_text += new_text
                        message_placeholder.markdown(generated_text)
                except json.JSONDecodeError:
                    pass

        # 将生成的模型（小助手）消息添加到历史消息列表
        st.session_state.messages.append({"role": "assistant", "content": generated_text})







# # 输入框
# user_input: str = st.text_area(
#     "请输入您的问题", height=200, placeholder="在这里输入您的问题或文本..."
# )
# prompt: str
# if has_chinese_char(user_input):
#     prompt = medical_chat_prompt_zh.format(user_input, "")
# else:
#     prompt = medical_chat_prompt_en.format(user_input, "")

# # 按钮
# if st.button("发送"):
#     if user_input.strip():
#         # 构造请求数据
#         data = {
#             "model": "./models/DeepSeek-R1-Distill-Qwen-32B",
#             "messages": [{"role": "user", "content": prompt}],
#         }

#         # 发送 POST 请求到远程服务器
#         try:
#             response = requests.post(REMOTE_SERVER_URL, json=data)
#             response.raise_for_status()  # 检查请求是否成功
#             result: Dict = response.json()
#             print(result)
#             # 显示结果
#             st.subheader("建议")

#             # 修正内容提取路径
#             content: str = (
#                 result.get("choices", [{}])[0]
#                 .get("message", {})
#                 .get("content", "亲，好像有问题，请重新试一下呢~")
#             )
#             content = content.split("</think>")[1]
#             if not content:
#                 content = "亲，好像有问题，请重新试一下呢~"

#             st.write(content)

#         except requests.exceptions.RequestException as e:
#             st.error(f"请求失败: {e}")
#     else:
#         st.warning("请输入一个有效的提示！")

# 添加一些说明
st.markdown(
    """
### 使用说明
- 输入您的问题后，点击"发送"按钮。
- 医学小助手的回答结果将显示在下方。
- 如果遇到问题，请检查远程服务器是否正常运行。
"""
)
