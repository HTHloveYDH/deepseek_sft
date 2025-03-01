import json
from typing import Dict, List, Union

import requests
import streamlit as st

from prompts import CHAT_PROMPTS_MAP
from utils import has_chinese_char


# 设置远程服务器的地址
REMOTE_SERVER_URL: str = (
    "http://localhost:8000/v1/chat/completions"  # 替换为你的服务器地址和端口
)
CHAT_MODEL: str = "./models/DeepSeek-R1-Distill-Qwen-32B"
MODE_MAP: Dict = {
    "医学": "medical",
    "闲聊": "chat",
}
prompt: str = ""
cot_finished: bool = False
new_text: str = ""
generated_text: str = ""

# Streamlit页面配置
st.set_page_config(
    page_title="AI助手（powered by VLLM Remote Inference）", layout="wide"
)
# 显示页面title
st.title("你的AI助手")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个人工智能助手"}]

# 显示聊天历史，不显示第一条系统消息
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 显示侧边栏在侧边栏中添加参数设置
st.sidebar.header("设置")

# 显示选择盒1
ai_assistant_mode: str = st.sidebar.radio(
    "AI助手模式：",
    options=["医学", "闲聊"],
    index=0,  # 默认选中第一个选项
)
intent: str = MODE_MAP.get(ai_assistant_mode, "medical")

# 显示选择盒2
chat_round_mode: str = st.sidebar.radio(
    "对话模式：",
    options=["单轮对话", "多轮对话"],
    index=0,  # 默认选中第一个选项
)

# 显示选择盒3
show_cot: str = st.sidebar.radio(
    "显示思考过程：",
    options=["是", "否"],
    index=0,  # 默认选中第一个选项
)

# 显示滑块控件1
temperature: float = st.sidebar.slider(
    "温度系数", min_value=0.0, max_value=100.0, value=50.0, step=0.5  # 默认值
)

# 显示滑块控件2
top_p: float = st.sidebar.slider(
    "采样概率", min_value=0.0, max_value=1.0, value=0.9, step=0.05  # 默认值
)

# 在机器人位置弹出默认的打招呼用户
with st.chat_message("assistant"):
    st.write("我是您的AI助手，请问有什么可以帮到您？")

if user_input := st.chat_input("请输入您的问题"):
    # 用户输入
    with st.chat_message("user"):
        st.markdown(user_input)
    if has_chinese_char(user_input):
        prompt = (
            CHAT_PROMPTS_MAP.get("zh", {"medical": CHAT_PROMPTS_MAP["zh"]["medical"]})
            .get(intent, CHAT_PROMPTS_MAP["zh"]["medical"])
            .format(user_input, "")
        )
    else:
        prompt = (
            CHAT_PROMPTS_MAP.get("en", {"medical": CHAT_PROMPTS_MAP["en"]["medical"]})
            .get(intent, CHAT_PROMPTS_MAP["en"]["medical"])
            .format(user_input, "")
        )
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )  # 添加用户消息添加到历史消息列表

    # 调用 vllm 服务并流式显示模型（助手）响应
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # 用户消息占位符
        generated_text = ""

        # 构造请求数据
        request_data = {
            "model": CHAT_MODEL,
            "messages": (
                st.session_state.messages[:-1] + [{"role": "user", "content": prompt}]
                if chat_round_mode == "多轮对话"
                else [{"role": "user", "content": prompt}]
            ),  # 不会改变st.session_state.messages的内容
            "stream": True,
        }

        # 发送流式请求
        response = requests.post(REMOTE_SERVER_URL, json=request_data, stream=True)

        # 逐行接收流式响应
        cot_finished = False
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                try:
                    chunk = json.loads(decoded_line[6:])  # 去除开头的"data: "部分
                    if "content" in chunk.get("choices", [{}])[0].get("delta", {}):
                        new_text = chunk["choices"][0]["delta"]["content"]
                        # 判断思考过程是否结束
                        if new_text == "</think>":
                            cot_finished = True
                            if show_cot == "是":
                                new_text = "\n--------------------------------------------------以上是思考过程---------------------------------------------------------"
                            else:
                                new_text = ""
                        if show_cot == "是" or cot_finished:
                            generated_text += new_text
                            message_placeholder.markdown(generated_text)
                except json.JSONDecodeError:
                    pass

        # 将生成的模型（助手）消息添加到历史消息列表
        st.session_state.messages.append(
            {"role": "assistant", "content": generated_text}
        )

# 添加一些说明
# st.markdown(
#     """
# ### 使用说明
# - 输入您的问题后，点击"发送"按钮。
# - AI助手的回答结果将显示在下方。
# - 如果遇到问题，请检查远程服务器是否正常运行。
# """
# )

# 显示上传按钮
upload_button = st.button("上传文件")
if upload_button:
    # 显示文件上传框
    uploaded_file = st.file_uploader(
        "请选择您的文件",
        type=["txt", "csv", "pdf", "docx", "xlsx", "pptx"],
        help="支持文本文件、CSV文件或PDF文件",
    )  # 上传文件
    if uploaded_file:
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

# 显示清空聊天历史按钮
clear_button = st.button("清空聊天")  # 清空聊天历史
if clear_button:
    st.session_state.messages = [{"role": "system", "content": "你是一个人工智能助手"}]
    st.rerun()
