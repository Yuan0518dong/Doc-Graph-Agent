import streamlit as st
import uuid
import sys
import os

# 确保能找到 src 目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.self_rag import app  # 导入你写好的 Agent
from langchain_core.messages import HumanMessage, AIMessage

# === 1. 基础配置 ===
st.set_page_config(page_title="项目申报醒题助手", layout="wide", page_icon="🚀")

# 强制设置中文字体显示（Streamlit 默认支持）
st.title("🛡️ 项目申报醒题助手")
st.markdown("---")

# === 2. 初始化 Session State (网页的记忆) ===
# 这里的 messages 存的是网页显示的对话，thread_id 存的是发给 Agent 的唯一标识
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "steps_log" not in st.session_state:
    st.session_state.steps_log = []

# === 3. 侧边栏：思维轨迹可视化 (Tracing) ===
with st.sidebar:
    st.header("🧠 Agent 思维引擎")
    st.caption(f"会话 ID: {st.session_state.thread_id}")

    st.subheader("思维轨迹 (Current Reasoning)")
    if st.session_state.steps_log:
        for i, step in enumerate(st.session_state.steps_log):
            st.info(f"{i + 1}. {step}")
    else:
        st.write("暂无轨迹，请开始提问。")

    if st.button("🔴 清空所有记忆"):
        st.session_state.chat_history = []
        st.session_state.steps_log = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# === 4. 主界面：聊天流显示 ===
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === 5. 用户输入与后端联动 ===
if prompt := st.chat_input("请输入你的问题，例如：本项目立项依据是否充分？"):

    # 1. 显示并记录用户提问
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端 Self-RAG Agent
    with st.chat_message("assistant"):
        # 创建一个空容器，用于后续“蹦字”
        response_placeholder = st.empty()
        full_response = ""

        with st.spinner("Agent 正在深度思考并校验原文..."):
            # 运行你的 LangGraph 逻辑
            result = app.invoke(
                {"messages": [HumanMessage(content=prompt)], "loop_count": 0},
                config={"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 15}
            )

            # 拿到最终答案
            answer = result["messages"][-1].content

            # 模拟流式输出 (Typewriter Effect)
            import time

            # 按照字符或者词切割（中文建议按字符）
            for char in answer:
                full_response += char
                # 在空容器里实时渲染当前已生成的文字，后面加个光标 ▌
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)  # 调节这个数字可以控制蹦字速度

            # 蹦字完成后，去掉光标，显示最终版
            response_placeholder.markdown(full_response)
            # === 原文溯源折叠框  ===
            # 我们检查最后一条系统通知里是否有原文（我们在 self_rag 里加过的）
            evidence = ""
            for m in reversed(result["messages"]):
                if isinstance(m, HumanMessage) and "【原文证据库】" in m.content:
                    evidence = m.content
                    break

            if evidence:
                with st.expander("🔍 查看申报书原文依据"):
                    st.caption("以下内容检索自底层知识库，由 Agent 质检通过：")
                    st.code(evidence, language="markdown")

            # 3. 把最终答案存入历史记录（注意：这里只存 AI 的回答，不存 evidence，以免重复）
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            # 4. 更新侧边栏的思维轨迹
            st.session_state.steps_log = result.get("steps", [])

            # 5. 触发页面重绘，让侧边栏刷新
            st.rerun()
