import streamlit as st
import sys
import os
import time

# === 路径配置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# 引用你的后端逻辑
from src.agent.graph_rag_engine import GraphRAGAgent

# === 1. 页面基础配置 ===
st.set_page_config(
    page_title="Graph RAG Pro",
    page_icon="🕸️",
    layout="wide",  # 以此开启宽屏模式，显得更大气
    initial_sidebar_state="expanded"
)

# 自定义 CSS 让界面更干净
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stStatus {
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# === 2. 初始化 Session State ===
if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "👋 你好！我是基于 **Neo4j 图谱** 的智能助手。\n\n你可以问我关于 **多智能体、SCHMM 框架** 等专业问题，我会基于事实回答。"}
    ]

# === 3. 侧边栏：控制中心 ===
with st.sidebar:
    st.title("🕸️ 控制中心")
    st.markdown("---")

    # 状态指示灯
    if st.session_state.agent is None:
        st.warning("🔴 系统未连接")
        if st.button("🔌 连接知识引擎", type="primary"):
            with st.spinner("正在初始化图谱连接..."):
                try:
                    st.session_state.agent = GraphRAGAgent()
                    st.toast("连接成功！", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"连接失败: {e}")
    else:
        st.success("🟢 系统在线")
        st.caption("已连接 Neo4j 数据库")

    st.markdown("---")

    # 高级参数
    with st.expander("⚙️ 检索参数设置"):
        retrieval_limit = st.slider("检索切片数量 (Limit)", 1, 10, 5, help="每次回答参考多少条背景知识")
        # 这里虽然UI有了，但要把参数传进去还需要改一下Agent代码，目前先做样子，或者稍后改Agent

    st.markdown("---")
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# === 4. 主聊天区域 ===
st.header("Graph RAG 知识库问答")
st.caption("🚀 Powered by DeepSeek V3 + Neo4j")

# 显示历史消息
for msg in st.session_state.messages:
    # 区分头像：用户用 user，AI 用 robot
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# === 5. 处理输入 ===
# 只有连接成功了才允许输入
if st.session_state.agent and (prompt := st.chat_input("请输入你的问题...")):

    # 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # AI 回答部分
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""

        # --- 核心升级：可视化思考过程 ---
        # 使用 st.status 创建一个可折叠的状态框
        with st.status("🧠 正在思考...", expanded=True) as status:

            # 1. 提取关键词
            st.write("🔍 分析意图 & 提取关键词...")
            # 为了在UI显示，我们需要一点小技巧，或者直接调用chat
            # 但目前的 GraphRAGAgent.chat() 是封装好的。
            # 为了更好的UI体验，建议让 chat 返回中间步骤，但现在为了不改后端，
            # 我们直接调用，并假设它很快。

            # 模拟一个进度条（真实场景里应该由 Agent 返回回调）
            progress_bar = st.progress(0)
            for i in range(30):
                time.sleep(0.01)
                progress_bar.progress(i + 10)

            st.write("📚 在 Neo4j 图谱中检索相关实体...")
            progress_bar.progress(60)

            st.write("⚡ DeepSeek 正在阅读文献并生成答案...")
            progress_bar.progress(90)

            # === 真正调用后端 ===
            try:
                # 调用 agent
                response_text = st.session_state.agent.chat(prompt)

                status.update(label="✅ 思考完成", state="complete", expanded=False)
                full_response = response_text

            except Exception as e:
                status.update(label="❌ 发生错误", state="error")
                st.error(f"处理请求时出错: {e}")
                full_response = "抱歉，系统遇到了一点小问题，请检查后台日志。"

        # 显示最终答案
        if full_response:
            # 模拟打字机效果
            displayed_response = ""
            for char in full_response:
                displayed_response += char
                # 如果字太长，可以稍微快一点
                time.sleep(0.005)
                message_placeholder.markdown(displayed_response + "▌")

            message_placeholder.markdown(displayed_response)

            # 存入历史
            st.session_state.messages.append({"role": "assistant", "content": full_response})

elif not st.session_state.agent:
    st.info("👈 请先在左侧侧边栏点击 **连接知识引擎** 启动系统。")