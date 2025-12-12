"""
会反思的“大脑” (The Orchestrator)
这是基于 LangGraph 构建的状态机，它定义了整个思考的闭环流程

它的核心逻辑不再是一条直线，而是一个有条件的循环：
思考 (Agent Node): “用户问了 Transformer，我要查一下。” -> 生成查询指令

执行 (Tool Node): 调用工具，查回来一段文字

质检 (Calling Grader): (关键点) 这里调用了 grader.py

情况 A (Pass): Grader 说 "yes"。 -> Agent 拿着资料生成最终回答 -> 结束

情况 B (Fail): Grader 说 "no"（资料无关）。 -> Agent 触发自我修正机制：“刚才查偏了，我要换个关键词重查。” -> 回到第 1 步 (Loop)

它赋予了 AI “自我纠错” 的能力。如果第一次没查对，它不会瞎回答，而是会尝试第二次、第三次，直到找到正确资料或达到最大重试次数
"""
import json
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
# 引入组件
from src.agent.tools import search_knowledge_base
from src.agent.grader import grade_document

# === 配置大脑 ===
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen2.5:1.5b",
    temperature=0,
)

# === 定义状态 ===
class AgentState(TypedDict):
    """
    这是一个列表。
    想象它是一个微信聊天记录。
    Agent 往里面加一条（思考），Tool 往里面加一条（资料），Grader 往里面加一条（通知）。

    为什么我们要用 messages[-1]？ 因为我们要看“最新那条消息”。
    """
    messages: Annotated[list[BaseMessage], operator.add] #遇到多个 messages，请使用 + 来合并，而不是覆盖
    loop_count: int # 防止死循环

# === 节点 1: 思考者 (Agent) ===
def agent_node(state: AgentState):
    """
    这是大脑决策中心，负责"变脸"（切换模式）
    [模式切换逻辑]:
    检查历史消息：看最后一条是不是 Grader 发回来的"资料有效"通知
    模式 A (搜查官模式):
       - 触发条件：还没查资料，或者 Grader 说"NO"（资料无效）。
       - 动作：必须输出 JSON {"action": "search"} 去调用工具，不准直接回答
    模式 B (作家模式):
       - 触发条件：Grader 刚才说了"YES"（资料有效）。
       - 动作：禁止再查资料！严禁输出 JSON！直接根据手头的资料写出最终答案
    """
    messages = state["messages"]
    loop_count = state.get("loop_count", 0)

    # === 修正后的侦探逻辑 ===
    # 我们不仅要看有没有"资料有效"，还要看它是不是"新鲜"的
    has_valid_context = False
    last_msg = messages[-1]

    # 逻辑 A：刚查完资料 -> 作家模式
    if isinstance(last_msg, HumanMessage) and "【系统通知】：资料有效" in last_msg.content:
        has_valid_context = True
    # 逻辑 B：用户发了新问题 -> 搜查官模式 (强制重置)
    elif isinstance(last_msg, HumanMessage) and "【系统通知】" not in last_msg.content:
        has_valid_context = False


    # === 动态变脸逻辑 ===
    if has_valid_context:
        # 【模式 B：作家模式】
        sys_prompt_content = """
                你是一个技术专家。资料库检索已完成。
                任务：根据资料回答问题。
                 重点：忽略实验数据表格，专注于解释【算法原理】和【架构设计】。
                """
        # 作家模式不需要修改用户消息
        final_messages = [SystemMessage(content=sys_prompt_content)] + messages

    else:
        # 【模式 A：搜查官模式】
        sys_prompt_content = """
                你是一个严谨的研究员。
                1. 遇到问题，**必须**先调用搜索工具。
                2. 格式：{"action": "search", "query": "关键词"}
                """

        # === 核心修复：末尾强指令 (Suffix Prompt) ===
        # 1.5B 模型记性不好，必须在最后一句狠狠踢它一脚
        # 我们复制一份消息列表，以免污染原始状态
        final_messages = messages.copy()

        # 找到最后一条用户消息
        if isinstance(final_messages[-1], HumanMessage):
            original_text = final_messages[-1].content
            # 只有当它还没被修改过时，才追加指令
            if "系统强制要求" not in original_text:
                forced_instruction = f"""
                        {original_text}

                        (系统强制要求：这是一个技术细节问题，你现在的知识库是空的。
                        你**必须**先输出 JSON 调用工具查询，**严禁**直接凭记忆回答！
                        格式示例：{{"action": "search", "query": "Transformer vs RNN advantages"}})
                        """
                final_messages[-1] = HumanMessage(content=forced_instruction)

        final_messages = [SystemMessage(content=sys_prompt_content)] + final_messages

    print(f"[Agent] 第 {loop_count + 1} 次思考 (模式: {'作家' if has_valid_context else '搜查官'})...")

    # 使用处理过的 final_messages 调用模型
    response = llm.invoke(final_messages)

    print(f"[Agent Output]: {response.content[:50]}...")

    return {"messages": [response], "loop_count": loop_count}

# === 节点 2: 路由 (Router) ===
def router_node(state: AgentState):
    """
    这个节点是路由导航
    如果Agent输出的JSON格式的内容，则会放行去tools节点
    否则会直接结束
    Agent -> Router -> (Tools OR End)
    Tools -> Agent (这就构成了环)
    """
    last_msg = state["messages"][-1]
    content = last_msg.content.strip()

    try:
        # 1. 尝试寻找 JSON 的大括号范围
        start = content.find('{')
        end = content.rfind('}') + 1

        # 2. 如果找到了括号，尝试解析
        if start != -1 and end != -1:
            json_str = content[start:end]
            data = json.loads(json_str)

            # 3. 检查字段是否匹配
            # 只要有 action 且是 search，就放行
            if data.get("action") == "search":
                return "tools"

    except Exception as e:
        print(f"[Router] JSON 解析失败: {e}")

    # 没抓到指令，或者指令不对，就结束
    return END

# === 节点 3: 执行与反思 (Tool + Grader) ===
def tool_and_grade_node(state: AgentState):
    """
    这是"手"和"质检员"的结合体。
    1. 执行: 调用 search_knowledge_base 工具真正去查 Neo4j/Chroma。
    2. 质检: 马上调用 grade_document 检查查回来的资料对不对

    [流程分支]:
    - 情况 A (YES): 资料有用 -> 告诉 Agent "资料齐了，请回答"。
    - 情况 B (NO):  资料垃圾 -> 告诉 Agent (注意是回给大脑!) "查偏了，请换个关键词重查"
    """
    messages = state["messages"]
    last_msg = messages[-1]
    loop_count = state.get("loop_count", 0)

    # 1. 解析查询词
    try:
        content = last_msg.content
        start = content.find('{')
        end = content.rfind('}') + 1
        data = json.loads(content[start:end])
        query = data["query"]

        print(f"[Tool] 执行搜索: {query}")
        doc_content = search_knowledge_base.invoke(query)

        # 2. 获取原始问题
        # 假设倒数第2条是 User 的问题（在 Agent 回复之前）
        # 这里为了简便，我们遍历找到最新的 HumanMessage
        user_question = "Unknown"
        for m in reversed(messages):
            if isinstance(m, HumanMessage) and "【系统通知】" not in m.content:
                user_question = m.content
                break

        # 3. Grader 介入质检
        score = grade_document(user_question, doc_content)

        if score == "yes":
            print("[Grader] 资料相关！通过！")
            return {
                "messages": [HumanMessage(content=f"【系统通知】：资料有效。\n内容：{doc_content}\n\n请回答。")],
                "loop_count": loop_count +1
            }
        else:
            print("[Grader] 资料无关！打回重写！")
            # 增加计数，防止死循环
            # 如果重试超过 2 次 (从 0 开始计数，所以是 0, 1, 2)
            if loop_count >= 2:
                print("[System] 重试次数过多，强制熔断！要求 Agent 强行回答。")

                # 这是一个"欺骗"指令：告诉 Agent 资料其实是有的，逼它回答
                # 这样可以打破"必须查工具"的死循环
                forced_instruction = f"""
                                【系统通知】：虽然资料可能不完美，但重试次数已达上限。
                                请忽略"必须查工具"的指令。
                                请根据以下现有信息（或你自己的知识）直接回答问题："{user_question}"
                                """
                return {
                    "messages": [HumanMessage(content=forced_instruction)],
                    "loop_count": loop_count + 1
                }

            return {
                "messages": [HumanMessage(
                    content=f"【系统通知】：你搜索的 '{query}' 结果与问题无关。\n请**更换关键词**重新尝试搜索。")],
                "loop_count": loop_count + 1
            }

    except Exception as e:
        return {"messages": [HumanMessage(content=f"工具调用错误: {e}")]}

# === 构建图谱 LangGraph===
workflow = StateGraph(AgentState)

workflow.add_node("agent",agent_node)
workflow.add_node("tools_grader",tool_and_grade_node)

workflow.set_entry_point("agent")

# 传入 "agent" 节点的输出
# 用 router_node 判断
# 根据返回值决定路径
workflow.add_conditional_edges(
    "agent",
    router_node,
    {
        "tools": "tools_grader",
        END: END
    }
)

# 核心闭环：工具执行完 -> 回到 Agent 根据反馈决定是"重查"还是"回答"
# tools_grader 执行完后，回到 agent 再思考一次
workflow.add_edge("tools_grader", "agent")

app = workflow.compile(checkpointer=memory)

# === 运行 ===
if __name__ == "__main__":
    print("🚀 启动 Self-RAG Agent (多轮对话 + 记忆版)...")

    # === 1. 配置记忆线程 ===
    # thread_id 就是用户的身份证，只要 ID 不变，记忆就在
    thread_id = "user_test_007"

    # 我们把 recursion_limit (防止死循环) 和 configurable (记忆ID) 合并到一个 config 里
    run_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 15  # 给足够多的重试机会
    }

    # === Round 1: 测试 Prompt 调优 (是否能答出原理) ===
    q1 = "Transformer 的核心机制是什么？"
    print(f"\n🗣️ User (Q1): {q1}")

    input1 = {"messages": [HumanMessage(content=q1)], "loop_count": 0}
    final_state1 = app.invoke(input1, config=run_config)

    print(f"🤖 Agent (A1): {final_state1['messages'][-1].content}")

    # === Round 2: 测试 Memory (是否记得'它'是谁) ===
    # 注意：这里我们只说"它"，如果 Agent 能回答出 Transformer 的优点，说明记忆生效了
    q2 = "它相比 RNN 有什么主要优势？"
    print(f"\n🗣️ User (Q2): {q2}")

    input2 = {"messages": [HumanMessage(content=q2)], "loop_count": 0}
    final_state2 = app.invoke(input2, config=run_config)

    print(f"🤖 Agent (A2): {final_state2['messages'][-1].content}")


















