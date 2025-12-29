import json
import operator
from typing import Annotated, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# 引入工具
from src.agent.tools import search_knowledge_base

# === 1. 核心修改：手动定义 Prompt ===
# 我们不再用 bind_tools，而是直接教模型怎么输出 JSON
SYSTEM_PROMPT = """你是一个智能项目申报助手。
你的任务是准确回答用户的技术问题。

### 你的工具箱：
你有且只有一个工具：
- 工具名称: search_knowledge_base
- 作用: 查询关于 Transformer、RNN、项目政策等背景知识。
- 参数: query (搜索词)

### 思考与行动规则：
1. 当用户问技术问题时，你**必须**先调用工具。
2. **如何调用工具？**
   请直接输出且仅输出以下 JSON 格式：
   {"action": "search", "query": "这里填搜索词"}
3. **不要**输出 <think> 标签，不要输出"好的我来查一下"这种废话，直接给 JSON。
"""

# === 2. 模型配置 ===
# 建议使用 qwen2.5:1.5b，因为它对指令遵循更好。deepseek-r1:1.5b 也可以试。
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen2.5:1.5b",  # 建议确认你拉取了 Qwen
    temperature=0,
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# === 3. 节点定义 ===

def agent_node(state: AgentState):
    """大脑节点"""
    messages = state["messages"]

    # 强行插入系统提示词 (放在第一位)
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    print("[Agent] 正在思考...")
    response = llm.invoke(messages)
    return {"messages": [response]}


def router_node(state: AgentState):
    """路由节点：更鲁棒的 JSON 解析"""
    last_msg = state["messages"][-1]
    content = last_msg.content.strip()

    print(f"[监控] 模型输出了: {content[:50]}...")

    try:
        # 1. 尝试寻找 JSON 的大括号范围
        start = content.find('{')
        end = content.rfind('}') + 1

        # 2. 如果找到了括号，尝试解析
        if start != -1 and end != -1:
            json_str = content[start:end]
            data = json.loads(json_str)

            # 3. 检查字段是否匹配
            if data.get("action") == "search":
                print("[Router] 捕获到工具调用指令！(JSON解析成功)")
                return "tools"
    except Exception as e:
        print(f"[Router] JSON 解析尝试失败: {e}")

    # 如果上面没返回，就说明没指令
    print("[Router] 无工具指令，结束。")
    return END

def tool_node(state: AgentState):
    """工具执行节点"""
    last_msg = state["messages"][-1]
    content = last_msg.content

    try:
        # 正则提取 JSON (简单版：找大括号)
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end]
        data = json.loads(json_str)

        query = data.get("query")
        print(f"[Tool] 执行搜索: {query}")

        # 调用真正的工具
        tool_result = search_knowledge_base.invoke(query)

        # === 加强反馈指令 ===
        # 我们要获取原始的用户问题，防止模型忘了
        # state["messages"][0] 是 SystemMessage
        # state["messages"][1] 通常是用户的第一个问题 HumanMessage
        original_question = "用户的问题"
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                original_question = m.content
                break

        # 构造一个带有强指令的反馈
        feedback = f"""
        【系统通知】：资料库查询已完成。

        检索到的上下文资料：
        ---------------------
        {tool_result}
        ---------------------

        【当前任务】：
        请立即忽略之前的客套话，**直接**根据上述资料，回答用户的问题："{original_question}"。
        不要说"好的"、"根据资料"，直接输出答案内容！
        """

        # 把这个反馈伪装成 HumanMessage 发回去
        return {"messages": [HumanMessage(content=feedback)]}

    except Exception as e:
        print(f" 解析失败: {e}")
        return {"messages": [HumanMessage(content="工具调用失败，请直接根据已有知识回答。")]}

# === 4. 构建图 ===
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router_node,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

app = workflow.compile()

# === 5. 运行 ===
if __name__ == "__main__":
    print(" 启动 Manual-ReAct Agent (JSON版)...")
    query = "Transformer 的核心机制是什么？"
    print(f" User: {query}")

    inputs = {"messages": [HumanMessage(content=query)]}

    # 使用 invoke 直接运行一次完整流程
    # 这里的递归限制 (recursion_limit) 防止死循环
    final_state = app.invoke(inputs, config={"recursion_limit": 10})

    print("\n最终回答:")
    print(final_state["messages"][-1].content)



    """
    1.感知 (User): 接收问题。

    2.决策 (Agent): 决定查库 ({"action":"search"...})。

    3.执行 (Tool): 你的 Python 代码捕获指令，去 Neo4j/Chroma 查回了资料。

    4.响应 (Agent): 拿到资料后，不再废话，直接生成了正确答案（“核心机制是注意力机制”）
"""







