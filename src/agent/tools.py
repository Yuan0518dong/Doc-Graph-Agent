from langchain_core.tools import tool
import sys
import os

# 引用我们之前写好的检索核心
# 确保路径能找到 src.retrieval
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.retrieval.graph_engine import GraphRetriever

# 初始化一个全局的检索器实例 (避免每次调用都重连数据库)
_retriever_instance = GraphRetriever()


@tool
def search_knowledge_base(query: str) -> str:
    """
    当需要查询专业知识、技术细节、实体关系或背景信息时，务必调用此工具。
    输入：一个查询字符串（例如："Transformer 架构" 或 "智能体协作"）。
    输出：来自图数据库的相关上下文文本。
    """
    print(f"🔧 [Tool] 收到查询请求: {query}")

    # 简单的预处理：把句子切成关键词列表 (GraphEngine 需要 List)
    # 这里简单按空格切分，或者直接把整个 query 当做一个关键词
    keywords = query.split()

    # 调用我们之前写好的 GraphEngine
    try:
        result = _retriever_instance.query_graph_context(keywords, limit=3)
        if not result:
            return "【数据库反馈】：没有找到包含这些关键词的实体或相关内容。"
        return result
    except Exception as e:
        return f"【工具报错】：查询过程中发生错误 - {e}"