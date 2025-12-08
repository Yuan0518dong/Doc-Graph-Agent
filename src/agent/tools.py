import sys
from pathlib import Path
from langchain_core.tools import tool

# 路径 hack，确保能导入 src
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.retrieval.rag_chat import HybridRAG


# === 单例模式管理 RAG 引擎 ===
class RAGSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("[System] 正在唤醒 RAG 引擎 (单例初始化)...")
            cls._instance = HybridRAG()
        return cls._instance


# ===  核心：定义工具 ===
@tool
def search_knowledge_base(query: str) -> str:
    """
        这是一个项目申报领域的专业知识库检索工具。
        当用户问到具体的政策文件、技术原理（如 Transformer）、写作规范或历史数据时，
          必须 使用此工具来获取事实依据。

        Args:
            query: 用户的搜索关键词或问题。

        Returns:
            str: 包含正文切片、章节来源和关联实体的结构化文本。
    """
    try:
        engine = RAGSingleton.get_instance()
        print(f"[Agent Tool] 正在查询: {query}")
        # 调用我们刚才重构的 search 方法
        return engine.search(query)
    except Exception as e:
        return f"查询出错: {str(e)}"


if __name__ == "__main__":
    # 模拟 Agent 调用
    print("测试工具封装...")
    # 模拟输入参数
    test_input = {"query": "Transformer 和 RNN 的区别"}
    # invoke 是 LangChain 工具的标准调用方式
    result = search_knowledge_base.invoke(test_input)
    print("\n工具返回结果:")
    print(result[:500] + "...")








