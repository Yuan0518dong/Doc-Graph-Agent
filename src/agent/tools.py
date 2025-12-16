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
        # 1. 获取引擎实例
        engine = RAGSingleton.get_instance()
        print(f"[Tool] 收到原始查询: {query}")

        # ===  核心修改：查询语义注入 (Query Injection) ===
        # 目的：防止 1.5B 模型被简单的数字吸引，强制检索“原理”类内容

        # A. 定义黑名单：如果用户问的是这些，我们就老实去查数据，不乱改
        data_keywords = ["多少", "分数", "数据", "几天", "BLEU", "时间", "参数量"]
        is_asking_data = any(k in query for k in data_keywords)

        # B. 实施增强：如果不是查数据，就强制加上“深度原理”滤镜
        if not is_asking_data:
            # 这里的关键词是精心挑选的，专门针对申报书/论文场景
            suffix = " 核心机制 技术原理 架构设计 实施方案 -Results -Experiment"
            enhanced_query = f"{query} {suffix}"
            print(f"[Tool] 查询已增强 (Logic Injection): {enhanced_query}")
        else:
            enhanced_query = query
            print(f"[Tool] 检测到数据类查询，保持原样。")

        # 2. 使用增强后的 query 调用底层引擎
        return engine.search(enhanced_query)

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








