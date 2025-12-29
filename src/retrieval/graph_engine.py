import os
from neo4j import GraphDatabase
from typing import List, Dict, Any

# === 配置 ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518"


class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def query_graph_context(self, keywords: List[str], limit: int = 5) -> str:
        """
        核心逻辑：根据关键词，在图谱中寻找相关实体，并返回它们关联的文本块。
        """
        if not keywords:
            return ""

        # 将关键词转换为小写，模糊匹配
        # 逻辑：找到名字包含关键词的实体 -> 找到该实体连接的切片 -> 返回切片文本
        cypher_query = """
        MATCH (e:Entity)
        WHERE any(word IN $keywords WHERE toLower(e.name) CONTAINS toLower(word))

        // 找到这个实体属于哪个切片
        MATCH (c:Chunk)-[:HAS_ENTITY]->(e)

        // 也可以找找这个实体的邻居（扩充上下文）
        OPTIONAL MATCH (e)-[r:RELATED]->(neighbor)

        // 聚合结果，按关联度排序（这里简单用实体出现的次数作为相关性）
        RETURN c.text as text, count(e) as score, collect(distinct e.name) as entities
        ORDER BY score DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, keywords=keywords, limit=limit)

            context_pieces = []
            for record in result:
                context_pieces.append(f"【相关原文 (包含实体: {record['entities']})】:\n{record['text']}\n")

            return "\n".join(context_pieces)

    def get_stats(self):
        """查看一下现在库里有多少货"""
        with self.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            return result.single()["count"]


# === 测试代码 ===
if __name__ == "__main__":
    # 简单的本地测试
    retriever = GraphRetriever()

    # 看看现在库里有多少实体
    count = retriever.get_stats()
    print(f"当前图谱实体总数: {count}")

    # 模拟搜索
    test_keywords = ["Agent", "Transformer", "RAG", "Model"]  # 你可以改在这个列表
    print(f"\n正在搜索关键词: {test_keywords} ...")

    context = retriever.query_graph_context(test_keywords)
    if context:
        print("✅ 检索成功！检索到的上下文示例：")
        print("-" * 50)
        print(context[:500] + "...")  # 只打印前500字
    else:
        print("⚠️ 未找到相关内容（可能是后台还在跑，还没存进去，或者关键词没对上）。")

    retriever.close()