import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# === 配置 ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518"
MODEL_NAME = 'all-MiniLM-L6-v2'


class VectorTester:
    def __init__(self):
        # 连接数据库
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # 加载同一个模型 (用于把你的问题变成向量)
        print("📥 加载模型中...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("✅ 准备就绪")

    def search(self, query, top_k=3):
        print(f"\n🔍 正在搜索: '{query}'")

        # 1. 把用户的问题变成向量
        query_embedding = self.model.encode(query, show_progress_bar=False).tolist()

        # 2. 在 Neo4j 里找最相似的邻居 (Vector Search)
        # 这里的 chunk_embedding_index 必须和你刚才创建的索引名字一致
        cql = """
        CALL db.index.vector.queryNodes('chunk_embedding_index', $k, $embedding)
        YIELD node, score
        RETURN node.text AS text, node.id AS id, score
        """

        with self.driver.session() as session:
            results = session.run(cql, k=top_k, embedding=query_embedding)

            print(f"🏆 找到最相关的 {top_k} 个片段：")
            print("-" * 50)
            for i, record in enumerate(results):
                score = record['score']
                text = record['text']
                # 截取前100个字显示
                preview = text[:100].replace('\n', ' ') + "..."

                print(f"[{i + 1}] 相似度: {score:.4f}")
                print(f"    内容: {preview}")
                print("-" * 50)

    def close(self):
        self.driver.close()


if __name__ == "__main__":
    tester = VectorTester()

    # === 在这里修改你想问的问题 ===
    # 试着用中文或英文问一些你觉得论文里会有的内容
    questions = [
        "How do multiple agents collaborate?",  # 多智能体如何协作？
        "What is the transformer architecture?",  # Transformer 架构是什么？
        "RLHF reinforcement learning"  # RLHF 强化学习
    ]

    for q in questions:
        tester.search(q)

    tester.close()