import os
import logging
import time
from pathlib import Path
from tqdm import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ===  路径与配置 ===
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# === Neo4j 配置 ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518"

# === 模型配置 (本地向量模型) ===
MODEL_NAME = 'all-MiniLM-L6-v2'  # 轻量级，速度快
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SemanticGraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print(f"正在加载嵌入模型 {MODEL_NAME} ...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("✅ 模型加载完成")

    def close(self):
        self.driver.close()

    def create_vector_index(self):
        """创建向量索引，让搜索变快"""
        with self.driver.session() as session:
            print("检查向量索引...")
            session.run("""
                CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                 `vector.dimensions`: 384,
                 `vector.similarity_function`: 'cosine'
                }}
            """)
            print("✅ 向量索引就绪")

    def build_embeddings(self):
        self.create_vector_index()

        with self.driver.session() as session:
            # 1. 检查还有多少没计算向量的 Chunk
            count_query = "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c)"
            total_remaining = session.run(count_query).single()[0]

            if total_remaining == 0:
                print("🎉 所有 Chunk 都有向量了，无需处理。")
                return

            print(f"[语义构建] 发现 {total_remaining} 个待处理 Chunk")
            pbar = tqdm(total=total_remaining, desc="计算向量")

            while True:
                # 2. 分批读取
                fetch_query = """
                MATCH (c:Chunk) WHERE c.embedding IS NULL
                RETURN c.id AS id, c.text AS text
                LIMIT $limit
                """
                results = list(session.run(fetch_query, limit=BATCH_SIZE))
                if not results: break

                # 3. 计算向量
                texts = [r["text"] for r in results]
                # 这里做个简单的防错，防止 text 为空
                valid_texts = [t if t else "" for t in texts]
                embeddings = self.model.encode(valid_texts, show_progress_bar=False)

                # 4. 批量写入
                update_query = """
                UNWIND $batches AS batch
                MATCH (c:Chunk {id: batch.id})
                SET c.embedding = batch.embedding
                """
                batches = [{"id": r["id"], "embedding": emb.tolist()} for r, emb in zip(results, embeddings)]
                session.run(update_query, batches=batches)

                pbar.update(len(results))

            pbar.close()
            print("\n🎉 向量计算完成！")


if __name__ == "__main__":
    builder = SemanticGraphBuilder()
    builder.build_embeddings()
    builder.close()