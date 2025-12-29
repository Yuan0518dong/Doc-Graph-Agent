import os
import logging
from pathlib import Path
from tqdm import tqdm
from neo4j import GraphDatabase
from langchain_text_splitters import MarkdownHeaderTextSplitter
# ✅ 使用 HuggingFace 本地运行向量模型 (不依赖 Ollama，更稳定)
from langchain_huggingface import HuggingFaceEmbeddings

# === 配置 ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518"

# 路径
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class KnowledgeGraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        #  这里会自动下载 BAAI/bge-small-zh-v1.5
        # 这是一个专门的向量模型，虽小(100MB)但中文检索能力极强，完全本地运行
        print("[系统] 加载本地 Embedding 模型 (BAAI/bge-small-zh-v1.5)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5"
        )
        print("模型加载完毕")

    def close(self):
        self.driver.close()

    def create_vector_index(self):
        with self.driver.session() as session:
            print("重建索引 (512维)...")
            session.run("DROP INDEX vector_index IF EXISTS")
            # bge-small-zh 是 512 维
            session.run("""
            CREATE VECTOR INDEX vector_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: 512, `vector.similarity_function`: 'cosine'}}
            """)

    def build(self):
        if not INPUT_DIR.exists(): return
        md_files = list(INPUT_DIR.glob("*.md"))
        # 按照 Markdown 标题切分
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])

        with self.driver.session() as session:
            for md_file in tqdm(md_files, desc="本地入库中"):
                file_name = md_file.stem
                if session.run("MATCH (d:Document {name: $name}) RETURN count(d)", name=file_name).single()[0] > 0:
                    continue

                content = md_file.read_text(encoding="utf-8")
                splits = splitter.split_text(content)
                session.run("MERGE (d:Document {name: $name})", name=file_name)

                for i, split in enumerate(splits):
                    # 本地 CPU 生成向量
                    vector = self.embeddings.embed_query(split.page_content)

                    session.run("""
                        MATCH (d:Document {name: $doc_name})
                        MERGE (c:Chunk {id: $id})
                        SET c.text = $text, c.embedding = $vector
                        MERGE (d)-[:HAS_CHUNK]->(c)
                    """, doc_name=file_name, id=f"{file_name}_{i}", text=split.page_content, vector=vector)


if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    builder.create_vector_index()  # 必须跑，重置索引
    builder.build()
    builder.close()
    print("🎉 本地入库完成！")