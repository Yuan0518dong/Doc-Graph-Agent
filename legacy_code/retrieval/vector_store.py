import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from tqdm import tqdm

# === 配置路径 ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "hierarchical_chunks.jsonl"
DB_PATH = BASE_DIR / "data" / "chroma_db"


class VectorStoreBuilder:
    def __init__(self):
        # 1. 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=str(DB_PATH))

        # 2. Embedding 模型
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # 3. 获取集合
        self.collection = self.client.get_or_create_collection(
            name="proposal_knowledge_base",
            embedding_function=self.emb_fn
        )

    def ingest(self):
        if not DATA_PATH.exists():
            print(f"找不到数据文件: {DATA_PATH}")
            return

        print(f"开始向量化 (Embedding)...")

        documents = []
        metadatas = []
        ids = []

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"共有 {len(lines)} 条数据待处理...")

        for idx, line in enumerate(tqdm(lines, desc="Vectorizing")):
            chunk = json.loads(line)
            chunk_id = chunk["id"]  # 使用 UUID

            # 构造内容
            content = f"Path: {chunk['metadata'].get('path', '')}\nContent: {chunk['content']}"
            documents.append(content)

            # === 修复核心：清洗 Metadata ===
            meta = chunk["metadata"].copy()  # 复制一份，别改坏了原数据

            # ChromaDB 不支持列表，所以要把 headers 转成字符串
            if "headers" in meta and isinstance(meta["headers"], list):
                meta["headers"] = " > ".join(meta["headers"])  # 变成 "第一章 > 1.1节" 这种字符串

            # 注入 chunk_id 方便后续查 Neo4j
            meta["chunk_id"] = chunk_id

            metadatas.append(meta)
            ids.append(chunk_id)

            # 批处理
            if len(documents) >= 100:
                self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                documents, metadatas, ids = [], [], []

        # 处理剩余
        if documents:
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

        print(f"向量库构建完成！共存储 {self.collection.count()} 个切片。")


if __name__ == "__main__":
    builder = VectorStoreBuilder()
    builder.ingest()