import chromadb
from chromadb.utils import embedding_functions
from neo4j import GraphDatabase
from openai import OpenAI
from pathlib import Path
import re

# === 全局配置 ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518"  # 密码

OLLAMA_API_KEY = "ollama"
OLLAMA_URL = "http://localhost:11434/v1"
MODEL_NAME = "qwen2.5:1.5b"  # 模型


class HybridRAG:
    def __init__(self):
        # 1. 连接 Chroma
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_collection(
            name="proposal_knowledge_base",
            embedding_function=self.emb_fn
        )

        # 2. 连接 Neo4j
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # 3. 连接 DeepSeek
        self.llm = OpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_URL)

    def close(self):
        self.driver.close()

    def clean_think(self, text):
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def get_graph_context(self, chunk_ids):
        """核心：去 Neo4j 里查"""
        if not chunk_ids: return []

        # 修正点：将 CONTAINS 改为 HAS_CHUNK
        cypher = """
        MATCH (c:Chunk)
        WHERE c.id IN $ids

        // 1. 找父章节 (Structure) - 注意方向是 c <-[HAS_CHUNK]- s
        OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(s:Section)

        // 2. 找关联实体 (Semantics)
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:Entity)

        RETURN 
            c.id AS id,
            s.title AS section,
            collect(e.name) AS entities
        """

        enriched_info = {}
        with self.driver.session() as session:
            result = session.run(cypher, ids=chunk_ids)
            for record in result:
                c_id = record["id"]
                enriched_info[c_id] = {
                    "section": record["section"] or "Unknown Section",
                    "entities": record["entities"]
                }
        return enriched_info

    # === 核心方法：供 Agent 调用 ===
    def search(self, query: str, top_k: int = 6) -> str:
        """
        只负责检索资料，不负责回答。返回拼接好的上下文 Context。
        """
        # 1. 向量检索
        results = self.collection.query(query_texts=[query], n_results=top_k)
        if not results['ids'][0]:
            return "未在数据库中找到相关信息。"

        ids = results['ids'][0]
        docs = results['documents'][0]

        # 2. 图谱检索
        graph_data = self.get_graph_context(ids)

        # 3. 组装成纯文本给 Agent 看
        context_parts = []
        for i, doc in enumerate(docs):
            c_id = ids[i]
            g_info = graph_data.get(c_id, {"section": "N/A", "entities": []})
            entities_str = ', '.join(g_info['entities']) if g_info['entities'] else "无"

            snippet = f"""
            [资料 {i + 1}]
            来源: {g_info['section']}
            关联实体: {entities_str}
            内容: {doc}
            """
            context_parts.append(snippet)

        return "\n".join(context_parts)


    def chat(self, query):
        print(f"\nUser: {query}")
        print("-" * 40)

        # Step 1: 向量检索
        print("1. ChromaDB: 正在定位切片...")
        results = self.collection.query(query_texts=[query], n_results=3)

        docs = results['documents'][0]
        ids = results['ids'][0]

        # Step 2: 图谱增强
        print(f"2. Neo4j: 正在扩展上下文 (IDs: {ids})...")
        graph_data = self.get_graph_context(ids)

        # 组装 Prompt
        context_parts = []
        for i, doc in enumerate(docs):
            c_id = ids[i]
            # 获取图谱信息
            g_info = graph_data.get(c_id, {"section": "N/A", "entities": []})

            # 这里的 entities 应该不为空了
            entities_str = ', '.join(g_info['entities']) if g_info['entities'] else "无关联实体"

            snippet = f"""
            [参考片段 {i + 1}]
            - 来源章节: {g_info['section']}
            - 知识图谱关联: {entities_str}
            - 正文内容: {doc}
            """
            context_parts.append(snippet)

        full_context = "\n".join(context_parts)

        # Step 3: DeepSeek 生成
        print("3. DeepSeek: 正在思考...")

        sys_prompt = "你是一个专业的项目申报助手。请根据提供的【图谱增强上下文】回答问题。"
        user_prompt = f"【上下文】：\n{full_context}\n\n【问题】：{query}"

        try:
            resp = self.llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            answer = self.clean_think(resp.choices[0].message.content)

            print("-" * 40)
            print(f"AI Answer:\n{answer}")
            print("-" * 40)

        except Exception as e:
            print(f"DeepSeek 调用失败: {e}")
            print("建议: 检查 Ollama 是否开启 (ollama serve)")


if __name__ == "__main__":
    bot = HybridRAG()
    # 测试 search 方法是否只返回字符串
    print(bot.search("Transformer 的优势"))
    bot.close()


    """
    Ablation Study (消融实验)
    目标： 理解 Chroma 和 Neo4j 的配合。

    操作： 
    打开 src/retrieval/rag_chat.py。
    找到 chat 函数。
    注释掉 Step 2 (图谱增强) 的代码，只保留 Chroma 检索，直接把 docs 喂给 DeepSeek。
    
    运行并提问："Transformer 和 RNN 的区别？"
    观察： 只有向量检索时，回答的质量下降了多少？是不是变得笼统了？丢失了哪些细节？
    收获： 这就是你论文里 Ablation Study (消融实验) 的雏形！你在面试时可以说：“我对比了纯向量检索和图增强检索，发现后者在逻辑关联性上提升了...”
    """