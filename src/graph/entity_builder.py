"""
(最终极速版)：多线程并发 + 强制打标 (防止死循环)
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from neo4j import GraphDatabase
from openai import OpenAI

# === 配置区域 ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518"

API_KEY = "sk-5f460d116b4243f498d356b5fb052fa5"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 5  # 保持 5 个线程并行

class SemanticGraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def close(self):
        self.driver.close()

    def extract_triples(self, text: str):
        if not text or len(text) < 10: return []

        prompt = f"""
        请从以下文本提取实体关系（三元组）。
        文本：{text[:1200]}
        
        要求：
        1. 严格输出 JSON 格式，包含 "triples" 列表。
        2. 关系 (relation) 可以是中文或英文。
        
        输出示例：
        {{
            "triples": [
                {{"head": "Transformer", "type": "技术", "relation": "replaces", "tail": "RNN", "tail_type": "技术"}}
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个输出 JSON 的工具。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=60
            )
            raw_content = response.choices[0].message.content
            data = json.loads(raw_content)
            return data.get("triples", [])
        except Exception:
            return []

    def process_single_chunk(self, record):
        """单个任务逻辑"""
        chunk_id = record["id"]
        text = record["text"]

        # 1. 提取
        triples = self.extract_triples(text)

        has_result = False

        # 2. 写入数据库 (独立 Session)
        with self.driver.session() as session:
            # ===  关键修正：不管 triples 是不是空，先打上标记！===
            # 这样下次运行，这个 Chunk 就不会再被查出来的。
            session.run("MATCH (c:Chunk {id: $id}) SET c.entity_processed = true", id=chunk_id)

            if triples:
                has_result = True
                for t in triples:
                    if "head" not in t or "tail" not in t: continue

                    session.run("""
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (h:Entity {name: $head})
                        ON CREATE SET h.type = $head_type
                        MERGE (t:Entity {name: $tail})
                        ON CREATE SET t.type = $tail_type
                        MERGE (h)-[r:RELATED {type: $relation}]->(t)
                        MERGE (c)-[:HAS_ENTITY]->(h)
                        MERGE (c)-[:HAS_ENTITY]->(t)
                    """,
                    chunk_id=chunk_id,
                    head=t["head"], head_type=t.get("type", "Concept"),
                    tail=t["tail"], tail_type=t.get("tail_type", "Concept"),
                    relation=t.get("relation", "RELATED")
                    )
        return has_result

    def build_semantics(self, limit=2000):
        print(f"启动极速提取 (线程: {MAX_WORKERS})...")

        chunks_to_process = []
        with self.driver.session() as session:
            # 这里的 WHERE 条件保证了只处理没打标记的
            result = session.run(f"""
                MATCH (c:Chunk) 
                WHERE c.text IS NOT NULL 
                  AND c.entity_processed IS NULL
                RETURN c.id AS id, c.text AS text 
                LIMIT {limit}
                """)
            chunks_to_process = [record for record in result]

        total = len(chunks_to_process)
        if total == 0:
            print("🎉 所有任务已完成！")
            return

        print(f"剩余 {total} 个切片，开始处理...")

        success_count = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(self.process_single_chunk, record) for record in chunks_to_process]

            for future in tqdm(as_completed(futures), total=total, desc="处理进度"):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"异常: {e}")

        print(f"\n✅ 本轮结束！有效提取: {success_count} 个。")

if __name__ == "__main__":
    builder = SemanticGraphBuilder()
    builder.build_semantics()
    builder.close()