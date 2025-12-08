"""
(隐式语义)：基于模型，构建文档的血肉（知识网）
    把扁平的 JSONL 数据，还原成 Neo4j 里立体的“树状结构”
"""
import json
import os
import time
import re
from tqdm import tqdm
from neo4j import GraphDatabase
from openai import OpenAI

# ===  配置区域 (DeepSeek R1 Local) ===
# 1. Neo4j 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518" # 数据库密码

# 2. Local LLM 配置 (Ollama)
API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "qwen2.5:1.5b" # 使用的本地模型

class SemanticGraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def close(self):
        self.driver.close()

    def clean_deepseek_output(self, content: str) -> str:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.replace("```json", "").replace("```", "").strip()
        return content

    def extract_triples(self, text: str):
        if not text: return [] # 防御性编程：如果是空的直接返回

        prompt = f"""
        你是一个知识图谱专家。请从以下文本中提取关键实体和关系。
        文本:
        {text[:1200]} 
        要求:
        1. 仅提取最重要的 3-5 个关系。
        2. 直接输出 JSON 数据，不要包含任何解释或思考过程。
        3. 实体类型 (type) 只能是: "Concept", "Method", "Metric", "Task"。
        4. 关系 (relation) 必须是大写英文，如: "USES", "IMPROVES", "SOLVES", "IS_A".
        输出格式必须是纯 JSON 列表:
        [
            {{"head": "Transformer", "head_type": "Method", "relation": "REPLACES", "tail": "RNN", "tail_type": "Method"}}
        ]
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
            )
            raw_content = response.choices[0].message.content
            cleaned_content = self.clean_deepseek_output(raw_content)

            try:
                data = json.loads(cleaned_content)
            except json.JSONDecodeError:
                match = re.search(r'\[.*\]', cleaned_content, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    return []

            if isinstance(data, list):
                return data
            return data.get("triples", [])

        except Exception as e:
            # print(f"提取失败: {e}")
            return []

    def build_semantics(self, limit=5):
        print(f"开始语义提取 (Model: {MODEL_NAME}, Limit: {limit})...")
        print("DeepSeek R1 正在本地思考中...")

        chunks_to_process = []
        with self.driver.session() as session:
            # 关键修正点：这里原来是 c.text，现在改成 c.content
            # 同时为了兼容后面的代码，我们用 AS text 把它别名化
            result = session.run(f"""
                MATCH (c:Chunk) 
                WHERE c.content IS NOT NULL
                RETURN c.id AS id, c.content AS text 
                LIMIT {limit}
                """)
            chunks_to_process = [record for record in result]

        print(f"选中 {len(chunks_to_process)} 个切片...")

        with self.driver.session() as session:
            # tqdm(...)：为遍历过程添加进度条，desc="Reasoning" 是进度条名称（显示 “Reasoning”），提升长任务的可视化体验
            for record in tqdm(chunks_to_process, desc="Reasoning"):
                chunk_id = record["id"]
                text = record["text"]

                triples = self.extract_triples(text)
                if not triples: continue

                for triple in triples:
                    if "head" not in triple or "tail" not in triple: continue

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
                    head=triple["head"], head_type=triple.get("head_type", "Concept"),
                    tail=triple["tail"], tail_type=triple.get("tail_type", "Concept"),
                    relation=triple.get("relation", "RELATED")
                    )

        print("语义增强完成！")

if __name__ == "__main__":
    builder = SemanticGraphBuilder()
    builder.build_semantics(limit=20)
    builder.close()