import json
from neo4j import GraphDatabase
from openai import OpenAI

# === 配置 ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518"
API_KEY = "sk-5f460d116b4243f498d356b5fb052fa5"
BASE_URL = "https://api.deepseek.com"


def debug_task():
    print("🔍 1. 连接数据库...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    with driver.session() as session:
        # 随机取 1 条文本不为空的切片
        print("🔍 2. 正在抓取一条测试数据...")
        result = session.run("MATCH (c:Chunk) WHERE c.text IS NOT NULL RETURN c.text AS text LIMIT 1")
        record = result.single()

        if not record:
            print("❌ 致命错误：数据库里没有任何文本数据！请先检查 build_graph.py 是否真的入库成功。")
            driver.close()
            return

        text = record["text"]
        print(f"\n📄 [原文片段] (前100字):\n{text[:100]}...\n")

        print("🔍 3. 正在发送给 DeepSeek (不做任何解析，只看原始回复)...")
        prompt = f"""
        请从以下文本提取实体关系（三元组）。
        文本：{text[:500]}

        要求：严格输出 JSON 格式，包含 "triples" 列表。
        例如：{{"triples": [{{"head": "A", "type": "T", "relation": "R", "tail": "B", "tail_type": "T"}}]}}
        """

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个输出 JSON 的工具。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},  # 强制 JSON
                temperature=0.1
            )
            raw_content = response.choices[0].message.content

            print("\n🤖 [DeepSeek 原始回复]:")
            print("-" * 50)
            print(raw_content)
            print("-" * 50)

            # 尝试解析
            print("\n🔍 4. 尝试代码解析...")
            data = json.loads(raw_content)
            triples = data.get("triples", [])
            print(f"✅ 解析成功！提取到了 {len(triples)} 个关系：")
            for t in triples:
                print(f"   - {t['head']} --[{t['relation']}]--> {t['tail']}")

        except Exception as e:
            print(f"\n❌ 调用或解析失败: {e}")

    driver.close()


if __name__ == "__main__":
    debug_task()