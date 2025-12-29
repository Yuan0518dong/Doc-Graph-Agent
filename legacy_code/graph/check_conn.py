# src/graph/check_conn.py
from neo4j import GraphDatabase

# 配置你的连接信息
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "zyyzdy0518")  # 改成你刚才设的密码

def verify_connection():
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            print("✅ Neo4j 连接成功！Operation Trojan Horse Base is Online.")

            # 顺便查一下版本，装个逼
            records, summary, keys = driver.execute_query(
                "CALL dbms.components() YIELD name, versions, edition"
            )
            for record in records:
                print(f"📊 Database: {record['name']} {record['versions'][0]} ({record['edition']})")

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("💡 提示: 检查 Neo4j Desktop 是否显示绿色 'Active' 状态")

if __name__ == "__main__":
    verify_connection()