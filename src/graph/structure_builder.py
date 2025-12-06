"""
(显式结构)：基于规则，构建文档的骨架（目录树）
"""
import json
import os
from neo4j import GraphDatabase
from pathlib import Path
from tqdm import tqdm

# === 配置 ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# 确保这里读取的是 markdown_splitter 生成的那个 jsonl
DATA_PATH = BASE_DIR / "data" / "processed" / "hierarchical_chunks.jsonl"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518"  # 密码


class KnowledgeGraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def clean_graph(self):
        """清空数据（不清空 Schema）"""
        print("正在清空旧图谱数据...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("数据库数据已清空")

    def init_schema(self):
        """初始化约束和索引 (使用 IF NOT EXISTS 防止报错)"""
        print("正在初始化 Schema (约束与索引)...")
        with self.driver.session() as session:
            # 1. Document 唯一性约束
            # 语法：如果不存在才创建
            session.run(
                "CREATE CONSTRAINT document_name_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE")

            # 2. Chunk 唯一性约束
            session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")

            # 3. Section 路径索引 (加速查询)
            session.run("CREATE INDEX section_path_index IF NOT EXISTS FOR (s:Section) ON (s.full_path)")

            print("Schema 初始化完毕 (跳过了已存在的规则)")

    def build_from_jsonl(self):
        """核心构建逻辑"""
        if not DATA_PATH.exists():
            print(f"未找到数据文件: {DATA_PATH}")
            return

        # 1. 先清空数据
        self.clean_graph()

        # 2. 初始化 Schema
        self.init_schema()

        print(f"开始构建图谱: {DATA_PATH.name}")

        # 读取文件统计行数
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        count = 0
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            # 使用 tqdm 显示进度条
            for line in tqdm(f, total=total_lines, desc="Building Graph"):
                chunk = json.loads(line)
                metadata = chunk["metadata"]

                # 提取关键字段
                # 兼容不同版本的字段名
                headers = metadata.get("headers", [])
                if not headers and "path" in metadata:
                    # 如果没有 headers 列表，尝试从 path 字符串解析
                    headers = [h.strip() for h in metadata["path"].split(">") if h.strip() != "Root"]

                chunk_id = chunk["id"]
                content = chunk["content"]
                source_doc = metadata.get("source", "Unknown Doc")
                # 构造一个全路径字符串用于索引
                path_str = " > ".join(headers)

                # 执行写入
                self._create_nodes(headers, content, chunk_id, source_doc, path_str)
                count += 1

        print(f"图谱构建完成！共处理 {count} 个切片。")

    def _create_nodes(self, headers, content, chunk_id, source_name, path_str):
        with self.driver.session() as session:
            # 我们把复杂逻辑拆解为 Python 循环，虽然比纯 Cypher 慢一点点，但极其稳定不容易写错

            # 1. 创建 Document 根节点
            session.run("MERGE (d:Document {name: $name})", name=source_name)

            # 2. 循环创建 Section 链
            parent_label = "Document"
            parent_key = "name"
            parent_val = source_name

            # 用于累积路径，确保 Section 唯一性 (避免不同章有相同的 "1.1 概述")
            current_full_path = source_name

            for h_title in headers:
                current_full_path += f" > {h_title}"

                # 链接 Parent -> Current Section
                if parent_label == "Document":
                    q = """
                    MATCH (p:Document {name: $p_val})
                    MERGE (s:Section {full_path: $full_path})
                    SET s.title = $title, s.source = $doc_name
                    MERGE (p)-[:HAS_SECTION]->(s)
                    """
                    session.run(q, p_val=parent_val, full_path=current_full_path, title=h_title, doc_name=source_name)
                else:
                    q = """
                    MATCH (p:Section {full_path: $p_val})
                    MERGE (s:Section {full_path: $full_path})
                    SET s.title = $title, s.source = $doc_name
                    MERGE (p)-[:HAS_SUBSECTION]->(s)
                    """
                    session.run(q, p_val=parent_val, full_path=current_full_path, title=h_title, doc_name=source_name)

                # 更新指针
                parent_label = "Section"
                parent_key = "full_path"
                parent_val = current_full_path

            # 3. 挂载 Chunk
            # 如果 headers 为空（只有 Root），直接挂在 Document 下
            if not headers:
                q = """
                MATCH (d:Document {name: $doc_name})
                MERGE (c:Chunk {id: $c_id})
                SET c.content = $content, c.path = $path
                MERGE (d)-[:HAS_CHUNK]->(c)
                """
                session.run(q, doc_name=source_name, c_id=chunk_id, content=content, path=path_str)
            else:
                q = """
                MATCH (s:Section {full_path: $s_path})
                MERGE (c:Chunk {id: $c_id})
                SET c.content = $content, c.path = $path
                MERGE (s)-[:HAS_CHUNK]->(c)
                """
                session.run(q, s_path=current_full_path, c_id=chunk_id, content=content, path=path_str)


if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    builder.build_from_jsonl()
    builder.close()