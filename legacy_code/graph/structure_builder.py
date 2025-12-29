import os
import logging
from pathlib import Path
from tqdm import tqdm
from neo4j import GraphDatabase
from langchain_text_splitters import MarkdownHeaderTextSplitter

# ===  路径配置 (自动定位) ===
# 确保这里能找到 data/processed 目录
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "processed"

# === Neo4j 数据库配置 ===
# 请确认你的 Neo4j Desktop 已经开启，且密码正确
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zyyzdy0518" 

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StructureGraphBuilder:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            print(" Neo4j 连接成功！")
        except Exception as e:
            print(f"Neo4j 连接失败: {e}")
            print(" 请检查 Neo4j Desktop 是否启动，或者密码是否写错。")
            raise e

    def close(self):
        self.driver.close()

    def build_structure(self):
        # 1. 扫描文件
        if not INPUT_DIR.exists():
            print(f" 找不到目录: {INPUT_DIR}")
            return

        md_files = list(INPUT_DIR.glob("*.md"))
        total_files = len(md_files)
        print(f"[入库引擎] 扫描到 {total_files} 个 Markdown 文件")

        if total_files == 0:
            print("还没有 Markdown 文件，请等待 pdf_parser.py 运行完成。")
            return

        # 2. 定义智能切分器 (按标题层级切分)
        # 这样 Agent 就能知道一段话是属于 "Introduction" 还是 "Conclusion"
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # 3. 开启会话 (流式处理)
        with self.driver.session() as session:
            # 建立索引以加速查询 (可选，但推荐)
            session.run("CREATE INDEX document_name IF NOT EXISTS FOR (d:Document) ON (d.name)")
            
            success_count = 0
            skip_count = 0

            # 4. 循环处理每个文件
            for md_file in tqdm(md_files, desc="正在入库 Neo4j"):
                try:
                    file_name = md_file.stem
                    
                    # === 关键：断点续传检查 ===
                    # 每次写入前，先问 Neo4j：“这篇论文你有了吗？”
                    result = session.run("MATCH (d:Document {name: $name}) RETURN count(d)", name=file_name).single()[0]
                    
                    if result > 0:
                        skip_count += 1
                        continue # 如果有了，直接处理下一篇

                    # === 读取与切分 ===
                    with open(md_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 1. 创建文档节点
                    session.run(
                        "MERGE (d:Document {name: $name}) SET d.path = $path, d.processed = true", 
                        name=file_name, path=str(md_file)
                    )

                    # 2. 切分内容
                    splits = splitter.split_text(content)
                    
                    # 3. 写入 Chunk 节点
                    for i, split in enumerate(splits):
                        chunk_id = f"{file_name}_chunk_{i}"
                        text = split.page_content
                        meta = split.metadata
                        
                        # 写入 Chunk 并关联到 Document
                        session.run("""
                            MATCH (d:Document {name: $doc_name})
                            MERGE (c:Chunk {id: $id})
                            SET c.text = $text,
                                c.h1 = $h1,
                                c.h2 = $h2
                            MERGE (d)-[:HAS_CHUNK]->(c)
                        """, 
                        doc_name=file_name, 
                        id=chunk_id, 
                        text=text,
                        h1=meta.get("h1", ""),
                        h2=meta.get("h2", "")
                        )
                        
                        # (可选) 可以在这里添加 Chunk 之间的 NEXT 关系
                    
                    success_count += 1

                except Exception as e:
                    logging.error(f"处理 {file_name} 失败: {e}")
                    continue
            
            print(f"\n 入库任务完成！")
            print(f" 新写入: {success_count} 篇")
            print(f" 跳过已存: {skip_count} 篇")

if __name__ == "__main__":
    builder = StructureGraphBuilder()
    builder.build_structure()
    builder.close()