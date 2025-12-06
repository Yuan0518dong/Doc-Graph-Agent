import re
import uuid
import json
from dataclasses import dataclass, asdict
from typing import List, Dict
from pathlib import Path

# === 路径配置 ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "hierarchical_chunks.jsonl"


@dataclass
class HeaderNode:
    level: int
    title: str


@dataclass
class MarkdownChunk:
    id: str  # 唯一标识符
    content: str  # 文本内容
    metadata: Dict  # 包含 path, source, headers(关键)


class MarkdownContextSplitter:
    def __init__(self):
        # 正则：匹配 Markdown 标题 (如 "## 1.1 背景")
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.*)')

    def split_text(self, text: str, source_name: str = "unknown") -> List[Dict]:
        lines = text.split('\n')
        chunks = []

        # 核心数据结构：栈
        header_stack: List[HeaderNode] = []
        content_buffer: List[str] = []

        for line in lines:
            match = self.header_pattern.match(line)

            if match:
                # 1. 遇到新标题 -> 结算上一段内容
                if content_buffer:
                    self._save_chunk(chunks, header_stack, content_buffer, source_name)
                    content_buffer = []  # 清空

                # 2. 栈操作 (维护层级)
                new_level = len(match.group(1))
                new_title = match.group(2).strip()

                # Pop: 如果新标题级别更高或相等，弹出栈顶
                while header_stack and header_stack[-1].level >= new_level:
                    header_stack.pop()

                # Push: 入栈当前标题
                header_stack.append(HeaderNode(level=new_level, title=new_title))

                # 把标题也加到正文里，保证语义连贯 (这对 Embedding 很有帮助)
                content_buffer.append(f"【{new_title}】")

            else:
                if line.strip():  # 忽略纯空行
                    content_buffer.append(line)

        # 3. 处理最后一段遗留文本
        if content_buffer:
            self._save_chunk(chunks, header_stack, content_buffer, source_name)

        return chunks

    def _save_chunk(self, chunks, header_stack, content_buffer, source_name):
        """辅助函数：打包数据"""
        full_text = "\n".join(content_buffer).strip()
        if not full_text:
            return

        # 生成面包屑路径: "Root > 1. 项目背景 > 1.1 现状"
        headers_list = [h.title for h in header_stack]
        current_path = " > ".join(headers_list) or "Root"

        chunk_obj = MarkdownChunk(
            id=str(uuid.uuid4()),
            content=full_text,
            metadata={
                "source": source_name,
                "path": current_path,
                "headers": headers_list,  # 必须加这个，给 Neo4j 用
                "level": len(header_stack)
            }
        )
        chunks.append(asdict(chunk_obj))


def process_all_markdowns():
    splitter = MarkdownContextSplitter()
    all_chunks = []

    # 扫描 Markdown 文件
    md_files = list(INPUT_DIR.glob("*.md"))
    if not md_files:
        print(f"错误：在 {INPUT_DIR} 没找到 .md 文件！请先运行 pdf_parser.py")
        return

    print(f"[AST Logic] 正在递归切分 {len(md_files)} 个 Markdown 文件...")

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        file_chunks = splitter.split_text(content, source_name=md_file.name)
        all_chunks.extend(file_chunks)
        print(f"  -> {md_file.name}: 生成 {len(file_chunks)} 个上下文语义块")

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"切分完成！已保存至: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    process_all_markdowns()