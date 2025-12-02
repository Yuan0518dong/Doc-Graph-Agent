import re
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class HeaderNode:
    level: int
    title: str


@dataclass
class MarkdownChunk:
    id: str  # 唯一标识符
    content: str  # 文本内容
    metadata: Dict  # 包含 path (路径), source (来源)


class MarkdownContextSplitter:
    def __init__(self):
        # 正则：匹配 Markdown 标题 (如 "## 1.1 背景")
        # ^(#{1,6}) —— 匹配行首 1～6 个 #
        # \s+ —— 至少一个空格
        # (.*) —— 标题内容
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.*)')

    def split_text(self, text: str, source_name: str = "unknown") -> List[Dict]:
        # 按行读取 Markdown，然后准备结果数组
        lines = text.split('\n')
        chunks = []

        # === 核心数据结构：栈 ===
        # 就像 LeetCode 394，用栈来记住我现在在哪个章节下面
        # header_stack 用来记住“我现在在哪个标题下面”
        # content_buffer 累积当前 chunk 的文本内容
        header_stack: List[HeaderNode] = []
        content_buffer: List[str] = []

        for line in lines:
            match = self.header_pattern.match(line)

            if match:
                # 1. 遇到新标题 -> 结算上一段内容
                if content_buffer:
                    # 面包屑路径: "Root > 1. 项目背景 > 1.1 现状"
                    current_path = " > ".join([h.title for h in header_stack]) or "Root"
                    full_text = "\n".join(content_buffer).strip()

                    if full_text:
                        chunks.append(asdict(MarkdownChunk(
                            id=str(uuid.uuid4()),
                            content=full_text,
                            metadata={
                                "source": source_name,
                                "path": current_path,
                                "level": len(header_stack)
                            }
                        )))
                    content_buffer = []  # 清空

                # 2. 栈操作 (维护层级)
                new_level = len(match.group(1))  # '#' 的数量
                new_title = match.group(2).strip()

                # Pop: 如果新标题级别更高或相等(比如从 1.1 到 1.2，或到 2.0)，弹出栈顶
                while header_stack and header_stack[-1].level >= new_level:
                    header_stack.pop()

                # Push: 入栈当前标题
                header_stack.append(HeaderNode(level=new_level, title=new_title))

                # 把标题也加到正文里，保证语义连贯
                content_buffer.append(f"【{new_title}】")

            else:
                # 普通文本 -> 放入缓存
                content_buffer.append(line)

        # 3. 处理最后一段遗留文本
        if content_buffer:
            current_path = " > ".join([h.title for h in header_stack]) or "Root"
            full_text = "\n".join(content_buffer).strip()
            if full_text:
                chunks.append(asdict(MarkdownChunk(
                    id=str(uuid.uuid4()),
                    content=full_text,
                    metadata={"source": source_name, "path": current_path, "level": len(header_stack)}
                )))

        return chunks