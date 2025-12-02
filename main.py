import json
from pathlib import Path
# 导入我们刚写的切分器
from src.processing.markdown_splitter import MarkdownContextSplitter

# 路径配置
BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "chunks_cache.jsonl"


def main():
    # 1. 扫描所有的 .md 文件 (由 pdf_parser.py 生成)
    md_files = list(PROCESSED_DIR.glob("*.md"))

    if not md_files:
        print("❌ 没找到 Markdown 文件！请先运行 pdf_parser.py 解析 PDF。")
        return

    print(f"📂 找到 {len(md_files)} 个文档，准备切分...")

    splitter = MarkdownContextSplitter()
    total_chunks = 0

    # 2. 打开输出文件 (JSONL 格式)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for md_file in md_files:
            print(f"⚡ Processing: {md_file.name}")

            with open(md_file, "r", encoding="utf-8") as f_in:
                text = f_in.read()

            # === 调用核心切分逻辑 ===
            chunks = splitter.split_text(text, source_name=md_file.name)

            # 写入文件
            for chunk in chunks:
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            print(f"   -> 生成 {len(chunks)} 个切片")
            total_chunks += len(chunks)

    print(f"\n✅ 全部完成！")
    print(f"📊 总共生成: {total_chunks} 个知识块")
    print(f"💾 结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()