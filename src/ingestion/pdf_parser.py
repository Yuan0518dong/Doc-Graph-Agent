import os
import time
import torch
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"


def convert_pdfs():
    if not INPUT_DIR.exists():
        print(f"❌ 错误：请先创建 {INPUT_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === 🔥 GPU 配置核心代码 ===
    # 检查是否有显卡
    use_gpu = torch.cuda.is_available()
    device_str = "CUDA" if use_gpu else "CPU"
    print(f"🚀 正在初始化 Docling 模型... (当前设备: {device_str})")

    # 配置管道选项
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # 开启OCR
    pipeline_options.do_table_structure = True  # 开启表格结构识别

    # 强制指定设备
    if use_gpu:
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.CUDA
        )

    # 使用配置初始化转换器
    converter = DocumentConverter(
        format_options={
            # 将配置应用到 PDF 处理流中
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 扫描文件
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print("⚠️ 警告：data/raw 为空！")
        return

    print(f"📂 发现 {len(pdf_files)} 个 PDF，开始处理...")

    for pdf_file in pdf_files:
        t0 = time.time()
        print(f"\n⚡ [{device_str}] 正在解析: {pdf_file.name} ...")

        try:
            result = converter.convert(pdf_file)
            md_content = result.document.export_to_markdown()

            output_path = OUTPUT_DIR / (pdf_file.stem + ".md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            print(f"✅ 成功! 耗时: {time.time() - t0:.2f}s")

        except Exception as e:
            print(f"❌ 失败: {e}")


if __name__ == "__main__":
    convert_pdfs()