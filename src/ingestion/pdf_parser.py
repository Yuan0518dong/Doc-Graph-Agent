import os
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat

# === 路径配置 ===
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_single_pdf(file_path):
    """
    子进程任务
    """
    try:
        safe_name = file_path.stem
        output_file = OUTPUT_DIR / f"{safe_name}.md"

        # 断点续传
        if output_file.exists() and output_file.stat().st_size > 100:
            return f"跳过: {safe_name}"

        # === 关键修改：强制使用 CPU ===
        # 1. 配置管道选项
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        # 强制指定 CPU
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.CPU
        )

        # 2. 初始化转换器
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # 3. 执行转换
        result = converter.convert(file_path)
        md_content = result.document.export_to_markdown()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        return f"✅ 成功: {safe_name}"

    except Exception as e:
        return f"失败: {file_path.name} - {str(e)[:100]}"


def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    total_files = len(pdf_files)

    if total_files == 0:
        print(f"警告: 还没下载完或者路径不对，暂时没找到 PDF。")
        return

    print(f"待处理: {total_files} 篇")

    # === 关键修改 ===
    # 你的 CPU 是 8 线程，内存 16GB。
    # 开 4 个进程比较稳，留一半资源给系统，防止内存溢出或卡死。
    max_workers = 1

    print(f"启用进程数: {max_workers} (平衡模式)")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_pdf, pdf_files), total=total_files, unit="file"))

    # 简单统计
    success_count = sum(1 for r in results if "✅" in r)
    duration = time.time() - start_time

    print(f"\n🎉 任务完成！")
    print(f"耗时: {duration / 3600:.2f} 小时")
    print(f"成功: {success_count} / {total_files}")


if __name__ == "__main__":
    main()