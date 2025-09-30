import argparse
import json
import os
import time

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 创建目录函数：确保输出目录存在
def create_directory(path):
    """如果目录不存在则创建，并打印创建信息"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


# 设置命令行参数解析器
parser = argparse.ArgumentParser(description="Process some text.")
parser.add_argument('--books_dir', type=str, required=True, help="包含原始文档的目录路径")
parser.add_argument('--model_path', type=str, required=True, help="嵌入模型路径")
args = parser.parse_args()

# 准备输入输出路径
book_files = os.listdir(args.books_dir)  # 获取原始文档列表
out_path = f'{args.books_dir.replace("/docs/", "/SemanticChunker/")}'  # 生成输出路径
create_directory(out_path)  # 确保输出目录存在
output_file = f"{out_path}/bge-large-en-v1.5.json"  # 输出文件名

# 主处理逻辑：仅当输出文件不存在时执行（或强制重新生成）
if not os.path.exists(output_file) or True:
    # 初始化嵌入模型（用于语义分割）
    embed_model = HuggingFaceEmbedding(model_name=args.model_path)
    
    # 创建语义分割器
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,  # 缓冲区大小
        breakpoint_percentile_threshold=74,  # 分割点百分位阈值
        embed_model=embed_model  # 使用的嵌入模型
    )

    # 记录处理开始时间
    start_time = time.time()
    print(f"Processing {len(os.listdir(args.books_dir))} files")
    
    # 加载文档数据
    documents = SimpleDirectoryReader(
        input_files=[os.path.join(args.books_dir, fn) for fn in os.listdir(args.books_dir)]
    ).load_data()
    
    # 执行语义分块处理
    nodes = splitter.get_nodes_from_documents(documents, show_progress=False)

    # 组织分块结果
    results = {}
    for node in nodes:
        file_name = node.extra_info['file_name']
        if file_name not in results:
            # 初始化文件的分块记录
            results[file_name] = {
                'filepath': file_name,
                'splits': []  # 存储分块内容
            }
        # 添加当前分块（文本内容和默认级别1）
        results[file_name]['splits'].append([node.text, 1])

    # 计算并打印处理时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{output_file} processed in: {execution_time:.2f} seconds")

    # 保存结果到JSON文件
    with open(output_file, 'w') as f:
        json.dump(results, f)
