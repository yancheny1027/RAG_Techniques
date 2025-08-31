import os
import sys
import argparse
import time
from dotenv import load_dotenv

# 项目根目录 = 当前脚本目录的上一级
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 指定 .env 文件路径
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from helper_functions import *
from evaluation.evalute_rag import *


class SimpleRAG:
    """
    用于处理文档分块和查询检索的简单 RAG 过程的类。
    """

    def __init__(self, path, chunk_size=1000, chunk_overlap=200, n_retrieved=2):
        """
        通过对 PDF 文档进行编码并创建检索器来初始化简单 RAG 检索器。

        参数：
            path （str）：要编码的 PDF 文件的路径。
            chunk_size （int）：每个文本块的大小（默认值：1000）。
            chunk_overlap （int）：连续块之间的重叠（默认值：200）。
            n_retrieved （int）：每个查询要检索的块数（默认值：2）。
        """
        print("\n--- 初始化简单 RAG 检索器 ---")

        # 使用 OpenAI 向量模型将 PDF 文档编码到向量存储中
        start_time = time.time()
        self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # 从向量存储创建检索器
        self.chunks_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        检索并显示给定查询的上下文。

        参数：
            query （str）：要检索上下文的查询。

        返回：
            元组：检索时间。
        """
        # 测量检索时间
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # 显示检索到的上下文
        show_context(context)


# 验证命令行输入的函数
def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size必须是正整数。")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap必须是非负整数。")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved必须是正整数。")
    return args


# 解析命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser(description="Encode a PDF document and test a simple RAG.")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change-1-10.pdf",
                        help="要编码的 PDF 文件的路径。")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="每个文本块的大小（默认值：1000）。")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="连续块之间的重叠（默认值：200）。")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="每个查询要检索的块数（默认值：2）。")
    parser.add_argument("--query", type=str, default="What is the main cause of climate change?",
                        help="用于测试检索器的查询（默认值：“气候变化的主要原因是什么？”)。")
    parser.add_argument("--evaluate", action="store_true",
                        help="是否评估检索器的性能（默认值：False）。")

    # 解析和验证参数
    return validate_args(parser.parse_args())


# 用于处理参数解析并调用 SimpleRAGRetriever 类的 Main 函数
def main(args):
    # 初始化 SimpleRAGRetriever
    simple_rag = SimpleRAG(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    # 根据查询检索上下文
    simple_rag.run(args.query)

    # 评估检索器在查询上的性能（如果请求）
    if args.evaluate:
        evaluate_rag(simple_rag.chunks_query_retriever)


if __name__ == '__main__':
    # 使用解析的参数调用 main 函数
    main(parse_args())
