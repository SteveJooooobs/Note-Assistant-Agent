from langchain.tools import tool
import os
import re
import sys
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 供 from retrieval import ... 解析
from config import KNOWLEDGE_BASE_SOURCE_PATH
from retrieval import dense_retrieve, bm25_retrieve, hybrid_retrieve, format_results


def get_tools(vectorstore, bm25_index: Optional[BM25Okapi] = None, chunks: Optional[list[Document]] = None):
    """创建并返回 Agent 工具列表。

    使用工厂函数模式，通过闭包将依赖（向量库、BM25 索引）注入到 @tool 中。
    每个 @tool 都是薄壳，核心检索逻辑在 retrieval.py 中。

    Args:
        vectorstore: Chroma 向量库实例（稠密检索需要）
        bm25_index: BM25Okapi 索引实例（BM25/混合检索需要）
        chunks: 原始 chunk 列表（BM25/混合检索时需要，用于按索引取 Document）
    """

    @tool
    def dense_retrieve_tool(query: str, k: int = 5) -> str:
        """语义搜索个人笔记，返回最相关的笔记片段及来源文件。

        稠密检索——按照语义相似度匹配。
        当需要按照用户语义匹配文档时使用此工具，例如：
        - "RAG的检索策略有哪些"
        - "Transformer注意力机制是怎么工作的"
        - "有没有关于模型部署的笔记"

        适合理解同义词、近义表达的开放式问答场景。
        如果此工具未找到结果，可以尝试 bm25_retrieve 工具做精确关键词匹配。

        Args:
            query: 用户的自然语言查询
            k: 返回结果数量，默认5
        """
        try:
            results = dense_retrieve(vectorstore, query, k)
            return format_results(results, method_label="稠密")
        except Exception as e:
            return f"稠密检索失败: {str(e)}"

    @tool
    def bm25_retrieve_tool(query: str, k: int = 5) -> str:
        """基于 BM25 关键词匹配搜索个人笔记，返回最相关的笔记片段及来源文件。

        稀疏检索——按照精确关键词匹配。
        当需要精确命中特定术语、产品名、型号时使用此工具，例如：
        - "KV-Cache是什么"
        - "BIO标签的标注规则"
        - "ONNX的部署流程"

        适合精确查找场景。如果检索无结果，可以尝试 dense_retrieve 工具做语义搜索。

        Args:
            query: 用户的查询关键词
            k: 返回结果数量，默认5
        """
        if bm25_index is None or chunks is None:
            return "BM25 索引未就绪，请确认已构建 BM25 索引后再使用。"
        try:
            results = bm25_retrieve(bm25_index, chunks, query, k)
            return format_results(results, method_label="BM25")
        except Exception as e:
            return f"BM25 检索失败: {str(e)}"

    @tool
    def hybrid_retrieve_tool(query: str, k: int = 5) -> str:
        """混合搜索个人笔记（稠密+稀疏），返回最相关的笔记片段及来源文件。

        同时使用语义匹配和关键词匹配，用 RRF 算法融合两种结果。
        当不确定该用哪种检索方式，或两者都想试时使用此工具。
        通常能取得最好的综合检索效果。

        如果此工具未找到结果，说明笔记库中确实没有相关内容。

        Args:
            query: 用户的查询内容
            k: 返回结果数量，默认5
        """
        if bm25_index is None or chunks is None:
            return "混合检索需要 BM25 索引，请确认已构建后再使用。"
        try:
            results = hybrid_retrieve(vectorstore, bm25_index, chunks, query, k)
            return format_results(results, method_label="混合")
        except Exception as e:
            return f"混合检索失败: {str(e)}"

    @tool
    def create_note(title: str, content: str, folder: str = "") -> str:
        """在笔记目录下创建新的 Markdown 笔记文件。

        当用户要求记录、新建笔记或保存内容时使用此工具，例如：
        - "帮我记一下RAG的核心知识点"
        - "写一篇关于Attention机制的笔记"
        - "新建一个部署流程的文档"

        Args:
            title: 笔记标题，将作为文件名（自动添加 .md 后缀）
            content: 笔记正文内容
            folder: 可选，存放的子目录名称，默认为笔记根目录
        """
        try:
            # 清洗标题中的非法文件名字符
            safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
            if not safe_title.endswith('.md'):
                safe_title += '.md'

            # 拼接目标路径
            if folder:
                safe_folder = re.sub(r'[\\/:*?"<>|]', '_', folder)
                target_dir = os.path.join(KNOWLEDGE_BASE_SOURCE_PATH, safe_folder)
            else:
                target_dir = KNOWLEDGE_BASE_SOURCE_PATH

            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, safe_title)

            if os.path.exists(file_path):
                return f"文件已存在: {file_path}，请使用不同的标题。"

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"笔记已创建: {file_path}"
        except Exception as e:
            return f"创建笔记失败: {str(e)}"

    @tool
    def create_folder(folder_name: str) -> str:
        """在笔记目录下创建新的子文件夹。

        当用户要求新建目录、整理分类时使用此工具，例如：
        - "建一个Prompt工程的目录"
        - "新建一个放微调相关笔记的文件夹"

        Args:
            folder_name: 文件夹名称，相对于笔记根目录
        """
        try:
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', folder_name)
            target_path = os.path.join(KNOWLEDGE_BASE_SOURCE_PATH, safe_name)

            if os.path.exists(target_path):
                return f"文件夹已存在: {target_path}"

            os.makedirs(target_path, exist_ok=True)
            return f"文件夹已创建: {target_path}"
        except Exception as e:
            return f"创建文件夹失败: {str(e)}"

    return [
        hybrid_retrieve_tool,
        dense_retrieve_tool,
        bm25_retrieve_tool,
        create_note,
        create_folder,
    ]
