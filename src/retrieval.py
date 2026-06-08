"""检索方法模块：稠密检索、BM25 稀疏检索、混合检索

职能边界：
- 提供三种检索方法的核心逻辑，返回 List[Document]
- 提供 BM25 索引的构建、持久化、加载
- 返回结果给 tools.py 中的 @tool 薄壳做格式化后输出
- 不接触 @tool 装饰器，不接触 Agent
"""

import pickle
import jieba
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma


# ============ BM25 索引持久化路径 ============
# 和 ChromaDB 放在同一目录下，同生命周期
BM25_INDEX_PATH = "./chroma_db/bm25_index.pkl"


# ============ 中文分词 ============

def _tokenize(text: str) -> List[str]:
    """中文分词，BM25 需要以词为单位建索引

    jieba 的 cut 返回生成器，用 list() 消费。
    默认精确模式，对技术笔记中的术语分词效果较好。
    """
    return list(jieba.cut(text))


# ============ BM25 索引管理 ============

def build_bm25_index(chunks: List[Document]) -> BM25Okapi:
    """从切好的 chunk 列表构建 BM25 索引

    Args:
        chunks: 经过两级切分的 Document 列表（即 final_chunks）

    Returns:
        BM25Okapi 实例，内部维护了 Term-Document 矩阵
    """
    corpus = [chunk.page_content for chunk in chunks]
    tokenized_corpus = [_tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def save_bm25_index(bm25: BM25Okapi, path: str = BM25_INDEX_PATH):
    """持久化 BM25 索引到磁盘

    用 pickle 序列化整个 BM25Okapi 对象。
    不持久化原始文本，只持久化索引结构（词频表、文档长度等），体积小、加载快。

    Args:
        bm25: BM25Okapi 实例
        path: 持久化路径，默认与 ChromaDB 同目录
    """
    with open(path, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25_index(path: str = BM25_INDEX_PATH) -> Optional[BM25Okapi]:
    """从磁盘加载 BM25 索引

    文件不存在时返回 None，调用方决定是否重建。

    Args:
        path: 持久化路径

    Returns:
        BM25Okapi 实例，或 None（索引不存在时）
    """
    if Path(path).exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ============ 检索方法 ============

def dense_retrieve(vectorstore: Chroma, query: str, k: int = 5) -> List[Document]:
    """稠密检索：基于语义相似度的向量检索

    将 query 用同一 embedding 模型转为向量，与库中所有 chunk 向量算余弦相似度，
    返回最相似的 top-k 个 Document。

    Args:
        vectorstore: Chroma 向量库实例（已加载 embedding_function）
        query: 用户的自然语言查询
        k: 返回结果数量

    Returns:
        按相似度降序排列的 Document 列表
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def bm25_retrieve(bm25: BM25Okapi, chunks: List[Document], query: str, k: int = 5) -> List[Document]:
    """稀疏检索：基于 BM25 关键词匹配

    将 query 中文分词后，与索引中每个文档的词频做 BM25 打分，
    返回分数最高的 top-k 个 Document。

    BM25 适合精确关键词匹配场景（如产品名、型号），
    与稠密检索（语义匹配）互补。

    Args:
        bm25: BM25Okapi 索引实例
        chunks: 与 BM25 索引构建时一致的 chunk 列表（索引位置对应）
        query: 用户的查询
        k: 返回结果数量

    Returns:
        按 BM25 分数降序排列的 Document 列表
    """
    tokenized_query = _tokenize(query)

    # get_scores 返回每个文档的 BM25 分数，索引与构建时传入的 corpus 顺序一致
    scores = bm25.get_scores(tokenized_query)

    # 按分数排序，取 top-k（只返回分数 > 0 的真正匹配结果）
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, score in indexed_scores[:k] if score > 0]

    # 按分数从高到低返回对应的 Document
    return [chunks[i] for i in top_k_indices]


def hybrid_retrieve(
    vectorstore: Chroma,
    bm25: BM25Okapi,
    chunks: List[Document],
    query: str,
    k: int = 5,
) -> List[Document]:
    """混合检索：稠密 + 稀疏结果用 RRF 融合

    同时调用 dense_retrieve 和 bm25_retrieve，然后用
    Reciprocal Rank Fusion (RRF) 算法对两组结果做排序融合。

    RRF 核心思想：文档在两种检索结果中排名越高，最终得分越高。
    公式：score(d) = Σ 1 / (rank(d) + 1)

    优势：
    - 不需要调权重（不像加权平均需要调稠密/稀疏比例）
    - 同时覆盖语义匹配和精确关键词匹配
    - 实现简单，效果稳定

    Args:
        vectorstore: Chroma 向量库实例
        bm25: BM25Okapi 索引实例
        chunks: chunk 列表
        query: 用户的查询
        k: 返回结果数量

    Returns:
        RRF 融合后按分数降序排列的 Document 列表（已去重）
    """
    dense_results = dense_retrieve(vectorstore, query, k)
    bm25_results = bm25_retrieve(bm25, chunks, query, k)

    # 用文档内容哈希做去重标识
    # 不直接用 source 路径，避免同一文件的不同 chunk 被误认为同一文档
    def doc_key(doc: Document) -> str:
        return doc.metadata.get("source", "") + "::" + doc.page_content[:80]

    # RRF 打分：每个结果中 rank 越靠前，得分贡献越大
    rrf_scores = {}

    for rank, doc in enumerate(dense_results):
        key = doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rank + 1)

    for rank, doc in enumerate(bm25_results):
        key = doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rank + 1)

    # 按 RRF 分数降序排列，取 top-k
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:k]

    # 按排序后的 key 从原始结果中取 Document，保持顺序
    seen = set()
    result = []
    for doc in dense_results + bm25_results:
        key = doc_key(doc)
        if key in sorted_keys and key not in seen:
            result.append(doc)
            seen.add(key)

    return result


# ============ 结果格式化 ============

def format_results(docs: List[Document]) -> str:
    """将 Document 列表格式化为可读字符串，供 @tool 返回给 Agent

    Args:
        docs: 检索返回的 Document 列表

    Returns:
        格式化后的文本，包含序号、来源文件和内容片段
    """
    if not docs:
        return "未找到相关笔记。"

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        results.append(f"【{i}】来源: {source}\n{doc.page_content}\n")

    return "\n".join(results)
