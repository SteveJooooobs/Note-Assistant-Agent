"""检索方法模块：稠密检索、BM25 稀疏检索、混合检索

职能边界：
- 提供三种检索方法的核心逻辑，返回 List[Tuple[Document, score]]
- 提供 BM25 索引的构建、持久化、加载
- 检索结果保留原始分数，供 tools.py 格式化后展示给用户
- 不接触 @tool 装饰器，不接触 Agent
"""

import pickle
import jieba
import sys
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BM25_INDEX_PATH, CHUNKS_PATH


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


def save_chunks(chunks: List[Document], path: str = CHUNKS_PATH):
    """持久化 chunk 列表到磁盘

    BM25 检索时需要通过索引位置反查 Document，必须和 BM25 索引搭配保存。

    Args:
        chunks: 经过两级切分的 Document 列表
        path: 持久化路径
    """
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(path: str = CHUNKS_PATH) -> Optional[List[Document]]:
    """从磁盘加载 chunk 列表

    文件不存在时返回 None。

    Args:
        path: 持久化路径

    Returns:
        Document 列表，或 None（文件不存在时）
    """
    if Path(path).exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None





# ============ 检索方法 ============

def dense_retrieve(vectorstore: Chroma, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """稠密检索：基于语义相似度的向量检索

    将 query 用同一 embedding 模型转为向量，与库中所有 chunk 向量算余弦相似度，
    返回最相似的 top-k 个 Document 及其分数。

    分数说明：ChromaDB 返回 L2 距离（越小越相关），
    此处转为 0~1 相似度（越大越相关），公式：sim = 1 / (1 + distance)。

    Args:
        vectorstore: Chroma 向量库实例（已加载 embedding_function）
        query: 用户的自然语言查询
        k: 返回结果数量

    Returns:
        按相似度降序排列的 (Document, score) 元组列表
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    # similarity_search_with_score 返回 (Document, L2_distance)，
    # 转为 (Document, 1/(1+distance)) 使分数在 (0,1] 且越大越相关
    return [(doc, round(1.0 / (1.0 + score), 4)) for doc, score in docs_with_scores]


def bm25_retrieve(bm25: BM25Okapi, chunks: List[Document], query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """稀疏检索：基于 BM25 关键词匹配

    将 query 中文分词后，与索引中每个文档的词频做 BM25 打分，
    返回分数最高的 top-k 个 Document 及其分数。

    BM25 适合精确关键词匹配场景（如产品名、型号），
    与稠密检索（语义匹配）互补。

    分数说明：BM25 原始分数，越高越相关，范围通常 0~20+，
    具体数值受语料大小、词频分布影响，横向对比同一次查询有意义。

    Args:
        bm25: BM25Okapi 索引实例
        chunks: 与 BM25 索引构建时一致的 chunk 列表（索引位置对应）
        query: 用户的查询
        k: 返回结果数量

    Returns:
        按 BM25 分数降序排列的 (Document, score) 元组列表
    """
    tokenized_query = _tokenize(query)

    # get_scores 返回每个文档的 BM25 分数，索引与构建时传入的 corpus 顺序一致
    scores = bm25.get_scores(tokenized_query)

    # 按分数排序，取 top-k（只返回分数 > 0 的真正匹配结果）
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_k = [(idx, score) for idx, score in indexed_scores[:k] if score > 0]

    # 按分数从高到低返回 (Document, score) 元组
    return [(chunks[idx], round(score, 4)) for idx, score in top_k]


def hybrid_retrieve(
    vectorstore: Chroma,
    bm25: BM25Okapi,
    chunks: List[Document],
    query: str,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """混合检索：稠密 + 稀疏结果用 RRF 融合

    同时调用 dense_retrieve 和 bm25_retrieve，然后用
    Reciprocal Rank Fusion (RRF) 算法对两组结果做排序融合。

    RRF 核心思想：文档在两种检索结果中排名越高，最终得分越高。
    公式：score(d) = Σ 1 / (rank(d) + 1)

    优势：
    - 不需要调权重（不像加权平均需要调稠密/稀疏比例）
    - 同时覆盖语义匹配和精确关键词匹配
    - 实现简单，效果稳定

    分数说明：RRF 融合分，范围 0~2，越大越相关。
    文档同时被稠密和 BM25 检索召回到较高位时得分最高。

    Args:
        vectorstore: Chroma 向量库实例
        bm25: BM25Okapi 索引实例
        chunks: chunk 列表
        query: 用户的查询
        k: 返回结果数量

    Returns:
        RRF 融合后按分数降序排列的 (Document, score) 元组列表（已去重）
    """
    # dense_retrieve / bm25_retrieve 现在返回 List[Tuple[Document, float]]
    # 这里只取 Document 用于 RRF 排序
    dense_results = [doc for doc, _ in dense_retrieve(vectorstore, query, k)]
    bm25_results = [doc for doc, _ in bm25_retrieve(bm25, chunks, query, k)]

    # 用文档内容哈希做去重标识
    def doc_key(doc: Document) -> str:
        return doc.metadata.get("source", "") + "::" + doc.page_content[:80]

    # RRF 打分
    rrf_scores: dict[str, float] = {}

    for rank, doc in enumerate(dense_results):
        key = doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rank + 1)

    for rank, doc in enumerate(bm25_results):
        key = doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rank + 1)

    # 按 RRF 分数降序排列，取 top-k
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    top_keys = {item[0] for item in sorted_items}

    # 按排序后的 key 取 Document，拼上 RRF 分数返回
    seen = set()
    result: List[Tuple[Document, float]] = []
    for doc in dense_results + bm25_results:
        key = doc_key(doc)
        if key in top_keys and key not in seen:
            result.append((doc, round(rrf_scores[key], 4)))
            seen.add(key)

    return result


# ============ 表格辅助函数 ============


def _visual_width(s: str) -> int:
    """计算字符串在终端中的视觉宽度（中文全角=2，ASCII半角=1）

    用于生成表格时对齐含中文的列，避免 Python 的 len() 把中文算成 1 列而导致错位。
    """
    return sum(2 if ord(c) > 127 else 1 for c in s)


def _pad_visual(s: str, width: int) -> str:
    """将字符串填充到指定的视觉宽度

    Args:
        s: 待填充的字符串
        width: 目标视觉宽度

    Returns:
        填充后的字符串（右侧补空格）
    """
    return s + " " * max(0, width - _visual_width(s))


def _content_preview(content: str, max_chars: int = 20) -> str:
    """提取内容前 max_chars 个字作为预览

    取文档内容的前 max_chars 个字符（去除首尾空白），超长则追加 '...'。
    用于在对比表格下方展示每篇文档的概要，帮助判断是否需要查看完整内容。

    Args:
        content: 文档正文
        max_chars: 预览字符数，默认 20

    Returns:
        预览字符串
    """
    text = content.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ============ 对比表格 ============


def build_comparison_table(
    dense_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    hybrid_results: List[Tuple[Document, float]],
) -> str:
    """构建三种检索策略的分数对比表格

    对同一查询，收集三路检索召回的每篇文档，展示其在每种策略下的分数。
    文档未被某策略召回时显示 '-'。

    列宽按视觉宽度对齐（中文全角=2，ASCII=1），确保终端显示不乱。

    Args:
        dense_results: 稠密检索结果列表 (Document, score)
        bm25_results: BM25 检索结果列表 (Document, score)
        hybrid_results: 混合检索结果列表 (Document, score)

    Returns:
        格式化的对比表格字符串
    """

    def doc_key(doc: Document) -> str:
        return doc.metadata.get("source", "") + "::" + doc.page_content[:80]

    # 收集所有唯一文档，记录每路得分
    all_docs: dict[str, dict] = {}

    for doc, score in dense_results:
        key = doc_key(doc)
        all_docs.setdefault(key, {"doc": doc, "dense": None, "bm25": None, "hybrid": None})
        all_docs[key]["dense"] = score

    for doc, score in bm25_results:
        key = doc_key(doc)
        all_docs.setdefault(key, {"doc": doc, "dense": None, "bm25": None, "hybrid": None})
        all_docs[key]["bm25"] = score

    for doc, score in hybrid_results:
        key = doc_key(doc)
        all_docs.setdefault(key, {"doc": doc, "dense": None, "bm25": None, "hybrid": None})
        all_docs[key]["hybrid"] = score

    if not all_docs:
        return "\n【三种检索策略分数对比】未召回任何文档\n"

    # 按混合分降序排列（None 当 0 处理）
    sorted_items = sorted(
        all_docs.items(),
        key=lambda x: x[1]["hybrid"] if x[1]["hybrid"] is not None else 0,
        reverse=True,
    )

    # 列宽设定（视觉宽度，ASCII=1，中文=2）
    COL_ID = 4       # " #  "
    W_SRC = 40       # 文档名
    W_DENSE = 10     # 稠密分
    W_BM25 = 10      # BM25 分
    W_HYBRID = 10    # 混合分
    # 总视觉宽度 = 列宽 + 4 个 " | " 分隔符（每个3视觉宽度）
    TOTAL_W = COL_ID + W_SRC + W_DENSE + W_BM25 + W_HYBRID + 4 * 3
    SEP = "-" * TOTAL_W

    def _col(content: str, width: int) -> str:
        return _pad_visual(content, width)

    header = (
        f"{' #':>{COL_ID}}"
        f" | {_col('文档来源', W_SRC)}"
        f" | {_col('稠密检索', W_DENSE)}"
        f" | {_col('BM25', W_BM25)}"
        f" | {_col('混合(RRF)', W_HYBRID)}"
    )

    div = (
        f"{'---':>{COL_ID}}"
        f" | {'---':>{W_SRC}}"
        f" | {'---':>{W_DENSE}}"
        f" | {'---':>{W_BM25}}"
        f" | {'---':>{W_HYBRID}}"
    )

    rows = [header, div]

    for rank, (key, data) in enumerate(sorted_items, 1):
        doc = data["doc"]
        src = doc.metadata.get("source", "未知来源")
        display_src = Path(src).name

        d = f"{data['dense']:.4f}" if data["dense"] is not None else "-"
        b = f"{data['bm25']:.4f}" if data["bm25"] is not None else "-"
        h = f"{data['hybrid']:.4f}" if data["hybrid"] is not None else "-"

        line = (
            f"{f' {rank}':>{COL_ID}}"
            f" | {_col(display_src, W_SRC)}"
            f" | {_col(d, W_DENSE)}"
            f" | {_col(b, W_BM25)}"
            f" | {_col(h, W_HYBRID)}"
        )
        rows.append(line)

    table = f"\n{'【三种检索策略分数对比】':^{TOTAL_W}}\n{SEP}\n" + "\n".join(rows) + f"\n{SEP}\n"

    # ========== 内容预览（与表格行号对应） ==========
    # " | " 分隔符 × 2 = 6，预览列宽 = 总宽 - 序号列 - 来源列 - 分隔符
    W_PREVIEW = TOTAL_W - COL_ID - W_SRC - 6

    preview_rows = [
        "",
        f"{'【内容预览】':^{TOTAL_W}}",
        SEP,
        f"{' #':>{COL_ID}}"
        f" | {_col('来源文件', W_SRC)}"
        f" | {_col('内容预览（前20字）', W_PREVIEW)}",
        f"{'---':>{COL_ID}}"
        f" | {'---':>{W_SRC}}"
        f" | {'---':>{W_PREVIEW}}",
    ]
    for rank, (key, data) in enumerate(sorted_items, 1):
        doc = data["doc"]
        src = doc.metadata.get("source", "未知来源")
        display_src = Path(src).name
        preview = _content_preview(doc.page_content, max_chars=20)
        preview_rows.append(
            f"{f' {rank}':>{COL_ID}}"
            f" | {_col(display_src, W_SRC)}"
            f" | {_col(preview, W_PREVIEW)}"
        )
    preview_rows.append(SEP)

    return table + "\n".join(preview_rows) + "\n"


def unified_retrieve(
    vectorstore: Chroma,
    bm25: BM25Okapi,
    chunks: List[Document],
    query: str,
    k: int = 5,
) -> str:
    """统一执行三种检索策略，返回对比表格

    同时运行稠密检索、BM25 稀疏检索和混合检索，
    只返回分数对比表格供用户快速比较，不返回文档具体内容。

    Args:
        vectorstore: Chroma 向量库实例
        bm25: BM25Okapi 索引实例
        chunks: chunk 列表
        query: 用户的查询
        k: 每种策略返回结果数量

    Returns:
        三种检索策略的分数对比表格
    """
    dense_results = dense_retrieve(vectorstore, query, k)
    bm25_ret = bm25_retrieve(bm25, chunks, query, k)
    hybrid_results = hybrid_retrieve(vectorstore, bm25, chunks, query, k)

    return build_comparison_table(dense_results, bm25_ret, hybrid_results)


# ============ 结果格式化 ============

def format_results(
    docs_with_scores: List[Tuple[Document, float]],
    method_label: str = "",
) -> str:
    """将 (Document, score) 列表格式化为可读字符串，供 @tool 返回给 Agent

    Args:
        docs_with_scores: 检索返回的 (Document, score) 元组列表
        method_label: 检索方法名称，如"稠密""BM25""混合"，用于分数标注

    Returns:
        格式化后的文本，包含序号、来源文件、分数和内容片段
    """
    if not docs_with_scores:
        return "未找到相关笔记。"

    # 构建分数标签，如 "[稠密: 0.87]"
    label_prefix = f"[{method_label}: " if method_label else "["

    results = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        source = doc.metadata.get("source", "未知来源")
        results.append(
            f"【{i}】来源: {source}  {label_prefix}{score}]\n{doc.page_content}\n"
        )

    return "\n".join(results)
