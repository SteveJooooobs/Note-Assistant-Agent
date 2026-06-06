from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
import json
import os
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EMBEDDING_MODEL, KNOWLEDGE_BASE_SOURCE_PATH, CHUNK_SIZE, CHUNK_OVERLAP



print("\n开始加载嵌入模型...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,         # huggingface下载很不稳定，选择直接下载好再本地加载
    model_kwargs={"device":"cpu"}      
)
print(f"\n嵌入模型加载完成,模型：{EMBEDDING_MODEL}")



# =============== 文档导入 ============
# 加载指定目录下所有.md文件
loader = DirectoryLoader(
    path=KNOWLEDGE_BASE_SOURCE_PATH,
    glob="**/*.md",         # 递归匹配所有.md文件
    loader_cls=TextLoader,  # 指定使用TextLoader加载纯文本
    loader_kwargs={"encoding": "utf-8"}      # 写死utf-8读,自动不好用
)
documents = loader.load()
print("\n=-=-=-=-=-=-=-=检查总笔记数量(文档数):", len(documents), "\n")
# print(f"\n=-=-=-=-=-=-=检查第一个文档内容:\n{documents[0]}")

# full_text = "\n\n".join([doc.page_content for doc in documents])        # 保存全部笔记文档--已弃用，逐文件切块使用下方循环完成




# ========= 文档切块逻辑 =============
# md文档初步按标题切
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "section")],
    strip_headers=False,        #保留标题文本
)

all_chunks = []
for doc in documents:
    # 
    chunks = md_splitter.split_text(doc.page_content)
    # 把原始文件的 metadata（含 source 路径）回填到每个 chunk
    for chunk in chunks:
        chunk.metadata = doc.metadata.copy()
    all_chunks.extend(chunks)
print('\n-------------all_chunks:',len(all_chunks))

# 过长的块再用字符级切分器切2次
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=['\n\n', '\n', '。', '，', ' ', '']   # 这里需要运行后看结果进行优化
)
# 兜底二次切分——RecursiveCharacterTextSplitter.split_documents 会自动保留 metadata
final_chunks = text_splitter.split_documents(all_chunks)        # 对单标题内超长文本切分后的chunks
print('\n-------------final_chunks:',len(final_chunks))



# ============= 构建向量库 ==============
def build_index(chunks, persist_dir):
    """全量构建向量索引"""
    if os.path.exists(persist_dir):
        print(f"向量库已存在: {persist_dir}，跳过构建。如需重建请手动删除该目录。")
        return Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    vectorstore = Chroma.from_documents(
        documents=chunks,                   # 待入库的Document列表,此处为final_chunks
        embedding=embeddings,               # 载入嵌入模型
        persist_directory=persist_dir       # 持久化目录，不传这个参数就是内存模式，进程结束就丢
    )
    return vectorstore



# ============= 增量更新辅助函数 ==============

def _load_index_record(record_path):
    """读取已入库文件的索引记录，文件不存在则返回空字典"""
    if not os.path.exists(record_path):
        return {}
    with open(record_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index_record(record_path, record):
    """保存索引记录到磁盘"""
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def _scan_md_files(docs_dir):
    """扫描目录下所有.md文件，返回 {相对路径: mtime_ISO字符串}

    mtime 使用 ISO 格式字符串存储，与 generate_index_record.py 脚本格式统一，
    避免 float 时间戳与字符串格式不匹配导致每个文件都被误判为"已修改"。
    """
    files = {}
    for root, dirs, filenames in os.walk(docs_dir):
        for fn in filenames:
            if fn.endswith(".md"):
                abs_path = os.path.join(root, fn)
                rel_path = os.path.relpath(abs_path, docs_dir)
                mtime_float = os.path.getmtime(abs_path)
                files[rel_path] = datetime.fromtimestamp(
                    mtime_float, tz=timezone.utc
                ).isoformat()
    return files


def _load_and_split_single_file(file_abs_path: str):
    """加载单个.md文件并切块，返回 chunk 列表（已携带 source metadata）"""
    loader = TextLoader(file_abs_path, encoding="utf-8")
    doc = loader.load()[0]          # TextLoader 单文件返回列表，取第一个
    chunks = md_splitter.split_text(doc.page_content)
    for chunk in chunks:
        chunk.metadata = doc.metadata.copy()   # 回填 source/mtime 等元数据
    return text_splitter.split_documents(chunks)  # 长块二次切分


def update_index(docs_dir, persist_dir, index_record_path):
    """增量更新向量索引

    对比 indexed_files.json 与当前文件列表，处理新增、修改、删除三类变更。
    """
    # 1. 打开已有向量库
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # 2. 读取已入库记录 + 扫描当前文件
    indexed = _load_index_record(index_record_path)
    current = _scan_md_files(docs_dir)

    # 3. 找出三类变更（mtime 均为 ISO 字符串，可直接比较）
    new_files = {k: v for k, v in current.items() if k not in indexed}
    modified_files = {k: v for k, v in current.items()
                      if k in indexed and indexed[k] != v}
    deleted_files = [k for k in indexed if k not in current]

    print(f"增量更新检测: 新增 {len(new_files)} | 修改 {len(modified_files)} | 删除 {len(deleted_files)}")

    # 4. 处理新增 + 修改（加载→切块→入库）
    changed = {**new_files, **modified_files}
    for rel_path in changed:
        abs_path = os.path.join(docs_dir, rel_path)
        chunks = _load_and_split_single_file(abs_path)
        if chunks:
            vectorstore.add_documents(chunks)
            print(f"  已入库: {rel_path} ({len(chunks)} chunks)")

    # 5. 处理删除（按 metadata.source 过滤删除）
    for rel_path in deleted_files:
        vectorstore.delete(where={"source": rel_path})
        print(f"  已删除: {rel_path}")

    # 6. 更新索引记录（current 已是 ISO 字符串格式，直接写入）
    _save_index_record(index_record_path, current)

    print("增量更新完成。")

# ============ 检索接口 ================
def retrieve(vectorstore, query, k=5):
    """语义检索, retrieve-召回"""
    retriever = vectorstore.as_retriever(search_kwargs={"k":k})
    return retriever.invoke(query)

# 测试期间使用全量构建方法，后期换成增量更新
vectorstore = build_index(final_chunks, "./chroma_db")  

docs = retrieve(vectorstore, "RAG检索策略有哪些")
for i, doc in enumerate(docs):
      print(f"[chunk {i}] 长度={len(doc.page_content)} | {doc.page_content[:80]}...")