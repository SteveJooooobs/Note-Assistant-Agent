import rag
from tools import get_tools
from agent import set_agent
from langchain_chroma import Chroma
from retrieval import (
    load_bm25_index,
    build_bm25_index,
    save_bm25_index,
    load_chunks,
    save_chunks,
)
from config import KNOWLEDGE_BASE_SOURCE_PATH

# ========== 加载已有向量库 ==========
vectorstore = Chroma(
    embedding_function=rag.embeddings,      # 共享 rag.py 已加载的 embedding 实例
    persist_directory="./chroma_db",
)

# ========== BM25 索引：存在则加载，不存在则重建 ==========
bm25_index = load_bm25_index()
chunks = load_chunks()  # BM25 检索需要 chunks 按索引取 Document

# 索引或 chunks 任一缺失都需要重建（避免 pickle 文件损坏或版本不一致）
if bm25_index is None or chunks is None:
    print("BM25 索引未找到或不完整，正在加载笔记并构建索引（首次运行较慢）...")
    final_chunks = rag.load_and_split_all(KNOWLEDGE_BASE_SOURCE_PATH)
    bm25_index = build_bm25_index(final_chunks)
    save_bm25_index(bm25_index)
    save_chunks(final_chunks)
    chunks = final_chunks
    print("BM25 索引及 chunks 已构建并持久化")
else:
    print("BM25 索引已就绪")

# ========== 组装 Agent ==========
tools = get_tools(vectorstore, bm25_index, chunks)
agent = set_agent(tools)

print("\n笔记助手已就绪（输入 quit / exit / q 退出）\n")

# ========== 交互循环 ==========
while True:
    user_input = input(">>> ")
    if user_input in ["quit", "exit", "q"]:
        print("已退出交互")
        break
    if not user_input.strip():
        continue

    # stream 逐节点输出，每个 chunk 是 {节点名: state_update}
    for chunk in agent.stream({"messages": [("user", user_input)]}):
        for node_name, node_output in chunk.items():
            if "messages" not in node_output:
                continue
            for msg in node_output["messages"]:
                msg_type = getattr(msg, "type", "")

                # AI 消息：可能包含 tool_calls 或文本回答
                if msg_type == "ai":
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            name = tc.get("name", "?")
                            args = tc.get("args", {})
                            print("=-=" * 5)
                            print(f"\n调用工具: {name}({args})")
                            print("=-=" * 5)
                            print("\n\n")
                    elif msg.content:
                        print(f"\n{msg.content}")

                # 工具返回结果
                elif msg_type == "tool":
                    preview = msg.content[:150].replace("\n", " ")
                    print("=-=" * 5)
                    print(f"\n   → 结果: {preview}...")
