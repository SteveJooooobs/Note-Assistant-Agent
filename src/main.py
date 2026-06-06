import rag
from tools import get_tools
from agent import set_agent
from langchain_chroma import Chroma

# 加载本地向量库
vectorstore = Chroma(
    embedding_function = rag.embeddings,        # 从rag加载embedding实例,节约重建耗时(导入时已经构建了embedding实例)
    persist_directory = "./chroma_db",
)

tools = get_tools(vectorstore)
agent = set_agent(tools)

print("\n笔记助手已就绪（输入 quit / exit / q 退出）\n")

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
                            print("=-="*5)
                            print(f"\n调用工具: {name}({args})")
                            print("=-="*5)
                            print("\n\n")
                    elif msg.content:
                        print(f"\n{msg.content}")

                # 工具返回结果
                elif msg_type == "tool":
                    preview = msg.content[:150].replace("\n", " ")
                    print("\n=-="*5)
                    print(f"\n   → 结果: {preview}...")