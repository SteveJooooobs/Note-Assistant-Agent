# Main 流水线指南

## 职责

`main.py` 负责把 rag、tools、agent 三个模块串起来，提供交互入口。

## 调用链

```
1. 检查 chroma_db 是否存在
   ├─ 不存在 → rag.build_index(final_chunks, persist_dir) → 全量建库
   └─ 存在 → Chroma(embedding_function=..., persist_directory=...) → 直接加载
        ↓
2. tools.get_tools(vectorstore) → 工具列表
        ↓
3. agent.create_agent(tools) → Agent 实例
        ↓
4. while True: 读用户输入 → agent.invoke(...) → 打印回答
```

## 各环节关键方法

### 环节 1：加载向量库

```python
from langchain_chroma import Chroma

# 从磁盘加载已有向量库（不重建）
vectorstore = Chroma(
    embedding_function=rag.embeddings,      # 用 rag.py 里已加载好的 embedding 实例
    persist_directory="./chroma_db",
)
```

注意：这里不能直接 `import rag` 后拿 `rag.embeddings`，因为 `rag.py` 模块顶层代码会执行加载文档、切块等全流程。需要把 embedding 和全量建库逻辑解耦——**在 rag.py 尾部加 `if __name__ == "__main__"`**，让 `import rag` 时只加载函数和 embedding 对象，不执行全量流程。

### 环节 2：获取工具

```python
from tools import get_tools
tools = get_tools(vectorstore)
```

### 环节 3：创建 Agent

```python
from agent import set_agent
agent = set_agent(tools)
```

### 环节 4：交互循环

关键方法：

| 方法 | 用途 |
|---|---|
| `agent.invoke({"messages": [("user", 用户输入)]})` | 同步执行，返回最终状态 |
| `result["messages"]` | 整个对话的消息列表 |
| `result["messages"][-1]` | 最后一条消息，通常是 AIMessage |
| `result["messages"][-1].content` | AI 的回答文本 |

`stream` 模式（可选，能看到 Agent 思考过程）：

```python
for chunk in agent.stream({"messages": [("user", query)]}):
    # chunk 是每步的输出片段
    # 可以筛选 AIMessage 打印
```

## 需要改 rag.py 的地方

目前 `rag.py` 第 16-170 行的代码在模块顶层，`import rag` 时会全部执行。需要：

1. 把 `embeddings` 和 `md_splitter`、`text_splitter` 这些**共用的对象**留在模块顶层
2. 把全量加载、切块、建库、测试代码包进 `if __name__ == "__main__"` 块

这样 `main.py` 才能安全地 `import rag` 拿 `embeddings`，同时 `python -m src.rag` 仍可独立运行建库。

## 交互循环伪代码

```
加载 vectorstore
获取 tools
创建 agent

while True:
    user_input = input("> ")
    if 退出条件:
        break
    result = agent.invoke({"messages": [("user", user_input)]})
    print(result["messages"][-1].content)
```
