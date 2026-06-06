# Agent 构建指南

## ReAct Agent 原理

ReAct（Reasoning + Acting）是 Agent 的基础范式。LLM 在"思考→行动→观察→再思考"的循环中运行：

```
用户提问 → LLM 思考 → 需要查笔记？→ 调用 search_notes("RAG检索策略")
                         ↓
                    拿到检索结果 → 再思考 → 还需要更多信息？
                         ↓ 不需要了
                    生成最终回答 → 返回用户
```

LangGraph 的 `create_react_agent` 把这个循环封装好了，你只需要提供 LLM + 工具列表。

## agent.py 结构

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import API_KEY, API_BASE_URL, AGENT_MODEL_NAME


def create_agent(tools):
    """创建 ReAct Agent 实例"""
    llm = ChatOpenAI(
        model=AGENT_MODEL_NAME,      # "deepseek-v4-flash"
        api_key=API_KEY,
        base_url=API_BASE_URL,        # "https://api.deepseek.com"
        temperature=0.3,              # 工具调用场景，低温度更稳定
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    return agent
```

## 关键参数说明

### ChatOpenAI

| 参数 | 值 | 说明 |
|---|---|---|
| `model` | `"deepseek-v4-flash"` | DeepSeek 的 fast 模型，工具调用够用 |
| `api_key` | 从 `.env` 加载 | 兼容 OpenAI SDK 格式 |
| `base_url` | `"https://api.deepseek.com"` | 指向 DeepSeek API |
| `temperature` | `0.3` | 工具调用场景需要确定性强，不要太高 |

> `ChatOpenAI` 兼容 DeepSeek API 是因为 DeepSeek 的 API 协议和 OpenAI 一致。这是业界常见做法——用 OpenAI SDK 调兼容厂商。

### create_react_agent

| 参数 | 类型 | 说明 |
|---|---|---|
| `model` | `BaseChatModel` | LLM 实例，用于推理和决策 |
| `tools` | `List[BaseTool]` | 工具列表，即 `get_tools()` 的返回值 |

## Agent 的运行方式

`create_react_agent` 返回的不是一个函数，而是一个**编译好的 LangGraph 图**。调用方式：

```python
# 方式一：invoke —— 同步执行，返回最终状态
result = agent.invoke({
    "messages": [("user", "RAG的检索策略有哪些？")]
})
# result["messages"][-1] 是最后一条 AI 消息

# 方式二：stream —— 流式输出，能看到每步的思考过程
for chunk in agent.stream({
    "messages": [("user", "帮我查一下Transformer注意力机制")]
}):
    print(chunk)  # 会输出多段：思考 → 工具调用 → 工具结果 → 最终回答
```

## 完整调用链（agent.py → main.py 的关系）

```
main.py:
  1. 调 rag.build_index() → 得到 vectorstore
  2. 调 tools.get_tools(vectorstore) → 得到工具列表
  3. 调 agent.create_agent(tools) → 得到 Agent 实例
  4. 用户输入 → agent.invoke({"messages": [...]})
  5. 解析 agent 返回的消息 → 打印给用户
```

agent.py **只负责创建 Agent**，不负责运行。运行逻辑在 `main.py` 里。

## 消息格式

LangGraph 使用 OpenAI 兼容的消息格式：

```python
# 单条消息的格式
{"messages": [("user", "用户的问题")]}

# Agent 处理过程中 messages 列表会自动增长：
# [HumanMessage, AIMessage(工具调用), ToolMessage(工具结果), AIMessage(最终回答)]
```

从 `result["messages"]` 中取最后一条 `AIMessage` 的 `.content` 就是给用户的回答。

## 注意事项

1. **API 调用的网络延迟**：Agent 每次决策要走一次 API，调用工具后又要走一次。一个简单的查询通常 2-3 次 API 调用。
2. **工具返回字符串即可**：Agent 会自动把工具返回值包装成 `ToolMessage` 注入上下文。
3. **错误处理在 main.py 做**：`agent.py` 不处理异常，把错误往外抛，`main.py` 统一 try/except。
