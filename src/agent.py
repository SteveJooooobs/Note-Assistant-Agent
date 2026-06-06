from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import API_KEY, API_BASE_URL, AGENT_MODEL_NAME

# 用set避免和langchain的create方法重名...
def set_agent(tools):
    """创建ReAct Agent实例"""
    llm = ChatOpenAI(
        model=AGENT_MODEL_NAME,
        api_key=API_KEY,
        base_url=API_BASE_URL,
        temperature=0.3     # 工具调用场景下，降低自由发挥的程度，采用低温
    )

    agent = create_agent(
        model=llm,
        tools=tools
    )

    return agent