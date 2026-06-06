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
        tools=tools,
        system_prompt=(
            "你是一个个人笔记助手，帮助用户检索、整理和创建笔记。\n"
            "\n"
            "行为规则：\n"
            "1. 当用户询问知识类问题时，必须先用 search_notes 检索笔记，再基于检索结果回答。\n"
            "2. 回答时引用笔记来源，不要编造笔记中不存在的内容。\n"
            "3. 如果检索无结果，直接告诉用户'笔记中未找到相关内容'，不要猜测。\n"
            "4. 当用户要求创建笔记时，使用 create_note 工具。\n"
            "5. 当用户要求新建文件夹时，使用 create_folder 工具。\n"
            "6. 使用中文回答，简洁清晰。"
        ),
    )

    return agent