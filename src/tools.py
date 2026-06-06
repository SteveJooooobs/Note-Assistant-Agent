from langchain.tools import tool
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import KNOWLEDGE_BASE_SOURCE_PATH


def get_tools(vectorstore):
    """创建并返回 Agent 工具列表。传入已构建的 Chroma 向量库实例。"""

    @tool
    def search_notes(query: str, k: int = 5) -> str:
        """语义搜索个人笔记，返回最相关的笔记片段及来源文件。

        当用户询问知识相关问题时使用此工具，例如：
        - "RAG的检索策略有哪些"
        - "Transformer注意力机制是怎么工作的"
        - "有没有关于模型部署的笔记"

        Args:
            query: 用户的自然语言查询
            k: 返回结果数量，默认5
        """
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)

            if not docs:
                return "未找到相关笔记。"

            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知来源")
                results.append(f"【{i}】来源: {source}\n{doc.page_content}\n")

            return "\n".join(results)
        except Exception as e:
            return f"搜索失败: {str(e)}"

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

    return [search_notes, create_note, create_folder]
