# 笔记助手 Agent

个人笔记助手，基于 LangChain + LangGraph + ChromaDB 搭建的 RAG Agent demo。支持语义检索笔记、新建笔记文件、新建文件夹。

## 功能

- **语义检索**：用自然语言搜索个人笔记，返回相关片段及来源文件
- **新建笔记**：通过对话创建 Markdown 笔记文件
- **新建文件夹**：管理笔记目录结构

## 技术栈

| 层 | 选型 |
|---|---|
| LLM | ChatOpenAI → DeepSeek API（`deepseek-v4-flash`） |
| Embedding | `BAAI/bge-base-zh-v1.5`，本地加载 |
| 向量库 | ChromaDB 持久化 |
| Agent | `langchain.agents.create_agent`（ReAct 范式） |
| 切块 | MarkdownHeaderTextSplitter（标题切） + RecursiveCharacterTextSplitter（兜底） |

## 项目结构

```
NoteAssistant/
├── src/
│   ├── rag.py          # RAG 管道：加载→切块→Embedding→ChromaDB→检索
│   ├── tools.py         # 3 个 @tool（search_notes / create_note / create_folder）
│   ├── agent.py         # create_agent + system_prompt
│   └── main.py          # CLI 交互入口
├── config.py            # 模块级常量
├── docs/                # 设计决策文档
├── scripts/             # 辅助脚本
└── notes_backup/        # 笔记源文件（示例知识库）
```

## 快速开始

### 1. 环境准备

```bash
git clone <repo-url>
cd NoteAssistant
pip install -r requirements.txt
```

### 2. 配置 API

```bash
cp .env.example .env
# 编辑 .env，填入 API_BASE_URL 和 API_KEY
```

### 3. 下载嵌入模型

```bash
pip install modelscope
modelscope download --model BAAI/bge-base-zh-v1.5 --local_dir ./models/bge-base-zh-v1.5
```

### 4. 构建向量库

```bash
python src/rag.py
```

### 5. 启动

```bash
python src/main.py
```

## 设计决策

| 文档 | 内容 |
|---|---|
| [嵌入模型选型](docs/嵌入模型选型.md) | 为什么选 bge-base-zh-v1.5，选型标准 |
| [文档切块策略](docs/文档切块策略.md) | 标题切块为主 + 中文分隔符兜底 |
| [增量更新策略](docs/增量更新策略.md) | 全量/增量设计，文件路径+mtime 去重 |
| [Agent 工具设计](docs/Agent工具设计.md) | 三个工具的边界决策 |
| [Agent 构建指南](docs/Agent构建指南.md) | create_agent 用法与消息格式 |
| [项目阶段总结](docs/项目阶段总结.md) | 当前完成度与遗留项 |

## 作者
Mjolnir[github:]
