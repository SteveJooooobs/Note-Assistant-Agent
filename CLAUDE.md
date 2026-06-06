# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

笔记助手 Agent — AI Agent 培训班第 9 周 demo 项目。用 LangChain + LangGraph + ChromaDB 搭建，功能为：检索笔记（RAG 语义搜索 Obsidian 笔记）、新建笔记文件、新建文件夹。

## 运行时代码路径

- `.env` — `API_BASE_URL`、`API_KEY`（DeepSeek API）
- `config.py` — 所有模块级常量，其他文件 `from config import XXX` 引用。非敏感常量直接硬编码，不用类包裹
- 笔记源目录 `notes_backup/pesonal_notes/`（18 个中文 .md 文件，10 个子目录）
- 构建顺序：`rag.py` → `tools.py` → `agent.py` → `main.py`

## 技术选型

| 层 | 选型 | 说明 |
|---|---|---|
| LLM | ChatOpenAI（DeepSeek API 兼容） | 模型 `deepseek-v4-flash` |
| Embedding | `BAAI/bge-base-zh-v1.5`（本地） | 768 维，缓存到 `./models/` |
| 向量库 | ChromaDB 持久化 | 存储到 `./chroma_db/` |
| Agent | `langgraph.prebuilt.create_react_agent` | 工具绑定，自动决策调用 |
| 切块 | MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter 兜底 | 按 `##` 标题切语义块，超长才二次切 |

## 设计决策（详见 docs/）

- **嵌入模型选型** → `docs/嵌入模型选型.md`：为什么选 `bge-base-zh-v1.5`，选型标准，候选对比
- **增量更新策略** → `docs/增量更新策略.md`：`build_index()` 全量 / `update_index()` 增量，用文件路径+mtime 做去重，metadata 存 source 和 mtime
- **文档切块策略** → `docs/文档切块策略.md`：Markdown 标题切块为主，中文分隔符兜底，保证 chunk 语义完整

## 编码约束

- 用户自己敲代码，Claude 只负责指导——解答方法名、参数、报错问题，**不代写**
- 笔记为中文，切块和 Embedding 策略需考虑中文语义
- 项目长期维护，每 1-2 周增量更新知识库，rag.py 设计时预留增量更新函数签名
