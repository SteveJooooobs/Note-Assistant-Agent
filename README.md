# 笔记助手 Agent

个人笔记助手，基于 LangChain + ChromaDB 搭建的 RAG Agent demo。
支持三种检索方式（稠密检索、BM25 稀疏检索、混合检索），以及新建笔记、新建文件夹。

## 功能

- **稠密检索**：基于语义相似度的向量检索，理解同义词和近义表达
- **BM25 稀疏检索**：基于关键词的精确匹配检索，适合术语和产品名查找
- **混合检索**：RRF 算法融合上述两种结果，兼顾语义和精确匹配
- **评分机制**：每种检索方法均返回文档级分数，可横向对比召回质量
- **新建笔记**：通过对话创建 Markdown 笔记文件
- **新建文件夹**：管理笔记目录结构

## 技术栈

| 层 | 选型 |
|---|---|
| LLM | ChatOpenAI → DeepSeek API（`deepseek-v4-flash`） |
| Embedding | `BAAI/bge-base-zh-v1.5`，本地加载 |
| 向量库 | ChromaDB 持久化 |
| 稀疏检索 | BM25Okapi（`rank_bm25`）+ jieba 中文分词 |
| 混合检索 | RRF（Reciprocal Rank Fusion）融合算法 |
| Agent | `langchain.agents.create_agent`（ReAct 范式） |
| 切块 | MarkdownHeaderTextSplitter（标题切） + RecursiveCharacterTextSplitter（兜底） |

## 项目结构

```
NoteAssistant/
├── src/
│   ├── rag.py          # RAG 管道：加载→切块→Embedding→ChromaDB→检索
│   ├── retrieval.py    # 检索方法模块：稠密/BM25/混合检索 + 评分 + 结果格式化
│   ├── tools.py        # 5 个 @tool 薄壳（3 检索 + 2 管理），调用 retrieval.py 核心逻辑
│   ├── agent.py        # create_agent + system_prompt
│   └── main.py         # CLI 交互入口，自动加载/构建 BM25 索引
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

## 评分机制

Agent 默认使用 `unified_retrieve_tool`（综合检索），同时运行三种检索策略，以**对比表格**展示每篇文档在三种策略下的得分：

```
            【三种检索策略分数对比】
────────────────────────────────────────────────────────────
 文档                       稠密检索     BM25  混合(RRF)
────────────────────────────────────────────────────────────
 RAG全链路与检索增强生成.md      0.5777  17.1912    1.2500
 RAG全链路与检索增强生成.md      0.6147  13.6187    1.2000
 大模型幻觉治理.md             0.5894        -    0.5000
 RAG全链路与检索增强生成.md           -  14.2614    0.3333
────────────────────────────────────────────────────────────
```

| 方法 | 分数含义 | 范围 | 说明 |
|---|---|---|---|
| 稠密检索 | 余弦相似度（归一化） | 0~1，越大越相关 | ChromaDB 返回 L2 距离，转为 `1/(1+distance)` |
| BM25 稀疏 | BM25 原始分数 | 0~∞，越大越相关 | 基于词频和逆文档频率的统计打分 |
| 混合检索 | RRF 融合分 | 0~2，越大越相关 | 文档在两路检索中排名越靠前，融合分越高 |

设计思路：
- **统一对比**：`unified_retrieve()` 同时执行三种检索，`build_comparison_table()` 收集所有唯一文档并构建对比表，'-' 表示该文档未被对应策略召回
- **默认综合检索**：Agent 收到知识类问题时优先调用 `unified_retrieve_tool`；如果无结果再回退到单个策略
- **分数透传**：对比表下方展示详细结果（按混合分排序），文档分数以 `[混合: 0.87]` 形式显示
- **不归一化**：稠密分和 BM25 分来自不同统计口径，各自在自己的体系内比较有意义，混合检索用 RRF 做了独立融合

## 设计决策

| 文档 | 内容 |
|---|---|
| [嵌入模型选型](docs/嵌入模型选型.md) | 为什么选 bge-base-zh-v1.5，选型标准 |
| [文档切块策略](docs/文档切块策略.md) | 标题切块为主 + 中文分隔符兜底 |
| [增量更新策略](docs/增量更新策略.md) | 全量/增量设计，文件路径+mtime 去重 |
| [Agent 工具设计](docs/Agent工具设计.md) | 工具边界设计与工厂函数模式 |
| [Agent 构建指南](docs/Agent构建指南.md) | create_agent 用法与消息格式 |
| [BM25 检索应用 QA](docs/BM25检索应用QA.md) | BM25 选型、持久化、增量、与向量检索组合 |
| [项目阶段总结](docs/项目阶段总结.md) | 当前完成度与遗留项 |

## 作者
Mjolnir[[github:](https://github.com/SteveJooooobs)]
