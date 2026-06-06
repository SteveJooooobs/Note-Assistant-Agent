from dotenv import load_dotenv
import os
load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
# 模型名
AGENT_MODEL_NAME = "deepseek-v4-flash"

# 文本切块大小
CHUNK_SIZE = 800
# overlap大小
CHUNK_OVERLAP = 100

# 嵌入模型
EMBEDDING_MODEL = "./models/bge-base-zh-v1.5"
# 嵌入模型本地缓存目录
# EMBEDDING_CACHE_DIR = "./models"

# 知识库根目录
KNOWLEDGE_BASE_SOURCE_PATH = "./notes_backup/personal_notes"
# 向量知识库目录
KNOWLEDGE_BASE_VECTOR_PATH = "./chroma_db"
