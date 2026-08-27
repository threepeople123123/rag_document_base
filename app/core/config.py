from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：app/core/config.py 向上两级
PROJECTING = Path(__file__).resolve().parents[2]

class Setting(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECTING / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 定义你要读取的配置项，字段名就是环境变量名
    app_name: str = "rag-document-base"
    debug: bool = False
    DATABASE_URL:str = "postgresql+asyncpg://postgres:wangjian020426@8.140.219.233:5433/rag_document_base"
    LOG_LEVER:int = 1

    MINIO_ENDPOINT:str = "http://8.140.219.233:9000"
    MINIO_ACCESS_KEY:str = "minioadmin"
    MINIO_SECRET_KEY:str = "minioadmin"
    MINIO_SECURE:bool = False
    BUCKET_NAME:str = "pythone-rag-document"
    REGION:str = "nanjing"

    CORS_ORIGINS:str = "localhost:5173"
    @property
    def cores_origin_list(self)->list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # 向量配置
    EMBEDDING_API_KEY:str = "sk-0363d3e4787e4ab19253e56309e0ff95"
    # aliyun最新根据业务空间id调用，猜测是为了分流和最近节点的服务器，更快
    EMBEDDING_BASE_URL:str = "https://llm-3l2i84ztuewo30qg.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL_NAME:str = "qwen3.7-text-embedding"
    EMBEDDING_DIMENSIONS:int = 1024
    CHUNK_SIZE:int = 600
    CHUNK_OVERLAP:int = 60

    UPLOAD_MAX_SIZE_MB:int = 10

    CHAT_API_KEY:str="sk-0363d3e4787e4ab19253e56309e0ff95"
    CHAT_BASE_URL:str = ""
    CHAT_MODEL:str = ""
    MINI_CHAT_MODEL: str = ""
    RETRIEVAL_TOP_K:int = 5
    RETRIEVAL_MIN_SCORE:str =""
    CHAT_HISTORY_WINDOW:str =""
    FINAL_TOP_K:int=5

    QUERY_ROUTE_ENABLED:bool = True
    MULTI_QUERY_COUNT:int = 3

    RRF_K: int = 60




@lru_cache
def get_setting() -> Setting:
    return Setting()


settings = get_setting()
