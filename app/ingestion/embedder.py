from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.exceptions import ConfigurationError

_embeddings: Embeddings | None = None

def get_embeddings()->Embeddings:
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    if not settings.EMBEDDING_API_KEY:
        raise ConfigurationError(code="api_key is null",message="向量 api_key 没有配置")
    _embeddings =  OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        base_url=settings.EMBEDDING_BASE_URL,
        api_key=settings.EMBEDDING_API_KEY,
        dimensions=settings.EMBEDDING_DIMENSIONS,
        chunk_size=settings.CHUNK_SIZE,
        check_embedding_ctx_length=False
    )
    return _embeddings
