from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import settings

_chat_model:BaseChatModel | None = None

def get_chat_model()->BaseChatModel:
    global _chat_model
    if _chat_model is not None:
        return _chat_model
    _chat_model = ChatOpenAI(
        base_url=settings.CHAT_BASE_URL,
        model=settings.CHAT_MODEL,
        api_key=settings.CHAT_API_KEY
        ,temperature=0
        ,streaming=True
    )

    return _chat_model