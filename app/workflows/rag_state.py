from typing import TypedDict, Literal
from uuid import UUID

from app.db.models import Message
from app.retrieval.vector_retrieval import RetrievalChunk


query_router = Literal["original","rewrite","hyde","multi_query"]

class RAGState(TypedDict, total=False):
    conversation_id:UUID

    question:str

    route:query_router

    rewritten_query:str | None

    hyde_answer:str|None

    multi_queries:list[str] | None

    chat_history:list[Message]

    query:str

    retrieved_chunks:list[RetrievalChunk]

    refused:bool

    answer:str

    user_message_id:UUID

    assistant_message_id:UUID
