from typing import TypedDict
from uuid import UUID

from app.db.models import Message
from app.retrieval.vector_retrieval import RetrievalChunk


class RAGState(TypedDict, total=False):
    conversation_id:UUID

    question:str

    chat_history:list[Message]

    query:str

    retrieved_chunks:list[RetrievalChunk]

    refused:bool

    answer:str

    user_message_id:UUID

    assistant_message_id:UUID
