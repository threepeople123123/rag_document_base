from datetime import datetime
from typing import Literal, Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

MessageRoleValue = Literal["user","assistant","system"]

class ConversationCreate(BaseModel):
    title:str = Field("新对话",min_length=1,max_length=256)

class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id :UUID
    title:str
    create_at: datetime
    update_at:datetime

class CitationRead(BaseModel):

    id :UUID
    ordinal:int
    document_id:UUID | None = None
    chunk_id : UUID | None = None
    document_name :str | None = None
    page_no : int | None = None
    quote : str

    @classmethod
    def from_orm(cls,citation)->"CitationRead":
        return cls(
            id = citation.id,
            ordinal = citation.ordinal,
            document_id = citation.document_id,
            chunk_id = citation.chunk_id,
            document_name = citation.document_name,
            page_no = citation.page_no,
            quote = citation.quote,
        )

class MessageRead(BaseModel):
    id:UUID
    role : MessageRoleValue
    content :str
    create_at: datetime
    citations:list[CitationRead] = Field(default_factory=list)

    @classmethod
    def from_orm(cls,message ) -> "MessageRead":
        return cls(
            id = message.id,
            role = message.role,
            content = message.content,
            create_at = message.create_at,
            citations = [CitationRead.from_orm(c) for c in message.citations]
            if message.role == "assistant"
            else []
        )

class ConversationDetail(BaseModel):
    conversation:ConversationRead
    message : list[MessageRead]

class ChatRequest(BaseModel):
    question : str = Field(min_length=1,max_length=2000)