from datetime import datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy.dialects.postgresql import JSONB, UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Text, DateTime, func, BigInteger, ForeignKey, Integer
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.orm import mapped_column

from app.core.config import settings
from app.db.base import Base


class DocumentStatus(str,Enum):
    """
    UPLOADING. 上传中
    PARSING  解析中
    INDEXING 切分 + 向量化 + 写chunk中
    READY    已经准备好
    FAILED   失败
    """
    UPLOADING = "uploading"
    PARSING = "parsing"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


# document 表对应的实体类
class Document(Base):
    __tablename__ = "document"
    id : Mapped[UUID] = mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid4)
    name : Mapped[str] = mapped_column(String(512),nullable= False)
    file_hash : Mapped[str] = mapped_column(String(64),nullable=False)
    mime_type : Mapped[str] = mapped_column(String(128),nullable=False)
    size: Mapped[int] = mapped_column(BigInteger,nullable=False)
    storage_provider: Mapped[str] = mapped_column(String(32), nullable=False, default="cos")
    cos_bucket: Mapped[str]= mapped_column(String(128), nullable=False)
    cos_object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    cos_region: Mapped[str]= mapped_column(String(64), nullable=False)

    status: Mapped[DocumentStatus] = mapped_column(String(32), nullable=False, default=DocumentStatus.UPLOADING)
    error_message: Mapped[str | None]= mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default = func.now(), nullable = False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default = func.now(),
        onupdate = func.now(),
        nullable = False,
    )
    chunks:Mapped[list["DocumentChunk"]] =relationship(
        back_populates = "document", cascade = "all, delete-orphan", passive_deletes = True
    )

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id : Mapped[UUID] = mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid4)
    document_id : Mapped[UUID] = mapped_column(
        UUID(as_uuid=True)
        ,ForeignKey("document.id"
        ,ondelete="CASCADE")
        ,nullable= False
        ,index=True
    )
    content : Mapped[str] = mapped_column(Text,nullable=False)
    embedding : Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIMENSIONS),nullable=False)
    page_no: Mapped[int | None] = mapped_column(Integer,nullable=True)
    section_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    chunk_index: Mapped[int]= mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(32), nullable=False,index=True)
    extra_metadata: Mapped[dict]= mapped_column("metadata", JSONB,nullable=False,default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default = func.now(), nullable = False
    )

    document:Mapped[Document] =relationship(
        back_populates = "chunks"
    )


class MessageRole(str,Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    __tablename__ = "conversation"
    id : Mapped[UUID] = mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid4)
    title : Mapped[str] = mapped_column(String(128),nullable=False,default="新对话")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default = func.now(),
        onupdate = func.now(),
        nullable = False,
    )
    messages : Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all,delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "message"
    id : Mapped[UUID] = mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid4)
    conversation_id : Mapped[UUID] = mapped_column(UUID(as_uuid=True),ForeignKey("conversation.id",ondelete="CASCADE"),nullable=False,index=True)

    role : Mapped[MessageRole] = mapped_column(String(16),nullable=False)
    content : Mapped[str] = mapped_column(Text,nullable=False)
    extra_metadata:Mapped[dict] = mapped_column("metadata",JSONB,nullable=False,default=dict)
    conversation : Mapped[Conversation] =  relationship(
        back_populates="messages"
    )
    citations:Mapped[list["AnswerCitation"]] = relationship(
        back_populates="message",
        cascade="all,delete-orphan",
        passive_deletes=True,
        order_by="AnswerCitation.ordinal"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AnswerCitation(Base):
    __tablename__ = "answer_citation"
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    message_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True),
                                                  ForeignKey("message.id", ondelete="CASCADE"), nullable=False,
                                                  index=True)
    ordinal: Mapped[int] = mapped_column(Integer,nullable=False)
    document_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True),
                                             ForeignKey("document.id", ondelete="SET NULL"), nullable=True,
                                             index=True)
    chunk_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True),
                                              ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True,
                                              index=True)
    document_name : Mapped[str] = mapped_column(String(512),nullable=False)
    page_no : Mapped[int | None] = mapped_column(Integer,nullable=True)
    quote : Mapped[str] = mapped_column(Text,nullable=False)
    message : Mapped[Message] = relationship(back_populates="citations")


