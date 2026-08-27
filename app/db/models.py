from datetime import datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy.dialects.postgresql import JSONB, UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Text, DateTime, func, BigInteger, ForeignKey, Integer, Column, Table, ARRAY, Boolean, \
    Float
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

class IngestionTaskType(str, Enum):
    """入库任务类型。

    ingest:  首次入库（解析 → 切分 → 全量 embedding → 写入）
    reindex: 增量重建（按 chunk_hash 对齐，仅对变化 chunk 重新 embedding）
    """

    INGEST = "ingest"
    REINDEX = "reindex"

class IngestionTaskStatus(str, Enum):
    """Celery 任务生命周期。

    pending: 已入库表、还没被 worker 拉走
    running: worker 已开始执行
    success / failed: 终态
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

class MessageRole(str, Enum):
    """消息角色。"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class UserStatus(str, Enum):
    """用户启用状态。"""

    ACTIVE = "active"
    DISABLED = "disabled"


# 用户 - 角色 多对多关系表。
# 不抽成 ORM 类是因为本身没有业务字段，纯关系；用 Table 让 SQLAlchemy 自动处理。
user_roles_table = Table(
    "user_roles",
    Base.metadata,
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "role_id",
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class User(Base):
    """用户主表。"""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # bcrypt hash，约 60 字符；预留 255 兼容未来切换算法
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[UserStatus] = mapped_column(
        String(16), nullable=False, default=UserStatus.ACTIVE
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    roles: Mapped[list["Role"]] = relationship(
        secondary=user_roles_table,
        back_populates="users",
        lazy="selectin",
    )


class Role(Base):
    """RBAC 角色。

    permission_tags：角色直接持有的权限标签数组；用户的有效权限 = 各角色 tags 的并集。
    特殊值 "*" 表示通配（admin）。
    """

    __tablename__ = "roles"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    permission_tags: Mapped[list[str]] = mapped_column(
        ARRAY(String()), nullable=False, default=list, server_default="{}"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    users: Mapped[list[User]] = relationship(
        secondary=user_roles_table,
        back_populates="roles",
    )

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
    ingestion_tasks: Mapped[list["IngestionTask"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="IngestionTask.created_at.desc()",
    )
    
class IngestionTask(Base):
    """文档入库任务记录。

    Celery 拉起 worker 前先在 DB 落一条 pending 行；worker 内根据生命周期更新
    running → success/failed。前端轮询 documents 接口附带 `latest_task` 即可
    展示进度（progress_total / progress_done）与失败原因。
    """

    __tablename__ = "ingestion_tasks"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    document_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_type: Mapped[IngestionTaskType] = mapped_column(String(16), nullable=False)
    status: Mapped[IngestionTaskStatus] = mapped_column(
        String(16), nullable=False, default=IngestionTaskStatus.PENDING
    )
    # Celery 当前 attempt 次数（Celery 内 retry 时 worker 写入），仅作展示用
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 进度：reindex 时 total=新增 chunks 数，done=已 embedding 的批次累计
    # ingest 走全量 embedding，total=切分后总 chunks 数
    progress_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="ingestion_tasks")



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


class EvaluationRunStatus(str, Enum):
    """评测 run 生命周期：BackgroundTasks 跑完前 RUNNING；正常结束 COMPLETED；
    主流程异常（不是单条 case 异常）置 FAILED 并写 error_message。"""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationRun(Base):
    """单次评测 run。

    评测集字段（dataset_name / dataset_size）记录这次 run 用了哪份 jsonl 多少条，
    便于历史 run 之间对比"扩集 → 指标变化"。
    """

    __tablename__ = "evaluation_runs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_size: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[EvaluationRunStatus] = mapped_column(String(16), nullable=False)

    progress_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 聚合指标：跑完后回填；中途为 None
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    refusal_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 首 token 延迟，拒答 case 不计入；rerank / 检索链路慢时这里会先涨
    avg_first_token_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    items: Mapped[list["EvaluationItem"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class EvaluationItem(Base):
    """单条 case 的输入快照 + 实际输出 + 指标 + Bad Case 归因。

    输入字段（question / expected_*）从 jsonl 复制过来，不再外键回评测集文件，
    这样评测集 jsonl 后续迭代不会污染历史 run 的对比基线。
    """

    __tablename__ = "evaluation_items"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 输入快照
    case_id: Mapped[str] = mapped_column(String(64), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    expected_document_names: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    expected_keywords: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    should_refuse: Mapped[bool] = mapped_column(Boolean, nullable=False)
    tags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # 实际输出快照
    actual_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    actual_refused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    citations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    retrieved_chunks_meta: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    query_route: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    agent_steps: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    verify_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 拒答 / 报错时为 None：没有真正生成 token，首 token 延迟无意义
    first_token_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 指标：RAGAS 4 项任一异常会落 None，前端按缺失隐藏对应单元格
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    # citation_hit：should_refuse=True 时 NULL（拒答 case 不参与命中率分母）
    citation_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    refusal_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Bad Case 归因：规则自动初判 + 前端 PATCH 覆盖
    is_bad_case: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    bad_case_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bad_case_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped[EvaluationRun] = relationship(back_populates="items")
