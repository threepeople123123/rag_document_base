from dataclasses import dataclass
from typing import TypedDict, Literal
from uuid import UUID

from app.db.models import Message
from app.retrieval.vector_retrieval import RetrievalChunk


query_router = Literal["original","rewrite","hyde","multi_query"]

@dataclass
class AgentSteps:
    round:int
    original:str
    hyde_answer:str
    multi_queries: list[str]
    rewritten_query: str
    route:query_router

class RAGState(TypedDict, total=False):
    # 会话id
    conversation_id:UUID

    # 提问
    question:str

    # 路由
    route:query_router

    # 重写
    rewritten_query:str

    # ai 虚拟回答
    hyde_answer:str

    # 多角度重写提问
    multi_queries:list[str]

    # 历史消息
    chat_history:list[Message]

    # 提问重新赋值
    query:str

    # 检索到的分片
    retrieved_chunks:list[RetrievalChunk]

    # 是否拒绝回答
    refused:bool

    # ai 最后回答
    answer:str

    # 用户消息id
    user_message_id:UUID

    # ai 回复消息id
    assistant_message_id:UUID

    # 每轮的执行计划
    agent_steps:list[AgentSteps]

