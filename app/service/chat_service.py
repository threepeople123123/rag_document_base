from typing import AsyncIterator
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.db.models import Conversation, Message, AnswerCitation
from app.db.session import AsyncSessionLocal
from app.repositories.citation_repo import AnswerCitationRepository
from app.repositories.conversation_repo import ConversationRepository
from app.retrieval.vector_retrieval import RetrievalChunk
from app.workflows.graph import get_rag_graph
from app.workflows.nodes import normalize_query
from app.workflows.nodes.load_context import load_context
from app.workflows.nodes.retrieve import retrieve
from app.workflows.nodes.stream_generate import stream_generate
from app.workflows.rag_state import RAGState

logger = get_logger(__name__)

def _serialize_citation(chunk:RetrievalChunk,ordinal:int) -> dict:
    return {
        "ordinal":ordinal,
        "chunk_id":str(chunk.chunk_id),
        "document_id":str(chunk.document_id),
        "document_name":chunk.document_name,
        "page_no":chunk.page_no,
        "section_path":chunk.section_path,
        "score":round(chunk.score,4),
        "quote":chunk.content,
    }

class ChatService:
    def __init__(self,session:AsyncSession):
        self.session = session

    async def create_conversation(self,title:str="新对话")->Conversation:
        repo = ConversationRepository(self.session)

        conversation = await repo.crate(title)
        await self.session.commit()
        await self.session.refresh(conversation)
        return conversation

    async def get_conversation(self,conversation_id:UUID)->Conversation :
        repo = ConversationRepository(self.session)
        conversation = await repo.get(conversation_id)
        if conversation is None:
            raise NotFoundError("conversation not found")
        return conversation

    async def list_message(self,conversation_id:UUID)->tuple[Conversation,list[Message]]:
        repo = ConversationRepository(self.session)
        conversation = await self.get_conversation(conversation_id)
        messages  = await repo.list_message(conversation_id)
        return conversation,messages

    async def stream_answer(self,conversation_id:UUID,question:str)->AsyncIterator[str]:
        await self.get_conversation(conversation_id)

        async with AsyncSessionLocal() as session:
            try:
                state:RAGState = {
                    "question":question,
                    "conversation_id":conversation_id
                }

                state.update(await load_context(state,session))

                # user消息落库
                await self._persist_user_message(state,session)

                final_state = await get_rag_graph().ainvoke(state)
                state.update(final_state)

                if state.get("refused"):
                    yield {
                        "event":"token",
                        "data": {
                            "delta" : state["answer"]
                        }
                    }
                else:
                    answer_parts: list[str] = []
                    async for delta in stream_generate(state):
                        answer_parts.append(delta)
                        print(f"ai返回消息：{delta}")
                        yield {
                            "event": "token",
                            "data": {
                                "delta": delta
                            }
                        }
                    state["answer"] = "".join(answer_parts)

                # assistant 消息 + citations 同事务落地
                await self._persist_assistant_message(state,session)

                yield {
                    "event":"message_end",
                    "data":{
                        "message_id": str(state["assistant_message_id"]),
                        "refused": str(state["refused"]),
                    }
                }
            except Exception as e:
                logger.error(f"失败：{e}")
                await session.rollback()
                yield {
                    "event":"error",
                    "data":{
                        "code":"chat_stream_error",
                        "message":str(e).strip() or "ai检索回答失败，请稍后重试"
                    }
                }
    async def _persist_user_message(self,state:RAGState,session:AsyncSession)->None:
        conv_repo = ConversationRepository(session)
        message = ConversationRepository.make_user_message(state["conversation_id"],state["question"])
        await conv_repo.add_message([message])
        await session.commit()
        state["user_message_id"] = message.id





    async def _persist_assistant_message(self,state:RAGState,session:AsyncSession)->None:
        conv_repo = ConversationRepository(session)
        answer_repo =  AnswerCitationRepository(session)

        assistant_msg = ConversationRepository.make_assistant_message(state["conversation_id"],state["answer"])
        await conv_repo.add_message([assistant_msg])
        await session.commit()

        if not state["refused"]:
            citations = [
                AnswerCitation(
                    message_id=assistant_msg.id,
                    ordinal=ordinal,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    chunk_id=chunk.chunk_id,
                    page_no=chunk.page_no
                    ,quote=chunk.content,
                )
                for ordinal , chunk in enumerate(state["retrieved_chunks"])
            ]
            await answer_repo.bulk_add(citations)
        await session.commit()
        state["assistant_message_id"] = assistant_msg.id





