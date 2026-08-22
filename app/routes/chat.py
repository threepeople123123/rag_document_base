from typing import AsyncIterable
from uuid import UUID

from fastapi import APIRouter
from fastapi.sse import ServerSentEvent

from app.api.deps import DbSession
from app.api.schemas.chat import ConversationRead, ConversationDetail, MessageRead, ChatRequest
from app.service import chat_service

router = APIRouter(prefix="/conversation",tags=["chat"])

@router.post("",response_model=ConversationRead,status_code=201,operation_id="createConversation")
async def create_conversation(conversation: ConversationRead,session:DbSession) ->ConversationRead:
    service = chat_service.ChatService(session)
    await service.create_conversation(conversation.title)
    return ConversationRead.model_validate(conversation)

@router.get("/{conversation_id}",response_model=ConversationDetail,operation_id="getConversation")
async def get_conversation(conversation_id:UUID,session:DbSession) ->ConversationDetail:
    service = chat_service.ChatService(session)
    conversation ,messages = await service.list_message(conversation_id)
    return ConversationDetail.model_validate(
        conversation=ConversationRead.model_validate(conversation),
        message=[MessageRead.from_orm(m) for m in messages]
    )

@router.post("/{conversation_id}/chat",operation_id="streamChat",response_model=ServerSentEvent)
async def stream_chat(conversation_id:UUID,payload:ChatRequest,session:DbSession) ->AsyncIterable[ServerSentEvent]:
    service = chat_service.ChatService(session)
    async for sse_event in service.stream_answer(conversation_id,payload.question):
        yield ServerSentEvent(
            data=sse_event["data"],
            event=sse_event["event"],
        )


