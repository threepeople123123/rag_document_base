from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Message, Conversation, MessageRole


class ConversationRepository:

    def __init__(self,session:AsyncSession):
        self.session = session

    async def crate(self,title:str = "新对话")->Conversation:
        conversation = Conversation(title=title)
        self.session.add(conversation)
        await self.session.refresh(conversation)
        return conversation

    async def get(self,conversation_id:UUID)->Conversation | None:
        return await self.session.get(Conversation, conversation_id)

    async def list_message(self,conversation_id:UUID)->list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id==conversation_id)
            .order_by(Message.created_at.asc(),Message.id.asc())
            .options(selectinload(Message.citations))
        )

        return list((await self.session.execute(stmt)).scalars().all())


    async def recent_messages(self,conversation_id:UUID,limit:int)->list[Message]:
        if limit <= 0:
            return []
        stmt = (
            select(Message)
            .where(Message.conversation_id==conversation_id)
            .order_by(Message.created_at.desc(),Message.id.desc())
            .limit(limit)
        )

        rows = list((await self.session.execute(stmt)).scalars().all())
        return list(reversed(rows))

    async def add_message(self,message:Sequence[Message])-> None:
        if not message:
            return None
        self.session.add_all(message)
        return await self.session.refresh(message)

    @staticmethod
    def make_user_message(conversation_id:UUID,content:str)->Message:
        return Message(conversation_id=conversation_id,content=content,role=MessageRole.USER)

    @staticmethod
    def make_assistant_message(conversation_id:UUID
                               ,content:str,*,extra_metadata:dict | None = None)->Message:
        return Message(conversation_id=conversation_id
                       ,content=content
                       ,role=MessageRole.ASSISTANT
                       ,extra_metadata=extra_metadata or {})

