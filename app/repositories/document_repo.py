from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentStatus


class DocumentRepository:
    def __init__(self,session:AsyncSession) -> None:
        self.session = session

    # id查询
    async def get_by_id(self,document_id:UUID)->Document | None:
        return await self.session.get(Document,document_id)

    # 条件查询
    async def get_by_hash(self,file_hash:str) ->Document | None:
        stmt = select(Document).where(Document.file_hash == file_hash)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    # 添加一条document
    async def add(self,document:Document)->Document:
        self.session.add(document)
        await self.session.flush()
        return document

    async def delete(self,document:Document):
        await self.session.delete(document)

    #
    async def update_status(self,document_id:UUID
                            ,status:DocumentStatus
                            ,*
                            ,error_message:str|None = None
                            )->None:
        document = await self.get_by_id(document_id)
        if document is None:
            return
        document.status = status
        if error_message is not None or status != DocumentStatus.FAILED:
            document.error_message = error_message

    # 查询分片后的文档
    async def list_paginated(
            self
            ,page:int
            ,page_size:int
            ,status:DocumentStatus | None
    )->tuple[list[Document] ,int]:

        offset = (page-1)*page_size

        # 查询文档分片，
        items_stmt = (
            select(Document)
            .where(Document.status == status)
            .order_by(Document.created_at.asc())
            .offset(offset)
            .limit(page_size)
        )

        # 查询文档分片数量
        count_stmt = (
            select(func.count())
            .select_from(Document)
        )

        items = (await self.session.execute(items_stmt)).scalars().all()
        total = (await self.session.execute(count_stmt)).scalar_one()

        return list(items) ,int(total)


