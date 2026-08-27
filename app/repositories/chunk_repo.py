from dataclasses import dataclass
from typing import Sequence
from uuid import UUID

from sqlalchemy import cast, delete, func, select
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.db.models import DocumentChunk, Document, DocumentStatus


@dataclass(frozen=True)
class ChunkStatus:

    """

    """
    total :int
    avg_length:int
    min_length:int
    max_length:int


class DocumentChunkRepository:
    def __init__(self,session:AsyncSession):
        self.session = session

    # 增加chunk
    async def bulk_add(self,document_chunk:Sequence[DocumentChunk])->None:
        if not document_chunk:
            return
        # 没有真正添加，需要调用flush方法
        self.session.add_all(document_chunk)
        await self.session.flush()

    # 根据document_id 删除
    async def delete_by_document(self,document_id:UUID)->None:
        stmt = delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        await self.session.execute(stmt)

    # 根据document_id 查询
    async def list_paginated_by_document(
            self
            ,document_id:UUID
            ,page:int
            ,page_size:int
    )->tuple[list[DocumentChunk],int] | None:
        if page<=0:
            return None
        offset = (page -1 )* page_size
        items_stmt = (select(DocumentChunk)
                .where(DocumentChunk.document == document_id)
                .order_by(DocumentChunk.created_at.asc())
                .offset(offset)
                .limit(page_size))

        items_count = (select(func.count())
                      .where(DocumentChunk.document == document_id)
                      )

        items = (await self.session.execute(items_stmt)).scalars().all()
        count = (await self.session.execute(items_count)).scalar_one()
        return list(items) ,int(count)

    async def get_for_document(self,document_id:UUID,document_chunk_id:UUID)->DocumentChunk | None:
        """
        根据文档id和分片id精确查询
        :param document_id:  文档id
        :param document_chunk_id:  分片id
        :return: none ｜ 分片文档
        """
        if (not document_id) or (not document_chunk_id):
            return None
        # 查询条件
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id==document_id
            ,DocumentChunk.id == document_chunk_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_stats(self,document_id:UUID)->ChunkStatus | None:
        """
        查询
        :param document_id:
        :return:
        """
        length = func.char_length(DocumentChunk.content)
        stmt = select(
            func.count().label("total")
            ,func.avg(length).label("avg_len")
            ,func.min(length).label("min_len")
            ,func.max().label("max_len")
        ).where(DocumentChunk.document_id == document_id)
        row = (await self.session.execute(stmt)).one()
        if not row.total:
            return None
        return ChunkStatus(
            total=int(row.total or 0)
            ,avg_length=int(row.avg_len or 0)
            ,min_length=int(row.min_len or 0)
            ,max_length=int(row.max_len or 0)
        )


    async def vector_search(self,query_embedding:list[float],top_k:int) -> list[tuple[DocumentChunk,float]]:
        """
        查询向量
        :param query_embedding: 向量
        :param top_k:  取多少条
        :return: 返回  top_k  数据
        """
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)

        stmt = (
            select(DocumentChunk,distance.label("distance"))
            .join(Document,Document.id == DocumentChunk.document_id)
            .where(Document.status == DocumentStatus.READY.value)
            .order_by(distance.asc())
            .limit(top_k)
            .options(selectinload(DocumentChunk.document))
        )

        result = (await self.session.execute(stmt)).all()
        return [(chunk,float(dist)) for chunk ,dist in result]

    async def keyword_search(self, query: str, top_k: int) -> list[tuple[DocumentChunk, float]]:
        """
        关键字全文检索（zhparser 中文分词），按 ts_rank 相关性分数从高到低返回。

        :param query: 用户关键字，按 chinese_zhparser 配置自动分词，多词默认 AND 语义
        :param top_k: 返回条数
        :return: [(DocumentChunk, score), ...]，score 越大越相关
        """
        if not query or not query.strip():
            return []

        # 显式 cast 成 regconfig，避免驱动对参数类型的推断问题
        config = cast("chinese_zhparser", REGCONFIG)
        tsv = func.to_tsvector(config, DocumentChunk.content)
        tsq = func.plainto_tsquery(config, query)
        score = func.ts_rank(tsv, tsq).label("score")

        stmt = (
            select(DocumentChunk, score)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == DocumentStatus.READY.value)
            .where(tsv.op("@@")(tsq))
            .order_by(score.desc())
            .limit(top_k)
            .options(selectinload(DocumentChunk.document))
        )

        result = (await self.session.execute(stmt)).all()
        return [(chunk, float(s)) for chunk, s in result]
