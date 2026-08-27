from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.embedder import get_embeddings
from app.repositories.chunk_repo import DocumentChunkRepository


@dataclass
class RetrievalChunk:
    chunk_id: UUID
    document_id:UUID
    document_name:str
    content:str
    page_no:int | None
    section_path : str | None
    score:float
    sources: tuple[str, ...] = field(default_factory=tuple)
    vector_rank: int | None = None
    vector_score: float | None = None  # 原始 cosine similarity（向量路命中时填充）
    keyword_rank: int | None = None
    keyword_score: float | None = None  # 原始 ts_rank（关键词路命中时填充）
    rrf_score: float | None = None
    # reranker query-chunk 成对打分的相关度，越大越相关
    # qwen3-rerank 输出 relevance_score ∈ [0, 1]
    rerank_score: float | None = None


class VectorRetrieval:
    def __init__(self,session:AsyncSession)->None:
        self.chunk_repo = DocumentChunkRepository(session)

    async def search(self,query:str,top_k:int)->list[RetrievalChunk]:
        embedding = await get_embeddings().aembed_query(query)
        rows  = await self.chunk_repo.vector_search(embedding,top_k)

        return [
            RetrievalChunk(
                chunk_id= chunk.id
                ,document_id=chunk.document_id
                ,document_name=chunk.document.name
                ,content=chunk.content
                ,page_no=chunk.page_no
                ,section_path=chunk.section_path
                ,score=1.0 - distance
            )
            for chunk, distance in rows
        ]