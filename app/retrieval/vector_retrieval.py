from dataclasses import dataclass
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