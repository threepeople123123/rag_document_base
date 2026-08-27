from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.chunk_repo import DocumentChunkRepository
from app.retrieval.vector_retrieval import RetrievalChunk


class KeywordRetrieval:
    def __init__(self,session:AsyncSession)->None:
        self.session = session

    async def search(self,query:str,top_k:int)->list[RetrievalChunk]:
        chunk_repo = DocumentChunkRepository(self.session)

        rows = await chunk_repo.keyword_search(query,top_k)

        return [
            RetrievalChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id
                ,document_name=chunk.document.name
                ,content=chunk.content
                ,page_no=chunk.page_no
                ,section_path=chunk.section_path
                ,score=ts_rank
                ,keyword_rank=rank
                ,sources=("keyword",)
                ,keyword_score=ts_rank
            )
            for rank,(chunk,ts_rank) in enumerate(rows ,start=1)
        ]