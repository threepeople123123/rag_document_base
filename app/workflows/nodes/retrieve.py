from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.retrieval.hybrid_retrieval import HybridRetrieval
from app.retrieval.vector_retrieval import RetrievalChunk
from app.workflows.rag_state import RAGState


async def retrieve(state:RAGState,session:AsyncSession) -> RAGState:
    multi_queries = state["multi_queries"]
    query = state["query"]
    retrieval_top_k = settings.RETRIEVAL_TOP_K
    final_top_k = settings.FINAL_TOP_K

    update= {"query":query,"multi_queries":multi_queries,}
    retrieval:list[RetrievalChunk] | None = None
    if multi_queries:
        #循环检索
        for multi_query in multi_queries:
            hybrid_retrieval = HybridRetrieval()
            retrieval:list[RetrievalChunk] = await hybrid_retrieval.search(query,retrieval_top_k,final_top_k)

    return RAGState(retrieved_chunks=retrieval)
