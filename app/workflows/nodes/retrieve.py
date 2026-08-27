from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.retrieval.hybrid_retrieval import HybridRetrieval
from app.retrieval.vector_retrieval import RetrievalChunk
from app.workflows.rag_state import RAGState


async def retrieve(state:RAGState) -> RAGState:
    multi_queries = state["multi_queries"]
    route:str = state["route"]
    query:str = state["query"]
    hyde_answer = state["hyde_answer"]
    retrieval_top_k = settings.RETRIEVAL_TOP_K
    final_top_k = settings.FINAL_TOP_K
    hybrid_retrieval = HybridRetrieval()

    retrieval:list[list[RetrievalChunk]] = []
    chunks:list[RetrievalChunk] = []
    if route == "multi_query" and multi_queries:
        #循环检索
        for multi_query in multi_queries:
            retrieval.append(await hybrid_retrieval.search(multi_query,retrieval_top_k,final_top_k))
        chunks = _merge_chunks(retrieval,retrieval_top_k)
    elif route== "original" or route== "rewrite" :
        chunks = await hybrid_retrieval.search(query,retrieval_top_k,final_top_k)
    elif route == "hyde" and hyde_answer:
        chunks = await hybrid_retrieval.search(hyde_answer, retrieval_top_k, final_top_k)

    return {"retrieved_chunks": chunks}

def _merge_chunks(bundles:list[list[RetrievalChunk]],recall_top_k:int)->list[RetrievalChunk]:
    chunk_map :dict[str,RetrievalChunk] = {}
    for bundle in bundles:
        for b in bundle:
            key = str(b.chunk_id)
            prev = chunk_map[key]
            if prev is None or ((b.rrf_score or 0.0) > (prev.rrf_score or 0.0)):
                chunk_map[key] = b
    ranked = sorted(chunk_map.values(),key=lambda c: c.rrf_score,reverse=True)
    return ranked[:recall_top_k]

