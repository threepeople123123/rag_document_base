from app.core.config import settings
from app.routes.reranker import get_reranker
from app.workflows.rag_state import RAGState


async def rerank_node(state:RAGState)->RAGState:
    # 判断是否需要召回
    chunks = state["retrieved_chunks"]
    query = state["query"]
    if len(chunks) < settings.RERANK_TOP_K:
        return {}
    rerank_document = await get_reranker().reranker(query=state.get("rewritten_query",default=query) , document= chunks, top_k= settings.RERANK_TOP_K)
    return {"retrieved_chunks":rerank_document}