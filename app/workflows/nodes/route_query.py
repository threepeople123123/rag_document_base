
from app.core.config import settings
from app.llm.query_rewirter import get_query_rewriter
from app.workflows import rag_state


async def route_query(state: rag_state.RAGState)-> rag_state.RAGState:
    if not settings.QUERY_ROUTE_ENABLED:
        return {"route": "original"}

    chat_history = state["chat_history"]
    question = state["question"]
    multi_query_count = settings.MULTI_QUERY_COUNT

    route_result =  await get_query_rewriter().optimize(question,chat_history,multi_query_count)

    update: rag_state.RAGState = {"route": route_result.route, "query": route_result.query}
    if route_result.rewritten_query is not None:
        update["rewritten_query"] = route_result.rewritten_query
    if route_result.hyde_answer is not None:
        update["hyde_answer"] = route_result.hyde_answer
    if route_result.multi_queries is not None:
        update["multi_queries"] = route_result.multi_queries
    return update




