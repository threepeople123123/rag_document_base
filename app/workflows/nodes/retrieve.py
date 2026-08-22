from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.llm.prompts import REFUSAL_ANSWER
from app.retrieval.vector_retrieval import VectorRetrieval
from app.workflows.rag_state import RAGState


async def retrieve(state:RAGState,session:AsyncSession) -> RAGState:
    retrieval = VectorRetrieval(session)

    chunks = await retrieval.search(state["query"],top_k=settings.CHAT_HISTORY_WINDOW)

    refused = not chunks or chunks[0].score < settings.RETRIEVAL_MIN_SCORE

    update:RAGState = {
        "retrieved_chunks":chunks,
        "refused":refused
    }
    if refused:
        update["answer"] = REFUSAL_ANSWER

    return update