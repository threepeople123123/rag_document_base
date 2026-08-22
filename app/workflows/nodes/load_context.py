from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repositories.conversation_repo import ConversationRepository
from app.workflows.rag_state import RAGState


async def load_context(state:RAGState,session:AsyncSession):
    repo = ConversationRepository(session)

    history = await repo.recent_messages(
        state["conversation_id"],limit=settings.CHAT_HISTORY_WINDOW * 2
    )
    return {"chat_history":history}