from typing import AsyncIterator

from app.llm.models import get_chat_model
from app.llm.prompts import build_answer_message
from app.workflows.rag_state import RAGState


async def stream_generate(state:RAGState)->AsyncIterator[str]:
    message = build_answer_message(
        question=state["question"],
        chunks=state["retrieved_chunks"],
        history=state["chat_history"],
    )

    async for chunk in get_chat_model().astream(message):
        text = chunk.content
        if not text:
            continue
        if isinstance(text,str):
            yield text
        else:
            yield "".join(part.get("text","") for part in text if isinstance(part,dict))
