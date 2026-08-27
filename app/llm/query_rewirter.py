from dataclasses import dataclass
from app.core.logging import get_logger
from app.db.models import Message, MessageRole
from app.llm.models import get_chat_model
from app.llm.prompts import route_message, rewrite_message, build_hyde_messages, build_multi_query_messages, \
    build_contextualize_messages
from app.workflows.rag_state import query_router


@dataclass(frozen=True)
class QueryRouteResult:
    query:str
    route:query_router
    rewritten_query: str | None = None
    hyde_answer: str | None = None
    multi_queries: list[str] | None = None

logger = get_logger(__name__)

class QueryRewriter:

    async def router(self,question:str)->str:
        router = await get_chat_model(True).ainvoke(route_message(question=question))
        raw = _extract_text(router.content).strip().lower()
        token = raw.strip("\"'`.。 ")
        for token in query_router:
            return token
        return "original"

    async def optimize(self,query:str,message:list[Message], multi_query_count: int) -> QueryRouteResult:
        # 第一步先路由，路由完成之后在改写
        try:
            router = await self.router(query)
        except Exception as e:
            logger.warning(f"路由失败:{e}，使用用户真实提问:{query}")
            return QueryRouteResult(query=query,route="original")
        return await self.apply_route(router,query,message,multi_query_count)


    async def apply_route(self,route:str,question:str,message:list[Message],multi_query_count: int)->QueryRouteResult:
        try:
            if route == "rewritten":
                rewrite_result = await self.rewrite(query=question,message=message)
                # rewrite 返回空也降级，避免后续用空字符串去 embedding
                if not rewrite_result:
                    return QueryRouteResult(route="original", query=question)
                return QueryRouteResult(
                    route="rewrite", query=rewrite_result, rewritten_query=rewrite_result
                )
            elif route == "hyde":
                hyde_answer = await self.hyde(question,message)
                if not hyde_answer:
                    return QueryRouteResult(route="original", query=question)
                # HyDE 用假设答案做检索；hyde_answer 字段额外保留同一份文本，供前端调试面板展示和落库审计
                return QueryRouteResult(
                    route="hyde", query=hyde_answer, hyde_answer=hyde_answer
                )
            elif route == "multi_query":
                multi_query_result = await self.multi_query(query=question,n=multi_query_count,message=message)
                if len(multi_query_result) < 2:
                    return QueryRouteResult(route="original", query=question)
                return QueryRouteResult(
                    route="multi_query", query=question, multi_queries=multi_query_result
                )
            return QueryRouteResult(route="original", query=question)
        except Exception:
            logger.exception(
                "apply_route 失败，降级到 original：route=%s question=%r",
                route,
                question,
            )
            return QueryRouteResult(route="original", query=question)

    async def rewrite(self,query:str,message:list[Message])->str:
        prompt = rewrite_message(query,message)
        response = await get_chat_model(True).ainvoke(prompt)
        return _extract_text(response.content).strip()

    async def hyde(self,query:str,message:list[Message])->str:
        prompt = build_hyde_messages(query,message)
        response = await get_chat_model(True).ainvoke(prompt)
        return _extract_text(response.content).strip()

    async def multi_query(self,query:str,n:int,message:list[Message])->list[str]:
        messages =  build_multi_query_messages(query,message)
        response = await get_chat_model().ainvoke(messages)
        text = _extract_text(response.content)
        queries = [line.strip(" -•*0123456789.、") for line in text.splitlines()]
        return [q for q in queries if q][:n]

    async def contextualize(self, question: str, history: list[Message]) -> str:
        """基于多轮历史把当前问题改写成独立完整问句。

        消解"它/这个/上面提到的"等指代、补省略，让后续 route_query / retrieve
        看到的 query 已经独立可检索。空历史直接回原问题；任何异常 / 改写为空 → 降级回原问题。
        """
        history_text = _format_history_text(history)
        if not history_text:
            return question
        try:
            messages = build_contextualize_messages(
                question=question, history=history_text
            )
            response = await get_chat_model().ainvoke(messages)
            rewritten = _extract_text(response.content).strip()
            return rewritten or question
        except Exception:
            logger.exception(
                "contextualize 调用失败，降级回原问题：question=%r", question
            )
            return question


_rewriter: QueryRewriter | None = None


def get_query_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter

def _extract_text(content: str | list[str | dict]) -> str:
    """兼容 langchain ChatModel 的 content 联合类型（参考 generate.py 同款处理）。"""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


_ROLE_LABEL: dict[MessageRole, str] = {
    MessageRole.USER: "用户",
    MessageRole.ASSISTANT: "助手",
    MessageRole.SYSTEM: "系统",
}

def _format_history_text(history: list[Message]) -> str:
    """把历史 Message 压成给 contextualize prompt 看的纯文本。

    只取 user / assistant，过滤 system；空内容跳过，避免把空消息塞进 prompt 浪费 token。
    """
    lines: list[str] = []
    for msg in history:
        role_label = _ROLE_LABEL.get(msg.role)
        if not role_label or not msg.content.strip():
            continue
        lines.append(f"{role_label}：{msg.content.strip()}")
    return "\n".join(lines)