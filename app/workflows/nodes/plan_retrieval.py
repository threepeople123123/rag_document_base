import json
from dataclasses import asdict
from typing import Literal

from app.core.config import settings
from app.llm.models import get_chat_model
from app.llm.prompts import build_agent_plan_messages
from app.llm.query_rewirter import get_query_rewriter
from app.workflows.rag_state import RAGState, AgentSteps
from pydantic import BaseModel, Field


class RetrievalDecision(BaseModel):
    action: Literal["proceed", "rewrite_query", "switch_route", "refuse"] = Field(..., description="""
    可选 Action：
    1,proceed: 选择 proceed 后，不需要继续检索;
    
    2,rewrite_query: 当前查询本身存在问题，需要重新组织查询表达后再次检索。
    
    3,switch_route: 
                * original：使用原始用户问题进行检索。
                * rewrite：使用查询改写后的问题进行检索。
                * hyde：生成假想答案后使用假想答案进行向量检索。
                * multi_query：从多个不同角度生成子查询并进行多路召回。
                
    4, refuse: 经过合理的多轮检索后，仍然没有找到与用户问题相关的知识内容，应判断知识库可能不覆盖该问题。
    """)
    reason: str = Field(..., description="说明，为什么这样做")
    new_query: str | None = Field(..., description="重新写的query")
    new_route: Literal["original", "rewrite", "hyde", "multi_query"] | None = Field(..., description="")

async def plan_retrieval(state: RAGState) -> RAGState:
    query = state["query"]

    # ============================================================
    # [修正] 原代码直接 state["agent_steps"] 下标取值：
    # RAGState 是 total=False 的 TypedDict，normalize_query 只写了 query，
    # 首次进入时这些 key 还不存在，会直接 KeyError。
    # 改用 .get() 兜底，行为不变、更安全。
    # ============================================================
    # agent_steps =  state["agent_steps"]
    # multi_queries =  state["multi_queries"]
    # route =  state["route"]
    # rewritten_query =  state["rewritten_query"]
    # hyde_answer =  state["hyde_answer"]
    agent_steps = list(state.get("agent_steps") or [])
    multi_queries = state.get("multi_queries")
    route = state.get("route") or "original"
    rewritten_query = state.get("rewritten_query")
    hyde_answer = state.get("hyde_answer")

    # ============================================================
    # [修正] 原代码在 if not agent_steps 分支里 append 后直接 return，
    # 导致第一轮只记录、不调 LLM、永远出不了决策（下游拿不到更新后的
    # query/route）。已把原来两段重复的 append 合并成下面这一次，
    # 首轮也会正常记录并继续规划。
    # ============================================================
    # if not agent_steps:
    #     agent_steps.append(
    #         AgentSteps(
    #             round=len(agent_steps)+1,
    #             original=query,
    #             rewritten_query=rewritten_query,
    #             hyde_answer=hyde_answer,
    #             route=route,
    #             multi_queries=multi_queries
    #         )
    #     )
    #     return {"agent_steps": agent_steps}
    #
    # agent_steps.append(
    #     AgentSteps(
    #         round=len(agent_steps) + 1,
    #         original=query,
    #         rewritten_query=rewritten_query,
    #         hyde_answer=hyde_answer,
    #         route=route,
    #         multi_queries=multi_queries
    #     )
    # )
    agent_steps.append(
        AgentSteps(
            round=len(agent_steps) + 1,
            original=query,
            rewritten_query=rewritten_query,
            hyde_answer=hyde_answer,
            route=route,
            multi_queries=multi_queries,
        )
    )

    history = json.dumps(
        [asdict(s) for s in agent_steps],  # ① dataclass → dict
        ensure_ascii=False,  # ② 中文不转成 \uXXXX，保持可读
        default=str,  # ③ 兜底：万一有 UUID/datetime 也能序列化
    )

    messages = build_agent_plan_messages(question=query,current_route=route,current_query=rewritten_query,history=history)

    structured_model = get_chat_model().with_structured_output(RetrievalDecision)

    # [修正] 原代码是同步 invoke，在 async 函数里会阻塞事件循环，改用 ainvoke
    # retrieval_decision =  structured_model.invoke(messages)
    retrieval_decision = await structured_model.ainvoke(messages)

    # ============================================================
    # [修正] 原代码 if isinstance(rewritten_query, RetrievalDecision)：
    # rewritten_query 是字符串不是决策对象，判断对象写错了；决策变量是
    # retrieval_decision。已注释，下面按 retrieval_decision.action 分派。
    # ============================================================
    # if isinstance(rewritten_query,RetrievalDecision):
    #
    #
    #
    #
    #
    #

    # ================= 应用决策，替换 state 供下游节点使用 =================
    update: RAGState = {}
    new_route = route
    new_query = query

    if retrieval_decision.action == "proceed":
        # 不调整：保持当前 query/route，什么都不覆盖
        pass
    elif retrieval_decision.action == "rewrite_query" and retrieval_decision.new_query:
        # 改 query，同时把 route 强制重置为 original：
        # 上一轮可能是 multi_query，残留的 multi_queries 会让下游 retrieve
        # 忽略本轮新 query，必须清干净
        new_query = retrieval_decision.new_query
        new_route = "original"
        update["query"] = new_query
        update["route"] = new_route
        update["rewritten_query"] = None
        update["hyde_answer"] = None
        update["multi_queries"] = None
    elif retrieval_decision.action == "switch_route" and retrieval_decision.new_route:
        # 真正切换路由：调 QueryRewriter 补齐目标路由对应字段，
        # 避免只换标签不换行为（rewrite/hyde/multi_query 各自需要重新生成内容）
        rewriter = get_query_rewriter()
        # 注意：apply_route 内部识别的是 "rewritten"（多个 t），不是决策里的 "rewrite"
        target_route = "rewritten" if retrieval_decision.new_route == "rewrite" else retrieval_decision.new_route
        result = await rewriter.apply_route(
            route=target_route,
            question=query,
            message=state.get("chat_history") or [],  # apply_route 的 message 是必传参数
            multi_query_count=settings.multi_query_count,
        )
        new_route = result.route
        new_query = result.query
        update["route"] = new_route
        update["query"] = new_query
        update["rewritten_query"] = result.rewritten_query
        update["hyde_answer"] = result.hyde_answer
        update["multi_queries"] = result.multi_queries
    elif retrieval_decision.action == "refuse":
        # 拒答：标记 refused，由下游 refuse 节点统一塞拒答文案
        update["refused"] = True

    # 本轮 agent_steps（含刚记录的一轮）写回 state
    update["agent_steps"] = agent_steps

    return update
