from langgraph.graph import StateGraph, START, END

from app.workflows.nodes.rerank import rerank_node
from app.workflows.nodes.normalize_query import normalize_query
from app.workflows.nodes.plan_retrieval import plan_retrieval
from app.workflows.nodes.refuse import refuse
from app.workflows.nodes.retrieve import retrieve
from app.workflows.nodes.route_query import route_query
from app.workflows.rag_state import RAGState


def _after_plan(state:RAGState)->str:
    # RAGState 是 total=False 的 TypedDict，refused 键只在 plan_retrieval
    # 走 refuse 分支时才会写入；其他分支没有该键，直接下标会 KeyError。
    refused = state.get("refused", False)
    if refused:
        return "refuse"
    return "retrieve"

def _after_retrieve(state:RAGState)->str:
    retrieved_chunks = state.get("retrieved_chunks") or []
    if len(retrieved_chunks) <= 0:
        return "refuse"
    return "retrieve"
def _build_graph():

    builder = StateGraph(RAGState)
    builder.add_node("normalize_query", normalize_query)
    builder.add_node("route_query", route_query)
    builder.add_node("plan_retrieval", plan_retrieval)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank_node)
    builder.add_node("refuse", refuse)

    builder.add_edge(START,"normalize_query")
    builder.add_edge("normalize_query","route_query")
    builder.add_edge("route_query","plan_retrieval")
    builder.add_conditional_edges(
        "plan_retrieval",
        _after_plan,
        {"retrieve": "retrieve", "refuse": "refuse"},
    )
    builder.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {"rerank": "rerank", "refuse": "refuse"},)
    builder.add_edge("rerank",END)
    builder.add_edge("refuse",END)

    return builder.compile()

_rag_graph = _build_graph()

def get_rag_graph():
    """对外暴露已编译好的子图；模块加载时一次编译，请求里直接复用。"""
    return _rag_graph
