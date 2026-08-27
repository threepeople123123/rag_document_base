import asyncio
from uuid import UUID

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.retrieval import keyword_retrieval
from app.retrieval.vector_retrieval import RetrievalChunk, VectorRetrieval


logger = get_logger(__name__)

class HybridRetrieval:

    async def search(self,query:str,recall_top_k:int,final_top_k: int)->list[RetrievalChunk]:
        vector ,keyword = await asyncio.gather(
            self._safe_search(VectorRetrieval,recall_top_k, query),
            self._safe_search(keyword_retrieval.KeywordRetrieval, recall_top_k, query),
        )
        return rrf(vector,keyword,final_top_k)


    async def _safe_search(self, retriever_cls: type[keyword_retrieval.KeywordRetrieval] | type[VectorRetrieval], recall_top_k:int,query:str)->list[RetrievalChunk]:

        try:
            async with AsyncSessionLocal() as session:
                retriever = retriever_cls(session)
                return await retriever.search(
                    query=query,top_k=recall_top_k
                )
        except Exception as e:
            logger.error(f"查询失败，原因：{e}")
            return []


def rrf(vector: list[RetrievalChunk], keyword: list[RetrievalChunk], final_top_k: int) -> list[RetrievalChunk]:
    """RRF 融合：结果只保留向量路命中的 chunk，关键词路只是给它们加分。

    语义：向量检索没查到的 chunk，即使关键词查到了也不进入结果。
    rrf_score = 1/(K + 向量名次) + (若关键词也命中) 1/(K + 关键词名次)
    """
    K = settings.RRF_K
    min_score = float(settings.RETRIEVAL_MIN_SCORE or 0)  # 向量路相似度阈值

    # 关键词路名次表：chunk_id -> rank（只对向量命中的 chunk 生效）
    keyword_rank_map: dict[UUID, int] = {
        chunk.chunk_id: rank
        for rank, chunk in enumerate(keyword, start=1)
    }

    # 向量路：过滤阈值后逐个算 RRF 分
    hits: list[RetrievalChunk] = []
    for rank, chunk in enumerate(vector, start=1):
        if chunk.score <= min_score:
            continue
        chunk.vector_rank = rank
        k_rank = keyword_rank_map.get(chunk.chunk_id)
        if k_rank is not None:
            chunk.keyword_rank = k_rank
        chunk.rrf_score = 1 / (K + rank) + (1 / (K + k_rank) if k_rank is not None else 0.0)
        hits.append(chunk)

    # 按 RRF 分数降序排序，取前 final_top_k 条
    return sorted(hits, key=lambda c: c.rrf_score or 0.0, reverse=True)[:final_top_k]
