import httpx

from app.core.config import settings
from app.retrieval.vector_retrieval import RetrievalChunk


class Reranker:

    def __init__(self)->None:
        self._http_client: httpx.AsyncClient | None = None

    def get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
        return self._http_client

    async def reranker(self,query:str,document:list[RetrievalChunk],top_k:int)->list[RetrievalChunk]:
        client =self.get_client()
        response = await client.post(
            url="https://api.deepai.org/v1/",
            json={
                "model": settings.RERANK_MODEL_NAME,
                "query": query,
                "documents": [c.content for c in document],
                "top_n": top_k
            }
        )
        response.raise_for_status()
        results = response.json()["results"]   # 已按 relevance_score 降序

        # 按 index 映射回原始分片，并写回分数
        ranked: list[RetrievalChunk] = []
        for r in results:
            chunk = document[r["index"]]
            chunk.rerank_score = r["relevance_score"]
            ranked.append(chunk)
        return ranked


_reranker: Reranker | None =None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker