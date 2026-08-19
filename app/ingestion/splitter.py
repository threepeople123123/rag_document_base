import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


def _build_splitter()->RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunked_overlay=settings.CHUNK_OVERLAP,
        separators=["\n\n","\n","。","，","！","？","：","；"," ",""],
        length_function=len,
        is_separator_regex=False
    )

# 分割
async def split(documents:list[Document])->list[Document]:
    splitter = _build_splitter()
    chunks = splitter.split_documents(documents)

    for index,chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] =index
        chunk.metadata["chunk_hash"] = hashlib.md5(
            chunk.page_content.encode("utf-8")
        ).hexdigest()

    return chunks

