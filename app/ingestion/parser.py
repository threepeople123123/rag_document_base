import asyncio
import io

from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from langchain_core.documents import Document

from app.core.exceptions import AppException
from app.core.logging import get_logger

logger = get_logger(__name__)

class DocumentParserError(AppException):
    code = "document_parser_error"
    message = "文档解析失败"
    http_status = 400

# 解析器
_converter:DocumentConverter | None = None

# 获取解析器
def _get_converter()->DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter

# 解析方法
def _convertor_sync(filename:str,content:bytes)->str:
    source = DocumentStream(name=filename,stream=io.BytesIO(content))
    result = _get_converter().convert(source)
    return result.document.export_to_markdown()

# 外部调用接口方法
async def parse(filename:str,content:bytes) -> list[Document]:
    try:
        # 将 _convertor_sync 方法异步化
        markdown:str = await asyncio.to_thread(_convertor_sync,filename,content)
    except Exception as  ex:
        logger.error("解析成markdown失败,error:",ex)
        raise DocumentParserError(f"docling 解析失败:{ex}") from ex

    if not markdown.strip():
        raise DocumentParserError("解析结果为空， 可能是文件损坏或者不支持")

    # 返回 langchain 文档对象
    return [
        Document(
            page_content=markdown,
            metadata={"source":filename}
        )
    ]