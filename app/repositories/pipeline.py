from uuid import UUID

from app.core.logging import get_logger
from app.db.models import DocumentStatus, DocumentChunk
from app.db.session import AsyncSessionLocal
from app.ingestion import splitter
from app.ingestion.embedder import get_embeddings
from app.ingestion.parser import parse
from app.repositories.chunk_repo import DocumentChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.store.file_service import get_file_service

logger = get_logger(__name__)

# 设置状态
async def _set_status(document_id:UUID,status:DocumentStatus,*,error_message:str|None= None):
    async with AsyncSessionLocal() as session:
        resp = DocumentRepository(session)
        await resp.update_status(document_id,status,error_message=error_message)
        await session.commit()

async def ingest_document(document_id:UUID)->None:
    """
    完整入库流程
    :param document_id: 文档id
    :return: none
    """
    logger.info("ingest start:document_id %s",document_id)

    try:
       async with AsyncSessionLocal() as session:
           doc_repo = DocumentRepository(session)
           document = await doc_repo.get_by_id(document_id)
           if document is None:
               logger.warning("document is none,document_id:%s",document_id)
               return
           object_key = document.cos_object_key
           name = document.name

       # 开始解析
       await _set_status(document_id,DocumentStatus.PARSING)
       content = await get_file_service().download(object_key)
       if len(content) <=0:
           raise FileNotFoundError("文件不存在,minio服务器上未找到")
       documents = await parse(filename=name,content=content)

       #
       await _set_status(document_id, DocumentStatus.INDEXING)
       chunks = await splitter.split(documents)

       if not chunks:
           raise ValueError("切分后没有任何数据，请检查文档")

       embedding_model = get_embeddings()
       embeddings = await embedding_model.aembed_documents(texts=[n.page_content for n in chunks])

       async with  AsyncSessionLocal() as session:
           chunk_repo = DocumentChunkRepository(session)
           chunk_repo.session.add_all(
               [
                   DocumentChunk(
                       document_id=document_id,
                       content=c.page_content,
                       embedding=vec,
                       page_no=c.metadata.get("page_no"),
                       section_path=c.metadata.get("section_path"),
                       chunk_index=c.metadata.get("chunk_index"),
                       chunk_hash=c.metadata.get("chunk_hash"),
                       extra_metadata=c.metadata
                   )
                   for c,vec in zip(chunks,embeddings,strict=True)
               ]
           )
           await session.commit()
       await _set_status(document_id,DocumentStatus.READY,error_message=None)
       logger.info("文档解析完成")
    except Exception as e:
        logger.error("文档解析失败",str(e))
        message = str(e).strip() or e.__class__.__name__
        await _set_status(document_id,DocumentStatus.FAILED,error_message=message)