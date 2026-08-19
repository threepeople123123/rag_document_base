import hashlib
from pathlib import PurePath
from uuid import UUID

from fastapi import UploadFile, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import ValidationError
from app.db.models import Document, DocumentStatus, DocumentChunk
from app.repositories.chunk_repo import DocumentChunkRepository, ChunkStatus
from app.repositories.document_repo import DocumentRepository
from app.repositories.pipeline import ingest_document
from app.routes.health import logger
from app.store.file_service import FileService, get_file_service

ALLOWED_SUFFIX = {
    "application/pdf" : ".pdf",                              # pdf
    "text/html" : ".html",                                    # html
    "application/msword": ".doc",                           # doc
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",  # docx
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "text/plain": ".md",  # md（部分浏览器报 text/plain）
}

_DELETE_STATUSES = {
    "uploading",
    "parsing",
    "indexing",
}

def _resolve_mime_and_suffix(file:UploadFile)->tuple[str,str]:
    """
    查询类型文件后缀，不支持的后缀抛出异常
    :param file: 上传的文件信息
    :return: 返回类型元祖
    """
    suffix = PurePath(file.filename or "").suffix.lower()
    content_type = file.content_type

    # 根据文件后缀查 MIME
    for mime_type, allowed_suffix in ALLOWED_SUFFIX.items():
        if suffix == allowed_suffix:
            return mime_type, allowed_suffix

    # 根据 MIME 查后缀
    if content_type in ALLOWED_SUFFIX:
        return content_type, ALLOWED_SUFFIX[content_type]

    raise ValidationError(
        f"不支持的文件类型，当前支持：{list(ALLOWED_SUFFIX.keys())}"
    )


class DocumentService:
    def __init__(self,session:AsyncSession,file_service:FileService|None =None)->None:
        self.session = session
        self.file_service = file_service or get_file_service()
        self.repo = DocumentRepository(session)
        self.chunk = DocumentChunkRepository(session)

    async def upload(self, file:UploadFile,background_tasks:BackgroundTasks)->Document:
        """
        上传文件，返回上传文档信息
        :param file: 文件
        :param background_tasks: 异步任务
        :return: 上传的文件转成文档信息
        """
        suffix , content_type = _resolve_mime_and_suffix(file)
        content = await file.read()
        max_size = settings.UPLOAD_MAX_SIZE_MB * 1024
        if len(content) < 0:
            raise ValidationError("文件为空，请重新上传")
        elif len(content) > max_size:
            raise ValidationError("文件过大，请重新上传")

        file_hash = hashlib.sha256(content).hexdigest()

        document =await self.repo.get_by_hash(file_hash)
        if document is not None:
            logger.info(f"文件已经存在，返回现有文件，file_name：{file.filename}")
            return document

        # 先上传文件，失败则不入库
        object_key = await self.file_service.upload(
            content=content,
            file_hash=file_hash,
            suffix=suffix,
            mine_type=content_type
        )
        document = Document(
            name=file.filename or f"{file_hash}{suffix}"
            , file_hash=file_hash
            , mime_type=content_type
            , size=len(content)
            , storage_provider="cos"
            , cos_bucket=self.file_service.get_bucket()
            , cos_object_key=object_key
            , cos_region=self.file_service.region()
            , status=DocumentStatus.UPLOADING
        )

        await self.repo.add(document)
        await self.session.commit()
        await self.session.refresh(document)

        background_tasks.add_task(ingest_document,document.id)

        return document

    async def get(self,document_id:UUID):
        document = await self.repo.get_by_id(document_id)
        if document is not None:
            return document
        raise FileNotFoundError("文档不存在")

    async def list_documents(self,page:int,page_no:int,*,status:DocumentStatus | None = None)->tuple[list[Document],int]:
        return await self.repo.list_paginated(page,page_no,status=status)

    async def delete(self,document_id:UUID):

        document = await self.repo.get_by_id(document_id)
        if document is None:
            raise FileNotFoundError("文件不存在")

        # 校验
        if document in _DELETE_STATUSES:
            raise ValidationError("文档正在处理中，不可以删除")

        await self.repo.delete(document)
        await self.session.commit()

        await self.file_service.remove(document.cos_object_key)
        logger.info(f"删除文档成功，document_id:{document_id}")

    async def retry(self,document_id:UUID,background_tasks:BackgroundTasks)->Document:
        """
        分片重试
        :param document_id:文档id
        :param background_tasks: 异步任务池
        :return: 文档信息
        """
        document = await self.repo.get_by_id(document_id)
        if document is None:
            raise FileNotFoundError("文档未找到")
        if document.status != DocumentStatus.FAILED:
            raise ValidationError("只有失败的文档支持重试")

        await self.chunk.delete_by_document(document_id)
        document.status = DocumentStatus.UPLOADING
        document.error_message = None
        await self.session.commit()
        await self.session.flush(document)

        # 执行分片操作
        background_tasks.add_task(ingest_document,document_id)

        return document

    async def list_chunks(self,document_id:UUID,page:int,page_size:int)->tuple[list[DocumentChunk],int ,ChunkStatus | None]:
        await self.get(document_id)
        chunks ,total= await self.chunk.list_paginated_by_document(document_id=document_id,page=page,page_size=page_size)
        chunk_status = await self.chunk.get_stats(document_id)

        return chunks,total,chunk_status

    async def get_chunk(self,document_id:UUID,chunk_id:UUID)->DocumentChunk:
        chunk = await self.chunk.get_for_document(document_id,chunk_id)
        if chunk is None:
            raise FileNotFoundError("分片不存在")
        return chunk












