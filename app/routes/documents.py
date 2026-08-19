from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Query,Response
from openai.types.responses import response_error

from app.api.deps import DbSession
from app.api.schemas.documents import DocumentRead, DocumentListResponse
from app.db.models import DocumentStatus
from app.service.document_service import DocumentService
from app.store.file_service import FileService

router = APIRouter(prefix="/documents",tags=["documents"])

@router.post("",response_model=DocumentRead,status_code=201,operation_id="uploadDocument")
async def upload_document(
        session:DbSession,
        background_tasks:BackgroundTasks,
        file:UploadFile = File(...,description="待上传文档（DOC / PDF / DOCX / MARKDOWN / HTML）"))->DocumentRead:
    document_service = DocumentService(session)
    document = await document_service.upload(file,background_tasks)
    return DocumentRead.model_validate(document)

@router.get("",response_model=DocumentListResponse,operation_id="listDocuments")
async def list_documents(
        session:DbSession,
        page:int = Query(1,ge=1)
        ,page_size :int = Query(1,ge=1,le=100)
        ,status:DocumentStatus | None = Query(None,description="按文档状态筛选"),)->DocumentListResponse:
    service = DocumentService(session)
    items ,total = await service.list_documents(page,page_size,status=DocumentStatus(status) if status else None)

    return DocumentListResponse(
        # 拿出items中的每个值在 model_validate 进行结构化
        items=[DocumentRead.model_validate(item) for item in items]
        , total=total, page=page, page_size=page_size
    )

@router.get("/{document_id}",response_model=DocumentRead,operation_id="getDocument")
async def get_document(document_id: UUID,session:DbSession)->DocumentRead:
    service = DocumentService(session)
    document = await service.get(document_id)
    return DocumentRead.model_validate(document)

@router.delete("/{document_id}",status_code=204,operation_id="deleteDocument")
async def delete_document(document_id:UUID,session:DbSession):
    service = DocumentService(DbSession)
    await service.delete(document_id)
    return Response(status_code=204)

@router.post("/{document_id}/retry",response_model=DocumentRead,operation_id="retryDocument")
async def retry_document(
        document_id:UUID,
        session:DbSession,
        background_tasks :BackgroundTasks,
)->DocumentRead:
    service = DocumentService(session)
    document = await service.retry(document_id,background_tasks)
    return DocumentRead.model_validate(document)

@router.get("/{document_id}/file",operation_id="downloadDocument")
async def download_document(document_id: UUID,session:DbSession)->Response:
    document_service = DocumentService(session)

    document = await document_service.get(document_id)
    if document  is None or document.cos_object_key is None:
        raise FileNotFoundError("文件未找到")

    content = await document_service.file_service.download(document.cos_object_key)


    return Response(
        content=content
    )


