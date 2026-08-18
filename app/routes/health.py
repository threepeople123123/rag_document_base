import io
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy.sql.expression import text
from starlette.concurrency import run_in_threadpool

from app.api.deps import DbSession
from app.core.exceptions import AppException
from app.core.logging import get_logger
from app.store.minio_client import get_minio_client

logger = get_logger(__name__)

router = APIRouter(prefix="/health",tags=["/health"])

healthStatusValue = Literal["ok","error"]

class HealthStatus(BaseModel):
    status : healthStatusValue
    detail:str | None = None


# definition interface
@router.get("/db",response_model=HealthStatus,operation_id="healthDb")
async def health_db(session:DbSession) -> HealthStatus:
    try:
        result = await session.execute(text("select 1"))
        health_status = HealthStatus(detail="ok",status="ok")
        health_status.detail = str(result.scalar_one())
        return health_status
    except Exception as ex:
        logger.error("查询数据库失败,",ex)
        raise AppException(code="error",message="查询数据库失败")

@router.put("/cos",response_model=HealthStatus,operation_id="healthDb")
async def health_db() -> HealthStatus:
    try:
        minio_client = get_minio_client()
        client = minio_client.get_client()
        bucket = minio_client.get_bucket()
        data = b"ok"
        upload = await run_in_threadpool(
            client.put_object
            ,bucket
            ,"health.txt"
            ,io.BytesIO(data)
            ,len(data)
        )
        print(upload)
        return HealthStatus(status="ok")
    except Exception as ex:
        logger.error("上传文件失败",ex)
        raise AppException(code="error",message="上传文件失败")