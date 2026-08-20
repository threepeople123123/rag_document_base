from minio import S3Error

from app.api.error_handles import logger
from app.core.config import settings
from app.store.minio_client import MinioClient, get_minio_client


class FileService:
    def __init__(self,minio_client:MinioClient | None = None):
        self._client = minio_client or get_minio_client()


    def get_bucket(self):
        return self._client.get_bucket()

    def region(self)->str:
        # 配置文件里面写死
        return settings.REGION

    @staticmethod
    def build_object_name(file_hash:str,suffix:str)->str:
        return f"document/{file_hash}{suffix}"

    # 下载文件
    async def download(self,object_key:str)->bytes:
        return await self._client.get_object(object_key)

    # 上传文件
    async def upload(self,*,content:bytes,file_hash:str,suffix:str,mine_type:str)->str:
        object_key = self.build_object_name(file_hash=file_hash,suffix=suffix)
        await self._client.put_object(object_key,content,content_type=mine_type)
        return object_key

    # 删除文件
    async def remove(self,object_name:str):
        try:
            await self._client.remove_object(object_name)
        except S3Error as ex:
            logger.warning("cos delete failed: object_name=%s err=%s",object_name,ex)

file_service : FileService | None= None

def get_file_service()->FileService:
    global file_service
    if file_service is None:
        file_service = FileService()
    return file_service