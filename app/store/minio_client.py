import asyncio
from datetime import timedelta
from typing import BinaryIO

from minio import Minio

from app.core.config import settings


class MinioClient:
    """MinIO 客户端封装。

    底层 minio-python 的 Minio 是同步阻塞的，这里把耗时操作包装成异步方法，
    调用方（FastAPI 接口）直接 await 即可，不会阻塞事件循环。
    """

    def __init__(self):
        # endpoint 只写 host:port，不带 http:// 前缀；secure 决定 http/https
        self._client: Minio = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self._bucket = settings.BUCKET_NAME

    def get_client(self) -> Minio:
        return self._client

    def get_bucket(self) -> str:
        return self._bucket

    async def put_object(
        self,
        object_name: str,
        data: bytes | BinaryIO,
        length: int | None = None,
        content_type: str = "application/octet-stream",
        bucket_name: str | None = None,
    ) -> str:
        """上传文件到 MinIO。

        :param object_name: 对象名（桶里的 key）
        :param data: 文件内容，支持 bytes 或文件流对象（UploadFile.file 等）
        :param length: 字节长度。bytes 可不传（自动算），文件流必须传
        :param content_type: MIME 类型
        :param bucket_name: 桶名，默认用配置里的 BUCKET_NAME
        :return: 上传后的对象名
        """
        bucket = bucket_name or self._bucket

        if length is None:
            if isinstance(data, bytes):
                length = len(data)
            else:
                raise ValueError("上传文件流时必须传入 length 参数")

        result = await asyncio.to_thread(
            self._client.put_object,
            bucket,
            object_name,
            data,
            length,
            content_type=content_type,
        )
        return result.object_name

    async def remove_object(
        self,
        object_name: str,
        bucket_name: str | None = None,
    ) -> None:
        """删除指定对象。"""
        bucket = bucket_name or self._bucket
        await asyncio.to_thread(self._client.remove_object, bucket, object_name)

    async def get_object(
        self,
        object_name: str,
        bucket_name: str | None = None,
    ) -> bytes:
        """读取对象内容（适合小文件，如文本/配置）。

        大文件建议用 presigned_get_object 生成临时链接，让前端直接下载。
        """
        bucket = bucket_name or self._bucket

        def _read() -> bytes:
            resp = self._client.get_object(bucket, object_name)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()

        return await asyncio.to_thread(_read)

    async def presigned_get_object(
        self,
        object_name: str,
        expires: int = 7 * 24 * 3600,
        bucket_name: str | None = None,
    ) -> str:
        """生成临时下载链接（默认 7 天有效），前端拿到 URL 直接下载，不经过后端转发。"""
        bucket = bucket_name or self._bucket
        return await asyncio.to_thread(
            self._client.presigned_get_object,
            bucket,
            object_name,
            expires=timedelta(seconds=expires),
        )


_min_client: MinioClient | None = None


def get_minio_client() -> MinioClient:
    """获取单例 MinioClient。"""
    global _min_client
    if _min_client is None:
        _min_client = MinioClient()
    return _min_client
