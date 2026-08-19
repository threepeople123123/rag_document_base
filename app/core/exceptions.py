from http import HTTPStatus


class AppException(Exception):

    code:str = "internal_error"
    message:str = "服务器内部错误"
    http_status :int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self,code :str | None = None ,*,message:str | None = None):
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        super().__init__(self.message)


class NotFoundError(AppException):

    code:str = "not_found"
    message:str = "资源不存在"
    http_status :int = HTTPStatus.NOT_FOUND

class PermissionDeniedError(AppException):

    code:str = "permission_denied"
    message:str = "无权限访问"
    http_status :int = HTTPStatus.FORBIDDEN

class ConfigurationError(AppException):

    code:str = "configuration_error"
    message:str = "服务配置缺失"
    http_status :int = HTTPStatus.SERVICE_UNAVAILABLE

class ValidationError(AppException):
    code: str = "validation_error"
    message: str = "参数校验失败"
    http_status: int = HTTPStatus.BAD_REQUEST

