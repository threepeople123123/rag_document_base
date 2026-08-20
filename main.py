import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.error_handles import register_error_handlers
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.routes import health, documents


def create_app()-> FastAPI:
    configure_logging()
    logger = get_logger(__name__)
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cores_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(documents.router,prefix="/api")
    app.include_router(health.router,prefix="/api")
    logger.info("app start complete")
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=8222)