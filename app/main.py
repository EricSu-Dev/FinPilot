"""FinPilot 的 FastAPI 应用入口。"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import auth, chat, diagnosis, portfolio
from app.api.rag import ingest_uploaded_report  # noqa: F401 - 保留共享入库逻辑供 chat 复用（rag 路由已下线）
from app.api.common import api_error
from app.models.chat import ChatConversation, ChatMessage  # noqa: F401 - 导入以使 metadata 包含 chat 表
from app.models.database import Base, engine
from app.models.user import User  # noqa: F401 - 导入以使 metadata 包含 users 表

app = FastAPI(title="FinPilot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(diagnosis.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    """在 API 启动时创建所有已声明的表。

    MySQL 不可达时**不阻塞启动**：诊断 / 财报等不依赖 DB 的端点仍可用，依赖 DB
    的端点（auth / chat / portfolio）会在请求时报错。MySQL 恢复后重启即可建表。
    """
    import logging

    log = logging.getLogger(__name__)
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:  # noqa: BLE001  启动期连不上 DB 不应让整个应用退出
        log.warning("启动时连不上 MySQL，跳过建表（auth/chat/portfolio 将不可用）：%s", exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """使用项目统一信封返回参数校验错误。

    只取 loc/msg——pydantic v2 的 errors() 里 ctx.error 可能是不可 JSON
    序列化的异常对象（自定义 field_validator raise ValueError 时），原样
    塞进 JSONResponse 会序列化失败。
    """
    details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"code": 422, "msg": "参数校验失败", "data": details})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """显式处理 HTTPException，返回统一信封而不是 Starlette 默认的 {"detail": ...}。

    必须写一个独立 handler 而不能靠 generic_exception_handler：后者会把
    HTTPException 的 status_code 覆盖成 500，并且丢失 WWW-Authenticate 头。
    """
    headers = getattr(exc, "headers", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "msg": str(exc.detail), "data": None},
        headers=headers,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """把未预期的服务端错误转换为统一的 JSON 错误格式。"""
    logger = logging.getLogger(__name__)
    logger.exception("未捕获的服务端错误: %s", exc)
    return api_error(str(exc), status_code=500)


@app.get("/health")
async def health_check():
    """供部署探针使用的简单健康检查端点。"""
    return JSONResponse(content={"code": 0, "msg": "ok", "data": {"status": "healthy"}})


if __name__ == "__main__":
    """用 uvicorn 在项目端口上直接运行 API。"""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8094, reload=False)
