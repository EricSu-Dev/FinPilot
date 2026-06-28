"""FastAPI 共享的 API 辅助函数。"""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import JSONResponse


def api_ok(data: Any = None, msg: str = "ok") -> JSONResponse:
    """返回项目统一的成功信封。"""
    return JSONResponse(content={"code": 0, "msg": msg, "data": data})


def api_error(message: str, status_code: int = 500) -> JSONResponse:
    """返回项目统一的错误信封。"""
    return JSONResponse(
        status_code=status_code,
        content={"code": status_code if status_code >= 400 else 500, "msg": message, "data": None},
    )


def sse_event(event: str, data: Any) -> str:
    """格式化一条 server-sent event 载荷。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"
