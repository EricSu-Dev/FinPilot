"""FastAPI 鉴权依赖。

提供 ``get_current_user``：从 ``Authorization: Bearer <token>`` 头解析 JWT
并加载对应的 ``User``。所有需要登录的端点通过 ``Depends(get_current_user)``
注入当前用户；持仓相关端点再以 ``user.id`` 做数据隔离。
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.user import User
from app.security import decode_token

bearer_scheme = HTTPBearer(auto_error=True)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """校验 Bearer token 并返回当前登录用户。"""
    token = credentials.credentials
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="token 缺少用户标识")
    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token 用户标识无效",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.get(User, user_id_int)
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user
