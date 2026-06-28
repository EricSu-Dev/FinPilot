"""密码哈希与 JWT 编解码工具。

刻意保持无状态：本模块只负责单向哈希和 token 的签发/校验，
不触碰数据库。用户查询由调用方（``app/api/deps.py``）完成。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.config import get_settings

settings = get_settings()


def hash_password(plain: str) -> str:
    """用 bcrypt 对明文密码做单向哈希，返回可持久化的字符串。"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文密码是否匹配已存储的 bcrypt 哈希。"""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int, username: str) -> str:
    """为指定用户签发一个带过期时间的 JWT。"""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """校验签名与过期时间，返回 payload；失败抛 ``jwt.PyJWTError`` 的子类。"""
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
