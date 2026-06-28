"""用户注册、登录与账号自助管理的 FastAPI 路由。"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.common import api_ok
from app.api.deps import get_current_user
from app.models.database import get_db
from app.models.user import User
from app.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    """注册载荷。"""

    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    """登录载荷。"""

    username: str
    password: str


class UpdateMeRequest(BaseModel):
    """修改个人资料载荷。

    ``username`` 与 ``new_password`` 均为可选，只传哪个就只改哪个。
    若提供 ``new_password``，必须同时提供 ``old_password`` 做二次校验，
    防止会话被拿到后无声改密。
    """

    username: Optional[str] = Field(default=None, min_length=3, max_length=64)
    old_password: Optional[str] = Field(default=None, max_length=128)
    new_password: Optional[str] = Field(default=None, min_length=6, max_length=128)


class ResetPasswordRequest(BaseModel):
    """忘记密码的重置载荷。

    按用户选定的"用户名直接重置"策略：仅凭用户名 + 新密码即可重置，
    不做额外校验。仅适用于学习演示场景，生产环境务必换邮件/短信验证码。
    """

    username: str = Field(min_length=3, max_length=64)
    new_password: str = Field(min_length=6, max_length=128)


def _public_user(user: User) -> dict:
    """对外暴露的用户视图，绝不包含 password_hash。"""
    return {"id": user.id, "username": user.username}


@router.post("/register")
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """注册一个新用户。用户名重复时返回 400。"""
    existing = db.scalar(select(User).where(User.username == request.username))
    if existing is not None:
        raise HTTPException(status_code=400, detail="用户名已被占用")
    user = User(username=request.username, password_hash=hash_password(request.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return api_ok(_public_user(user))


@router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """校验凭据并签发 JWT。凭据错误时返回 401。"""
    user = db.scalar(select(User).where(User.username == request.username))
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user.id, user.username)
    return api_ok(
        {
            "access_token": token,
            "token_type": "bearer",
            "user": _public_user(user),
        }
    )


@router.put("/me")
async def update_me(
    request: UpdateMeRequest,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """修改当前登录用户的用户名和/或密码。

    - 改密码必须校验旧密码；
    - 改用户名要保证不与他人重复；
    - 用户名一旦变更，旧 token 里的 ``username`` 即过时，故重新签发 token。
    """
    changed = False

    if request.new_password:
        if not request.old_password:
            raise HTTPException(status_code=400, detail="修改密码需提供原密码")
        if not verify_password(request.old_password, current.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="原密码错误",
            )
        if verify_password(request.new_password, current.password_hash):
            raise HTTPException(status_code=400, detail="新密码不能与原密码相同")
        current.password_hash = hash_password(request.new_password)
        changed = True

    if request.username and request.username != current.username:
        duplicate = db.scalar(select(User).where(User.username == request.username))
        if duplicate is not None:
            raise HTTPException(status_code=400, detail="用户名已被占用")
        current.username = request.username
        changed = True

    if not changed:
        raise HTTPException(status_code=400, detail="未提供任何需要修改的字段")

    db.commit()
    db.refresh(current)

    # 用户名变了旧 token 的 username 字段会失效，统一重签发，前端直接替换本地 token。
    token = create_access_token(current.id, current.username)
    return api_ok(
        {
            "access_token": token,
            "token_type": "bearer",
            "user": _public_user(current),
        }
    )


@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """忘记密码：凭用户名直接重置密码（免登录）。

    用户不存在时仍返回 400，便于前端给出明确提示。该策略无任何身份校验，
    仅为学习演示用途，切勿用于真实生产环境。
    """
    user = db.scalar(select(User).where(User.username == request.username))
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if verify_password(request.new_password, user.password_hash):
        raise HTTPException(status_code=400, detail="新密码不能与原密码相同")
    user.password_hash = hash_password(request.new_password)
    db.commit()
    return api_ok(msg="密码重置成功，请用新密码登录")
