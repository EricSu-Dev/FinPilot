"""组合 CRUD 与分析的 FastAPI 路由。

所有端点都要求登录，并以当前登录用户的 id 做持仓隔离。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.api.common import api_error, api_ok
from app.api.deps import get_current_user
from app.data_source.code_validation import verify_security_code
from app.models.portfolio_crud import add_position, delete_position, update_position
from app.models.user import User
from app.tools.portfolio_tools import build_portfolio_summary

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class PortfolioCreateRequest(BaseModel):
    """用于创建一行持仓的载荷。"""

    code: str
    name: str
    type: Literal["stock", "fund"]
    shares: Decimal
    cost_price: Decimal

    @field_validator("code")
    @classmethod
    def _validate_code(cls, v: str) -> str:
        """代码必须是 6 位数字，挡住随意乱输的字符串。"""
        v = v.strip()
        if not v.isdigit() or len(v) != 6:
            raise ValueError("证券代码必须是 6 位数字")
        return v


class PortfolioUpdateRequest(BaseModel):
    """用于更新一行持仓的载荷。"""

    shares: Decimal
    cost_price: Decimal


def _serialize_decimal(value: Any) -> Any:
    """把 Decimal 值转换为 JSON 友好的基础类型。"""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_serialize_decimal(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_decimal(item) for key, item in value.items()}
    return value


def _to_public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """把组合汇总渲染为前端友好的结构。"""
    return _serialize_decimal(summary)


@router.get("")
async def list_portfolio(current_user: User = Depends(get_current_user)):
    """返回当前用户的持仓，并用实时盈亏数据补全。"""
    try:
        summary = build_portfolio_summary(current_user.id)
        return api_ok(_to_public_summary(summary))
    except Exception as exc:
        return api_error(str(exc))


@router.get("/validate")
async def validate_position_code(
    code: str,
    type: str,
    current_user: User = Depends(get_current_user),
):
    """校验证券代码真实性，股票顺带返回名称供前端回填。

    供前端新增表单代码失焦时调用：股票带出名称预览，基金只确认代码存在。
    不创建持仓，仅做只读校验。
    """
    try:
        if not code.isdigit() or len(code) != 6:
            return api_ok({"valid": False, "name": None, "message": "证券代码必须是 6 位数字"})
        resolved_name, err = await run_in_threadpool(verify_security_code, code, type)
        if err:
            return api_ok({"valid": False, "name": None, "message": err})
        return api_ok({"valid": True, "name": resolved_name, "message": None})
    except Exception as exc:
        return api_error(str(exc))


@router.post("")
async def create_portfolio_item(
    request: PortfolioCreateRequest,
    current_user: User = Depends(get_current_user),
):
    """为当前用户创建一条新持仓。

    提交前校验代码真实性：股票校验行情并以行情名回填，基金校验净值存在
    且名称必填。校验涉及网络 IO，放线程池执行避免阻塞事件循环。
    """
    try:
        resolved_name, err = await run_in_threadpool(
            verify_security_code, request.code, request.type
        )
        if err:
            return api_error(err, status_code=400)
        # 股票名称以行情名为准回填；基金名称由用户填写（必填）
        if request.type == "stock":
            final_name = resolved_name or request.name
        else:
            if not request.name.strip():
                return api_error("基金名称不能为空，请填写", status_code=400)
            final_name = request.name
        position = add_position(
            current_user.id,
            request.code,
            final_name,
            request.type,
            request.shares,
            request.cost_price,
        )
        return api_ok(_serialize_decimal({
            "id": position.id,
            "code": position.code,
            "name": position.name,
            "type": position.type,
            "shares": position.shares,
            "cost_price": position.cost_price,
        }))
    except Exception as exc:
        return api_error(str(exc))


@router.put("/{id}")
async def modify_portfolio_item(
    id: int,
    request: PortfolioUpdateRequest,
    current_user: User = Depends(get_current_user),
):
    """更新当前用户名下的一条持仓。"""
    try:
        position = update_position(current_user.id, id, request.shares, request.cost_price)
        return api_ok(_serialize_decimal({
            "id": position.id,
            "code": position.code,
            "name": position.name,
            "type": position.type,
            "shares": position.shares,
            "cost_price": position.cost_price,
        }))
    except Exception as exc:
        return api_error(str(exc))


@router.delete("/{id}")
async def remove_portfolio_item(
    id: int,
    current_user: User = Depends(get_current_user),
):
    """删除当前用户名下的一条持仓。"""
    try:
        deleted = delete_position(current_user.id, id)
        return api_ok({"deleted": deleted})
    except Exception as exc:
        return api_error(str(exc))


