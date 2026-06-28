"""Portfolio 表的 CRUD 辅助函数。

项目把数据库访问封装在服务风格的辅助函数之后，这样应用的其余部分可以把
持仓变更当作小型、显式的操作，而不必手写 SQL。

所有函数都以 ``user_id`` 作为第一道隔离：查询、修改、删除都限定在
当前用户名下，避免越权访问他人持仓。
"""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.database import SessionLocal
from app.models.portfolio import Portfolio


def _to_decimal(value: Decimal | float | int | str) -> Decimal:
    """把灵活的数值输入转换为 Decimal，以保证稳定持久化。"""
    return value if isinstance(value, Decimal) else Decimal(str(value))


@contextmanager
def _session_scope() -> Iterator[Session]:
    """提供一个短生命周期的 SQLAlchemy 会话，并处理提交/回滚。

    关闭 ``expire_on_commit``：本模块的函数会把 ORM 对象返回到 session
    生命周期之外（路由层、Agent 工具层都会在 session 关闭后读取属性）。
    默认的 expire_on_commit=True 会在 commit 后把属性标记为过期，导致
    关闭后访问触发 DetachedInstanceError（表现为 GET /portfolio 返回 500）。
    关闭后，commit 不再使已加载的标量属性失效，session 关闭后读取安全。
    """
    db = SessionLocal(expire_on_commit=False)
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def add_position(
    user_id: int,
    code: str,
    name: str,
    type: str,
    shares: Decimal | float | int | str,
    cost_price: Decimal | float | int | str,
) -> Portfolio:
    """向当前用户的 Portfolio 表插入一条新持仓。

    Parameters
    ----------
    user_id:
        持仓所属用户。
    code:
        证券代码，例如 ``600519`` 或 ``110010``。
    name:
        展示给用户看的证券名称。
    type:
        持仓类型，通常为 ``stock`` 或 ``fund``。
    shares:
        持有数量。
    cost_price:
        每股或每份基金的平均成本。

    Returns
    -------
    Portfolio
        持久化之后新创建的 ORM 对象。
    """
    with _session_scope() as db:
        position = Portfolio(
            user_id=user_id,
            code=code.strip(),
            name=name.strip(),
            type=type.strip(),
            shares=_to_decimal(shares),
            cost_price=_to_decimal(cost_price),
        )
        db.add(position)
        db.flush()
        db.refresh(position)
        return position


def get_all_positions(user_id: int) -> list[Portfolio]:
    """返回指定用户的全部持仓行，按创建时间排序。"""
    with _session_scope() as db:
        return list(
            db.scalars(
                select(Portfolio)
                .where(Portfolio.user_id == user_id)
                .order_by(Portfolio.create_time.asc())
            ).all()
        )


def update_position(
    user_id: int,
    id: int,
    shares: Decimal | float | int | str,
    cost_price: Decimal | float | int | str,
) -> Portfolio:
    """更新当前用户名下某条持仓的数量与成本价。

    若该 id 不属于当前用户，视为不存在，抛出 ValueError——避免泄露
    其他用户持仓是否存在。
    """
    with _session_scope() as db:
        position = db.scalar(
            select(Portfolio).where(Portfolio.id == id, Portfolio.user_id == user_id)
        )
        if position is None:
            raise ValueError(f"未找到 Portfolio 行 {id}")
        position.shares = _to_decimal(shares)
        position.cost_price = _to_decimal(cost_price)
        db.flush()
        db.refresh(position)
        return position


def delete_position(user_id: int, id: int) -> bool:
    """从当前用户的 Portfolio 表删除一条持仓。

    Returns
    -------
    bool
        删除了行返回 ``True``，否则（含 id 属于他人）返回 ``False``。
    """
    with _session_scope() as db:
        position = db.scalar(
            select(Portfolio).where(Portfolio.id == id, Portfolio.user_id == user_id)
        )
        if position is None:
            return False
        db.delete(position)
        return True
