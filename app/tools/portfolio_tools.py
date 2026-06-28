"""用于汇总当前用户持仓的工具。

本模块从 MySQL 读取指定用户的持仓，并通过 akshare 用最新行情价对每行
数据进行补全，这样 ``/api/portfolio`` 列表端点就能返回带实时盈亏的持仓视图。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.data_source.fund import get_fund_nav
from app.data_source.stock import get_stock_realtime
from app.models.portfolio_crud import get_all_positions


def _to_decimal(value: Any) -> Decimal:
    """将一个值转换为 Decimal，同时保证缺失值安全。"""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _latest_price_for_position(code: str, position_type: str) -> tuple[Decimal | None, str | None]:
    """获取股票或基金持仓的最新可交易价格。"""
    try:
        if position_type == "fund":
            nav = get_fund_nav(code)
            latest = nav.get("unit_nav")
            if latest is None:
                latest = nav.get("accumulated_nav")
            return (Decimal(str(latest)) if latest is not None else None, None)

        realtime = get_stock_realtime(code)
        latest = realtime.get("latest_price")
        return (Decimal(str(latest)) if latest is not None else None, None)
    except Exception as exc:
        return None, str(exc)


def build_portfolio_summary(user_id: int) -> dict[str, Any]:
    """从 MySQL 加载指定用户的持仓，并用当前行情价进行补全。"""
    positions = get_all_positions(user_id)
    rows: list[dict[str, Any]] = []
    total_cost = Decimal("0")
    total_market_value = Decimal("0")

    for position in positions:
        shares = _to_decimal(position.shares)
        cost_price = _to_decimal(position.cost_price)
        latest_price, error = _latest_price_for_position(position.code, position.type)

        cost_value = shares * cost_price
        market_value = shares * latest_price if latest_price is not None else None
        pnl = market_value - cost_value if market_value is not None else None
        pnl_pct = (pnl / cost_value) if pnl is not None and cost_value != 0 else None

        total_cost += cost_value
        if market_value is not None:
            total_market_value += market_value

        rows.append(
            {
                "id": position.id,
                "code": position.code,
                "name": position.name,
                "type": position.type,
                "shares": shares,
                "cost_price": cost_price,
                "latest_price": latest_price,
                "cost_value": cost_value,
                "market_value": market_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "error": error,
            }
        )

    total_pnl = total_market_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) if total_cost != 0 else None
    return {
        "rows": rows,
        "total_cost": total_cost,
        "total_market_value": total_market_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
    }
