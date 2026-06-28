"""把第三方数据转换为稳定 Python 值的共享辅助函数。"""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd


def clean_value(value: Any) -> Any:
    """把 pandas/numpy 标量值转换为 JSON 友好的 Python 值。"""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def to_float(value: Any) -> float | None:
    """安全地把原始表格值转换为浮点数。"""
    cleaned = clean_value(value)
    if cleaned in (None, "", "-"):
        return None
    try:
        return float(str(cleaned).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def to_decimal(value: Any) -> Decimal | None:
    """安全地把原始表格值转换为 Decimal，用于 ORM 写入。"""
    cleaned = clean_value(value)
    if cleaned in (None, "", "-"):
        return None
    try:
        return Decimal(str(cleaned).replace(",", "").replace("%", ""))
    except (InvalidOperation, ValueError):
        return None


def to_date(value: Any) -> date | None:
    """安全地把原始值转换为日期。"""
    cleaned = clean_value(value)
    if cleaned in (None, "", "-"):
        return None
    parsed = pd.to_datetime(cleaned, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def first_existing(row: pd.Series, names: list[str]) -> Any:
    """返回 pandas 行中第一个存在的列值。"""
    for name in names:
        if name in row:
            return row[name]
    return None


def ensure_dataframe(value: Any, source_name: str) -> pd.DataFrame:
    """校验 akshare 的返回是否为 DataFrame。"""
    if not isinstance(value, pd.DataFrame):
        raise RuntimeError(f"{source_name} 返回了 {type(value).__name__}，期望返回 DataFrame")
    return value
