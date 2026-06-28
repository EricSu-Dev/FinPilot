"""用于基金数据访问、业绩、费率与轻量估值的 LangChain 工具。"""

import logging
import re
from datetime import date, datetime
from typing import Any, TypedDict

from langchain_core.tools import tool

from app.data_source.common import clean_value
from app.data_source.fund import (
    FundHolding,
    get_fund_info,
    get_fund_industry_allocation,
    get_fund_nav,
    get_fund_performance,
    get_fund_fee,
    get_fund_top_holdings,
)
from app.data_source.stock import get_stock_realtime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 是否是有效 A 股代码（6 位数字或含 sh/sz 前缀）——用于过滤 QDII 美股/港股
# ---------------------------------------------------------------------------

_A_SHARE_CODE_RE = re.compile(r"^(sh\.|sz\.|SH|SZ|6|0|3|8)\d{5,6}$", re.IGNORECASE)


def _is_ashare_code(code: str) -> bool:
    """判断代码是否是 A 股格式（排除 QDII 美股/港股如 NVDA、0700.HK 等）。"""
    s = str(code).strip()
    if not s:
        return False
    # 纯 6 位数字 → A 股
    if re.fullmatch(r"\d{6}", s):
        return True
    # 带前缀的 9 位 → baostock 格式
    if re.fullmatch(r"(sh|sz)\.\d{6}", s, re.IGNORECASE):
        return True
    return False


class FundValuationResult(TypedDict):
    """根据基金前十大持仓估算的短期变动结果。"""

    code: str
    estimated_change_percent: float | None
    data_cutoff_date: str
    disclaimer: str
    holdings: list[dict[str, Any]]


def _latest_completed_quarter_end(reference_date: date | None = None) -> date:
    """返回最近一个已结束季度的末尾日期。

    基金持仓按季度披露且存在滞后，因此工具应给出一个保守的截止日期，
    而不是暗示数据为当日实时。
    """
    current = reference_date or datetime.now().date()
    if current.month <= 3:
        return date(current.year - 1, 12, 31)
    if current.month <= 6:
        return date(current.year, 3, 31)
    if current.month <= 9:
        return date(current.year, 6, 30)
    return date(current.year, 9, 30)


@tool
def fund_info_tool(code: str) -> dict[str, Any]:
    """获取公募基金的基本资料。

    Parameters
    ----------
    code:
        基金代码，例如 ``110010``。

    Returns
    -------
    dict[str, Any]
        归一化后的字典，包含基金代码、名称和类型。适合用于筛选，
        以及把代码转换为可读的基金标签。
    """
    return get_fund_info(code)


@tool
def fund_nav_tool(code: str) -> dict[str, Any]:
    """获取公募基金最新的净值信息。

    Parameters
    ----------
    code:
        基金代码，例如 ``110010``。

    Returns
    -------
    dict[str, Any]
        归一化后的字典，包含 ``code``、``nav_date``、``unit_nav``
        和 ``accumulated_nav``。这些值为数据源层返回的最新净值快照。
    """
    return get_fund_nav(code)


@tool
def fund_valuation_tool(code: str) -> FundValuationResult:
    """根据基金前十大持仓估算其短期净值方向。

    对非 A 股持仓（QDII 美股/港股）优雅跳过，不触发 baostock 查询。

    Parameters
    ----------
    code:
        基金代码，例如 ``110010``。

    Returns
    -------
    FundValuationResult
        包含以下字段的字典：
        ``estimated_change_percent``：前十大持仓实时涨跌的加权平均值。
        ``data_cutoff_date``：作为披露截止提示的最近一个已结束季度末日期。
        ``disclaimer``：明确说明结果仅为粗略估算，不构成投资建议。
        ``holdings``：每只持仓的明细，包括股票代码、股票名称、披露权重、
        实时涨跌幅以及对估算值的贡献。

    Notes
    -----
    结果有意保持近似。它使用披露的前十大持仓权重与这些持仓当前的实时变动，
    因此更适合用于解释和提供背景，而非用于预测。非 A 股持仓涨跌记为 None。
    """
    holdings: list[FundHolding] = get_fund_top_holdings(code)
    cutoff_date = _latest_completed_quarter_end().isoformat()
    weighted_holdings: list[dict[str, Any]] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for holding in holdings:
        stock_code = clean_value(holding.get("stock_code"))
        weight = holding.get("weight")
        if not stock_code or weight is None:
            continue

        # 非 A 股代码（QDII 美股/港股）直接跳过实时查询
        change_percent: float | None = None
        if _is_ashare_code(str(stock_code)):
            try:
                realtime = get_stock_realtime(str(stock_code))
                change_percent = realtime.get("change_percent")
            except Exception:
                logger.debug("持仓 %s 实时行情获取失败，跳过", stock_code)
                change_percent = None
        else:
            logger.debug("持仓 %s 非 A 股代码，跳过实时查询", stock_code)

        if change_percent is None:
            weighted_holdings.append(
                {
                    "stock_code": stock_code,
                    "stock_name": clean_value(holding.get("stock_name")),
                    "weight": float(weight),
                    "change_percent": None,
                    "contribution": None,
                }
            )
            continue

        weight_value = float(weight)
        contribution = weight_value * float(change_percent)
        weighted_sum += contribution
        total_weight += weight_value
        weighted_holdings.append(
            {
                "stock_code": stock_code,
                "stock_name": clean_value(holding.get("stock_name")),
                "weight": weight_value,
                "change_percent": float(change_percent),
                "contribution": contribution,
            }
        )

    # 若无任何 A 股持仓可估算（如债基、QDII 美股/港股基金），优雅降级：
    # 保留持仓明细（供 holdings_analysis 使用），estimated_change_percent 为 None。
    if total_weight <= 0:
        disclaimer = (
            "本基金无可估算的 A 股持仓实时涨跌（可能为债基或 QDII 基金），"
            "estimated_change_percent 为空。持仓明细仍可用于持仓分析。"
            "不构成投资建议、不是预测、也不是推荐。"
        )
        return {
            "code": code.strip(),
            "estimated_change_percent": None,
            "data_cutoff_date": cutoff_date,
            "disclaimer": disclaimer,
            "holdings": weighted_holdings,
        }

    estimated_change = weighted_sum / total_weight
    disclaimer = (
        "本估算基于披露的前十大持仓及其当前实时涨跌。仅为近似的环境参考信号，"
        "不构成投资建议、不是预测、也不是推荐。"
    )
    return {
        "code": code.strip(),
        "estimated_change_percent": round(estimated_change, 4),
        "data_cutoff_date": cutoff_date,
        "disclaimer": disclaimer,
        "holdings": weighted_holdings,
    }


# ---------------------------------------------------------------------------
# 新增：业绩 / 费率 / 行业配置
# ---------------------------------------------------------------------------


@tool
def fund_performance_tool(code: str) -> dict[str, Any]:
    """获取基金业绩与波动数据（多周期收益、波动率、最大回撤）。

    Parameters
    ----------
    code:
        基金代码，例如 ``007575``。

    Returns
    -------
    dict[str, Any]
        包含近1周/1月/3月/6月/1年/3年/今年来/成立来收益率、
        年化波动率、最大回撤、最新单日涨跌幅等指标。
    """
    return get_fund_performance(code)


@tool
def fund_fee_tool(code: str) -> dict[str, Any]:
    """获取基金费率与交易规则（申购/赎回/管理/托管/销售服务费）。

    Parameters
    ----------
    code:
        基金代码，例如 ``007575``。

    Returns
    -------
    dict[str, Any]
        包含申购费率(前端/后端)、赎回费率(分档)、管理费率、托管费率、
        销售服务费率、最低申购金额等。
    """
    return get_fund_fee(code)


@tool
def fund_industry_allocation_tool(code: str) -> dict[str, Any]:
    """获取基金行业配置（行业集中度）。

    Parameters
    ----------
    code:
        基金代码，例如 ``007575``。

    Returns
    -------
    dict[str, Any]
        包含行业名称和占比权重列表，对应蚂蚁财富的"持仓分析-行业集中度"。
    """
    return get_fund_industry_allocation(code)
