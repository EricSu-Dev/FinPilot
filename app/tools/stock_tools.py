"""用于股票数据访问的 LangChain 工具。"""

from typing import Any

from langchain_core.tools import tool

from app.data_source.stock import (
    get_stock_financial_detail,
    get_stock_history_kline,
    get_stock_industry,
    get_stock_realtime,
)


@tool
def stock_realtime_tool(code: str) -> dict[str, Any]:
    """获取单只 A 股的实时行情数据。

    Parameters
    ----------
    code:
        股票代码，例如 ``600519``。该工具期望的是不带交易所前缀的
        沪深 A 股代码。

    Returns
    -------
    dict[str, Any]
        归一化后的字典，包含以下字段：
        ``code``、``name``、``latest_price``、``change_percent``、``volume``、
        ``pe_ratio`` 和 ``pb_ratio``。数值为浮点数或 ``None``。

    Notes
    -----
    这是对 baostock 股票实时数据源的一层薄封装。
    """
    return get_stock_realtime(code)


@tool
def stock_financial_detail_tool(code: str) -> dict[str, Any]:
    """获取单只 A 股的详细财务指标，覆盖盈利/成长/偿债/运营/杜邦五个维度。

    Parameters
    ----------
    code:
        股票代码，例如 ``600519`` 或 ``300308``。

    Returns
    -------
    dict[str, Any]
        包含以下分组的归一化字典：
        - 盈利：``roe``、``gross_margin``、``net_profit_margin``、
          ``net_profit_yi``（亿元）、``revenue_yi``（亿元）、``eps``
        - 成长：``net_profit_growth``、``asset_growth``、``equity_growth``
        - 偿债：``current_ratio``、``quick_ratio``、``debt_ratio``
        - 运营：``asset_turnover``
        - 杜邦：``dupont_net_margin``、``dupont_asset_turnover``、
          ``dupont_equity_multiplier``
        另含 ``report_date`` 指明数据对应的报告期。所有比率均为小数形式
        （如 0.44 表示 44%），绝对值已折算成亿元。
    """
    return get_stock_financial_detail(code)


@tool
def stock_history_kline_tool(code: str) -> dict[str, Any]:
    """获取单只 A 股近 120 个交易日的日 K 线及派生技术指标。

    Parameters
    ----------
    code:
        股票代码，例如 ``300308``。

    Returns
    -------
    dict[str, Any]
        包含最新价、均线（MA5/MA10/MA20/MA60）、RSI(14)、近 20 日高低点
        （作为支撑/压力位参考）、20 日均量与量比、换手率、趋势判断，
        以及最近 10 个交易日的 K 线摘要（``recent_kline``）。
        非交易日自动取最近一个有数据的交易日。
    """
    return get_stock_history_kline(code, days=120)


@tool
def stock_industry_tool(code: str) -> dict[str, Any]:
    """获取单只 A 股的所属行业与证监会行业分类。

    Parameters
    ----------
    code:
        股票代码，例如 ``300308``。

    Returns
    -------
    dict[str, Any]
        归一化后的字典，包含 ``code``、``industry``（所属行业）和
        ``industry_classification``（证监会行业分类）。用于补充公司业务
        背景与同业对比。
    """
    return get_stock_industry(code)
