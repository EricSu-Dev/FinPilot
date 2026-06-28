"""市场热点数据获取工具。

提供 A 股市场实时热点信息：涨幅榜板块、涨停股、资金流向等。
数据源用 akshare（东方财富/新浪源），移动网络下优先走新浪源兜底。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import akshare as ak
import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _safe_akshare_call(func, *args, **kwargs) -> Any:
    """调用 akshare 接口，失败返回 None 而非抛异常。"""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.warning("akshare 调用失败 %s: %s", func.__name__, exc)
        return None


def _format_sector_table(df: pd.DataFrame | None, top_n: int = 5, ascending: bool = True) -> str:
    """把板块 DataFrame 格式化成紧凑文本，取前 N 行。

    ascending=True 取涨幅前 N，False 取跌幅前 N。
    """
    if df is None or df.empty:
        return "暂无数据"
    try:
        df_sorted = df.sort_values(by="涨跌幅", ascending=not ascending).head(top_n)
        lines = []
        for _, row in df_sorted.iterrows():
            name = row.get("板块名称", "未知")
            change = row.get("涨跌幅", 0)
            code = row.get("板块代码", "")
            lines.append(f"- {name}({code})：涨跌幅 {change:+.2f}%")
        return "\n".join(lines) if lines else "暂无数据"
    except Exception as exc:
        logger.warning("格式化板块数据失败: %s", exc)
        return "数据格式化失败"


def _get_top_sectors() -> tuple[str, str | None]:
    """获取涨幅前 5 的行业板块。返回 (格式化文本, 日期由 _get_market_fund_flow 统一提供)。"""
    df = _safe_akshare_call(ak.stock_board_industry_name_em)
    if df is None or df.empty:
        return "暂无数据", None
    return _format_sector_table(df, 5, ascending=True), None


def _get_bottom_sectors() -> tuple[str, str | None]:
    """获取跌幅前 5 的行业板块。返回 (格式化文本, 日期由 _get_market_fund_flow 统一提供)。"""
    df = _safe_akshare_call(ak.stock_board_industry_name_em)
    if df is None or df.empty:
        return "暂无数据", None
    return _format_sector_table(df, 5, ascending=False), None


def _get_top_concepts() -> tuple[str, str | None]:
    """获取涨幅前 5 的概念板块。返回 (格式化文本, 日期由 _get_market_fund_flow 统一提供)。"""
    df = _safe_akshare_call(ak.stock_board_concept_name_em)
    if df is None or df.empty:
        return "暂无数据", None
    return _format_sector_table(df, 5), None


def _get_zt_pool() -> tuple[str, str | None]:
    """获取涨停池概况（数量 + 前 5 只涨停股）。返回 (格式化文本, 日期由 _get_market_fund_flow 统一提供)。"""
    today = datetime.now().strftime("%Y%m%d")
    df = _safe_akshare_call(ak.stock_zt_pool_em, date=today)
    if df is None or df.empty:
        return "暂无涨停股数据（可能为非交易时段）", None
    count = len(df)
    top5 = df.head(5)
    lines = [f"涨停股共 {count} 只，前 5 只："]
    for _, row in top5.iterrows():
        code = row.get("代码", "")
        name = row.get("名称", "")
        industry = row.get("所属行业", "")
        zt_stat = row.get("涨停统计", "")
        lines.append(f"- {name}({code})：{industry} | 涨停统计 {zt_stat}")
    return "\n".join(lines), None


def _get_market_fund_flow() -> tuple[str, str | None]:
    """获取大盘资金流向概况。返回 (格式化文本, 数据日期)。"""
    df = _safe_akshare_call(ak.stock_market_fund_flow)
    if df is None or df.empty:
        return "暂无大盘资金流向数据", None
    try:
        latest = df.iloc[-1]  # 数据按日期升序排列，取最后一条即最新交易日
        date = latest.get("日期", None)
        data_date = str(date) if date else None
        main_net = latest.get("主力净流入-净额", 0)
        main_net_str = f"{main_net / 1e8:+.2f}亿" if main_net else "未知"
        super_large_net = latest.get("超大单净流入-净额", 0)
        super_large_str = f"{super_large_net / 1e8:+.2f}亿" if super_large_net else "未知"
        return (
            f"大盘资金流向：\n"
            f"- 主力净流入：{main_net_str}\n"
            f"- 超大单净流入：{super_large_str}"
        ), data_date
    except Exception as exc:
        logger.warning("解析资金流向失败: %s", exc)
        return "资金流向数据解析失败", None


def _get_major_indices() -> tuple[str, str | None]:
    """获取主要大盘指数实时行情。返回 (格式化文本, 日期由 _get_market_fund_flow 统一提供)。

    包括：上证指数、深证成指、创业板指、科创50、北证50、沪深300、中证500。
    """
    df = _safe_akshare_call(ak.stock_zh_index_spot_em)
    if df is None or df.empty:
        return "暂无大盘指数数据", None

    # 主要指数代码映射
    target_indices = {
        "000001": "上证指数",
        "399001": "深证成指",
        "399006": "创业板指",
        "000688": "科创50",
        "899050": "北证50",
        "000300": "沪深300",
        "000905": "中证500",
    }

    # 尝试匹配指数
    lines = []

    for _, row in df.iterrows():
        code = str(row.get("代码", ""))
        # 尝试匹配（akshare 返回的代码格式可能带后缀如 .SH）
        matched_name = None
        for target_code, target_name in target_indices.items():
            if target_code in code or code.startswith(target_code):
                matched_name = target_name
                break

        if matched_name:
            try:
                price = row.get("最新价", 0)
                change_pct = row.get("涨跌幅", 0)
                change_amt = row.get("涨跌额", 0)
                lines.append(f"- {matched_name}：{price:.2f} ({change_pct:+.2f}%, {change_amt:+.2f})")
            except Exception:
                lines.append(f"- {matched_name}：数据解析失败")

    return "\n".join(lines) if lines else "未找到主要指数数据", None


@tool
def market_hotspots_tool() -> str:
    """获取今日 A 股市场热点：大盘指数、涨幅榜/跌幅榜板块、涨停股、大盘资金流向。

    当用户问"今天市场怎么样"、"有什么热点"、"哪些板块涨得好"、
    "今天涨停的有哪些"、"大盘指数表现如何"等市场概况问题时调用。
    返回结构化的热点数据，由你综合解读成口语化的市场简报。

    注意：非交易时段（周末、节假日、收盘后）调用时，akshare 返回的是上一个
    交易日的数据。工具会明确标注数据实际日期，请在回答中如实告知用户。
    """
    indices, _ = _get_major_indices()
    sectors, _ = _get_top_sectors()
    bottom_sectors, _ = _get_bottom_sectors()
    concepts, _ = _get_top_concepts()
    zt_pool, _ = _get_zt_pool()
    fund_flow, data_date = _get_market_fund_flow()
    if data_date:
        date_note = f"（数据日期：{data_date}，非交易时段返回的是上一个交易日数据）"
    else:
        date_note = "（数据来自最新交易日）"

    return (
        f"【A 股市场热点数据】{date_note}\n\n"
        f"一、主要大盘指数：\n{indices}\n\n"
        f"二、行业板块涨幅前 5：\n{sectors}\n\n"
        f"三、行业板块跌幅前 5：\n{bottom_sectors}\n\n"
        f"四、概念板块涨幅前 5：\n{concepts}\n\n"
        f"五、涨停池：\n{zt_pool}\n\n"
        f"六、大盘资金流向：\n{fund_flow}\n\n"
        f"【提示】以上为行情数据，请结合这些事实向用户解读市场热点、"
        f"驱动逻辑（可调用 web_search_tool 补充相关新闻）、以及潜在风险提示。"
        f"如果数据日期与当前日期不同，请明确告知用户这是上一个交易日的数据。"
    )
