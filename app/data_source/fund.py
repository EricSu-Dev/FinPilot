"""基于 akshare 构建的公募基金数据封装。"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, TypedDict

import akshare as ak
import pandas as pd

from app.data_source.common import clean_value, ensure_dataframe, first_existing, to_float

logger = logging.getLogger(__name__)


# 全市场公募基金列表缓存：ak.fund_name_em() 一次拉一万多条，多处分用；
# 带 5 分钟 TTL，避免每次 get_fund_info / 按名搜索都重新拉。
_fund_name_cache: dict[str, Any] = {"df": None, "ts": 0.0}
_FUND_NAME_TTL = 300.0


def _get_fund_name_df() -> pd.DataFrame:
    """返回带 TTL 缓存的全市场公募基金列表 DataFrame。"""
    now = time.time()
    cached = _fund_name_cache["df"]
    if cached is not None and now - _fund_name_cache["ts"] < _FUND_NAME_TTL:
        return cached
    data = ensure_dataframe(ak.fund_name_em(), "fund_name_em")
    _fund_name_cache["df"] = data
    _fund_name_cache["ts"] = now
    return data


class FundInfo(TypedDict):
    """公募基金基本信息。"""

    code: str
    name: str | None
    type: str | None


class FundNav(TypedDict):
    """基金最新净值信息。"""

    code: str
    nav_date: str | None
    unit_nav: float | None
    accumulated_nav: float | None


class FundHolding(TypedDict):
    """公募基金的一只前十大持仓。"""

    stock_code: str | None
    stock_name: str | None
    weight: float | None


def get_fund_info(code: str) -> FundInfo:
    """获取公募基金基本信息。"""
    normalized_code = code.strip()
    try:
        data = _get_fund_name_df()
    except Exception as exc:
        raise RuntimeError("从 akshare 获取公募基金列表失败") from exc

    matched = data[data["基金代码"].astype(str) == normalized_code]
    if matched.empty:
        raise ValueError(f"在公募基金列表中未找到基金代码 {normalized_code}")

    row = matched.iloc[0]
    return {
        "code": normalized_code,
        "name": clean_value(first_existing(row, ["基金简称", "基金名称"])),
        "type": clean_value(first_existing(row, ["基金类型", "类型"])),
    }


def resolve_fund_code(keyword: str) -> tuple[str | None, str | None, list[dict[str, str]]]:
    """把用户输入的代码或名称解析成 6 位基金代码。

    返回 ``(code, name, candidates)``：
    - 6 位数字：直接当代码返回。
    - 名称：在全市场基金列表里按"基金简称"包含匹配。唯一匹配则 code 填上；
      多个匹配则 code=None、candidates 列前 10 个；无匹配则全 None。
    """
    kw = (keyword or "").strip()
    if not kw:
        return None, None, []
    if kw.isdigit() and len(kw) == 6:
        return kw, None, []

    try:
        data = _get_fund_name_df()
    except Exception:
        return None, None, []

    name_col = None
    for col in ("基金简称", "基金名称", "名称"):
        if col in data.columns:
            name_col = col
            break
    if name_col is None:
        return None, None, []

    matched = data[data[name_col].astype(str).str.contains(kw, na=False)]
    if matched.empty:
        return None, None, []

    candidates: list[dict[str, str]] = []
    for _, row in matched.head(10).iterrows():
        candidates.append(
            {"code": str(row["基金代码"]), "name": str(row[name_col])}
        )

    if len(candidates) == 1:
        return candidates[0]["code"], candidates[0]["name"], candidates
    return None, None, candidates


def get_fund_nav(code: str) -> FundNav:
    """获取公募基金最新的单位净值与累计净值。"""
    normalized_code = code.strip()
    try:
        data = ensure_dataframe(
            ak.fund_open_fund_info_em(symbol=normalized_code, indicator="单位净值走势"),
            "fund_open_fund_info_em",
        )
    except Exception as exc:
        raise RuntimeError(f"获取基金 {normalized_code} 的净值数据失败") from exc

    if data.empty:
        raise ValueError(f"未找到基金 {normalized_code} 的净值数据")

    date_column = "净值日期" if "净值日期" in data.columns else "x"
    data = data.copy()
    if date_column in data.columns:
        data["__nav_date"] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.sort_values("__nav_date")
    latest = data.iloc[-1]

    return {
        "code": normalized_code,
        "nav_date": clean_value(first_existing(latest, ["净值日期", "x"])),
        "unit_nav": to_float(first_existing(latest, ["单位净值", "y"])),
        "accumulated_nav": to_float(first_existing(latest, ["累计净值"])),
    }


def _candidate_holding_years() -> list[str]:
    """返回最近的候选年份，因为 akshare 的基金持仓接口按年份查询。"""
    current_year = datetime.now().year
    return [str(year) for year in range(current_year, current_year - 4, -1)]


def get_fund_top_holdings(code: str) -> list[FundHolding]:
    """获取公募基金的前十大股票持仓。"""
    normalized_code = code.strip()
    last_error: Exception | None = None
    for year in _candidate_holding_years():
        try:
            data = ensure_dataframe(
                ak.fund_portfolio_hold_em(symbol=normalized_code, date=year),
                "fund_portfolio_hold_em",
            )
        except Exception as exc:
            last_error = exc
            continue

        if data.empty:
            continue

        holdings: list[FundHolding] = []
        for _, row in data.head(10).iterrows():
            holdings.append(
                {
                    "stock_code": clean_value(first_existing(row, ["股票代码", "代码"])),
                    "stock_name": clean_value(first_existing(row, ["股票名称", "名称"])),
                    "weight": to_float(first_existing(row, ["占净值比例", "持仓占比", "持股占净值比"])),
                }
            )
        return holdings

    if last_error is not None:
        raise RuntimeError(f"获取基金 {normalized_code} 的前十大持仓失败") from last_error
    raise ValueError(f"未找到基金 {normalized_code} 的前十大持仓数据")


# ---------------------------------------------------------------------------
# 业绩与波动（多周期收益 + 波动率 + 最大回撤）
# ---------------------------------------------------------------------------


class FundPerformance(TypedDict):
    """基金业绩与波动数据。"""

    code: str
    nav_date: str | None                    # 最新净值日期
    unit_nav: float | None                  # 最新单位净值
    accumulated_nav: float | None           # 最新累计净值
    daily_change: float | None              # 最新单日涨跌幅(%)
    return_1w: float | None                 # 近1周收益率(%)
    return_1m: float | None                 # 近1月收益率(%)
    return_3m: float | None                 # 近3月收益率(%)
    return_6m: float | None                 # 近6月收益率(%)
    return_1y: float | None                 # 近1年收益率(%)
    return_3y: float | None                 # 近3年收益率(%)
    return_ytd: float | None                # 今年来收益率(%)
    return_since_inception: float | None    # 成立来收益率(%)
    annualized_volatility: float | None     # 年化波动率(%)
    max_drawdown: float | None              # 最大回撤(%)
    data_source: str                        # 数据来源说明


def _compute_return(nav_series: pd.Series, days_back: int) -> float | None:
    """从累计净值序列中计算 N 天前的收益率。"""
    if len(nav_series) < 2 or days_back >= len(nav_series):
        return None
    end_val = nav_series.iloc[-1]
    start_idx = len(nav_series) - 1 - days_back
    if start_idx < 0:
        return None
    start_val = nav_series.iloc[start_idx]
    if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
        return None
    return round((end_val / start_val - 1) * 100, 2)


def _compute_ytd_return(nav_series: pd.Series, date_series: pd.Series) -> float | None:
    """计算今年来收益率。"""
    if len(nav_series) < 2:
        return None
    current_year = datetime.now().year
    year_start_mask = date_series.dt.year == current_year
    year_start_indices = year_start_mask[year_start_mask].index
    if len(year_start_indices) == 0:
        return None
    start_val = nav_series.iloc[year_start_indices[0]]
    end_val = nav_series.iloc[-1]
    if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
        return None
    return round((end_val / start_val - 1) * 100, 2)


def _compute_since_inception(nav_series: pd.Series) -> float | None:
    """计算成立来收益率。"""
    if len(nav_series) < 2:
        return None
    start_val = nav_series.iloc[0]
    end_val = nav_series.iloc[-1]
    if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
        return None
    return round((end_val / start_val - 1) * 100, 2)


def _compute_annualized_volatility(daily_returns: pd.Series) -> float | None:
    """计算年化波动率(日收益标准差 × √250)。"""
    if len(daily_returns) < 20:
        return None
    std = daily_returns.std()
    if pd.isna(std):
        return None
    return round(std * (250**0.5) * 100, 2)


def _compute_max_drawdown(nav_series: pd.Series) -> float | None:
    """计算最大回撤。"""
    if len(nav_series) < 2:
        return None
    peak = nav_series.expanding().max()
    drawdown = (nav_series - peak) / peak
    min_dd = drawdown.min()
    if pd.isna(min_dd):
        return None
    return round(min_dd * 100, 2)


def get_fund_performance(code: str) -> FundPerformance:
    """获取基金业绩与波动数据（多周期收益、波动率、最大回撤）。

    通过 akshare ``fund_open_fund_info_em`` 获取累计净值走势全序列，
    自算多周期收益率、年化波动率、最大回撤。eastmoney 接口失败时
    所有数值字段返回 None，data_source 标注降级状态。
    """
    normalized_code = code.strip()

    # 尝试获取累计净值走势时序
    nav_series: pd.Series | None = None
    date_series: pd.Series | None = None
    daily_returns: pd.Series | None = None
    latest_nav_date: str | None = None
    latest_unit: float | None = None
    latest_acc: float | None = None
    source_label = "akshare eastmoney 累计净值时序"

    try:
        data = ensure_dataframe(
            ak.fund_open_fund_info_em(symbol=normalized_code, indicator="累计净值走势"),
            "fund_open_fund_info_em",
        )
        if not data.empty:
            data = data.copy()
            # 列名可能是 净值日期/x + 累计净值/y + 单位净值
            date_col = "净值日期" if "净值日期" in data.columns else "x"
            acc_col = "累计净值" if "累计净值" in data.columns else "y"
            unit_col = "单位净值" if "单位净值" in data.columns else None

            data["_date"] = pd.to_datetime(data[date_col], errors="coerce")
            data = data.dropna(subset=["_date"]).sort_values("_date")

            nav_series = data[acc_col].astype(float)
            date_series = data["_date"]

            # 单位净值列可能不存在（累计净值走势里不一定有）
            if unit_col and unit_col in data.columns:
                latest_unit = to_float(data[unit_col].iloc[-1])
            # 从 fund_nav 获取最新单位净值（如有）
            try:
                nav_info = get_fund_nav(normalized_code)
                latest_unit = nav_info.get("unit_nav") or latest_unit
                latest_nav_date = nav_info.get("nav_date")
                latest_acc = nav_info.get("accumulated_nav") or to_float(nav_series.iloc[-1])
            except Exception:
                latest_acc = to_float(nav_series.iloc[-1]) if nav_series is not None else None

            # 日收益率序列
            if nav_series is not None and len(nav_series) > 1:
                daily_returns = nav_series.pct_change().dropna()
    except Exception as exc:
        logger.warning("fund_open_fund_info_em(累计净值走势) 失败: %s", exc)
        source_label = "eastmoney 接口失败，数据全部缺失"

    # 计算各项指标
    daily_change = None
    if daily_returns is not None and len(daily_returns) > 0:
        daily_change = round(daily_returns.iloc[-1] * 100, 2)

    return_1w = _compute_return(nav_series, 5) if nav_series is not None else None
    return_1m = _compute_return(nav_series, 22) if nav_series is not None else None
    return_3m = _compute_return(nav_series, 66) if nav_series is not None else None
    return_6m = _compute_return(nav_series, 132) if nav_series is not None else None
    return_1y = _compute_return(nav_series, 250) if nav_series is not None else None
    return_3y = _compute_return(nav_series, 750) if nav_series is not None else None
    return_ytd = _compute_ytd_return(nav_series, date_series) if nav_series is not None else None
    return_since_inception = _compute_since_inception(nav_series) if nav_series is not None else None
    annualized_vol = _compute_annualized_volatility(daily_returns) if daily_returns is not None else None
    max_dd = _compute_max_drawdown(nav_series) if nav_series is not None else None

    return {
        "code": normalized_code,
        "nav_date": latest_nav_date,
        "unit_nav": latest_unit,
        "accumulated_nav": latest_acc,
        "daily_change": daily_change,
        "return_1w": return_1w,
        "return_1m": return_1m,
        "return_3m": return_3m,
        "return_6m": return_6m,
        "return_1y": return_1y,
        "return_3y": return_3y,
        "return_ytd": return_ytd,
        "return_since_inception": return_since_inception,
        "annualized_volatility": annualized_vol,
        "max_drawdown": max_dd,
        "data_source": source_label,
    }


# ---------------------------------------------------------------------------
# 费率与交易规则
# ---------------------------------------------------------------------------


class FundFee(TypedDict):
    """基金费率与交易规则。"""

    code: str
    purchase_fee_front: str | None       # 申购费率（前端），可能为分档文本
    purchase_fee_back: str | None        # 申购费率（后端）
    redemption_fee: str | None           # 赎回费率（分档文本）
    management_fee: str | None           # 管理费率原始文本（如"0.60%(每年)"）
    custody_fee: str | None              # 托管费率原始文本
    sales_service_fee: str | None        # 销售服务费原始文本（C类份额特征）
    min_purchase_amount: str | None      # 最低申购金额
    data_source: str


def get_fund_fee(code: str) -> FundFee:
    """获取基金费率与交易规则。

    通过 akshare ``fund_fee_em`` 分别查询申购费率(前端)、赎回费率、
    运作费用（管理费/托管费/销售服务费）。eastmoney 接口失败时所有
    字段返回 None，LLM 会在报告里标注数据缺口。
    """
    normalized_code = code.strip()
    purchase_front: str | None = None
    purchase_back: str | None = None
    redemption: str | None = None
    management: str | None = None
    custody: str | None = None
    sales_service: str | None = None
    min_purchase: str | None = None
    source_label = "akshare eastmoney fund_fee_em"

    # 申购费率（前端）—— 注意：akshare 要求半角括号
    try:
        data = ensure_dataframe(
            ak.fund_fee_em(symbol=normalized_code, indicator="申购费率(前端)"),
            "fund_fee_em 申购费率前端",
        )
        if not data.empty:
            purchase_front = _fee_table_to_text(data)
    except Exception as exc:
        logger.warning("fund_fee_em(申购费率前端) 失败: %s", exc)

    # 赎回费率
    try:
        data = ensure_dataframe(
            ak.fund_fee_em(symbol=normalized_code, indicator="赎回费率"),
            "fund_fee_em 赎回费率",
        )
        if not data.empty:
            redemption = _fee_table_to_text(data)
    except Exception as exc:
        logger.warning("fund_fee_em(赎回费率) 失败: %s", exc)

    # 运作费用—— akshare 返回编号列(0,1,2,3,4,5)，单行 key-value 布局：
    # (0,1): 管理费率→值, (2,3): 托管费率→值, (4,5): 销售服务费率→值
    try:
        data = ensure_dataframe(
            ak.fund_fee_em(symbol=normalized_code, indicator="运作费用"),
            "fund_fee_em 运作费用",
        )
        if not data.empty:
            row = data.iloc[0]
            cols_list = list(data.columns)
            vals_list = [str(clean_value(row[c])) for c in cols_list]
            # 遍历 key-value 对（相邻两列），存原始文本
            for i in range(0, len(vals_list) - 1, 2):
                key = vals_list[i]
                value = vals_list[i + 1]
                if "管理费" in key:
                    management = value
                elif "托管费" in key:
                    custody = value
                elif "销售服务" in key or "服务费" in key:
                    sales_service = value
    except Exception as exc:
        logger.warning("fund_fee_em(运作费用) 失败: %s", exc)

    # 最低申购金额—— fund_purchase_em 加载全量数据太重，跳过
    # 如有需要可从 fund_open_fund_info_em(indicator="费率") 另取
    min_purchase = None

    if all(v is None for v in [purchase_front, redemption, management, custody, sales_service]):
        source_label = "eastmoney 接口失败，费率数据全部缺失"

    return {
        "code": normalized_code,
        "purchase_fee_front": purchase_front,
        "purchase_fee_back": purchase_back,
        "redemption_fee": redemption,
        "management_fee": management,
        "custody_fee": custody,
        "sales_service_fee": sales_service,
        "min_purchase_amount": min_purchase,
        "data_source": source_label,
    }


def _fee_table_to_text(data: pd.DataFrame) -> str:
    """把费率分档表（如 申购金额区间 → 费率）转为可读文本。"""
    lines: list[str] = []
    for _, row in data.head(8).iterrows():
        parts = [clean_value(v) for v in row.values if clean_value(v) is not None]
        if parts:
            lines.append(" / ".join(str(p) for p in parts))
    return "; ".join(lines) if lines else None


# ---------------------------------------------------------------------------
# 行业配置（行业集中度）
# ---------------------------------------------------------------------------


class FundIndustryAllocation(TypedDict):
    """基金行业配置。"""

    code: str
    report_date: str | None               # 报告期
    industries: list[dict[str, Any]]       # [{name, weight}]
    data_source: str


def get_fund_industry_allocation(code: str) -> FundIndustryAllocation:
    """获取基金行业配置（行业集中度）。

    通过 akshare ``fund_portfolio_industry_allocation_em`` 查询最近年份
    的行业配置。eastmoney 接口失败时 industries 为空列表。
    """
    normalized_code = code.strip()
    current_year = datetime.now().year
    industries: list[dict[str, Any]] = []
    report_date: str | None = None
    source_label = "akshare eastmoney fund_portfolio_industry_allocation_em"

    for year in range(current_year, current_year - 4, -1):
        try:
            data = ensure_dataframe(
                ak.fund_portfolio_industry_allocation_em(symbol=normalized_code, date=str(year)),
                "fund_portfolio_industry_allocation_em",
            )
            if data.empty:
                continue
            # 列名可能是: 行业 / 占净值比例 / 报告期
            report_date = clean_value(
                first_existing(data.iloc[0], ["报告期", "日期", "截止日期"]) if "报告期" in data.columns or "日期" in data.columns or "截止日期" in data.columns else None
            )
            # 对列名做宽容匹配
            name_col = _find_col(data, ["行业", "行业名称", "板块"])
            weight_col = _find_col(data, ["占净值比例", "比例", "持仓占比", "占比"])
            if name_col and weight_col:
                for _, row in data.head(10).iterrows():
                    industries.append({
                        "name": clean_value(row[name_col]),
                        "weight": to_float(row[weight_col]),
                    })
            break  # 找到数据就停
        except Exception as exc:
            logger.warning("fund_portfolio_industry_allocation_em(%s, %s) 失败: %s", normalized_code, year, exc)
            continue

    if not industries:
        source_label = "eastmoney 接口失败，行业配置数据缺失"

    return {
        "code": normalized_code,
        "report_date": report_date,
        "industries": industries,
        "data_source": source_label,
    }


def _find_col(data: pd.DataFrame, candidates: list[str]) -> str | None:
    """在 DataFrame 列名中找第一个包含候选关键词的列。"""
    for candidate in candidates:
        for col in data.columns:
            if candidate in str(col):
                return col
    return None
