"""基于 baostock 构建的股票行情数据封装。"""

from __future__ import annotations

import atexit
import contextlib
import io
import logging
import threading
import time
from datetime import date, timedelta
from typing import Any, Callable, TypedDict

import baostock as bs

from app.data_source.common import to_float

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baostock 会话 —— 首次查询时惰性登录，退出时登出。
# ---------------------------------------------------------------------------
# baostock 的 login 会启动后台接收线程；移动网络下连不上行情服务器时，该线程
# 会在底层文件描述符上直接输出 WinError 10057 / "接收数据异常"（绕过 Python 的
# sys.stderr 重定向，无法靠 redirect_stdout 拦截）。若在模块导入时登录，每次
# 启动都会刷这些吓人的日志，看起来像后端启动失败。改为惰性登录后，启动完全不
# 触发 baostock 连接；真正查询时若连不上，则回退 akshare 新浪源（见各调用点）。
def _bs_do_login() -> bool:
    """执行一次 baostock 登录，抑制其连接错误输出。返回是否登录成功。"""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            lg = bs.login()
    except Exception as exc:
        logger.warning("baostock 登录失败，后续查询将回退 akshare 新浪源: %s", exc)
        return False
    # 连不上行情服务器时 login 仍可能返回，但 error_code 非 0
    if getattr(lg, "error_code", "0") != "0":
        logger.warning("baostock 登录失败(error_code=%s)，后续查询回退 akshare", getattr(lg, "error_code", "?"))
        return False
    return True


_bs_logged_in = False       # 是否已完成登录探测
_bs_available = False       # baostock 当前是否可用（移动网络下常连不上）
_bs_probe_started = False   # 后台探测线程是否已启动
_bs_login_lock = threading.Lock()


def _probe_bs_login() -> None:
    """后台执行 baostock 登录探测，更新可用标志。

    baostock login 在连不上行情服务器时要等 TCP 超时（移动网络下约 20s）
    才返回。同步 login 会把首次请求阻塞 20s+。改为后台探测：首次请求
    立即返回 False 走 akshare，探测完成后更新标志供后续请求使用。
    """
    global _bs_logged_in, _bs_available
    avail = _bs_do_login()
    with _bs_login_lock:
        _bs_available = avail
        _bs_logged_in = True


def _ensure_bs_login() -> bool:
    """返回 baostock 是否可用。首次调用启动后台探测，不阻塞当前请求。

    探测完成前返回 False（走 akshare）；完成后有线网络恢复 baostock，
    移动网络维持 False。登录失败后 ``_run_bs_query`` 直接抛错让上层回退
    akshare，省去每行持仓的 baostock 试错 + relogin 开销。
    """
    global _bs_probe_started
    if _bs_logged_in:
        return _bs_available
    if not _bs_probe_started:
        with _bs_login_lock:
            if not _bs_probe_started:
                _bs_probe_started = True
                threading.Thread(target=_probe_bs_login, daemon=True).start()
    return _bs_available


def _safe_bs_logout() -> None:
    """退出时若曾登录则登出，抑制其错误输出。"""
    global _bs_logged_in
    if not _bs_logged_in:
        return
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bs.logout()
    except Exception:
        pass


atexit.register(_safe_bs_logout)


def _bs_relogin() -> None:
    """重新登录 baostock。

    长驻后端进程里，baostock 服务端会话会过期，表现为后续 query 返回
    ``error_code != "0"`` 且无数据。先 logout 再 login 即可恢复。
    """
    global _bs_available
    try:
        bs.logout()
    except Exception:
        # logout 失败不应阻塞重新登录。
        pass
    _bs_available = _bs_do_login()


def _run_bs_query(query_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """执行一次 baostock 查询，遇到会话类错误则重新登录后重试一次。

    baostock 的 query 在会话过期时返回非零 ``error_code``，此时直接重试无效，
    必须先重新登录。这里在第一次查询失败后自动 relogin 并重试一次，绝大多数
    会话过期场景都能自愈，调用方无需感知。

    注意：本包装只处理“快速返回错误”的情况，无法处理 baostock 服务端卡死
    （socket hang）——后者需要外层加超时，baostock 原生不支持。
    """
    if not _ensure_bs_login():
        raise RuntimeError("baostock 不可用，应回退 akshare")
    try:
        rs = query_fn(*args, **kwargs)
    except IndexError as exc:
        raise RuntimeError("baostock 查询内部 IndexError，应回退 akshare") from exc
    if getattr(rs, "error_code", "0") != "0":
        _bs_relogin()
        if not _bs_available:
            raise RuntimeError("baostock 重连失败，应回退 akshare")
        try:
            rs = query_fn(*args, **kwargs)
        except IndexError as exc:
            raise RuntimeError("baostock 重试查询内部 IndexError，应回退 akshare") from exc
    return rs


class StockRealtime(TypedDict):
    """单只股票的实时行情。"""

    code: str
    name: str | None
    latest_price: float | None
    change_percent: float | None
    volume: float | None
    pe_ratio: float | None
    pb_ratio: float | None


class StockKLineItem(TypedDict):
    """单日 K 线摘要，供技术面分析使用。"""

    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    change_percent: float | None


# ---------------------------------------------------------------------------
# 代码格式辅助 —— baostock 期望 "sh.600519" / "sz.000001"。
# ---------------------------------------------------------------------------

_SH_PREFIXES = ("60", "68")
_SZ_PREFIXES = ("00", "30")


def _to_bs_code(code: str) -> str:
    """把纯数字股票代码转换为 baostock 格式。"""
    code = code.strip()
    if code.startswith(("sh.", "sz.")):
        return code
    if any(code.startswith(p) for p in _SH_PREFIXES):
        return f"sh.{code}"
    if any(code.startswith(p) for p in _SZ_PREFIXES):
        return f"sz.{code}"
    return f"sz.{code}"


def _to_ak_code(code: str) -> str:
    """把纯数字股票代码转换为 akshare 新浪源格式（sz300308 / sh600519）。"""
    code = code.strip()
    if code.startswith(("sh", "sz")) and "." not in code:
        return code.lower()
    if any(code.startswith(p) for p in _SH_PREFIXES):
        return f"sh{code}"
    if any(code.startswith(p) for p in _SZ_PREFIXES):
        return f"sz{code}"
    return f"sz{code}"


# ---------------------------------------------------------------------------
# 核心数据访问。
# ---------------------------------------------------------------------------


def _fetch_k_data_row(bs_code: str, fields: str) -> list[str] | None:
    """获取 *bs_code* 最近一条日 K 线数据。

    查询一个滚动 7 天窗口，这样非交易日也能取到最后一个有数据的交易日。
    当代码不存在或窗口内无数据时返回 ``None``。
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=7)
    rs = _run_bs_query(
        bs.query_history_k_data_plus,
        bs_code,
        fields,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        frequency="d",
        adjustflag="2",  # 前复权
    )
    last_row = None
    while rs.error_code == "0" and rs.next():
        last_row = rs.get_row_data()
    return last_row


def _fetch_stock_name(bs_code: str) -> str | None:
    """从 baostock 查询股票的中文名称。"""
    rs = _run_bs_query(bs.query_stock_basic, code=bs_code)
    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        name = row[1] if len(row) > 1 else None
        return name if name else None
    return None


# 新浪全市场 spot 缓存：ak.stock_zh_a_spot() 一次拉全 A 股，多行持仓
# 共享同一份；带 60s TTL，避免每次 GET /api/portfolio 都重新拉取。
_ak_spot_cache: dict[str, Any] = {"df": None, "ts": 0.0}
_AK_SPOT_TTL = 60.0


def _get_ak_spot_df():
    """返回带 TTL 缓存的新浪全市场 spot DataFrame。

    当 akshare 调用失败时，若缓存中仍有旧数据（即使已过期），也降级返回
    旧数据避免整页不可用；仅在连缓存都没有时才抛错。
    """
    now = time.time()
    cached = _ak_spot_cache["df"]
    if cached is not None and now - _ak_spot_cache["ts"] < _AK_SPOT_TTL:
        return cached
    import akshare as ak

    try:
        df = ak.stock_zh_a_spot()
    except Exception as exc:
        # 网络波动下 ak 可能炸，若有旧缓存则降级返回旧数据
        if cached is not None:
            logger.warning("akshare stock_zh_a_spot() 失败，降级使用过期缓存(age=%.0fs): %s",
                           now - _ak_spot_cache["ts"], exc)
            return cached
        raise RuntimeError("akshare 新浪源全市场行情拉取失败") from exc

    _ak_spot_cache["df"] = df
    _ak_spot_cache["ts"] = now
    return df


def resolve_stock_code(keyword: str) -> tuple[str | None, str | None, list[dict[str, str]]]:
    """把用户输入的代码或名称解析成 6 位股票代码。

    返回 ``(code, name, candidates)``：
    - 输入是 6 位数字：直接当代码返回（code=输入, name=None, candidates=[]），
      名称留给后续行情接口补。
    - 输入是名称：在新浪全 A 股 spot 里按名称包含匹配。唯一匹配则 code 填上；
      多个匹配则 code=None、candidates 列出前 10 个供用户选择；无匹配则全 None。
    """
    kw = (keyword or "").strip()
    if not kw:
        return None, None, []
    if kw.isdigit() and len(kw) == 6:
        return kw, None, []

    df = _get_ak_spot_df()
    if df is None or df.empty:
        return None, None, []
    code_col = "代码" if "代码" in df.columns else df.columns[0]
    name_col = "名称" if "名称" in df.columns else None
    if name_col is None:
        return None, None, []

    matched = df[df[name_col].astype(str).str.contains(kw, na=False)]
    if matched.empty:
        return None, None, []

    candidates: list[dict[str, str]] = []
    for _, row in matched.head(10).iterrows():
        raw_code = str(row[code_col])
        # 新浪源代码列形如 "sz300308"，取后 6 位
        code = raw_code[-6:] if len(raw_code) >= 6 else raw_code
        candidates.append({"code": code, "name": str(row[name_col])})

    if len(candidates) == 1:
        return candidates[0]["code"], candidates[0]["name"], candidates
    return None, None, candidates


def _realtime_via_akshare(code: str) -> dict[str, Any]:
    """akshare 新浪源实时行情：baostock 不可用时的兜底。

    新浪 spot 不带 PE/PB，故估值字段返回 None，由下游标注缺口。
    走 sina 线路，移动网络下比东方财富源更稳。
    """
    try:
        df = _get_ak_spot_df()
        if df is None or df.empty:
            raise ValueError(f"新浪源未找到股票 {code} 的行情数据")
        # 新浪源代码列形如 "sz300308"
        code_col = "代码" if "代码" in df.columns else df.columns[0]
        matched = df[df[code_col].astype(str).str.endswith(code)]
        if matched.empty:
            raise ValueError(f"新浪源未匹配到股票 {code}")
        row = matched.iloc[0]

        def _get(col: str) -> float | None:
            return to_float(row[col]) if col in row else None

        return {
            "code": code,
            "name": row.get("名称") if "名称" in row else None,
            "latest_price": _get("最新价"),
            "change_percent": _get("涨跌幅"),
            "volume": _get("成交量"),
            "pe_ratio": None,  # 新浪 spot 无 PE
            "pb_ratio": None,  # 新浪 spot 无 PB
        }
    except Exception as exc:
        raise RuntimeError(f"akshare 新浪源实时行情获取失败(code={code}): {exc}") from exc


def get_stock_realtime(code: str) -> StockRealtime:
    """获取单只 A 股的最新日报行情。

    优先 baostock；baostock 不可用（移动网络常对其连接重置）时回退到
    akshare 新浪源。非交易日会自动返回最近一个交易日的数据。
    """
    try:
        bs_code = _to_bs_code(code)
        row = _fetch_k_data_row(
            bs_code,
            "date,code,close,pctChg,volume,peTTM,pbMRQ",
        )
        if row is not None:
            name = _fetch_stock_name(bs_code)
            return {
                "code": code,
                "name": name,
                "latest_price": to_float(row[2]) if len(row) > 2 else None,
                "change_percent": to_float(row[3]) if len(row) > 3 else None,
                "volume": to_float(row[4]) if len(row) > 4 else None,
                "pe_ratio": to_float(row[5]) if len(row) > 5 else None,
                "pb_ratio": to_float(row[6]) if len(row) > 6 else None,
            }
        logger.warning("baostock 实时行情返回空，回退 akshare 新浪源")
    except Exception as exc:
        logger.warning("baostock 实时行情取数失败，回退 akshare 新浪源: %s", exc)

    return _realtime_via_akshare(code)


# ---------------------------------------------------------------------------
# 扩展数据：历史 K 线 + 技术指标、详细财务、行业分类。
# 这些函数为深度诊断提供更丰富的上下文，均基于 baostock 已有接口，
# 不引入新的数据源依赖。
# ---------------------------------------------------------------------------


def _fetch_latest_annual(
    bs_code: str,
    query: Callable[..., Any],
    min_len: int,
) -> list[str] | None:
    """向前回溯最多 3 年，取最新一份非空年报数据行。

    不同财务接口（利润/资产负债/运营/成长/杜邦）共用这套回溯逻辑，
    抽出来避免到处复制粘贴同一段 year 循环。
    """
    today = date.today()
    for offset in range(3):
        year = today.year - offset
        rs = _run_bs_query(query, bs_code, year=year, quarter=4)
        last = None
        while rs.error_code == "0" and rs.next():
            last = rs.get_row_data()
        if last is not None and len(last) >= min_len:
            return last
    return None


def _to_yi(value: float | None) -> float | None:
    """把以“元”为单位的绝对值换算成“亿元”并保留两位小数。

    baostock 的净利润、营收等字段返回的是元，直接塞进 Prompt 数字太长
    不利于 LLM 解读，统一折算成亿元更接近研报口径。
    """
    if value is None:
        return None
    return round(value / 1e8, 2)


def _mean(values: list[float]) -> float | None:
    """对非空列表求均值，空列表返回 None。"""
    if not values:
        return None
    return sum(values) / len(values)


def _compute_ma(closes: list[float], window: int) -> float | None:
    """计算最近 ``window`` 期的简单移动平均。数据不足时返回 None。"""
    if len(closes) < window:
        return None
    return round(sum(closes[-window:]) / window, 2)


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """计算 RSI(N)，使用标准的简单平均法。

    数据不足 period+1 个时返回 None。RSI 衡量近期上涨幅度占涨跌总幅度的
    比例，是诊断超买/超卖状态的核心指标之一。
    """
    if len(closes) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    # 取最后 period+1 个收盘价，计算 period 个日变化。
    tail = closes[-(period + 1):]
    for i in range(1, len(tail)):
        diff = tail[i] - tail[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def _build_kline_result(code: str, ohlc: list[dict[str, Any]]) -> dict[str, Any]:
    """从归一化的 OHLC 行计算技术指标，baostock 与 akshare 路径共用。

    ``ohlc`` 每行含 ``date/open/high/low/close/volume/turnover/change_percent``。
    返回结构与原 ``get_stock_history_kline`` 一致，保证下游 Prompt 无感知。
    """
    if not ohlc:
        raise ValueError(f"股票 {code} 无有效K线数据")

    closes = [r.get("close") for r in ohlc]
    volumes = [r.get("volume") for r in ohlc]
    latest = ohlc[-1]

    recent_20_closes = [c for c in closes[-20:] if c is not None]
    recent_high_20 = max(recent_20_closes) if recent_20_closes else None
    recent_low_20 = min(recent_20_closes) if recent_20_closes else None

    latest_volume = latest.get("volume")
    avg_volume_20 = _mean([v for v in volumes[-20:] if v is not None])
    volume_ratio = (
        round(latest_volume / avg_volume_20, 2)
        if latest_volume and avg_volume_20
        else None
    )

    ma5 = _compute_ma(closes, 5)
    ma20 = _compute_ma(closes, 20)
    if ma5 is not None and ma20 is not None:
        if ma5 > ma20 * 1.01:
            trend = "短期多头（MA5 在 MA20 之上）"
        elif ma5 < ma20 * 0.99:
            trend = "短期空头（MA5 在 MA20 之下）"
        else:
            trend = "短期震荡（MA5 与 MA20 缠绕）"
    else:
        trend = "数据不足，无法判断趋势"

    recent_kline: list[StockKLineItem] = [
        {
            "date": str(r.get("date")),
            "open": r.get("open"),
            "high": r.get("high"),
            "low": r.get("low"),
            "close": r.get("close"),
            "volume": r.get("volume"),
            "change_percent": r.get("change_percent"),
        }
        for r in ohlc[-10:]
    ]

    return {
        "code": code,
        "latest_date": str(latest.get("date")),
        "latest_price": latest.get("close"),
        "ma5": ma5,
        "ma10": _compute_ma(closes, 10),
        "ma20": ma20,
        "ma60": _compute_ma(closes, 60),
        "rsi14": _compute_rsi(closes, 14),
        "recent_high_20": round(recent_high_20, 2) if recent_high_20 else None,
        "recent_low_20": round(recent_low_20, 2) if recent_low_20 else None,
        "support": round(recent_low_20, 2) if recent_low_20 else None,
        "resistance": round(recent_high_20, 2) if recent_high_20 else None,
        "avg_volume_20": round(avg_volume_20, 2) if avg_volume_20 else None,
        "latest_volume": latest_volume,
        "volume_ratio": volume_ratio,
        "turnover_rate_latest": latest.get("turnover"),
        "trend": trend,
        "recent_kline": recent_kline,
    }


def _kline_via_baostock(code: str, days: int) -> list[dict[str, Any]]:
    """baostock 路径：取近 days 日 K 线，归一化为 OHLC 行。"""
    bs_code = _to_bs_code(code)
    end_date = date.today()
    start_date = end_date - timedelta(days=int(days * 1.5))
    rs = _run_bs_query(
        bs.query_history_k_data_plus,
        bs_code,
        "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        frequency="d",
        adjustflag="2",  # 前复权
    )
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    # 截取最近 days 个交易日，裁掉 close 为空的停牌行。
    rows = [r for r in rows[-days:] if to_float(r[4]) is not None]
    return [
        {
            "date": r[0],
            "open": to_float(r[1]),
            "high": to_float(r[2]),
            "low": to_float(r[3]),
            "close": to_float(r[4]),
            "volume": to_float(r[5]),
            "turnover": to_float(r[7]),
            "change_percent": to_float(r[8]),
        }
        for r in rows
    ]


def _kline_via_akshare(code: str, days: int) -> list[dict[str, Any]]:
    """akshare 新浪源路径：baostock 不可用时的兜底，走不同线路更稳。

    新浪日 K 不带 PE/PB，turnover 字段名不同，这里统一归一化。
    """
    import akshare as ak

    ak_code = _to_ak_code(code)
    df = ak.stock_zh_a_daily(symbol=ak_code, adjust="qfq")  # 前复权
    if df is None or df.empty:
        return []
    df = df.tail(days)
    ohlc: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        close = to_float(row.get("close"))
        if close is None:
            continue
        ohlc.append(
            {
                "date": str(row.get("date")),
                "open": to_float(row.get("open")),
                "high": to_float(row.get("high")),
                "low": to_float(row.get("low")),
                "close": close,
                "volume": to_float(row.get("volume")),
                "turnover": to_float(row.get("turnover")),
                "change_percent": None,  # 新浪日 K 无涨跌幅字段，留空
            }
        )
    return ohlc


def get_stock_history_kline(code: str, days: int = 120) -> dict[str, Any]:
    """获取近 ``days`` 个交易日的日 K 线并派生技术指标。

    优先 baostock；baostock 不可用（移动网络常对其连接重置）时回退到
    akshare 新浪源。两者数据都归一化后走同一套指标计算，下游无感知。
    返回结构包含均线、RSI、支撑/压力位、量比、趋势与最近 10 日 K 线摘要。
    """
    try:
        ohlc = _kline_via_baostock(code, days)
        if ohlc:
            return _build_kline_result(code, ohlc)
    except Exception as exc:
        logger.warning("baostock K线取数失败，回退 akshare 新浪源: %s", exc)

    ohlc = _kline_via_akshare(code, days)
    return _build_kline_result(code, ohlc)


def get_stock_financial_detail(code: str) -> dict[str, Any]:
    """获取单只 A 股的详细财务指标，覆盖盈利、成长、偿债、运营、杜邦五个维度。

    聚合利润表、资产负债表、运营能力、杜邦分析与成长能力，给深度基本面分析
    提供更完整的底料。所有绝对值字段（净利润、营收）已折算成亿元，便于 LLM 直接引用。
    """
    bs_code = _to_bs_code(code)

    result: dict[str, Any] = {
        "code": code,
        "report_date": None,
        # 盈利能力
        "roe": None,
        "gross_margin": None,
        "net_profit_margin": None,
        "net_profit_yi": None,
        "revenue_yi": None,
        "eps": None,
        # 成长能力
        "net_profit_growth": None,
        "asset_growth": None,
        "equity_growth": None,
        # 偿债与资本结构
        "current_ratio": None,
        "quick_ratio": None,
        "debt_ratio": None,
        # 运营能力
        "asset_turnover": None,
        # 杜邦分解
        "dupont_net_margin": None,
        "dupont_asset_turnover": None,
        "dupont_equity_multiplier": None,
    }

    # 利润表：roeAvg, npMargin, gpMargin, netProfit, epsTTM, MBRevenue
    profit = _fetch_latest_annual(bs_code, bs.query_profit_data, 7)
    if profit is not None:
        result["report_date"] = str(profit[2]) if len(profit) > 2 else None
        result["roe"] = to_float(profit[3])
        result["net_profit_margin"] = to_float(profit[4])
        result["gross_margin"] = to_float(profit[5])
        result["net_profit_yi"] = _to_yi(to_float(profit[6]))
        result["eps"] = to_float(profit[7]) if len(profit) > 7 else None
        result["revenue_yi"] = _to_yi(to_float(profit[8])) if len(profit) > 8 else None

    # 成长能力：YOYEquity, YOYAsset, YOYNI
    growth = _fetch_latest_annual(bs_code, bs.query_growth_data, 6)
    if growth is not None:
        result["equity_growth"] = to_float(growth[3])
        result["asset_growth"] = to_float(growth[4])
        result["net_profit_growth"] = to_float(growth[5])

    # 资产负债：currentRatio, quickRatio, liabilityToAsset
    balance = _fetch_latest_annual(bs_code, bs.query_balance_data, 6)
    if balance is not None:
        result["current_ratio"] = to_float(balance[3])
        result["quick_ratio"] = to_float(balance[4])
        result["debt_ratio"] = to_float(balance[7]) if len(balance) > 7 else None

    # 运营能力：AssetTurnRatio 在最后一列
    operation = _fetch_latest_annual(bs_code, bs.query_operation_data, 6)
    if operation is not None:
        result["asset_turnover"] = to_float(operation[8]) if len(operation) > 8 else None

    # 杜邦分解：dupontROE 同 ROE，这里取净利率/周转率/权益乘数三项拆解
    dupont = _fetch_latest_annual(bs_code, bs.query_dupont_data, 6)
    if dupont is not None:
        result["dupont_net_margin"] = to_float(dupont[7]) if len(dupont) > 7 else None
        result["dupont_asset_turnover"] = to_float(dupont[5]) if len(dupont) > 5 else None
        result["dupont_equity_multiplier"] = to_float(dupont[4]) if len(dupont) > 4 else None

    return result


def get_stock_industry(code: str) -> dict[str, Any]:
    """获取股票的所属行业与证监会行业分类。

    行业信息用于在诊断中补充公司业务背景、做同业对比，也方便 LLM 结合
    自身对该行业的认知展开分析。
    """
    bs_code = _to_bs_code(code)
    rs = _run_bs_query(bs.query_stock_industry, code=bs_code)
    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        # 字段：updateDate, code, code_name, industry, industryClassification
        return {
            "code": code,
            "industry": row[3] if len(row) > 3 else None,
            "industry_classification": row[4] if len(row) > 4 else None,
        }
    return {
        "code": code,
        "industry": None,
        "industry_classification": None,
    }
