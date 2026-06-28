"""证券代码真实性校验。

在新增持仓、发起诊断前，先确认用户输入的代码确实存在行情或净值，避免拿一个
不存在的代码跑完整条（很慢的）诊断流水线，最后只得到一份"数据缺失"的空报告。

只做存在性校验（联网拉一次实时行情 / 最新净值）；6 位数字的格式校验由调用方
负责（pydantic 的 field_validator / pattern，或端点内联判断），这样格式错误
能直接 422 返回、不耗网络。
"""

from __future__ import annotations

from app.data_source.fund import get_fund_nav
from app.data_source.stock import get_stock_realtime


def verify_security_code(code: str, ptype: str) -> tuple[str | None, str | None]:
    """校验证券代码真实性，返回 ``(名称, 错误信息)``。成功时 err 为 None。

    股票：用实时行情验证代码存在并返回行情名（供前端回填）；
    基金：用最新净值验证代码存在，名称返回 None（拉全市场基金列表取名太重，
    基金名由用户填写）。

    与 ``app/api/portfolio.py`` 原 ``_verify_code_exists`` 行为一致，抽出来供
    portfolio 与 diagnosis 两个路由共用，避免两份校验逻辑漂移。
    """
    code = (code or "").strip()
    if ptype == "stock":
        try:
            realtime = get_stock_realtime(code)
        except Exception as exc:
            return None, f"无法验证股票 {code}：{exc}"
        if not realtime.get("latest_price"):
            return None, f"未找到股票 {code} 的行情，请检查代码"
        return realtime.get("name"), None

    if ptype == "fund":
        try:
            nav = get_fund_nav(code)
        except Exception as exc:
            return None, f"无法验证基金 {code}：{exc}"
        if nav.get("unit_nav") is None and nav.get("accumulated_nav") is None:
            return None, f"未找到基金 {code} 的净值，请检查代码"
        return None, None

    return None, f"不支持的类型：{ptype}"
