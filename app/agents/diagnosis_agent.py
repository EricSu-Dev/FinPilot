"""LangGraph diagnosis agent for stocks and funds.

This module is the core of FinPilot's first-tier reasoning flow. It does four
things in order:

1. Fetch a rich data bundle for a diagnosis (行情/财务/K线技术指标/行业/联网搜索
   for stocks; 基本信息/净值/业绩/持仓/费率/行业配置/联网搜索 for funds).
2. Ask the LLM to turn that data into a structured, multi-dimensional analysis —
   stocks use a 7-dimension sell-side note frame, funds use a 5-segment Ant
   Fortune-aligned frame (核心诊断→业绩波动→持仓→费率→投资建议).
3. Run a safety pass that locks the disclaimer to a fixed project-wide wording
   and audits (logs) any direct trade instruction wording rather than silently
   rewriting it, so the practical-advice / investment-advice section can still
   carry concrete risk thresholds such as stop-loss levels and position caps.

The design intentionally keeps every intermediate value inside the LangGraph
state object so the graph is easy to inspect in notebooks and easy to extend
later without introducing hidden globals.
"""

from __future__ import annotations

import json
import logging
import re
from pprint import pformat
from typing import Annotated, Any, Literal, Optional, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.llm import llm
from app.tools.fund_tools import (
    fund_fee_tool,
    fund_industry_allocation_tool,
    fund_info_tool,
    fund_nav_tool,
    fund_performance_tool,
    fund_valuation_tool,
)
from app.tools.stock_tools import (
    stock_financial_detail_tool,
    stock_history_kline_tool,
    stock_industry_tool,
    stock_realtime_tool,
)
from app.tools.web_search_tool import web_search_tool

logger = logging.getLogger(__name__)

DEFAULT_DISCLAIMER = (
    "本结果仅用于信息整理和数据解读，不构成投资建议、收益承诺或交易指令。"
)


# ---------------------------------------------------------------------------
# 股票诊断结果（7 维度，保持不变）
# ---------------------------------------------------------------------------


class DiagnosisResult(BaseModel):
    """Structured diagnosis result returned by the LLM for stocks.

    The model intentionally mirrors the dimensions of a professional diagnostic
    note: a business overview, four analysis angles (fundamental / technical /
    capital flow / risk), a synthesis, and concrete practical reference split
    by investor type. Every text field is free-form so the LLM can lay out
    reasoning, but ``risk_level`` and ``disclaimer`` stay constrained.
    """

    name: str = Field(default="", description="标的名称，从已抓取数据中提取，不依赖 LLM 输出。")
    business_overview: str = Field(
        description="公司业务概述：主营业务、产业链位置、行业地位、核心竞争力、主要客户与市场。"
    )
    fundamental_analysis: str = Field(
        description="基本面分析：盈利能力、成长性、财务健康、估值水平、杜邦拆解。"
    )
    technical_analysis: str = Field(
        description="技术面分析：价格位置、均线排列、RSI、支撑压力位、量价关系、趋势。"
    )
    capital_flow_analysis: str = Field(
        description="资金面分析：基于量价换手推断 + 联网搜索到的资金/机构动向。"
    )
    risk_analysis: str = Field(
        description="风险提示：具体风险项（估值/行业/政策/公司特定/地缘），结合最新事件。"
    )
    comprehensive_diagnosis: str = Field(
        description="综合诊断：核心投资逻辑、风险评级依据、标的定位（价值/成长/周期/防御）。"
    )
    practical_advice: str = Field(
        description="实操参考：分持仓者/观望者/短线三类，给出观察指标与风险阈值。"
    )
    one_sentence_summary: str = Field(description="一句话总结核心结论。")
    risk_level: Literal["低", "中", "高"] = Field(description="综合风险评级。")
    disclaimer: str = Field(description="固定免责声明文本，必须保持中性且不包含投资建议。")


# ---------------------------------------------------------------------------
# 基金诊断结果（蚂蚁财富 5 段结构，独立字段）
# ---------------------------------------------------------------------------


class FundDiagnosisResult(BaseModel):
    """Structured diagnosis result for funds, aligned with Ant Fortune layout.

    Ant Fortune's fund diagnosis uses 5 segments: core diagnosis at the top,
    then performance & volatility, holdings analysis, fee & trading rules,
    investment advice. We add a separate risk_analysis field to ensure thorough
    coverage. ``risk_level`` and ``disclaimer`` stay constrained.
    """

    name: str = Field(default="", description="基金名称，从已抓取数据中提取，不依赖 LLM 输出。")
    core_diagnosis: str = Field(
        description="核心诊断：一句话定性该基金的风格与定位（如'高弹性进攻品种'、'稳健防守型'）。"
    )
    performance_volatility: str = Field(
        description="业绩与波动：多周期收益率（今年来/近1周/1月/3月/1年/3年）、"
        "最新单日涨跌、年化波动率、最大回撤、风险等级。"
    )
    holdings_analysis: str = Field(
        description="持仓分析：核心赛道/投资主题、十大重仓股（代码/名称/权重）、"
        "行业集中度、风格特征、持仓披露滞后提示。"
    )
    fee_and_trading: str = Field(
        description="费率与交易规则：申购费率、赎回费率(分档)、管理费/托管费/销售服务费、"
        "最低申购金额、持有期限建议（如'持有30天以上免赎回费'）。"
    )
    risk_analysis: str = Field(
        description="风险提示：回撤/波动风险、持仓集中风险、风格漂移、披露滞后风险、地缘/政策风险。"
    )
    investment_advice: str = Field(
        description="投资建议与观察点：基金定位（卫星/底仓）、入场时机（左侧/右侧）、"
        "定投vs一次性建议、同赛道对比选择、观察指标。"
        "注意：不要说'您目前未持有该基金'（系统暂无持仓功能）。"
    )
    risk_level: Literal["低", "中", "高"] = Field(description="综合风险评级。")
    disclaimer: str = Field(description="固定免责声明文本，必须保持中性且不包含投资建议。")


def _append_messages(existing: list[BaseMessage] | None, new_messages: list[BaseMessage] | None) -> list[BaseMessage]:
    """Merge two message lists for LangGraph state updates."""
    return [*(existing or []), *(new_messages or [])]


class DiagnosisState(TypedDict, total=False):
    """LangGraph state passed between diagnosis nodes."""

    target_code: str
    target_type: Literal["stock", "fund"]
    # 股票专用数据
    realtime_data: dict[str, Any]
    financial_detail_data: dict[str, Any]
    kline_data: dict[str, Any]
    industry_data: dict[str, Any]
    # 基金专用数据
    fund_basic_info: dict[str, Any]
    fund_nav: dict[str, Any]
    fund_valuation: dict[str, Any]
    fund_performance_data: dict[str, Any]
    fund_fee_data: dict[str, Any]
    fund_industry_allocation_data: dict[str, Any]
    # 共用：联网搜索摘要
    web_search_data: dict[str, Any]
    # 结果（股票 DiagnosisResult 或基金 FundDiagnosisResult）
    analysis_result: Optional[Union[DiagnosisResult, FundDiagnosisResult]]
    messages: Annotated[list[BaseMessage], _append_messages]
    error: Optional[str]


def _debug_dump(title: str, payload: Any, debug: bool) -> None:
    """Print a compact, readable snapshot of a node's input or output."""
    if not debug:
        return
    print(f"\n[{title}]")
    print(pformat(payload, width=120, sort_dicts=False))


def _serialize_data(payload: Any) -> str:
    """Serialize nested dictionaries and model objects into stable prompt text."""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _kline_prompt_summary(kline: dict[str, Any]) -> dict[str, Any]:
    """Trim the raw kline payload for the prompt.

    The full kline dict carries a 10-day ``recent_kline`` list plus a handful of
    derived indicators. We pass indicators verbatim but only keep the last 5
    days of raw candles to keep token usage bounded while still showing the
    LLM the most recent price action.
    """
    if not kline:
        return {}
    summary = {k: v for k, v in kline.items() if k != "recent_kline"}
    summary["recent_kline_last_5"] = kline.get("recent_kline", [])[-5:]
    return summary


# ---------------------------------------------------------------------------
# 股票 Prompt（保持不变）
# ---------------------------------------------------------------------------


def _build_stock_prompt(state: DiagnosisState) -> str:
    """Format stock state into a multi-dimensional diagnosis prompt."""
    target_code = state.get("target_code", "")
    stock_name = state.get("realtime_data", {}).get("name") or ""
    kline = state.get("kline_data", {}) or {}
    financial = state.get("financial_detail_data", {}) or {}

    context = {
        "标的信息": {
            "代码": target_code,
            "名称": stock_name,
            "行业": state.get("industry_data", {}),
        },
        "数据基准日期": {
            "行情日期": kline.get("latest_date") or state.get("realtime_data", {}).get("latest_date"),
            "财报报告期": financial.get("report_date"),
            "说明": "行情日期为最近一个交易日；财报为最近一期年报，存在披露滞后。",
        },
        "实时行情": state.get("realtime_data", {}),
        "详细财务": financial,
        "技术指标与近期K线": _kline_prompt_summary(kline),
        "联网搜索摘要": state.get("web_search_data", {}),
    }

    json_schema = """{
  "business_overview": "公司业务概述",
  "fundamental_analysis": "基本面分析",
  "technical_analysis": "技术面分析",
  "capital_flow_analysis": "资金面分析",
  "risk_analysis": "风险提示",
  "comprehensive_diagnosis": "综合诊断",
  "practical_advice": "实操参考",
  "one_sentence_summary": "一句话总结",
  "risk_level": "低|中|高",
  "disclaimer": "固定免责声明"
}"""

    return (
        "你是 FinPilot 的金融诊断分析师。基于下方结构化数据 + 公开搜索摘要 + 你的金融知识，"
        "对该标的做多维度深度诊断。风格：专业、客观、结构化、数据驱动。\n\n"
        "【数据说明】\n"
        "- \"实时行情/详细财务/技术指标\"来自交易所与上市公司公开数据，权威可信。\n"
        "- \"联网搜索摘要\"来自公开网页，可能含噪声，仅作背景参考，引用时需谨慎并标注来源性质。\n"
        "- 你的训练知识可补充公司业务背景、行业格局、已知风险事件，但需与当日数据区分措辞。\n\n"
        "【分析框架】严格按以下 7 个维度输出，每个维度都要有实质内容：\n\n"
        "1. business_overview（公司业务概述）\n"
        "   - 主营业务、产品/服务、所处产业链位置；行业地位、核心竞争力、主要客户与市场。\n"
        "   - 可结合你的知识补充，措辞用\"公开信息表明/行业常识\"，不要冒充当日数据。\n\n"
        "2. fundamental_analysis（基本面分析）\n"
        "   - 盈利能力：ROE、毛利率、净利率水平，并做行业对比判断强弱。\n"
        "   - 成长性：净利润增速、营收规模与增速、资产扩张速度。\n"
        "   - 财务健康：资产负债率、流动/速动比率、现金流质量（CFOToNP 等）。\n"
        "   - 估值：PE/PB 绝对水平，结合成长性判断估值是否合理（高估/合理/低估）。\n"
        "   - 杜邦：指出 ROE 由净利率×周转率×杠杆哪一项主导。\n"
        "   - 数据异常校验：对偏离行业常识的极端值（如资产负债率<5%、ROE>50%、PE>100、净利增速>100%），"
        "在该值后标注【数据待复核】并简述为何反常（是真实现象还是数据源可能出错），不要默认采信。\n\n"
        "3. technical_analysis（技术面分析）\n"
        "   - 价格位置：最新价相对 MA5/MA10/MA20/MA60 的位置（上方/下方/缠绕）。\n"
        "   - 趋势：多头/空头/震荡，依据均线排列与 trend 字段。\n"
        "   - 超买超卖：RSI14 数值解读（>70 偏超买，<30 偏超卖）。\n"
        "   - 支撑/压力位必须分层标注：短期支撑/压力（近5日高低、MA5）、"
        "中期支撑/压力（MA20、近20日高低 support/resistance）、终极支撑（MA60 或前低）。"
        "不要把不同层级的点位混为一谈。\n"
        "   - 量价：量比、换手率、近期成交量变化，判断资金参与度。\n\n"
        "4. capital_flow_analysis（资金面分析）\n"
        "   - 从量价、换手率、量比推断资金活跃度与方向。\n"
        "   - 结合搜索摘要中的资金流向、机构动向、融资余额等信息。\n"
        "   - 必须明确数据缺口：baostock 不提供主力资金流明细，相关判断属推断。\n\n"
        "5. risk_analysis（风险提示）\n"
        "   - 列具体风险项，不要泛泛而谈：估值风险、行业/政策风险、公司特定风险"
        "（客户集中、减持、诉讼等）、地缘/宏观风险。\n"
        "   - 结合搜索摘要中的最新事件（如出口管制清单、机构评级变动、重大公告）。\n\n"
        "6. comprehensive_diagnosis（综合诊断）\n"
        "   - 多维度综合，给出核心投资逻辑；明确风险评级（低/中/高）的依据。\n"
        "   - 定位标的类型：价值/成长/周期/防御，适合哪类风险偏好的投资者。\n\n"
        "7. practical_advice（实操参考）\n"
        "   - 分三类：持仓者 / 观望者 / 短线博弈者。\n"
        "   - 给出可执行的观察指标与风险阈值：支撑位止损、仓位上限、入场区间、触发条件。\n"
        "   - 短线博弈策略必须计算并标注盈亏比（目标位收益 vs 止损位风险），"
        "盈亏比<1.5 的策略需显式提示\"性价比偏低\"。\n"
        "   - 措辞强调是\"风险控制参考\"而非交易指令。\n\n"
        "one_sentence_summary：一句话概括核心结论（可含控仓/止损/不追高等风险控制要点）。\n"
        "risk_level：只从 低/中/高 中选一个。\n"
        f"disclaimer：必须原样输出：{DEFAULT_DISCLAIMER}\n\n"
        "【数据质量与溯源要求】\n"
        "1. 所有定性判断（PEG 合理与否、估值高低、盈利强弱）必须附计算过程或判断基准，"
        "禁止只用\"尚可/偏高/合理\"这类无标准的主观词。例：PE97.89、增速115.57%→PEG≈0.85<1，"
        "若明年增速降至50%则PEG升至约1.96，估值将承压。\n"
        "2. 引用结构化数据时标注数据基准日期与来源（行情日期、财报报告期、baostock/搜索摘要）。\n"
        "3. 数据缺口必须具体到指标名与来源：写明是哪个指标、来自哪个接口/报表未返回"
        "（如\"经营性现金流净额未由 baostock cash_flow 接口提供\"），不要只写\"现金流数据缺失\"。\n"
        "4. 极端财务值按框架第2条标注【数据待复核】。\n\n"
        "【写作要求】\n"
        "1. 用具体数字支撑观点，不写空话；数据缺口要明确指出，绝不编造数字。\n"
        "2. 区分\"数据显示\"（结构化数据）与\"公开信息表明/行业常识\"（你的知识或搜索摘要）。\n"
        "3. 风险提示要具体到该公司，不要套话；实操参考要给阈值和区间，而非\"建议关注\"这种废话。\n"
        "4. 整体语气专业冷静，不渲染情绪。\n\n"
        "【输出格式】只输出以下 JSON，不要 markdown 代码块、不要任何其他文字：\n"
        f"{json_schema}\n\n"
        "【诊断数据】\n"
        f"{_serialize_data(context)}"
    )


# ---------------------------------------------------------------------------
# 基金 Prompt（蚂蚁财富 5 段结构）
# ---------------------------------------------------------------------------


def _build_fund_prompt(state: DiagnosisState) -> str:
    """Format fund state into a 5-segment Ant Fortune-aligned diagnosis prompt."""
    target_code = state.get("target_code", "")
    fund_name = state.get("fund_basic_info", {}).get("name") or ""

    context = {
        "标的信息": {
            "代码": target_code,
            "名称": fund_name,
            "类型": "基金",
        },
        "数据基准日期": {
            "净值日期": state.get("fund_nav", {}).get("nav_date") or state.get("fund_performance_data", {}).get("nav_date"),
            "持仓披露期": state.get("fund_valuation", {}).get("data_cutoff_date"),
            "费率来源": state.get("fund_fee_data", {}).get("data_source"),
            "说明": "净值日期为最近一个交易日；持仓基于季报，存在披露滞后；费率来自基金公开资料。",
        },
        "基金基本信息": state.get("fund_basic_info", {}),
        "最新净值": state.get("fund_nav", {}),
        "业绩与波动": state.get("fund_performance_data", {}),
        "持仓估值估算": state.get("fund_valuation", {}),
        "行业配置": state.get("fund_industry_allocation_data", {}),
        "费率与交易规则": state.get("fund_fee_data", {}),
        "联网搜索摘要": state.get("web_search_data", {}),
    }

    json_schema = """{
  "core_diagnosis": "核心诊断：一句话定性该基金的风格与定位",
  "performance_volatility": "业绩与波动：多周期收益、波动率、回撤、风险等级",
  "holdings_analysis": "持仓分析：核心赛道、十大重仓股、行业集中度、风格特征",
  "fee_and_trading": "费率与交易规则：申购/赎回/管理费分档、持有期限建议",
  "risk_analysis": "风险提示：回撤/集中度/风格漂移/披露滞后/地缘风险",
  "investment_advice": "投资建议与观察点：定位/入场时机/对比选择/定投建议",
  "risk_level": "低|中|高",
  "disclaimer": "固定免责声明"
}"""

    return (
        "你是 FinPilot 的基金诊断分析师。基于下方结构化数据 + 公开搜索摘要 + 你的金融知识，"
        "对该基金做多维度深度诊断。风格：专业、客观、结构化、数据驱动。\n\n"
        "【输出结构】严格按蚂蚁财富 5 段结构输出，核心诊断置顶：\n\n"
        "1. core_diagnosis（核心诊断）\n"
        "   - 一句话定性该基金的风格与定位（如'高弹性进攻品种'、'稳健防守型'、"
        "'周期跟踪型'、'红利防守型'）。\n"
        "   - 必须点明核心赛道/投资主题（如'AI算力硬件（光模块/PCB/通信设备）'、"
        "'消费医药'、'全球科技'、'高股息防御'），这是该基金最关键的识别信息。\n"
        "   - 明确该基金属于哪类风险偏好的投资者（适合风险承受能力较强/适中/较弱的投资者）。\n\n"
        "2. performance_volatility（业绩与波动）\n"
        "   - 用表格格式列出多周期收益率：今年来、近1周、近1月、近3月、近6月、近1年、近3年、"
        "成立来，每行标注 维度 / 关键数据 / 评价（如'跑赢多数同类'、'回撤明显'）。\n"
        "   - 最新单日涨跌幅、年化波动率、最大回撤。\n"
        "   - 风险等级（低/中低/中/中高/高）及其依据。\n"
        "   - 对长期趋势的整体定性（如'长期趋势向上'、'震荡反复'）。\n\n"
        "3. holdings_analysis（持仓分析）\n"
        "   - 核心赛道/投资主题（如'AI算力基础设施'、'消费医药'、'全球科技'）。\n"
        "   - 十大重仓股列表：代码、名称、权重（%）。\n"
        "   - 行业集中度：前3行业占比，持仓风格（进攻/防御/均衡）。\n"
        "   - 持仓披露滞后提示：基于季报，权重可能已变动。\n"
        "   - 对 QDII 基金，说明海外持仓与 A 股持仓的差异。\n\n"
        "4. fee_and_trading（费率与交易规则）\n"
        "   - 申购费率（前端/后端）、赎回费率（分档列出，如持有<7天1.5%、"
        "7-30天0.75%、30天以上0%）、管理费率、托管费率、销售服务费率。\n"
        "   - 最低申购金额。\n"
        "   - 持有期限建议（如'建议持有30天以上以免除赎回费'、'适合中长线配置'）。\n\n"
        "5. risk_analysis（风险提示）\n"
        "   - 回撤/波动风险：最大回撤幅度、波动率水平、与同类对比。\n"
        "   - 持仓集中风险：重仓股集中度、赛道单一风险。\n"
        "   - 风格漂移风险：持仓与基金宣称风格的偏离。\n"
        "   - 披露滞后风险：季报滞后2个月，持仓可能已大幅变化。\n"
        "   - 地缘/政策风险：对 QDII 基金需特别标注汇率风险、海外政策风险。\n\n"
        "6. investment_advice（投资建议与观察点）\n"
        "   - 定位清晰：该基金适合作为组合中的'卫星'仓位还是'底仓'，建议占比上限。\n"
        "   - 入场时机：左侧布局思路（回调分批建仓/定投）、右侧观察思路（等待企稳信号）。"
        "给出具体的观察指标（如纳斯达克止跌回升、净值回撤到支撑位）。\n"
        "   - 对比选择：若有同赛道基金，简要对比差异，提示重叠风险。\n"
        "   - 定投 vs 一次性：根据波动率推荐定投或一次性，附判断基准。\n"
        "   - 不要说'您目前未持有该基金'或'您持仓中已有某某基金'——系统暂无持仓功能。\n\n"
        "risk_level：只从 低/中/高 中选一个。\n"
        f"disclaimer：必须原样输出：{DEFAULT_DISCLAIMER}\n\n"
        "【数据质量与溯源要求】\n"
        "1. 所有定性判断必须附判断基准或计算过程，禁止只用\"尚可/偏高\"等无标准主观词。"
        "例：近1年+96.79%、同类平均+45%→跑赢同类约50个百分点，爆发力强但波动大。\n"
        "2. 引用数据标注净值日期、持仓披露期、费率来源与来源类型（akshare eastmoney/搜索摘要/行业常识）。\n"
        "3. 数据缺口具体到指标名与来源：写明是哪个指标、来自哪个接口未返回"
        "（如\"基金规模未由 akshare fund_name_em 返回\"），不要只写\"数据缺失\"。\n"
        "4. 前十大持仓基于季报披露，需提示权重可能已变动（披露滞后约2个月）。"
        "QDII 持仓实时涨跌不可获取时需显式标注。\n"
        "5. 极端业绩值（如近1年收益>100%、最大回撤>-40%）需标注【数据待复核】并简述反常原因。\n\n"
        "【写作要求】\n"
        "1. 用具体数字支撑观点，不写空话；数据缺口明确指出，绝不编造数字。\n"
        "2. 区分\"数据显示\"（结构化数据）与\"公开信息表明/行业常识\"（搜索摘要或你的知识）。\n"
        "3. 风险提示要具体到该基金，不要套话；投资建议要给具体观察指标与入场条件。\n"
        "4. 整体语气专业冷静，不渲染情绪。\n\n"
        "【输出格式】只输出以下 JSON，不要 markdown 代码块、不要任何其他文字：\n"
        f"{json_schema}\n\n"
        "【诊断数据】\n"
        f"{_serialize_data(context)}"
    )


def _build_diagnosis_prompt(state: DiagnosisState) -> str:
    """Dispatch to the stock or fund prompt builder based on target type."""
    if state.get("target_type") == "fund":
        return _build_fund_prompt(state)
    return _build_stock_prompt(state)


# Patterns that look like a direct buy/sell instruction. Kept for auditing only;
# the risk node no longer rewrites them so that practical advice can stay concrete.
_ACTIONABLE_PATTERNS = [
    r"建议.*买入",
    r"建议.*卖出",
    r"推荐.*买入",
    r"推荐.*卖出",
    r"买入",
    r"卖出",
    r"加仓",
    r"减仓",
    r"建仓",
    r"平仓",
    r"抄底",
    r"追高",
    r"止损",
    r"止盈",
    r"持有",
    r"看多",
    r"看空",
]


def _contains_actionable_language(text: str) -> bool:
    """Detect wording that looks like a direct buy/sell instruction."""
    return any(re.search(pattern, text) for pattern in _ACTIONABLE_PATTERNS)


# ---------------------------------------------------------------------------
# 数据抓取
# ---------------------------------------------------------------------------


def _fetch_stock_bundle(target_code: str) -> dict[str, Any]:
    """Fetch the full stock data bundle through the tool layer.

    Pulls realtime quote, detailed financials, 120-day kline + technical
    indicators, industry classification, and a web search for recent news /
    capital flow / analyst views. All five are gathered here so the analysis
    node receives one consolidated context.
    """
    realtime = stock_realtime_tool.invoke({"code": target_code})
    financial_detail = stock_financial_detail_tool.invoke({"code": target_code})
    kline = stock_history_kline_tool.invoke({"code": target_code})
    industry = stock_industry_tool.invoke({"code": target_code})

    stock_name = realtime.get("name") or target_code
    web_query = f"{stock_name} {target_code} 最新消息 资金流向 机构评级 风险事件"
    web_search = web_search_tool.invoke({"query": web_query})

    return {
        "realtime_data": realtime,
        "financial_detail_data": financial_detail,
        "kline_data": kline,
        "industry_data": industry,
        "web_search_data": web_search,
    }


def _fetch_fund_bundle(target_code: str) -> dict[str, Any]:
    """Fetch fund basic info, NAV, performance, holdings, fee, industry, and web search.

    每个数据源独立 try/except，单个工具失败（如债基无持仓、QDII 无 A 股实时）
    不会拖垮整条诊断，对应字段降级为空 dict，LLM 会在报告里标注数据缺口。
    """
    bundle: dict[str, Any] = {}

    def _safe_invoke(tool, key: str) -> None:
        try:
            bundle[key] = tool.invoke({"code": target_code})
        except Exception as exc:
            logger.warning("基金工具 %s 调用失败: %s", tool.name, exc)
            bundle[key] = {}

    _safe_invoke(fund_info_tool, "fund_basic_info")
    _safe_invoke(fund_nav_tool, "fund_nav")
    _safe_invoke(fund_valuation_tool, "fund_valuation")
    _safe_invoke(fund_performance_tool, "fund_performance_data")
    _safe_invoke(fund_fee_tool, "fund_fee_data")
    _safe_invoke(fund_industry_allocation_tool, "fund_industry_allocation_data")

    # 联网搜索
    fund_name = bundle.get("fund_basic_info", {}).get("name") or target_code
    web_query = f"{fund_name} {target_code} 基金 最新消息 规模变化 申赎 风险事件 机构评级"
    try:
        bundle["web_search_data"] = web_search_tool.invoke({"query": web_query})
    except Exception as exc:
        logger.warning("基金联网搜索失败: %s", exc)
        bundle["web_search_data"] = {}

    return bundle


# ---------------------------------------------------------------------------
# LangGraph pipeline
# ---------------------------------------------------------------------------


def build_diagnosis_graph(debug: bool = False):
    """Build a compiled LangGraph diagnosis pipeline.

    The `debug` flag is intentionally captured by closure rather than stored in
    state. That keeps the state payload clean while still allowing the notebook
    to print each node's input and output when we want to inspect the flow.
    """

    workflow: StateGraph[DiagnosisState] = StateGraph(DiagnosisState)

    def fetch_data_node(state: DiagnosisState) -> dict[str, Any]:
        # This node gathers the full upstream data bundle for the analysis step.
        _debug_dump("fetch_data_node input", state, debug)
        target_code = state.get("target_code", "").strip()
        target_type = state.get("target_type")

        if not target_code:
            result = {"error": "target_code is empty"}
            _debug_dump("fetch_data_node output", result, debug)
            return result

        try:
            if target_type == "stock":
                bundle = _fetch_stock_bundle(target_code)
            elif target_type == "fund":
                bundle = _fetch_fund_bundle(target_code)
            else:
                raise ValueError(f"Unsupported target_type: {target_type!r}")

            result = {
                **bundle,
                "error": None,
                "messages": [
                    HumanMessage(
                        content=(
                            f"已完成 {target_type} {target_code} 的数据抓取，"
                            f"准备进入结构化分析。"
                        )
                    )
                ],
            }
        except Exception as exc:
            result = {
                "error": str(exc),
                "messages": [AIMessage(content=f"数据抓取失败: {exc}")],
            }

        _debug_dump("fetch_data_node output", result, debug)
        return result

    def _extract_json(text: str) -> str:
        """Strip markdown fences and whitespace around a JSON payload."""
        text = text.strip()
        if text.startswith("```"):
            text = text.lstrip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

    def analyze_node(state: DiagnosisState) -> dict[str, Any]:
        # This node turns the consolidated market data into a structured diagnosis.
        # Stock → DiagnosisResult, Fund → FundDiagnosisResult.
        _debug_dump("analyze_node input", state, debug)
        target_type = state.get("target_type", "stock")
        prompt = _build_diagnosis_prompt(state)

        # System prompt varies by target type
        if target_type == "fund":
            system_content = (
                "你是一位专业的基金诊断分析师。"
                "输出要按蚂蚁财富 5 段结构（核心诊断→业绩波动→持仓→费率→投资建议），"
                "结构化、克制、可审计，基于数据与公开信息给出深度分析。"
                "投资建议可以给出具体观察指标与入场条件，"
                "但需标注是风险控制参考而非交易指令。"
                "不要说'您目前未持有该基金'。"
                "你只输出 JSON，不输出任何其他内容、不输出 markdown 代码块。"
            )
        else:
            system_content = (
                "你是一位专业的金融诊断分析师。"
                "输出要结构化、克制、可审计，基于数据与公开信息给出多维度深度分析。"
                "实操参考部分可以给出具体的风险阈值（止损位、仓位上限、入场区间），"
                "但需标注是风险控制参考而非交易指令。"
                "你只输出 JSON，不输出任何其他内容、不输出 markdown 代码块。"
            )

        try:
            raw = llm.invoke(
                [
                    SystemMessage(content=system_content),
                    HumanMessage(content=prompt),
                ]
            )
            raw_text = raw.content if hasattr(raw, "content") else str(raw)
            json_text = _extract_json(raw_text)
            parsed = json.loads(json_text)

            # 从已抓取数据中提取标的名称并注入结果，避免 LLM 写错名称。
            name_source = state.get("realtime_data") or state.get("fund_basic_info") or {}
            parsed.setdefault("name", name_source.get("name") or "")
            parsed["disclaimer"] = DEFAULT_DISCLAIMER

            # 按类型选择 result class
            if target_type == "fund":
                result = FundDiagnosisResult(**parsed)
            else:
                result = DiagnosisResult(**parsed)

            result_payload = result.model_dump()
            update = {
                "analysis_result": result,
                "messages": [
                    HumanMessage(content=prompt),
                    AIMessage(content=json.dumps(result_payload, ensure_ascii=False)),
                ],
                "error": None,
            }
        except Exception as exc:
            update = {"error": f"analysis failed: {exc}"}

        _debug_dump("analyze_node output", update, debug)
        return update

    def risk_check_node(state: DiagnosisState) -> dict[str, Any]:
        # Final safety gate: lock the disclaimer and audit any actionable wording.
        _debug_dump("risk_check_node input", state, debug)
        result = state.get("analysis_result")
        if result is None:
            update = {"error": "analysis_result is missing"}
            _debug_dump("risk_check_node output", update, debug)
            return update

        updated = result.model_copy(update={"disclaimer": DEFAULT_DISCLAIMER})

        # 审计：扫描所有文本字段是否出现指令性措辞。仅记录日志，不再改写，
        # 以保留实操参考中的具体风险阈值（止损/仓位/入场区间）。免责声明已锁定。
        if isinstance(updated, FundDiagnosisResult):
            text_fields = [
                updated.core_diagnosis,
                updated.performance_volatility,
                updated.holdings_analysis,
                updated.fee_and_trading,
                updated.risk_analysis,
                updated.investment_advice,
            ]
        else:
            text_fields = [
                updated.business_overview,
                updated.fundamental_analysis,
                updated.technical_analysis,
                updated.capital_flow_analysis,
                updated.risk_analysis,
                updated.comprehensive_diagnosis,
                updated.practical_advice,
                updated.one_sentence_summary,
            ]

        flagged = [f for f in text_fields if _contains_actionable_language(f)]
        if flagged:
            logger.info(
                "诊断结果中出现指令性/风险控制措辞（已保留以提供具体参考，免责声明已锁定）: "
                "%d 个字段",
                len(flagged),
            )

        update = {
            "analysis_result": updated,
            "messages": [
                AIMessage(
                    content=(
                        "风险检查已完成，免责声明已锁定；"
                        "实操参考中的风险控制措辞已保留并审计记录。"
                    )
                )
            ],
        }
        _debug_dump("risk_check_node output", update, debug)
        return update

    def route_after_fetch(state: DiagnosisState) -> Literal["analyze", "end"]:
        """Route to analysis only when data fetching succeeded."""
        return "end" if state.get("error") else "analyze"

    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("risk_check", risk_check_node)

    workflow.set_entry_point("fetch_data")
    workflow.add_conditional_edges(
        "fetch_data",
        route_after_fetch,
        {
            "analyze": "analyze",
            "end": END,
        },
    )
    workflow.add_edge("analyze", "risk_check")
    workflow.add_edge("risk_check", END)

    return workflow.compile(debug=debug)
