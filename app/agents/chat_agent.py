"""FinPilot 对话首页的工具调用型 Agent。

把项目已有的能力（股基诊断、组合分析、财报 RAG、联网搜索）包装成
LangChain ``@tool``，再用 ``langgraph.prebuilt.create_react_agent`` 组装成一个
能自主决定"调哪个工具"的对话 agent。普通闲聊不触发任何工具，直接由 LLM 回答。

设计要点：
- ``streaming_llm``（app/llm.py，streaming=True，此前一直闲置）在这里启用，
  配合 create_react_agent 的 ``astream(stream_mode="messages")`` 实现 token 级流式。
- 需要请求级上下文的工具（``analyze_my_portfolio`` 需要 user_id；
  ``query_uploaded_report`` 需要当前会话的 active_report_id）用闭包在
  ``build_chat_agent`` 里构造，把 user_id / conversation_id 绑进去。
- 诊断/组合工具不把完整结构化 JSON 原样塞回 LLM，而是裁剪成紧凑文本摘要，
  既省 token，也让 agent 的口语化解说更自然。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app.agents import chat_memory
from app.agents.diagnosis_agent import (
    DiagnosisResult,
    FundDiagnosisResult,
    build_diagnosis_graph,
)
from app.agents.portfolio_agent import build_portfolio_graph
from app.data_source.fund import resolve_fund_code
from app.data_source.stock import resolve_stock_code
from app.llm import streaming_llm
from app.models.portfolio_crud import get_all_positions
from app.rag.qa_chain import answer_report_question
from app.rag.retriever import find_report_id_for_stock, report_age_days, report_vintage
from app.tools.market_hotspots_tool import market_hotspots_tool
from app.tools.web_search_tool import web_search_tool

logger = logging.getLogger(__name__)

# 单个维度摘要的字符上限。诊断结果每个维度都是大段文本，全量回塞会让
# agent 的回答冗长且爆 token；裁剪到首段要点供其口语化解说即可。
_FIELD_LIMIT = 400


def _trim(text: Any, limit: int = _FIELD_LIMIT) -> str:
    """把任意值转为字符串并截断到安全长度。"""
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    return s if len(s) <= limit else s[:limit] + "…"


def _vintage_note(report_id: str) -> str:
    """给财报工具的返回拼一段报告期提示，让模型在回答里说清数据出处与时效。

    财报有时效性：若语料库里的报告已较陈旧（>18 个月），必须提醒模型在回答里
    显式标注报告期并提示数据可能过时、建议上传最新财报，避免把旧数据当现状讲。
    """
    label, is_stale = report_vintage(report_id)
    if is_stale:
        return (
            f"【时效提醒】该回答基于「{label}」财报，已较为陈旧，数据可能过时。"
            f"回答时必须明确告知用户数据来自{label}，并提示如需最新数据请上传较新的财报。\n"
        )
    return f"【报告期】{label}。回答时请告知用户数据来自该报告期。\n"


def _summarize_stock_result(code: str, result: DiagnosisResult) -> str:
    """把股票 DiagnosisResult 裁剪成供 agent 解说的紧凑文本。"""
    return (
        f"标的：{result.name or code}（{code}）  风险评级：{result.risk_level}\n"
        f"一句话总结：{_trim(result.one_sentence_summary, 300)}\n"
        f"- 业务概述：{_trim(result.business_overview)}\n"
        f"- 基本面：{_trim(result.fundamental_analysis)}\n"
        f"- 技术面：{_trim(result.technical_analysis)}\n"
        f"- 资金面：{_trim(result.capital_flow_analysis)}\n"
        f"- 风险提示：{_trim(result.risk_analysis)}\n"
        f"- 综合诊断：{_trim(result.comprehensive_diagnosis)}\n"
        f"- 实操参考：{_trim(result.practical_advice)}\n"
        f"免责声明：{result.disclaimer}"
    )


def _summarize_fund_result(code: str, result: FundDiagnosisResult) -> str:
    """把基金 FundDiagnosisResult 裁剪成供 agent 解说的紧凑文本。"""
    return (
        f"标的：{result.name or code}（{code}）  风险评级：{result.risk_level}\n"
        f"核心诊断：{_trim(result.core_diagnosis, 300)}\n"
        f"- 业绩与波动：{_trim(result.performance_volatility)}\n"
        f"- 持仓分析：{_trim(result.holdings_analysis)}\n"
        f"- 费率与交易：{_trim(result.fee_and_trading)}\n"
        f"- 风险提示：{_trim(result.risk_analysis)}\n"
        f"- 投资建议：{_trim(result.investment_advice)}\n"
        f"免责声明：{result.disclaimer}"
    )


def _run_diagnosis(keyword: str, target_type: str) -> str:
    """跑一次诊断图并把结果裁剪成文本摘要。供 diagnose_stock/fund 复用。

    keyword 可以是 6 位代码，也可以是名称（如"贵州茅台"）。先解析成 6 位代码
    再跑诊断图；名称有多个匹配时返回候选列表让用户指定。
    """
    kw = (keyword or "").strip()
    if not kw:
        return "诊断失败：未提供标的代码或名称。"

    # 名称→代码解析
    if target_type == "stock":
        code, _name, candidates = resolve_stock_code(kw)
    else:
        code, _name, candidates = resolve_fund_code(kw)

    if code is None:
        if candidates:
            type_label = "股票" if target_type == "stock" else "基金"
            lines = [f"找到多个与「{kw}」匹配的{type_label}，请告诉我要诊断哪一个（回复代码即可）："]
            for c in candidates[:10]:
                lines.append(f"- {c['code']} {c['name']}")
            return "\n".join(lines)
        type_label = "股票" if target_type == "stock" else "基金"
        return f"未找到与「{kw}」匹配的{type_label}，请确认名称或代码后重试。"

    try:
        state = build_diagnosis_graph().invoke(
            {"target_code": code, "target_type": target_type, "messages": []}
        )
    except Exception as exc:  # noqa: BLE001  工具失败要回给 agent 而非中断对话
        logger.exception("chat 诊断工具异常 %s %s", target_type, code)
        return f"诊断失败：{exc}"

    if state.get("error"):
        return f"诊断失败：{state['error']}"

    result = state.get("analysis_result")
    if result is None:
        return "诊断失败：未生成分析结果。"

    if isinstance(result, FundDiagnosisResult):
        return _summarize_fund_result(code, result)
    return _summarize_stock_result(code, result)


@tool
def diagnose_stock(code_or_name: str) -> str:
    """对 A 股股票做多维度深度诊断（业务/基本面/技术面/资金面/风险/实操参考）。

    当用户想了解某只股票的投资价值、风险或操作参考时调用。输入可以是 6 位股票代码
    （如 600519、300308），也可以是股票名称（如"贵州茅台"、"中际旭创"）。
    名称会自动解析成代码；若有多个匹配会返回候选列表让用户指定。返回结构化诊断
    结果的紧凑摘要，由你向用户口语化解说。
    """
    return _run_diagnosis(code_or_name, "stock")


@tool
def diagnose_fund(code_or_name: str) -> str:
    """对公募基金做诊断（核心定位/业绩波动/持仓/费率/投资建议）。

    当用户想了解某只基金的风格、业绩、持仓或是否值得持有时调用。输入可以是 6 位
    基金代码（如 110011、005827），也可以是基金名称（如"易方达蓝筹精选"）。
    名称会自动解析成代码；多个匹配会返回候选列表。返回诊断结果的紧凑摘要。
    """
    return _run_diagnosis(code_or_name, "fund")


@tool
def query_stock_report(stock_name_or_code: str, question: str) -> str:
    """针对某只 A 股股票的财报回答问题（基于已上传的财报语料库做检索）。

    当用户同时提到某只股票和财报、想问财报里的具体内容时调用——例如
    "贵州茅台财报里毛利率多少"、"查一下中际旭创年报的营收增长"、"300308 的研发费用占比"。
    输入股票名称或 6 位代码 + 具体问题。会自动解析名称→代码，在已上传的财报语料库中
    找该股票最新的那份（data/chroma_db 下 {code} 或 {code}_{year}_{quarter}）做混合
    检索并据检索内容回答，找不到会明说。若该股票从未上传过财报，返回提示让用户先上传。
    跨会话可用：只要该股票的财报曾被上传过，任何会话都能查。
    """
    code, name, candidates = resolve_stock_code(stock_name_or_code)
    if code is None:
        if candidates:
            lines = ["找到多个匹配的股票，请指定具体代码后重试："]
            for c in candidates[:10]:
                lines.append(f"- {c['code']} {c['name']}")
            return "\n".join(lines)
        return f"未找到与「{stock_name_or_code}」匹配的股票，请确认名称或代码。"

    report_id = find_report_id_for_stock(code)
    if report_id is None:
        label = f"{name}（{code}）" if name else f"{code}"
        return (
            f"还没有上传过「{label}」的财报。请先点输入框上方的「上传财报PDF」按钮，"
            f"上传该股票的财报（填代码+年份+季度），之后就能问我关于它的问题了。"
        )

    try:
        result = answer_report_question(question, report_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("query_stock_report 失败 report_id=%s", report_id)
        return f"财报问答失败：{exc}"

    answer = result.get("answer", "")
    pages = result.get("page_numbers", [])
    suffix = f"（来源页码：{pages}）" if pages else ""
    return f"{_vintage_note(report_id)}{answer}{suffix}"


# "分析财报"用的通用问题：让 RAG 给出一份关键财务指标的概述，而不是只答某个具体问题。
_FINANCIAL_OVERVIEW_QUESTION = (
    "请基于财报内容，概述该公司最新报告期的营收、净利润、毛利率、期间费用、"
    "现金流等关键财务指标及同比变化，并点出值得关注的亮点与风险。"
)


@tool
def analyze_financial_report(stock_code: str, stock_name: str = "") -> str:
    """分析某只 A 股股票的财报，给出关键财务指标概述。

    当用户点"分析财报"按钮、提供股票名称和代码、想看该公司财报整体情况时调用
    （区别于 query_stock_report：那个是回答某个具体问题，这个是给整体概述）。

    流程按向量库里该股票财报的新旧分支：
    - 有且最近一年内：直接基于该财报检索作答，不联网。
    - 有但超过一年（或报告期未知）：基于该财报检索 + 额外联网搜索补充最新信息。
    - 没有：只联网搜索，回复时提示"向量库无该公司财报，信息来自搜索可能不够准确，
      建议上传相关财报PDF"。

    输入：6 位股票代码 + 股票名称（名称用于联网搜索关键词，没填就用代码）。
    """
    code = (stock_code or "").strip()
    name = (stock_name or "").strip() or code
    if not code:
        return "分析失败：未提供股票代码。"

    report_id = find_report_id_for_stock(code)

    # 分支 1：向量库无该财报 → 只搜索 + 提示上传
    if report_id is None:
        try:
            sr = web_search_tool.invoke({"query": f"{name} {code} 财报 营收 净利润 业绩 最新"})
            search_text = sr.get("results", "") if isinstance(sr, dict) else str(sr)
        except Exception as exc:  # noqa: BLE001
            search_text = f"联网搜索失败：{exc}"
        return (
            f"【提示】目前向量库没有 {name}（{code}）的财报，以下信息通过联网搜索获取，"
            f"可能不够准确。如想保证准确，建议点「上传财报PDF」上传该公司的财报后再分析。\n\n"
            f"【联网搜索结果】\n{_trim(search_text, 2500)}"
        )

    # 有财报：先检索
    label, _is_stale = report_vintage(report_id)
    try:
        result = answer_report_question(_FINANCIAL_OVERVIEW_QUESTION, report_id)
        retrieve_answer = result.get("answer", "")
    except Exception as exc:  # noqa: BLE001
        retrieve_answer = f"财报检索失败：{exc}"

    age_days = report_age_days(report_id)
    # 最近一年内（age ≤365）且 age 已知 → 不补搜索
    if age_days is not None and age_days <= 365:
        return (
            f"【报告期】{label}（最近一年内，直接基于财报作答，未联网）\n"
            f"【财报检索】{retrieve_answer}"
        )

    # 超过一年 或 报告期未知 → 检索 + 补搜索
    age_note = "已较旧" if age_days is not None else "报告期未知"
    try:
        sr = web_search_tool.invoke({"query": f"{name} {code} 最新财报 营收 净利润 业绩"})
        search_text = sr.get("results", "") if isinstance(sr, dict) else str(sr)
    except Exception as exc:  # noqa: BLE001
        search_text = f"联网搜索失败：{exc}"
    return (
        f"【报告期】{label}（{age_note}，额外联网搜索补充最新信息）\n"
        f"【财报检索】{retrieve_answer}\n\n"
        f"【联网搜索补充】\n{_trim(search_text, 2500)}"
    )


def _build_portfolio_tools(user_id: int, conversation_id: str) -> list:
    """构造需要请求级上下文的工具（持仓分析 / 财报问答）。

    user_id 用于持仓隔离，conversation_id 用于定位当前会话绑定的财报 report_id。
    用闭包绑定，避免工具签名暴露这些内部参数给 LLM。
    """

    @tool
    def analyze_my_portfolio() -> str:
        """分析当前登录用户的持仓组合是否合理。

        当用户问"我的持仓怎么样"、"分析一下我的组合"、"持仓合理吗"等时调用。
        会读取该用户全部持仓，逐个诊断并给出组合层面的集中度与整体风险评级。
        若用户没有持仓，返回提示其先去"我的持仓"页添加。
        """
        try:
            positions = get_all_positions(user_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("读取持仓失败 user_id=%s", user_id)
            return f"读取持仓失败：{exc}"

        if not positions:
            return (
                "当前没有任何持仓。请先到「我的持仓」页面添加持仓（股票或基金代码、"
                "数量、成本价），再来让我分析组合。"
            )

        holdings = [
            {
                "code": p.code,
                "name": p.name,
                "type": p.type,
                "shares": float(p.shares) if p.shares is not None else 0.0,
                "cost_price": float(p.cost_price) if p.cost_price is not None else 0.0,
            }
            for p in positions
        ]

        try:
            state = build_portfolio_graph().invoke({"holdings": holdings, "messages": []})
        except Exception as exc:  # noqa: BLE001
            logger.exception("组合分析失败 user_id=%s", user_id)
            return f"组合分析失败：{exc}"

        if state.get("error"):
            return f"组合分析失败：{state['error']}"

        report = state.get("report")
        if report is None:
            return "组合分析失败：未生成报告。"

        lines = [
            f"组合概览：{report.summary}",
            f"集中度分析：{report.concentration_analysis}",
            f"整体风险评级：{report.risk_level}",
            "各持仓：",
        ]
        for h in report.holdings:
            lines.append(
                f"- {h.name}（{h.code}，{h.type}）：{_trim(h.summary, 200)}（风险：{h.risk_level}）"
            )
        lines.append(f"免责声明：{report.disclaimer}")
        return "\n".join(lines)

    @tool
    def query_uploaded_report(question: str) -> str:
        """针对当前会话已上传的财报 PDF 回答问题。

        当用户上传财报后追问"毛利率怎么样"、"营收增长情况"、"现金流如何"
        等财报内容时调用。只能基于该财报检索到的内容回答，找不到会明说。
        若本会话尚未上传财报，提示用户先点"上传财报PDF"按钮。
        """
        report_id = chat_memory.get_active_report(user_id, conversation_id)
        if not report_id:
            return (
                "当前会话还没有上传财报。请先点输入框上方的「上传财报PDF」按钮"
                "上传一份财报，之后就可以问我关于它的问题了。"
            )
        try:
            result = answer_report_question(question, report_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("财报问答失败 report_id=%s", report_id)
            return f"财报问答失败：{exc}"

        answer = result.get("answer", "")
        pages = result.get("page_numbers", [])
        suffix = f"（来源页码：{pages}）" if pages else ""
        return f"{_vintage_note(report_id)}{answer}{suffix}"

    return [analyze_my_portfolio, query_uploaded_report]


SYSTEM_PROMPT = """你是 FinPilot 的 AI 金融助手，通过对话帮助用户分析持仓、诊断股基、解读财报、联网查询，也可以普通闲聊。

你可以调用以下工具，按需自主决定是否调用、调用哪个：
- diagnose_stock(code_or_name)：A 股股票多维度诊断。输入 6 位代码或股票名称（如"贵州茅台"）。
- diagnose_fund(code_or_name)：公募基金诊断。输入 6 位代码或基金名称（如"易方达全球成长精选"）。
- analyze_my_portfolio()：分析当前用户持仓组合是否合理（无需参数，自动读取该用户持仓）。
- query_stock_report(stock_name_or_code, question)：按股票名/代码查其财报里的具体数据。当用户问某只股票财报的某个具体指标时调用（如"贵州茅台财报毛利率多少"）。跨会话可用。
- analyze_financial_report(stock_code, stock_name)：分析某股票财报的整体情况（关键财务指标概述）。当用户点"分析财报"按钮、或说"分析X的财报"想看整体时调用。工具会按向量库财报新旧自动决定要不要补联网搜索，无需你额外调 web_search。
- query_uploaded_report(question)：针对本会话刚上传的财报 PDF 回答问题（无需指定股票，用户刚上传完追问时用）。
- market_hotspots_tool()：获取今日 A 股市场热点（主要大盘指数、行业板块涨幅榜/跌幅榜、概念板块涨幅榜、涨停股、大盘资金流向）。当用户问"今天市场怎么样"、"有什么热点"、"哪些板块涨得好"、"大盘指数表现如何"时调用。返回结构化数据后，你需结合这些事实 + 可选的 web_search 补充新闻，综合解读成口语化的市场简报。
- web_search_tool(query)：联网搜索最新消息、资金动向、机构观点等公开信息。

回答风格：
1. 用自然、口语化的中文回答，像一个专业的金融顾问在和你聊天，不要生硬地罗列字段。
2. 调用工具拿到结构化结果后，要把关键结论和数据用自己的话讲清楚，挑重点说，不要把工具返回的原文整段复读。
3. 客观、数据驱动；数据缺口要如实说明，绝不编造数字。
4. 涉及具体操作（止损位、仓位等）时，强调是"风险控制参考"而非交易指令，并保留免责声明。
5. diagnose 工具接受代码或名称：用户给名称就直接传名称，工具会自动解析成代码；若返回多个候选，把候选清单原样转达给用户让其指定。
6. 财报工具有时效性：工具返回的开头会标注报告期（如"2023年年报"），若标"时效提醒"说明该财报已较陈旧。回答时必须把报告期讲给用户，陈旧的要明确提示"数据来自X年报、可能过时、建议查看最新财报"，不要把旧数据当现状陈述。
7. 不在工具范围内的闲聊正常回答即可，不必强行调用工具。
"""


def build_chat_agent(user_id: int, conversation_id: str):
    """构造一个绑定了用户上下文的对话 agent。

    返回一个已编译的 langgraph react agent，可直接 ``astream`` 流式调用。
    """
    tools = [
        diagnose_stock,
        diagnose_fund,
        query_stock_report,
        analyze_financial_report,
        market_hotspots_tool,
        web_search_tool,
        *_build_portfolio_tools(user_id, conversation_id),
    ]
    return create_react_agent(streaming_llm, tools, prompt=SYSTEM_PROMPT)
