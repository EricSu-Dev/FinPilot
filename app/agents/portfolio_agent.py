"""在诊断 Agent 之上构建的 LangGraph 组合 Agent。

这一层不去搞隐藏记忆之类的花招。它只是对每个持仓运行一次诊断图，
再把每个持仓的结果聚合为组合层面的报告。这让行为确定、易于测试，
也便于在 Text2SQL 阶段到来之前进行推理。
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from statistics import mean
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.agents.diagnosis_agent import (
    DEFAULT_DISCLAIMER,
    DiagnosisResult,
    FundDiagnosisResult,
    build_diagnosis_graph,
)
from app.observability import elapsed_ms, log_agent_event, new_trace_id

logger = logging.getLogger(__name__)

PORTFOLIO_ANALYSIS_TIMEOUT_SECONDS = 60.0


class PortfolioHoldingInput(TypedDict, total=False):
    """单个组合持仓的输入结构。"""

    code: str
    name: str
    type: Literal["stock", "fund"]
    sector: str
    shares: float
    cost_price: float


class PortfolioHoldingSummary(BaseModel):
    """聚合组合报告中单个持仓的诊断。"""

    code: str
    name: str
    type: Literal["stock", "fund"]
    summary: str
    risk_level: Literal["低", "中", "高"]


class PortfolioReport(BaseModel):
    """由 Agent 返回的组合层面诊断。"""

    summary: str = Field(description="组合层面的一句话总结。")
    concentration_analysis: str = Field(description="行业或类型集中度分析。")
    risk_level: Literal["低", "中", "高"] = Field(description="组合整体风险评级。")
    holdings: list[PortfolioHoldingSummary] = Field(description="每个持仓的简要诊断。")
    disclaimer: str = Field(description="固定免责声明文本。")


class PortfolioState(TypedDict, total=False):
    """组合分析流程的 LangGraph 状态。"""

    holdings: list[PortfolioHoldingInput]
    holding_reports: list[dict[str, Any]]
    report: Optional[PortfolioReport]
    error: Optional[str]
    observability_trace_id: str
    messages: Annotated[list[BaseMessage], append_messages]


def append_messages(existing: list[BaseMessage] | None, new_messages: list[BaseMessage] | None) -> list[BaseMessage]:
    """合并两条消息列表，用于 LangGraph 状态更新。"""
    return [*(existing or []), *(new_messages or [])]


def _holding_bucket(holding: PortfolioHoldingInput) -> str:
    """返回单个持仓可用的最佳集中度分桶。"""
    return holding.get("sector") or holding.get("type") or "unknown"


def _risk_score(risk_level: Literal["低", "中", "高"]) -> int:
    """把中文风险标签转换为可排序的分数。"""
    return {"低": 1, "中": 2, "高": 3}[risk_level]


def _score_to_risk_label(score: float) -> Literal["低", "中", "高"]:
    """把数值分数映射回项目使用的风险标签。"""
    if score >= 2.5:
        return "高"
    if score >= 1.5:
        return "中"
    return "低"


def _aggregate_concentration(holdings: list[PortfolioHoldingInput]) -> str:
    """描述组合是否集中在单一行业或类型上。"""
    buckets = [_holding_bucket(holding) for holding in holdings]
    if not buckets:
        return "当前没有足够的持仓数据，无法判断集中度。"

    counts = Counter(buckets)
    dominant_bucket, dominant_count = counts.most_common(1)[0]
    ratio = dominant_count / len(buckets)

    if ratio >= 0.6:
        tone = "集中度偏高"
    elif ratio >= 0.4:
        tone = "存在一定集中"
    else:
        tone = "分布相对分散"

    return (
        f"按当前输入数据统计，主要集中在 {dominant_bucket}，占比约 {ratio:.0%}，"
        f"{tone}。分桶明细：{dict(counts)}。"
    )


def _aggregate_overall_risk(holding_reports: list[dict[str, Any]], holdings: list[PortfolioHoldingInput]) -> Literal["低", "中", "高"]:
    """根据持仓级别结果与集中度推导一个简单的组合风险等级。"""
    if not holding_reports:
        return "高"

    base_score = mean(_risk_score(report["risk_level"]) for report in holding_reports)
    concentration_buckets = Counter(_holding_bucket(holding) for holding in holdings)
    concentration_ratio = 0.0
    if holdings:
        concentration_ratio = concentration_buckets.most_common(1)[0][1] / len(holdings)

    concentration_score = 3 if concentration_ratio >= 0.6 else 2 if concentration_ratio >= 0.4 else 1
    final_score = max(base_score, float(concentration_score))
    return _score_to_risk_label(final_score)


def build_portfolio_graph(debug: bool = False):
    """构建一个已编译的图：诊断每个持仓，再汇总整个组合。"""

    diagnosis_graph = build_diagnosis_graph(debug=debug)
    workflow: StateGraph[PortfolioState] = StateGraph(PortfolioState)

    def diagnose_one(holding: PortfolioHoldingInput, trace_id: str, holding_index: int) -> dict[str, Any]:
        """对单条持仓跑一次诊断图，返回 holding_report dict。

        抽出来是为了能并发：每条持仓的诊断彼此独立（各自抓数据 + 各自调 LLM），
        串行跑会让 N 条持仓的等待时间累加；并发后墙钟时间≈最慢一条。compiled
        的 diagnosis_graph 是无状态的，可安全被多线程并发 invoke。
        """
        code = holding.get("code", "").strip()
        holding_type = holding.get("type", "stock")
        holding_started_at = time.perf_counter()
        log_agent_event(
            "portfolio.holding.start",
            trace_id=trace_id,
            holding_index=holding_index,
            code=code,
            holding_type=holding_type,
        )
        if not code:
            log_agent_event(
                "portfolio.holding.error",
                level=logging.WARNING,
                trace_id=trace_id,
                holding_index=holding_index,
                holding_type=holding_type,
                duration_ms=elapsed_ms(holding_started_at),
                error="empty code",
            )
            return {
                "code": "",
                "name": holding.get("name", ""),
                "type": holding_type,
                "summary": "持仓代码缺失，无法完成诊断。",
                "risk_level": "高",
            }

        try:
            diagnosis_state = diagnosis_graph.invoke(
                {
                    "target_code": code,
                    "target_type": holding_type,
                    "messages": [],
                    "observability_trace_id": trace_id,
                }
            )
        except Exception as exc:
            log_agent_event(
                "portfolio.holding.error",
                level=logging.ERROR,
                trace_id=trace_id,
                holding_index=holding_index,
                code=code,
                holding_type=holding_type,
                duration_ms=elapsed_ms(holding_started_at),
                error=str(exc),
            )
            logger.exception("portfolio holding diagnosis failed code=%s type=%s", code, holding_type)
            return {
                "code": code,
                "name": holding.get("name", ""),
                "type": holding_type,
                "summary": f"诊断失败：{exc}",
                "risk_level": "高",
            }
        diagnosis_result = diagnosis_state.get("analysis_result")
        if diagnosis_state.get("error") or diagnosis_result is None:
            log_agent_event(
                "portfolio.holding.error",
                level=logging.WARNING,
                trace_id=trace_id,
                holding_index=holding_index,
                code=code,
                holding_type=holding_type,
                duration_ms=elapsed_ms(holding_started_at),
                error=str(diagnosis_state.get("error", "missing analysis_result")),
            )
            return {
                "code": code,
                "name": holding.get("name", ""),
                "type": holding_type,
                "summary": diagnosis_state.get("error", "诊断失败。"),
                "risk_level": "高",
            }

        # 股票诊断结果有 one_sentence_summary，基金诊断结果（FundDiagnosisResult）
        # 没有，用 core_diagnosis（一句话定性）作为该持仓的摘要。
        if isinstance(diagnosis_result, FundDiagnosisResult):
            summary_text = diagnosis_result.core_diagnosis
        else:
            summary_text = diagnosis_result.one_sentence_summary
        log_agent_event(
            "portfolio.holding.done",
            trace_id=trace_id,
            holding_index=holding_index,
            code=code,
            holding_type=holding_type,
            duration_ms=elapsed_ms(holding_started_at),
            risk_level=diagnosis_result.risk_level,
        )
        return {
            "code": code,
            "name": holding.get("name", code),
            "type": holding_type,
            "summary": summary_text,
            "risk_level": diagnosis_result.risk_level,
        }

    def diagnose_holdings_node(state: PortfolioState) -> dict[str, Any]:
        # 该节点在多个持仓上展开，并复用诊断图，使每个持仓都走相同的推理流水线。
        # 各持仓诊断相互独立，用线程池并发跑，把 N 条持仓的墙钟时间从累加降到≈最慢一条。
        holdings = state.get("holdings", [])
        trace_id = state.get("observability_trace_id") or new_trace_id("portfolio")
        node_started_at = time.perf_counter()
        if not holdings:
            log_agent_event(
                "portfolio.holdings.done",
                trace_id=trace_id,
                holdings_count=0,
                duration_ms=elapsed_ms(node_started_at),
            )
            return {"holding_reports": [], "error": None, "observability_trace_id": trace_id}

        # 单条持仓诊断涉及数据抓取（akshare/baostock/ddgs）与 LLM 调用，都是 IO 密集。
        # baostock 使用全局会话，并发过高容易出现连接/文件描述符异常；worker 数保守封顶 2。
        max_workers = min(len(holdings), 2)
        log_agent_event(
            "portfolio.holdings.start",
            trace_id=trace_id,
            holdings_count=len(holdings),
            max_workers=max_workers,
        )
        ordered: list[dict[str, Any] | None] = [None] * len(holdings)
        pool = ThreadPoolExecutor(max_workers=max_workers)
        future_to_idx = {
            pool.submit(diagnose_one, h, trace_id, i): i for i, h in enumerate(holdings)
        }
        timed_out = False
        try:
            for future in as_completed(future_to_idx, timeout=PORTFOLIO_ANALYSIS_TIMEOUT_SECONDS):
                idx = future_to_idx[future]
                try:
                    ordered[idx] = future.result()
                except Exception as exc:
                    holding = holdings[idx]
                    log_agent_event(
                        "portfolio.holding.future_error",
                        level=logging.ERROR,
                        trace_id=trace_id,
                        holding_index=idx,
                        code=holding.get("code"),
                        holding_type=holding.get("type"),
                        error=str(exc),
                    )
                    ordered[idx] = {
                        "code": holding.get("code", ""),
                        "name": holding.get("name", ""),
                        "type": holding.get("type", "stock"),
                        "summary": f"诊断失败：{exc}",
                        "risk_level": "高",
                    }
        except TimeoutError:
            timed_out = True
            log_agent_event(
                "portfolio.holdings.timeout",
                level=logging.WARNING,
                trace_id=trace_id,
                holdings_count=len(holdings),
                timeout_seconds=PORTFOLIO_ANALYSIS_TIMEOUT_SECONDS,
                duration_ms=elapsed_ms(node_started_at),
            )
        finally:
            for future in future_to_idx:
                if not future.done():
                    future.cancel()
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pool.shutdown(wait=False)

        for idx, report in enumerate(ordered):
            if report is not None:
                continue
            holding = holdings[idx]
            ordered[idx] = {
                "code": holding.get("code", ""),
                "name": holding.get("name", ""),
                "type": holding.get("type", "stock"),
                "summary": (
                    "数据源响应超时，已停止等待该持仓的深度诊断。"
                    "可能是 baostock/akshare 行情源连接异常或外部接口限流，请稍后重试。"
                ),
                "risk_level": "高",
            }
        log_agent_event(
            "portfolio.holdings.done",
            trace_id=trace_id,
            holdings_count=len(holdings),
            timed_out=timed_out,
            duration_ms=elapsed_ms(node_started_at),
        )
        return {"holding_reports": ordered, "error": None, "observability_trace_id": trace_id}

    def summarize_portfolio_node(state: PortfolioState) -> dict[str, Any]:
        # 该节点把各持仓的诊断结果折叠为紧凑的组合视图，无需再次调用模型。
        holdings = state.get("holdings", [])
        holding_reports = state.get("holding_reports", [])
        concentration_analysis = _aggregate_concentration(holdings)
        risk_level = _aggregate_overall_risk(holding_reports, holdings)

        holding_models = [
            PortfolioHoldingSummary(
                code=str(report.get("code", "")),
                name=str(report.get("name", "")),
                type=report.get("type", "stock"),
                summary=str(report.get("summary", "")),
                risk_level=report.get("risk_level", "中"),
            )
            for report in holding_reports
        ]

        summary = (
            f"组合共包含 {len(holding_models)} 个标的，整体风险评级为 {risk_level}，"
            f"主要集中情况如下：{concentration_analysis}"
        )
        timeout_count = sum(
            1 for report in holding_reports if "数据源响应超时" in str(report.get("summary", ""))
        )
        if timeout_count:
            summary += f" 其中 {timeout_count} 个持仓因外部行情源响应超时，已停止等待并按高风险数据缺口处理。"
        report = PortfolioReport(
            summary=summary,
            concentration_analysis=concentration_analysis,
            risk_level=risk_level,
            holdings=holding_models,
            disclaimer=DEFAULT_DISCLAIMER,
        )
        log_agent_event(
            "portfolio.summary.done",
            trace_id=state.get("observability_trace_id"),
            holdings_count=len(holding_models),
            risk_level=risk_level,
        )
        return {"report": report}

    def route_after_diagnosis(state: PortfolioState) -> Literal["summarize", "end"]:
        """仅当持仓级别诊断成功时才路由到汇总步骤。"""
        return "end" if state.get("error") else "summarize"

    # 第一个节点把组合展开为针对每个持仓的诊断调用。
    workflow.add_node("diagnose_holdings", diagnose_holdings_node)
    # 第二个节点把持仓级别结果折叠回单份组合报告。
    workflow.add_node("summarize", summarize_portfolio_node)

    # 组合图以诊断开始，因为汇总步骤需要先有持仓级别的依据。
    workflow.set_entry_point("diagnose_holdings")
    # 如果诊断意外失败，提前停止，以免生成误导性的汇总。
    workflow.add_conditional_edges(
        "diagnose_holdings",
        route_after_diagnosis,
        {
            "summarize": "summarize",
            "end": END,
        },
    )
    # 汇总完成后，图即可结束，因为组合报告已完整。
    workflow.add_edge("summarize", END)

    return workflow.compile(debug=debug)
