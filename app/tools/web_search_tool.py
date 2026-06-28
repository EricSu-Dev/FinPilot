"""联网搜索工具，用于补充实时新闻、资金动向与机构观点。

诊断报告的数据层（baostock / akshare）只能给出结构化的历史与财务数据，
无法覆盖“今日发生了什么”——例如最新公告、主力资金流向、机构目标价调整、
突发政策或地缘事件。这个工具把这些实时信息从公开网页抓回来，塞进诊断上下文，
让 LLM 的资金面与风险提示分析有据可依，而不是只能靠训练记忆泛泛而谈。

搜索采用两级降级：先用 ddgs（免费、无需 key），失败再回退到 Tavily
（需 key，专为 AI agent 设计，返回结构化清洗片段）。两者都失败时返回空，
诊断流程仍可基于结构化数据继续，不会崩溃。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from app.config import get_settings

logger = logging.getLogger(__name__)

# 单个搜索结果字符数上限。网页摘要拼起来很容易很长，截断避免把 Prompt 撑爆。
_MAX_SEARCH_CHARS = 2500


def _truncate(text: str) -> str:
    """把搜索结果截断到安全长度，保护下游 Prompt 的 token 预算。"""
    return text[:_MAX_SEARCH_CHARS]


def _search_ddgs(query: str, num_results: int) -> str:
    """用 DuckDuckGo 搜索，失败抛异常由上层兜底。"""
    from langchain_community.tools import DuckDuckGoSearchResults

    search = DuckDuckGoSearchResults(num_results=num_results)
    results = search.invoke(query)
    if isinstance(results, str):
        return results
    return str(results)


def _search_tavily(query: str, num_results: int) -> str:
    """用 Tavily 搜索，失败抛异常由上层兜底。

    Tavily 返回 list[dict]（含 title/url/content），这里拼成可读文本塞进 Prompt。
    需要在 .env 配置 TAVILY_API_KEY，否则视为不可用。
    """
    api_key = get_settings().TAVILY_API_KEY.strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY 未配置")

    from langchain_community.tools import TavilySearchResults

    search = TavilySearchResults(max_results=num_results, api_key=api_key)
    results = search.invoke(query)
    if not results:
        return ""
    # results 是 list[dict]，含 url / content / title，拼成紧凑文本
    chunks = []
    for item in results:
        if isinstance(item, dict):
            content = item.get("content") or ""
            title = item.get("title") or ""
            if content:
                chunks.append(f"[{title}] {content}" if title else content)
    return "\n".join(chunks)


def _safe_search(query: str, num_results: int = 5) -> tuple[str, str]:
    """两级降级搜索：ddgs 失败则回退 Tavily。

    返回 ``(结果文本, 使用的引擎)``。两个引擎都失败时返回 ``("", "none")``，
    由调用方据 ``ok`` 字段降级，不抛异常以保证诊断流程不中断。
    """
    # 第一级：ddgs（免费）
    try:
        text = _search_ddgs(query, num_results)
        if text and text.strip():
            return _truncate(text), "ddgs"
        logger.info("ddgs 返回空结果，回退 Tavily")
    except Exception as exc:  # noqa: BLE001  搜索失败不应中断诊断
        logger.info("ddgs 搜索失败，回退 Tavily: %s", exc)

    # 第二级：Tavily 兜底
    try:
        text = _search_tavily(query, num_results)
        if text and text.strip():
            return _truncate(text), "tavily"
        logger.info("Tavily 返回空结果")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tavily 搜索也失败，将仅基于结构化数据生成诊断: %s", exc)

    return "", "none"


@tool
def web_search_tool(query: str) -> dict[str, Any]:
    """联网搜索公开网页，获取标的的最新消息、资金动向与机构观点。

    Parameters
    ----------
    query:
        搜索关键词，例如 ``中际旭创 最新消息 资金流向 机构评级``。

    Returns
    -------
    dict[str, Any]
        包含 ``query``、``results``（拼接后的网页摘要文本）、``ok``
        （是否成功拿到结果）和 ``engine``（实际使用的引擎：ddgs/tavily/none）。
        当两个引擎都失败时 ``ok`` 为 False、``results`` 为空串，调用方应据此
        降级而非报错。

    Notes
    -----
    采用 ddgs → Tavily 两级降级。ddgs 免费但偶发不稳定；Tavily 需在 .env 配置
    ``TAVILY_API_KEY``，返回专为 LLM 清洗过的结构化片段。结果为公开网页摘要，
    可能含噪声，仅供 LLM 做背景参考，不作为任何交易依据。
    """
    results, engine = _safe_search(query, num_results=5)
    return {
        "query": query,
        "results": results,
        "ok": bool(results),
        "engine": engine,
    }
