"""财报的 RAG 问答链。"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - 兼容旧版 LangChain 布局的回退。
    from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import llm
from app.rag.retriever import corpus_is_ready, hybrid_retrieve

RAG_DISCLAIMER = "文档中未找到相关信息"

# 向量库后台构建中时给用户的提示。
# 后台线程需要 30+ 秒才能开始写入 embedding，全部 chunks 写完约需 3-6 分钟。
_BUILDING_NO_RESULT_MSG = (
    "财报向量库正在后台构建中，关键词匹配暂未找到相关内容。"
    "请稍等 1-2 分钟后重新提问，届时将获得基于向量检索的精确回答。"
)
_BUILDING_FALLBACK_NOTE = (
    "【提示】财报向量库仍在后台构建中，当前回答基于关键词匹配，精度有限。"
    "建议 2-5 分钟后重新提问以获得更准确的检索结果。\n\n"
)


def _source_chunk_payload(document: Document) -> dict[str, Any]:
    """把 LangChain document 转换为可序列化的块载荷。"""
    metadata = dict(document.metadata or {})
    return {
        "content": document.page_content,
        "metadata": metadata,
        "page": metadata.get("page"),
    }


def _page_numbers(documents: list[Document]) -> list[int]:
    """返回一组文档中排序去重后的页码。"""
    pages = sorted({int(doc.metadata.get("page")) for doc in documents if doc.metadata and doc.metadata.get("page") is not None})
    return pages


def answer_report_question(query: str, stock_code: str, top_k: int = 12) -> dict[str, Any]:
    """仅根据检索到的块回答关于财报的问题。

    向量库后台构建中时会通过 corpus_is_ready 自动检测，未就绪时：
    - BM25 无结果 -> 提示用户稍后重试（而非报"未找到"误导用户）
    - BM25 有结果 -> 在回答前附加精度提示
    """
    source_chunks = hybrid_retrieve(query=query, stock_code=stock_code, top_k=top_k)
    page_numbers = _page_numbers(source_chunks)
    is_building = not corpus_is_ready(stock_code)

    if not source_chunks:
        if is_building:
            return {
                "answer": _BUILDING_NO_RESULT_MSG,
                "source_chunks": [],
                "page_numbers": [],
            }
        return {
            "answer": RAG_DISCLAIMER,
            "source_chunks": [],
            "page_numbers": [],
        }

    context_lines = []
    for index, chunk in enumerate(source_chunks, start=1):
        metadata = chunk.metadata or {}
        page = metadata.get("page", "未知")
        content_type = metadata.get("content_type", "paragraph")
        context_lines.append(
            f"[chunk {index} | page {page} | {content_type}]\n{chunk.page_content}"
        )

    prompt_prefix = _BUILDING_FALLBACK_NOTE if is_building else ""
    prompt = (
        prompt_prefix +
        "你是 FinPilot 的财报问答助手，只能根据给定的检索内容回答，必须用中文。\n\n"
        "判断与回答规则：\n"
        "1. 先判断检索内容是否与问题相关。\n"
        "2. 仅当检索内容与问题完全无关时，才回复\"文档中未找到相关信息\"。\n"
        "3. 若检索到相关财务数据（哪怕不完整），必须基于已有数据回答或分析，"
        "不要因为不完整就拒绝作答。\n"
        "   - 对分析/评价类问题（如\"评价这家公司\"），把检索到的财务数据组织成结构化评价："
        "营收、净利、增长率、现金流、资产负债等维度，有数据的列出数值并标注页码，"
        "没检索到的维度明确写\"未在检索片段中找到\"，不要编造。\n"
        "4. 关键结论后标注来源页码，例如\"(来源：第3页、第5页)\"。\n"
        "5. 不要输出投资建议，不要编造财务数据，不要补全缺失数字。"
        f"\n\n问题：{query}\n\n检索内容：\n" + "\n\n".join(context_lines)
    )

    # DashScope 的 DeepSeek 兼容接口不支持 with_structured_output 所用的
    # response_format（会返回 400 "This response_format type is unavailable now"），
    # 因此改为普通 invoke，直接取文本内容作为答案；输出格式已在 prompt 中约束。
    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你只能根据给定的检索内容回答，而且必须输出中文。"
                    "如果信息不足，必须明确说明文档中未找到相关信息。"
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    answer = getattr(result, "content", None)
    if not isinstance(answer, str):
        answer = str(answer) if answer is not None else ""

    return {
        "answer": answer.strip() or RAG_DISCLAIMER,
        "source_chunks": [_source_chunk_payload(chunk) for chunk in source_chunks],
        "page_numbers": page_numbers,
    }
