"""财报 RAG 的混合检索。

本模块组合了三种检索信号：
1. 向量相似度（阿里百炼 DashScope Embeddings），用于语义召回。
2. BM25，用于精确词项与股票代码/金融术语的召回。
3. 词项重叠重排，用于最终排序（轻量级，不依赖本地模型）。

这种组合对财报很重要，因为有用的事实往往位于表格中，使用诸如"毛利率"或
"研发"这类精确术语，并且在不同章节中的表述可能略有差异。
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from functools import lru_cache
from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - 兼容旧版 LangChain 布局的回退。
    from langchain.schema import Document

from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

from app.rag.chunker import chunk_documents
from app.rag.loader import load_financial_report_documents
from app.config import get_settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_ROOT = DATA_DIR / "chroma_db"

# 百炼最新一代向量模型（Qwen3-Embedding），支持 100+ 语种，2K 维度。
DASHSCOPE_MODEL = "text-embedding-v4"
# v4 每批最多 10 条（v1/v2 是 25 条），每条最长 8192 token。
# LangChain 的 DashScopeEmbeddings 内部已做分批，无需手动处理。

# 向量库后台构建中的标记文件。上传接口立即返回后会启动后台线程构建向量库，
# 期间提问端看到该标记会等待构建完成再返回回答。
# 标记超过此秒数视为废弃（上次构建崩溃残留），提问端不再等待直接自行重建。
_BUILDING_SUFFIX = ".building"
_BUILDING_STALE_SECONDS = 1800

# DashScope OpenAI 兼容端点（避免 DashScope 原生 API 的 SSL 兼容性问题）
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _ensure_dir(path: Path) -> Path:
    """目录不存在时创建它。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_text(value: Any) -> str:
    """把一个值转换为稳定的检索文本。"""
    if value is None:
        return ""
    return str(value).strip()


def _doc_fingerprint(document: Document) -> str:
    """为检索结果去重构建稳定的指纹。"""
    metadata = document.metadata or {}
    key = "|".join(
        [
            _safe_text(metadata.get("filename")),
            _safe_text(metadata.get("page")),
            _safe_text(metadata.get("content_type")),
            _safe_text(metadata.get("chunk_index")),
            document.page_content,
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _find_pdf_for_stock(stock_code: str) -> Path:
    """在 data/ 下定位该 report_id 对应的财报 PDF。

    优先精确匹配 ``{report_id}.pdf``，避免 report_id 作为子串时误命中其他财报
    （例如 report_id="300308" 会子串匹配到 "300308_2025_4.pdf"）。找不到精确
    名时再回退到子串匹配，兼容旧的非规范命名文件。
    """
    normalized = stock_code.strip()
    exact = DATA_DIR / f"{normalized}.pdf"
    if exact.is_file():
        return exact
    candidates = sorted(
        [path for path in DATA_DIR.glob("*.pdf") if normalized.lower() in path.name.lower()]
        + [path for path in DATA_DIR.glob("*.PDF") if normalized.lower() in path.name.lower()]
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {DATA_DIR} 中未找到 report_id {normalized} 对应的财报 PDF。"
            f"请先通过首页上传按钮或 /api/chat/upload 上传该财报。"
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------


def _build_embeddings():
    """通过 DashScope OpenAI 兼容端点创建 Embeddings 封装。

    原用 langchain_community.DashScopeEmbeddings（服务器 SSL 报错），后换
    langchain_openai.OpenAIEmbeddings（v1.3.3 用新版 Responses API 格式，
    DashScope 兼容端点不认）。现直接用 openai SDK 薄封装，格式完全可控。
    """
    from openai import OpenAI

    settings = get_settings()
    api_key = (settings.DASHSCOPE_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY 未配置，财报 RAG 的向量检索不可用。"
            "请在 .env 中设置有效的阿里云百炼 API Key。"
        )
    return _DashScopeOpenAIWrapper(
        client=OpenAI(
            api_key=api_key,
            base_url=_DASHSCOPE_BASE_URL,
        ),
        model=DASHSCOPE_MODEL,
    )


class _DashScopeOpenAIWrapper:
    """薄封装 openai.OpenAI，对外暴露 LangChain Embeddings 兼容接口。

    直接调用 client.embeddings.create()，参数格式完全可控，不依赖
    langchain_openai 的中间层（其新版本可能改变请求格式导致兼容性问题）。
    """

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # 兼容端点每次最多 10 条，手动分批
        all_embeddings: list[list[float]] = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=1024,
            )
            all_embeddings.extend(d.embedding for d in resp.data)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=1024,
        )
        return resp.data[0].embedding


class TermOverlapReranker:
    """基于词项重叠度对候选块重新排序。

    轻量级重排器，不依赖本地模型，适合低配服务器（2GB 内存）。
    """

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """按词项重叠度打分后返回最相关的文档。"""
        if not documents:
            return []

        query_terms = set(re.findall(r"[一-鿿]+|[A-Za-z0-9_.%/-]+", query.lower()))
        if not query_terms:
            query_terms = {char for char in query.lower() if not char.isspace()}
        scored: list[tuple[float, Document]] = []
        for document in documents:
            text = document.page_content.lower()
            overlap = sum(1 for term in query_terms if term in text)
            scored.append((float(overlap), document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


def _building_marker(stock_code: str) -> Path:
    """返回向量库构建中的标记文件路径。"""
    return CHROMA_ROOT / stock_code.strip() / _BUILDING_SUFFIX


def corpus_is_ready(stock_code: str) -> bool:
    """向量库是否已构建完毕、可供查询。

    有 .building 标记且未过期 = 后台构建中，未就绪。
    标记已过期（上次构建崩溃残留）= 视为未构建，返回 False（触发重建）。
    无标记且目录存在 = 就绪。
    """
    marker = _building_marker(stock_code)
    if marker.exists():
        try:
            age = time.time() - marker.stat().st_mtime
        except Exception:
            age = 0
        if age < _BUILDING_STALE_SECONDS:
            return False  # 仍在构建中
        # 标记过期，清理掉
        logger.warning("[检索] .building 标记已过期(%.0fs)，视为废弃并清理 %s", age, stock_code)
        try:
            marker.unlink(missing_ok=True)
        except Exception:
            pass
    stock_dir = CHROMA_ROOT / stock_code.strip()
    return stock_dir.is_dir()


def _wait_for_corpus_build(stock_code: str, timeout: int = 900) -> bool:
    """等待后台构建完成。timeout 秒内未完成返回 False。"""
    marker = _building_marker(stock_code)
    if not marker.exists():
        return True  # 没人构建，无需等
    logger.info("[入库] step5 向量库后台构建中，等待完成 report_id=%s ...", stock_code)
    deadline = time.monotonic() + timeout
    while marker.exists():
        if time.monotonic() > deadline:
            logger.error("[入库] step5 等待后台构建超时 report_id=%s", stock_code)
            return False
        time.sleep(2)
    logger.info("[入库] step5 后台构建已完成 report_id=%s", stock_code)
    return True


def _load_or_build_vectorstore(stock_code: str, documents: list[Document], *, is_background_build: bool = False) -> tuple[Chroma, list[Document]]:
    """有已持久化的 Chroma 存储时加载，否则从头构建。

    chromadb 大版本升级后 SQLite schema 可能不兼容（如出现 Rust backend 的
    ``no such table: tenants``），此时自动清理旧库重建，避免上传卡死。

    上传端点改为异步后，可能出现"向量库正在后台构建、同时用户提问"的并发
    场景——此时等待后台构建完成再加载，避免提问端另起一个构建造成冲突。
    """
    import gc, shutil
    from datetime import datetime

    embedding_function = _build_embeddings()
    logger.info("[入库] step5 嵌入模型已初始化 model=%s", DASHSCOPE_MODEL)

    stock_dir = _ensure_dir(CHROMA_ROOT / stock_code.strip())

    def _create_vs() -> Chroma:
        return Chroma(
            collection_name=f"financial_report_{stock_code.strip()}",
            persist_directory=str(stock_dir),
            embedding_function=embedding_function,
        )

    # 先尝试打开已有库
    try:
        vectorstore = _create_vs()
        existing_count = vectorstore._collection.count()
        logger.info("[入库] step5 已有向量库 count=%d", existing_count)
    except Exception:
        logger.warning(
            "[入库] step5 Chroma 打开 %s 失败（schema 不兼容或损坏），将删除旧库重建",
            stock_code,
        )
        # 释放可能持有的文件句柄后删除旧库
        reset_corpus_cache()
        gc.collect()
        for _ in range(3):
            try:
                if stock_dir.exists():
                    shutil.rmtree(stock_dir)
                break
            except Exception:
                time.sleep(0.5)
        stock_dir = _ensure_dir(CHROMA_ROOT / stock_code.strip())
        vectorstore = _create_vs()
        existing_count = 0
        logger.info("[入库] step5 旧库已删除，新库已创建")

    if existing_count > 0:
        stored = vectorstore.get(include=["documents", "metadatas"])
        loaded_docs: list[Document] = []
        for doc_text, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
            loaded_docs.append(Document(page_content=doc_text, metadata=metadata or {}))
        logger.info("[入库] step6 从已有库加载 docs=%d", len(loaded_docs))
        return vectorstore, loaded_docs

    if not documents:
        # 无文档且库为空：可能在后台构建中，等待完成
        if _building_marker(stock_code).exists():
            if _wait_for_corpus_build(stock_code):
                # 重新打开已构建好的库
                vectorstore = _create_vs()
                stored = vectorstore.get(include=["documents", "metadatas"])
                loaded_docs: list[Document] = []
                for doc_text, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
                    loaded_docs.append(Document(page_content=doc_text, metadata=metadata or {}))
                logger.info("[入库] step6 等待后台构建后加载 docs=%d", len(loaded_docs))
                return vectorstore, loaded_docs
            raise RuntimeError(f"向量库 {stock_code} 后台构建超时，请稍后重试")
        raise ValueError(f"没有可用的源文档来为 {stock_code} 构建向量库")

    # 有文档但库为空：检查是否后台正在构建
    # （提问端自己解析了 PDF 带文档进来，但后台线程可能正在写同一个库）
    # is_background_build=True 表示调用者就是后台构建线程本人，直接跳过等待继续构建。
    if not is_background_build and _building_marker(stock_code).exists():
        logger.info("[入库] step6 检测到后台构建中，等待完成以避免竞态 report_id=%s", stock_code)
        if _wait_for_corpus_build(stock_code):
            vectorstore = _create_vs()
            stored = vectorstore.get(include=["documents", "metadatas"])
            loaded_docs: list[Document] = []
            for doc_text, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
                loaded_docs.append(Document(page_content=doc_text, metadata=metadata or {}))
            logger.info("[入库] step6 等待后台构建后加载 docs=%d", len(loaded_docs))
            return vectorstore, loaded_docs
        raise RuntimeError(f"向量库 {stock_code} 后台构建超时，请稍后重试")

    logger.info("[入库] step6 开始生成 embedding 并写入向量库 docs=%d ...", len(documents))
    t_emb0 = datetime.now()
    vectorstore.add_documents(documents)
    t_emb1 = datetime.now()
    logger.info("[入库] step6 embedding 生成完成 耗时=%.1fs", (t_emb1 - t_emb0).total_seconds())

    # chromadb >= 0.5 用 PersistentClient 自动持久化，persist() 已移除。
    try:
        vectorstore.persist()
    except AttributeError:
        pass
    logger.info("[入库] step6 向量库已持久化到 %s", stock_dir)
    return vectorstore, documents


@lru_cache(maxsize=8)
def _prepare_stock_corpus(stock_code: str, *, is_background_build: bool = False) -> tuple[Chroma, tuple[Document, ...]]:
    """每个会话内为单只股票加载或构建一次持久化语料库。"""
    from datetime import datetime

    t0 = datetime.now()
    logger.info("[入库] step1 开始处理 report_id=%s", stock_code)

    pdf_path = _find_pdf_for_stock(stock_code)
    logger.info("[入库] step2 找到 PDF: %s", pdf_path)

    source_docs = load_financial_report_documents(pdf_path, stock_code=stock_code)
    t1 = datetime.now()
    logger.info("[入库] step3 PDF 解析完成 docs=%d 耗时=%.1fs", len(source_docs), (t1 - t0).total_seconds())

    chunks = tuple(chunk_documents(source_docs))
    t2 = datetime.now()
    logger.info("[入库] step4 分块完成 chunks=%d 耗时=%.1fs", len(chunks), (t2 - t1).total_seconds())

    vectorstore, indexed_docs = _load_or_build_vectorstore(stock_code, list(chunks), is_background_build=is_background_build)
    t3 = datetime.now()
    logger.info("[入库] step7 向量库构建完成 耗时=%.1fs 总耗时=%.1fs", (t3 - t2).total_seconds(), (t3 - t0).total_seconds())
    return vectorstore, tuple(indexed_docs)


def _bm25_retrieve(documents: list[Document], query: str, top_n: int) -> list[Document]:
    """使用 BM25 检索精确词项匹配。"""
    if not documents:
        return []

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = top_n
    try:
        return retriever.invoke(query)
    except Exception:
        try:
            return retriever.get_relevant_documents(query)
        except Exception:
            return []


def _vector_retrieve(vectorstore: Chroma, query: str, top_n: int) -> list[Document]:
    """从 Chroma 检索语义相似的块。"""
    try:
        return vectorstore.similarity_search(query, k=top_n)
    except Exception:
        return []


def _unique_documents(documents: list[Document]) -> list[Document]:
    """对检索到的文档去重，同时保留顺序。"""
    seen: set[str] = set()
    unique_docs: list[Document] = []
    for document in documents:
        fingerprint = _doc_fingerprint(document)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_docs.append(document)
    return unique_docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_corpus_for_stock(stock_code: str, *, is_background_build: bool = False) -> list[Document]:
    """为某只股票代码加载、分块并持久化报告语料库。"""
    _, indexed_docs = _prepare_stock_corpus(stock_code, is_background_build=is_background_build)
    return list(indexed_docs)


def reset_corpus_cache() -> None:
    """清理语料库缓存，释放对 Chroma 句柄的引用。

    上传重传同一 report_id 时，持久化目录会被 rmtree 重建。但 ``_prepare_stock_corpus``
    的 ``@lru_cache`` 仍持有上一次构建的 Chroma vectorstore 对象，它在 Windows 上
    占着库文件句柄（sqlite + mmap），导致 rmtree 抛 ``WinError 32``。删库前调用本
    函数清空缓存，让旧 vectorstore 可被回收、释放句柄；清空后下次查询会重新加载
    持久化库。
    """
    _prepare_stock_corpus.cache_clear()


def find_report_id_for_stock(code: str) -> str | None:
    """在 data/chroma_db/ 下找该股票已上传的财报语料库 report_id。

    report_id 命名为 ``{code}``（单份模式）或 ``{code}_{year}_{quarter}``（多季度）。
    有多份时取最新（year+quarter 最大；纯 ``{code}`` 视为最旧）。返回最新那份的
    report_id；该股票从未上传过财报则返回 None。

    供对话 agent 的"按股票名查财报"工具使用——只要某只股票的财报曾被上传过
    （语料落盘在 chroma_db），跨会话也能查到。
    """
    code = code.strip()
    if not code or not CHROMA_ROOT.exists():
        return None

    candidates: list[tuple[str, int, int]] = []  # (report_id, year, quarter)
    for entry in CHROMA_ROOT.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name == code:
            # 单份模式，视为最旧（-1, -1），有季度数据时会被更具体的覆盖
            candidates.append((name, -1, -1))
        elif name.startswith(f"{code}_"):
            parts = name.split("_")
            if len(parts) == 3:
                try:
                    candidates.append((name, int(parts[1]), int(parts[2])))
                except ValueError:
                    continue

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[1], t[2]))
    return candidates[-1][0]


def report_vintage(report_id: str) -> tuple[str, bool]:
    """从 report_id 解析报告期，返回 (中文标签, 是否陈旧)。

    report_id 形如 ``{code}_{year}_{quarter}``（quarter：1=一季报、2=半年报、
    3=三季报、4=年报）或 ``{code}``（单份模式，无报告期信息）。

    陈旧判定：报告期末超过 18 个月则视为陈旧——财报有时效性，拿两年前的年报
    回答"现在怎么样"会误导用户，需要显式提示。标签始终返回，让模型在回答里
    说清报告期，是否陈旧由 is_stale 标记。
    """
    from datetime import date

    today = date.today()
    parts = report_id.split("_")
    if len(parts) == 3:
        try:
            year = int(parts[1])
            quarter = int(parts[2])
        except ValueError:
            return ("未标注报告期", False)
        quarter_label = {1: "一季报", 2: "半年报", 3: "三季报", 4: "年报"}.get(quarter, f"第{quarter}期")
        end_month = {1: 3, 2: 6, 3: 9, 4: 12}.get(quarter, 12)
        label = f"{year}年{quarter_label}"
        try:
            period_end = date(year, end_month, 28)  # 月末近似，够用
            is_stale = (today - period_end).days > 548  # ~18 个月
        except ValueError:
            is_stale = False
        return label, is_stale
    # 单份模式 report_id 只有 code，没有年份季度，无法判断报告期
    return ("未标注报告期（单份模式）", False)


def report_age_days(report_id: str) -> int | None:
    """report_id 的报告期末距今多少天。无日期（单份模式 {code}）返回 None。

    供"分析财报"工具判断该财报是最近一年内（不补搜索）还是超过一年（补搜索）。
    """
    from datetime import date

    parts = report_id.split("_")
    if len(parts) != 3:
        return None
    try:
        year = int(parts[1])
        quarter = int(parts[2])
    except ValueError:
        return None
    end_month = {1: 3, 2: 6, 3: 9, 4: 12}.get(quarter, 12)
    try:
        period_end = date(year, end_month, 28)  # 月末近似
        return (date.today() - period_end).days
    except ValueError:
        return None


def _bm25_retrieve_fast(stock_code: str, query: str, top_n: int) -> list[Document]:
    """加载 PDF 分块后直接用 BM25 检索（跳过 Chroma 向量库）。

    向量库后台构建中时调用，秒级返回，不会因等待构建而超时。
    低配服务器上 PDF 解析与分块可能因内存不足失败，异常时返回空列表。
    """
    from datetime import datetime

    t0 = datetime.now()
    try:
        pdf_path = _find_pdf_for_stock(stock_code)
        source_docs = load_financial_report_documents(pdf_path, stock_code=stock_code)
        chunks = list(chunk_documents(source_docs))
        t1 = datetime.now()
        logger.info("[BM25快速] PDF解析+分块完成 docs=%d chunks=%d 耗时=%.1fs", len(source_docs), len(chunks), (t1 - t0).total_seconds())
        result = _bm25_retrieve(chunks, query, top_n)
        t2 = datetime.now()
        logger.info("[BM25快速] 检索完成 found=%d 耗时=%.1fs", len(result), (t2 - t1).total_seconds())
        return result
    except Exception:
        logger.exception("[BM25快速] 失败（可能 OOM）report_id=%s", stock_code)
        return []


def hybrid_retrieve(query: str, stock_code: str, top_k: int = 5) -> list[Document]:
    """运行向量检索、BM25 与重排，返回最佳块。

    向量库后台构建中时退化为纯 BM25（秒回），避免等待构建导致 HTTP 超时。
    构建完成后恢复完整的向量 + BM25 混合检索。
    """
    def _bm25_fallback() -> list[Document]:
        candidates = _bm25_retrieve_fast(stock_code, query, max(top_k * 3, top_k))
        reranker = TermOverlapReranker()
        return reranker.rerank(query, candidates, top_k)

    # 向量库仍在后台构建中 → BM25 秒回，不等待
    if not corpus_is_ready(stock_code):
        logger.info("[检索] 向量库未就绪，降级为纯 BM25 report_id=%s", stock_code)
        return _bm25_fallback()

    try:
        vectorstore, indexed_docs = _prepare_stock_corpus(stock_code)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[检索] 向量库加载或构建失败，回退到 BM25 report_id=%s: %s", stock_code, exc)
        return _bm25_fallback()

    vector_candidates = _vector_retrieve(vectorstore, query, max(top_k * 3, top_k))
    keyword_candidates = _bm25_retrieve(list(indexed_docs), query, max(top_k * 3, top_k))
    merged = _unique_documents([*vector_candidates, *keyword_candidates])

    reranker = TermOverlapReranker()
    return reranker.rerank(query, merged, top_k=top_k)
