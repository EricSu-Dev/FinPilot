"""财报 RAG 的共享入库逻辑。

原先这里还挂着 `/api/report/upload` 与 `/api/report/query` 两个 HTTP 端点（供
独立的「财报解读」页用）。该页与对话首页功能重叠、体验也不如首页（无时效标注、
不能跨会话、不能多轮追问），已下线；这两个端点随之删除。

保留的是 ``ingest_uploaded_report``——对话首页的 ``/api/chat/upload`` 仍在调用它
把 PDF 落盘并构建 Chroma 语料库。问答检索则由对话 agent 的工具
（``query_stock_report`` / ``query_uploaded_report``）直接调
``answer_report_question`` 完成，不再走 HTTP 端点。
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.rag.retriever import (
    build_corpus_for_stock,
    reset_corpus_cache,
    CHROMA_ROOT,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

# Chroma 的 collection 名只接受 [a-zA-Z0-9._-]（3-512 字符）。财报语料库标识
# report_id 会直接拼进 collection 名与持久化目录名，因此必须满足该约束——
# 这也是不传股票代码、从中文文件名推导时会 500 的根因。
REPORT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{3,512}$")

# 背景构建用进程间锁超时（秒），防止 OOM/API 卡死导致提问端无限等。
_BUILD_LOCK_TIMEOUT = 900


def _build_report_id(code: str, year: int | None, quarter: int | None) -> str:
    """根据股票代码 + 年份 + 季度拼出财报语料库标识 report_id。

    格式为 `代码_年份_季度`，季度映射：1季度=1、半年报=2、3季度=3、年报=4。
    未提供年份/季度时退化为纯代码（向后兼容单份财报的场景）。
    """
    code = code.strip()
    if not code:
        raise ValueError("缺少股票代码 code。")
    if year is not None and quarter is not None:
        if quarter not in (1, 2, 3, 4):
            raise ValueError("quarter 必须为 1-4（1季度=1，半年报=2，3季度=3，年报=4）。")
        report_id = f"{code}_{int(year)}_{int(quarter)}"
    else:
        report_id = code
    if not REPORT_ID_PATTERN.match(report_id):
        raise ValueError(
            f"财报标识 report_id 非法：{report_id!r}。"
            f"只能包含字母、数字、点、下划线、短横线（3-512 字符）；"
            f"请提供 ASCII 股票代码（如 300308），不要用中文文件名推导。"
        )
    return report_id


def _rmtree_with_retry(path: Path, retries: int = 3, delay: float = 0.5) -> None:
    """删除目录，Windows 下文件句柄释放有延迟时重试。

    Chroma 的 sqlite/mmap 句柄即使清了 lru_cache、做了 gc，在 Windows 上也可能
    需要一小段才真正释放，直接 ``shutil.rmtree`` 容易抛 WinError 32。重试几次
    通常即可成功。
    """
    import gc
    import time

    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except Exception:
            if attempt == retries - 1:
                raise
            gc.collect()
            time.sleep(delay * (attempt + 1))


def ingest_uploaded_report(
    file: UploadFile,
    code: str,
    year: int | None,
    quarter: int | None,
) -> dict[str, Any]:
    """把上传的财报 PDF 落盘，清理旧缓存，返回 report_id。

    向量语料库构建已移至后台异步执行（``build_report_corpus_background``），
    避免 4000+ chunk 的 embedding API 调用阻塞 HTTP 响应导致超时。

    供对话首页的 ``/api/chat/upload`` 调用。返回 ``{"report_id": ...,
    "status": "saved"}``；调用方负责包 api_ok/api_error。
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    report_id = _build_report_id(code, year, quarter)
    logger.info("[上传] step1 report_id=%s code=%s year=%s quarter=%s", report_id, code, year, quarter)

    destination = UPLOAD_DIR / f"{report_id}.pdf"

    # 先写新文件再清旧：避免删到客户端正在读取的同名源文件（命令行 curl 场景）。
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logger.info("[上传] step2 PDF 已写入 uploads/  size=%d", destination.stat().st_size)

    # 仅清理同一 report_id 的旧 PDF（同公司同季度重传覆盖）；不同季度/年份的
    # 文件名中 report_id 不同，不会被误删，从而共存。
    for stale_pdf in DATA_DIR.glob(f"*{report_id}*.pdf"):
        if stale_pdf.is_file() and stale_pdf != destination:
            stale_pdf.unlink()
    for stale_pdf in UPLOAD_DIR.glob(f"*{report_id}*.pdf"):
        if stale_pdf.is_file() and stale_pdf != destination:
            stale_pdf.unlink()

    # 在 data/ 下保留规范化副本，检索器按 report_id 定位该文件。
    canonical_path = DATA_DIR / f"{report_id}.pdf"
    shutil.copyfile(destination, canonical_path)
    logger.info("[上传] step3 PDF 已复制到 data/  canonical=%s", canonical_path)

    chroma_cache = DATA_DIR / "chroma_db" / report_id
    if chroma_cache.exists():
        logger.info("[上传] step4 发现旧向量库，准备删除 %s", chroma_cache)
        reset_corpus_cache()
        _rmtree_with_retry(chroma_cache)
        logger.info("[上传] step4 旧向量库已删除")

    return {"report_id": report_id, "status": "saved"}


def build_report_corpus_background(report_id: str) -> None:
    """在后台构建向量语料库（耗时较长，由上传端点以线程触发）。

    构建前创建 ``.building`` 标记文件，构建后删除。提问端看到标记会等待
    构建完成再返回回答，避免"上传成功但问不了"。
    """
    marker = CHROMA_ROOT / report_id.strip() / ".building"
    logger.info("[后台入库] 开始构建向量语料库 report_id=%s ...", report_id)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        chunks = build_corpus_for_stock(report_id, is_background_build=True)
        logger.info("[后台入库] 完成 report_id=%s chunks=%d", report_id, len(chunks))
    except Exception:
        logger.exception("[后台入库] 失败 report_id=%s", report_id)
    finally:
        try:
            marker.unlink(missing_ok=True)
        except Exception:
            pass
