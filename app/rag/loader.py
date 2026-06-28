"""财报文档的 PDF 加载器。

加载器把段落与表格分开保存，这样下游的分块与检索可以独立地处理叙述性段落与
表格化事实。对于表格，我们保留行列结构，而不是把单元格拍平成单段文字。这对
财报很重要，因为一个数字的位移可能彻底改变某个指标的含义。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

try:  # LangChain 在新版本中把 Document 跨包迁移了。
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - 兼容旧版 LangChain 布局的回退。
    from langchain.schema import Document


def _normalize_stock_code(stock_code: str | None, pdf_path: Path) -> str:
    """当调用方未提供股票代码时，推导出一个稳定的股票代码。"""
    if stock_code:
        return stock_code.strip()

    stem = pdf_path.stem
    for token in stem.replace("-", "_").split("_"):
        if token:
            return token
    return stem


def _rect_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """两个矩形重叠时返回 True。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _page_tables(page: fitz.Page) -> list[tuple[tuple[float, float, float, float], list[list[Any]]]]:
    """从页面中提取表格，同时保留每个表格的边界框。"""
    tables: list[tuple[tuple[float, float, float, float], list[list[Any]]]] = []
    if not hasattr(page, "find_tables"):
        return tables

    try:
        table_finder = page.find_tables()
    except Exception:
        return tables

    for table in getattr(table_finder, "tables", []) or []:
        bbox = tuple(getattr(table, "bbox", (0.0, 0.0, 0.0, 0.0)))  # type: ignore[arg-type]
        try:
            rows = table.extract()
        except Exception:
            rows = []
        if rows:
            tables.append((bbox, rows))
    return tables


def _table_rows_to_text(rows: list[list[Any]]) -> str:
    """把表格矩阵转换为以制表符分隔的文本块。"""
    lines: list[str] = []
    for row in rows:
        cells = []
        for cell in row:
            if cell is None:
                cells.append("")
            else:
                cells.append(str(cell).strip())
        lines.append("\t".join(cells).rstrip())
    return "\n".join(lines).strip()


def _blocks_to_documents(
    blocks: list[tuple[Any, ...]],
    *,
    pdf_name: str,
    stock_code: str,
    page_number: int,
    extracted_at: str,
    table_bboxes: list[tuple[float, float, float, float]],
) -> list[Document]:
    """把页面文本块转为 LangChain Document，同时跳过表格区域。"""
    documents: list[Document] = []
    for index, block in enumerate(blocks):
        if len(block) < 5:
            continue

        x0, y0, x1, y1 = map(float, block[:4])
        text = str(block[4]).strip()
        if not text:
            continue

        block_bbox = (x0, y0, x1, y1)
        if any(_rect_overlaps(block_bbox, table_bbox) for table_bbox in table_bboxes):
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "filename": pdf_name,
                    "page": page_number,
                    "stock_code": stock_code,
                    "extracted_at": extracted_at,
                    "content_type": "paragraph",
                    "block_index": index,
                },
            )
        )
    return documents


def load_financial_report_documents(pdf_path: str | Path, stock_code: str | None = None) -> list[Document]:
    """把财报 PDF 解析为段落与表格 Document。

    每个返回的 Document 在元数据中携带来源文件名、从 1 开始计数的页码、
    股票代码与抽取时间戳。这些元数据之后能帮助检索器与 QA 链引用确切的
    来源页。
    """
    source_path = Path(pdf_path)
    if not source_path.exists():
        raise FileNotFoundError(f"未找到财报 PDF：{source_path}")

    resolved_stock_code = _normalize_stock_code(stock_code, source_path)
    extracted_at = datetime.now(timezone.utc).isoformat()
    documents: list[Document] = []

    with fitz.open(source_path) as pdf:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_number = page_index + 1
            table_entries = _page_tables(page)
            table_bboxes = [bbox for bbox, _ in table_entries]

            for table_index, (bbox, rows) in enumerate(table_entries):
                table_text = _table_rows_to_text(rows)
                if not table_text:
                    continue

                documents.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "filename": source_path.name,
                            "page": page_number,
                            "stock_code": resolved_stock_code,
                            "extracted_at": extracted_at,
                            "content_type": "table",
                            "table_index": table_index,
                            "bbox": ",".join(str(value) for value in bbox),
                            "row_count": len(rows),
                            "column_count": max((len(row) for row in rows), default=0),
                        },
                    )
                )

            blocks = page.get_text("blocks", sort=True)
            documents.extend(
                _blocks_to_documents(
                    blocks,
                    pdf_name=source_path.name,
                    stock_code=resolved_stock_code,
                    page_number=page_number,
                    extracted_at=extracted_at,
                    table_bboxes=table_bboxes,
                )
            )

    return documents
