"""财报文档的分块工具。"""

from __future__ import annotations

from typing import Any

try:  # LangChain 的分割器在新版中迁入了单独的包。
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - 兼容旧版 LangChain 布局的回退。
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - 兼容旧版 LangChain 布局的回退。
    from langchain.schema import Document


def chunk_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """把 Document 切成更小的块，同时保留来源元数据。"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []

    for parent_index, document in enumerate(documents):
        split_chunks = splitter.split_text(document.page_content)
        for chunk_index, chunk_text in enumerate(split_chunks):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "parent_index": parent_index,
                    "chunk_index": chunk_index,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            )
            chunks.append(Document(page_content=chunk_text, metadata=metadata))

    return chunks
