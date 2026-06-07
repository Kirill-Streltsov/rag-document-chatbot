"""Ingestion pipeline: PDF -> text -> chunks -> FAISS vector store.

This is the "document in" half of the resume's "document in, questions out".
Each step is a small pure-ish function so it can be unit-tested and reused
(e.g. by the retrieval-tuning script in ``scripts/eval_retrieval.py``).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config


def load_pdf(path: str | os.PathLike) -> list[Document]:
    """Parse a PDF into one Document per page, preserving page metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such PDF: {path}")
    docs = PyPDFLoader(str(path)).load()
    # Normalize the source to the clean basename (the loader defaults it to the
    # full path) so citations read as "report.pdf, page 3" not a temp path.
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def load_pdf_bytes(data: bytes, filename: str) -> list[Document]:
    """Parse a PDF provided as raw bytes (e.g. a Streamlit upload).

    PyPDFLoader needs a real path, so we round-trip through a temp file.
    """
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        docs = load_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)
    # Overwrite the temp path with the real filename for readable citations.
    for d in docs:
        d.metadata["source"] = filename
    return docs


def split_documents(
    docs: Iterable[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split page-level Documents into overlapping chunks.

    Overlap keeps sentences that straddle a chunk boundary retrievable from
    both sides, which measurably improves recall (see docs/EXPERIMENTS.md).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer splitting on natural boundaries before falling back to chars.
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(list(docs))


def build_vectorstore(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed chunks and index them in an in-memory FAISS store."""
    if not chunks:
        raise ValueError("Cannot build a vector store from zero chunks.")
    return FAISS.from_documents(chunks, embeddings)


def ingest_pdfs(
    sources: list[tuple[bytes, str]],
    embeddings: Embeddings,
    cfg: Config,
) -> tuple[FAISS, int]:
    """End-to-end ingest for one or more uploaded PDFs.

    ``sources`` is a list of ``(pdf_bytes, filename)`` tuples.
    Returns the FAISS store and the number of chunks indexed.
    """
    pages: list[Document] = []
    for data, name in sources:
        pages.extend(load_pdf_bytes(data, name))
    chunks = split_documents(pages, cfg.chunk_size, cfg.chunk_overlap)
    store = build_vectorstore(chunks, embeddings)
    return store, len(chunks)
