from __future__ import annotations

import pytest

from ragchat.ingest import (
    build_vectorstore,
    load_pdf,
    load_pdf_bytes,
    split_documents,
)


def test_load_pdf_parses_pages_and_tags_source(sample_pdf_path):
    docs = load_pdf(sample_pdf_path)
    assert len(docs) >= 3  # the sample report spans multiple pages
    assert all(d.metadata.get("source") == sample_pdf_path.name for d in docs)
    text = " ".join(d.page_content for d in docs)
    assert "84.6 million" in text
    assert "Jane Doe" in text


def test_load_pdf_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nope.pdf")


def test_load_pdf_bytes_uses_given_filename(sample_pdf_path):
    data = sample_pdf_path.read_bytes()
    docs = load_pdf_bytes(data, "renamed.pdf")
    assert docs and all(d.metadata["source"] == "renamed.pdf" for d in docs)


def test_split_documents_creates_overlap(sample_pages):
    chunks = split_documents(sample_pages, chunk_size=60, chunk_overlap=20)
    # Small chunks over the fixture text should produce several pieces...
    assert len(chunks) > len(sample_pages)
    # ...each carrying a start_index so provenance is traceable.
    assert all("start_index" in c.metadata for c in chunks)
    # page metadata is preserved through the split
    assert {c.metadata["page"] for c in chunks} == {0, 1}


def test_split_documents_rejects_bad_overlap(sample_pages):
    with pytest.raises(ValueError):
        split_documents(sample_pages, chunk_size=100, chunk_overlap=100)


def test_build_vectorstore_indexes_all_chunks(sample_pages, fake_embeddings):
    chunks = split_documents(sample_pages, chunk_size=200, chunk_overlap=40)
    store = build_vectorstore(chunks, fake_embeddings)
    # FAISS index should hold exactly one vector per chunk.
    assert store.index.ntotal == len(chunks)


def test_build_vectorstore_empty_raises(fake_embeddings):
    with pytest.raises(ValueError):
        build_vectorstore([], fake_embeddings)
