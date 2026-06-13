from __future__ import annotations

from ragchat.chain import answer, build_qa_chain, build_retriever
from ragchat.ingest import build_vectorstore, split_documents


def _make_chain(sample_pages, fake_embeddings, fake_llm, k=2):
    chunks = split_documents(sample_pages, chunk_size=200, chunk_overlap=40)
    store = build_vectorstore(chunks, fake_embeddings)
    retriever = build_retriever(store, top_k=k)
    return build_qa_chain(fake_llm, retriever)


def test_chain_returns_answer_and_sources(sample_pages, fake_embeddings, fake_llm):
    chain = _make_chain(sample_pages, fake_embeddings, fake_llm, k=2)
    out = answer(chain, "What was the 2024 revenue?")
    assert out["answer"] == "The 2024 revenue was 84.6 million euros."
    assert len(out["sources"]) == 2


def test_sources_are_one_indexed_and_shaped(sample_pages, fake_embeddings, fake_llm):
    chain = _make_chain(sample_pages, fake_embeddings, fake_llm, k=1)
    out = answer(chain, "Who is the CEO?")
    assert len(out["sources"]) == 1
    src = out["sources"][0]
    assert set(src) == {"source", "page", "snippet"}
    assert src["page"] >= 1  # pages shown 1-indexed to humans
    assert len(src["snippet"]) <= 300


def test_top_k_limits_retrieved_sources(sample_pages, fake_embeddings, fake_llm):
    chain = _make_chain(sample_pages, fake_embeddings, fake_llm, k=1)
    out = answer(chain, "anything")
    assert len(out["sources"]) == 1


def test_chain_invoke_shape(sample_pages, fake_embeddings, fake_llm):
    """The raw chain should expose input, context and answer keys."""
    chain = _make_chain(sample_pages, fake_embeddings, fake_llm)
    result = chain.invoke("What was the revenue?")
    assert set(result) >= {"input", "context", "answer"}
    assert result["input"] == "What was the revenue?"
