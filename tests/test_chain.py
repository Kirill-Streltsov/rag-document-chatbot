from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragchat.chain import (
    MAX_HISTORY_MESSAGES,
    answer,
    build_qa_chain,
    build_retriever,
    history_to_messages,
)
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


def test_first_question_is_not_rewritten(sample_pages, fake_embeddings, fake_llm):
    chain = _make_chain(sample_pages, fake_embeddings, fake_llm)
    out = answer(chain, "What was the 2024 revenue?", history=[])
    # No history -> no rewrite call, the question is searched as asked.
    assert out["question"] == "What was the 2024 revenue?"
    assert out["answer"] == "The 2024 revenue was 84.6 million euros."


def test_follow_up_is_rewritten_before_retrieval(sample_pages, fake_embeddings):
    # The fake model answers the rewrite prompt first, then the QA prompt.
    llm = FakeListChatModel(
        responses=['"Who was the CEO of Acme in 2024?"', "Jane Doe."]
    )
    chain = _make_chain(sample_pages, fake_embeddings, llm)
    history = [
        {"role": "user", "content": "What was the 2024 revenue?"},
        {"role": "assistant", "content": "84.6 million euros."},
    ]
    out = answer(chain, "And who ran the company?", history=history)
    assert out["question"] == "Who was the CEO of Acme in 2024?"  # quotes stripped
    assert out["answer"] == "Jane Doe."


def test_empty_rewrite_falls_back_to_original(sample_pages, fake_embeddings):
    llm = FakeListChatModel(responses=["   ", "Some answer."])
    chain = _make_chain(sample_pages, fake_embeddings, llm)
    history = [{"role": "user", "content": "Hi"}]
    out = answer(chain, "What was the revenue?", history=history)
    assert out["question"] == "What was the revenue?"


def test_history_is_trimmed_to_recent_turns():
    history = [{"role": "user", "content": str(i)} for i in range(20)]
    history.append({"role": "assistant", "content": "x"})
    messages = history_to_messages(history)
    assert len(messages) == MAX_HISTORY_MESSAGES
    assert messages[-1].content == "x"
