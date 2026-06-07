"""Retrieval-augmented QA chain.

Wires a retriever + a chat model + the anti-hallucination prompt into a single
LangChain runnable using LCEL (LangChain Expression Language) primitives from
``langchain-core``. This is the same idea as the classic ``RetrievalQA`` helper
(retrieve relevant chunks, stuff them into the prompt context, answer) but built
from stable core primitives so it doesn't depend on the moving
``langchain.chains`` API (removed in LangChain 1.x).

The LLM and retriever are injected, not constructed here, which keeps this
module trivially unit-testable with a fake model and keeps both the embedding
model and the LLM swappable from the outside.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStore

from .prompts import build_prompt


def build_retriever(vectorstore: VectorStore, top_k: int):
    """A similarity retriever returning the ``top_k`` closest chunks."""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def _format_docs_for_prompt(docs: list[Document]) -> str:
    """Render retrieved chunks into the ``{context}`` block the prompt expects.

    Each chunk is prefixed with its source + page so the model can ground its
    answer and, if asked, cite where a fact came from.
    """
    blocks = []
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source", "?")
        page = (meta.get("page", 0) or 0) + 1
        blocks.append(f"[source: {source}, page {page}]\n{d.page_content}")
    return "\n\n".join(blocks)


def build_qa_chain(llm: BaseChatModel, retriever) -> Runnable:
    """Compose retriever + prompt + LLM into a retrieval-QA runnable.

    ``chain.invoke("<question>")`` returns a dict with:
        - ``input``:   the original question
        - ``context``: the list of retrieved source Documents (for citations)
        - ``answer``:  the model's grounded response string
    """
    prompt = build_prompt()

    # Given {input, context:[Document]}, produce just the answer string.
    generate = (
        {
            "context": lambda x: _format_docs_for_prompt(x["context"]),
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Retrieve first (the retriever takes the raw question string), keep the
    # docs, then attach the generated answer alongside them.
    retrieve = RunnableParallel(
        input=RunnablePassthrough(),
        context=retriever,
    )
    return retrieve | RunnablePassthrough.assign(answer=generate)


def _format_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Compact, UI-friendly view of the chunks used to answer."""
    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append(
            {
                "source": meta.get("source", "?"),
                # PyPDF pages are 0-indexed; show 1-indexed to humans.
                "page": (meta.get("page", 0) or 0) + 1,
                "snippet": d.page_content.strip()[:300],
            }
        )
    return sources


def answer(chain: Runnable, question: str) -> dict[str, Any]:
    """Run a question through the chain and return a normalized result.

    Returns ``{"answer": str, "sources": list[dict]}``.
    """
    result = chain.invoke(question)
    return {
        "answer": (result.get("answer") or "").strip(),
        "sources": _format_sources(result.get("context", [])),
    }
