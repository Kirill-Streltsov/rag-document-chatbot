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

Follow-up questions ("and in 2023?") are supported: when there is chat history,
the question is first rewritten into a standalone one, and that rewritten
question is what gets searched for and answered.
"""

from __future__ import annotations

from operator import itemgetter
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStore

from .prompts import build_condense_prompt, build_prompt

# How much of the conversation the question rewriter sees (messages, not turns).
MAX_HISTORY_MESSAGES = 6
# A rewritten question longer than this is treated as the model rambling.
MAX_REWRITE_CHARS = 400


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


def _as_payload(x: str | dict) -> dict[str, Any]:
    """Accept a bare question or ``{"input": ..., "chat_history": [...]}``."""
    if isinstance(x, str):
        return {"input": x, "chat_history": []}
    return {"input": x["input"], "chat_history": list(x.get("chat_history") or [])}


def build_qa_chain(llm: BaseChatModel, retriever) -> Runnable:
    """Compose retriever + prompt + LLM into a retrieval-QA runnable.

    ``chain.invoke("<question>")`` or
    ``chain.invoke({"input": "<question>", "chat_history": [messages]})``
    returns a dict with:
        - ``input``:    the original question
        - ``question``: the standalone question that was searched for (equal
                        to ``input`` when there is no history)
        - ``context``:  the list of retrieved source Documents (for citations)
        - ``answer``:   the model's grounded response string
    """
    prompt = build_prompt()
    condense = build_condense_prompt() | llm | StrOutputParser()

    def standalone_question(x: dict[str, Any]) -> str:
        if not x["chat_history"]:
            return x["input"]  # first question: no extra LLM call
        rewritten = condense.invoke(x).strip().strip('"').strip()
        # Fall back to the original wording if the model returns nothing usable.
        if not rewritten or len(rewritten) > MAX_REWRITE_CHARS:
            return x["input"]
        return rewritten

    # Given {question, context:[Document]}, produce just the answer string.
    generate = (
        {
            "context": lambda x: _format_docs_for_prompt(x["context"]),
            "input": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Each step keeps the keys it was given and adds one more, so the retrieved
    # docs are still there next to the answer for the citations.
    return (
        RunnableLambda(_as_payload)
        | RunnablePassthrough.assign(question=RunnableLambda(standalone_question))
        | RunnablePassthrough.assign(context=itemgetter("question") | retriever)
        | RunnablePassthrough.assign(answer=generate)
    )


def history_to_messages(history: list[dict[str, Any]]) -> list[BaseMessage]:
    """Turn UI chat messages (``{"role", "content"}``) into LangChain messages.

    Only the most recent turns are kept; that is all a follow-up needs, and it
    keeps the rewrite prompt small.
    """
    messages: list[BaseMessage] = []
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        if msg.get("role") == "user":
            messages.append(HumanMessage(msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(msg["content"]))
    return messages


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


def answer(
    chain: Runnable, question: str, history: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Run a question (plus any earlier chat turns) through the chain.

    ``history`` is the conversation so far as ``{"role", "content"}`` dicts.
    Returns ``{"answer": str, "question": str, "sources": list[dict]}``, where
    ``question`` is the standalone question that was actually searched for.
    """
    result = chain.invoke(
        {"input": question, "chat_history": history_to_messages(history or [])}
    )
    return {
        "answer": (result.get("answer") or "").strip(),
        "question": result.get("question", question),
        "sources": _format_sources(result.get("context", [])),
    }
