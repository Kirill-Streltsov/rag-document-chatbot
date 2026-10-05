"""Prompt templates.

The system prompt is the main lever against hallucination: it constrains the
model to the retrieved context and gives it an explicit escape hatch
("I don't know") so it doesn't invent answers when the documents are silent.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are a precise assistant that answers questions about the \
user's uploaded document(s).

Rules:
- Answer ONLY using the context below. Do not use outside knowledge.
- If the answer is not contained in the context, say exactly:
  "I couldn't find that in the document." Do not guess or fabricate.
- Be concise and factual. When useful, quote short phrases from the context.
- If the question is ambiguous, answer for the most likely interpretation and
  say what you assumed.

Context:
{context}"""

HUMAN_PROMPT = "{input}"


def build_prompt() -> ChatPromptTemplate:
    """The QA prompt consumed by LangChain's stuff-documents chain.

    Must expose a ``{context}`` variable (filled with retrieved chunks) and an
    ``{input}`` variable (the user's question).
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )


# Follow-ups like "and in 2023?" only make sense next to the earlier turns, but
# the retriever sees one string. This prompt turns such a follow-up into a
# question that stands on its own before it is searched for.
CONDENSE_PROMPT = """Rewrite the user's latest question so that it can be \
understood without the conversation before it, by resolving references such \
as "it", "they" or "the year before" from the earlier turns.

Rules:
- Keep the language and the meaning of the question.
- If it is already self-contained, return it unchanged.
- Do not answer it. Reply with the rewritten question only."""


def build_condense_prompt() -> ChatPromptTemplate:
    """Prompt that rewrites a follow-up into a standalone question.

    Exposes ``{chat_history}`` (a list of messages) and ``{input}``.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
