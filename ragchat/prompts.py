"""Prompt templates.

The system prompt is the main lever against hallucination: it constrains the
model to the retrieved context and gives it an explicit escape hatch
("I don't know") so it doesn't invent answers when the documents are silent.
"""

from langchain_core.prompts import ChatPromptTemplate

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
