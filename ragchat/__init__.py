"""ragchat - a modular, swappable RAG pipeline for chatting with your PDFs.

Public API:
    Config           - runtime configuration (env / secrets / UI overrides)
    build_embeddings - embedding-model factory (HuggingFace or OpenAI)
    build_llm        - LLM factory (HuggingFace or OpenAI)
    ingest_pdfs      - parse -> chunk -> embed -> FAISS index
    build_qa_chain   - LangChain retrieval chain with an anti-hallucination prompt
    answer           - run a question through the chain and return answer + sources
"""

from .chain import answer, build_qa_chain, build_retriever
from .config import Config
from .embeddings import build_embeddings
from .ingest import build_vectorstore, ingest_pdfs, load_pdf, split_documents
from .llm import build_llm

__all__ = [
    "Config",
    "build_embeddings",
    "build_llm",
    "ingest_pdfs",
    "load_pdf",
    "split_documents",
    "build_vectorstore",
    "build_retriever",
    "build_qa_chain",
    "answer",
]

__version__ = "1.0.0"
