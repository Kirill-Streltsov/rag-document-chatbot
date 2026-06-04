"""Embedding-model factory.

Returns a LangChain ``Embeddings`` object chosen by ``Config.embedding_backend``.
Both branches implement the same interface, which is what makes the embedding
model swappable without changing the ingest/retrieval code.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .config import Config


def build_embeddings(cfg: Config) -> Embeddings:
    """Build the embedding model for the configured backend.

    - ``huggingface``: local sentence-transformers model (CPU, no API key).
    - ``openai``: OpenAI embeddings (requires ``openai_api_key``).
    """
    if cfg.embedding_backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not cfg.openai_api_key:
            raise ValueError(
                "embedding_backend='openai' requires an OpenAI API key "
                "(set OPENAI_API_KEY or pass it in the UI)."
            )
        return OpenAIEmbeddings(
            model=cfg.openai_embedding_model,
            api_key=cfg.openai_api_key,
        )

    if cfg.embedding_backend == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        # normalized embeddings + cosine similarity is the standard setup for
        # MiniLM-family sentence encoders.
        return HuggingFaceEmbeddings(
            model_name=cfg.hf_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"Unknown embedding_backend: {cfg.embedding_backend!r}")
