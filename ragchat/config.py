"""Central configuration for the RAG pipeline.

Everything that decides *which* models run and *how* documents are chunked and
retrieved lives here, so the embedding model and the LLM stay swappable without
touching the pipeline code (see README -> Architecture). Values are resolved
from, in order of precedence:

    1. explicit arguments / UI overrides (highest)
    2. environment variables (incl. Streamlit `st.secrets`, which are exported
       into the environment on Streamlit Community Cloud)
    3. the defaults below (lowest)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

Backend = Literal["huggingface", "openai"]


def _get(name: str, default: str) -> str:
    """Read an env var, treating empty/whitespace values as unset."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass(frozen=True)
class Config:
    # --- backend selection -------------------------------------------------
    # "huggingface" keeps the demo fully free (local embeddings + free hosted
    # LLM); "openai" is the drop-in premium alternative documented in the README.
    embedding_backend: Backend = "huggingface"
    llm_backend: Backend = "huggingface"

    # --- HuggingFace models ------------------------------------------------
    # Small, fast, 384-dim sentence embedder - runs locally on CPU, no key.
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Instruction-tuned chat model served via the free HF Inference API.
    # Non-gated (Apache-2.0) so a fresh free token works without a license
    # click. Swap via HF_CHAT_MODEL - see README for alternatives.
    hf_chat_model: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_token: str = ""

    # --- OpenAI models (optional) -----------------------------------------
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # --- chunking (empirically tuned - see docs/EXPERIMENTS.md) ------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- retrieval / generation -------------------------------------------
    top_k: int = 4
    temperature: float = 0.0
    max_new_tokens: int = 512

    # metadata for the UI / logs
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Config:
        """Build a Config from environment variables and Streamlit secrets."""
        return cls(
            embedding_backend=_get("EMBEDDING_BACKEND", cls.embedding_backend),  # type: ignore[arg-type]
            llm_backend=_get("LLM_BACKEND", cls.llm_backend),  # type: ignore[arg-type]
            hf_embedding_model=_get("HF_EMBEDDING_MODEL", cls.hf_embedding_model),
            hf_chat_model=_get("HF_CHAT_MODEL", cls.hf_chat_model),
            # Accept the common aliases people set on hosting platforms.
            hf_token=_get("HF_TOKEN", _get("HUGGINGFACEHUB_API_TOKEN", "")),
            openai_embedding_model=_get("OPENAI_EMBEDDING_MODEL", cls.openai_embedding_model),
            openai_chat_model=_get("OPENAI_CHAT_MODEL", cls.openai_chat_model),
            openai_api_key=_get("OPENAI_API_KEY", ""),
            chunk_size=int(_get("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(_get("CHUNK_OVERLAP", str(cls.chunk_overlap))),
            top_k=int(_get("TOP_K", str(cls.top_k))),
            temperature=float(_get("TEMPERATURE", str(cls.temperature))),
            max_new_tokens=int(_get("MAX_NEW_TOKENS", str(cls.max_new_tokens))),
        )

    def with_overrides(self, **kwargs) -> Config:
        """Return a copy with the given fields replaced (used by the UI sidebar)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)

    # --- convenience -------------------------------------------------------
    @property
    def needs_hf_token(self) -> bool:
        # Local sentence-transformers embeddings need no token; only the free
        # hosted HF Inference LLM does.
        return self.llm_backend == "huggingface"

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems (empty = OK)."""
        problems: list[str] = []
        if self.llm_backend == "openai" and not self.openai_api_key:
            problems.append("LLM backend is 'openai' but OPENAI_API_KEY is not set.")
        if self.embedding_backend == "openai" and not self.openai_api_key:
            problems.append("Embedding backend is 'openai' but OPENAI_API_KEY is not set.")
        if self.llm_backend == "huggingface" and not self.hf_token:
            problems.append(
                "LLM backend is 'huggingface' but no HF token is set "
                "(HF_TOKEN / HUGGINGFACEHUB_API_TOKEN). Get a free token at "
                "https://huggingface.co/settings/tokens."
            )
        if self.chunk_overlap >= self.chunk_size:
            problems.append("chunk_overlap must be smaller than chunk_size.")
        return problems
