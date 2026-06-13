"""Factory selection + error-path tests.

These stay offline: they only exercise backend selection and the guardrails that
raise before any network call or model download happens.
"""

from __future__ import annotations

import pytest

from ragchat.config import Config
from ragchat.embeddings import build_embeddings
from ragchat.llm import build_llm


def test_embeddings_openai_without_key_raises():
    with pytest.raises(ValueError, match="OpenAI API key"):
        build_embeddings(Config(embedding_backend="openai", openai_api_key=""))


def test_embeddings_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown embedding_backend"):
        build_embeddings(Config(embedding_backend="bogus"))  # type: ignore[arg-type]


def test_llm_openai_without_key_raises():
    with pytest.raises(ValueError, match="OpenAI API key"):
        build_llm(Config(llm_backend="openai", openai_api_key=""))


def test_llm_hf_without_token_raises():
    with pytest.raises(ValueError, match="HuggingFace token"):
        build_llm(Config(llm_backend="huggingface", hf_token=""))


def test_llm_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown llm_backend"):
        build_llm(Config(llm_backend="bogus"))  # type: ignore[arg-type]
