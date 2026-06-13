from __future__ import annotations

from ragchat.config import Config


def test_defaults_are_free_stack():
    cfg = Config()
    assert cfg.embedding_backend == "huggingface"
    assert cfg.llm_backend == "huggingface"
    assert cfg.chunk_overlap < cfg.chunk_size


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("CHUNK_SIZE", "777")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_xyz")
    cfg = Config.from_env()
    assert cfg.llm_backend == "openai"
    assert cfg.chunk_size == 777
    assert cfg.top_k == 9
    # HF token alias is honored
    assert cfg.hf_token == "hf_xyz"


def test_empty_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "   ")  # whitespace treated as unset
    assert Config.from_env().chunk_size == Config().chunk_size


def test_with_overrides_ignores_none():
    cfg = Config().with_overrides(top_k=7, chunk_size=None)
    assert cfg.top_k == 7
    assert cfg.chunk_size == Config().chunk_size


def test_validate_flags_missing_openai_key():
    problems = Config(llm_backend="openai", openai_api_key="").validate()
    assert any("OPENAI_API_KEY" in p for p in problems)


def test_validate_flags_missing_hf_token():
    problems = Config(llm_backend="huggingface", hf_token="").validate()
    assert any("HF" in p or "HuggingFace" in p for p in problems)


def test_validate_flags_bad_overlap():
    problems = Config(chunk_size=100, chunk_overlap=100, hf_token="x").validate()
    assert any("overlap" in p for p in problems)


def test_needs_hf_token_only_for_hf_llm():
    assert Config(llm_backend="huggingface").needs_hf_token is True
    assert Config(llm_backend="openai").needs_hf_token is False
