"""LLM factory.

Returns a LangChain chat model chosen by ``Config.llm_backend``. Like the
embedding factory, both branches return objects that satisfy the same
``BaseChatModel`` interface, so the generation model is swappable without
touching the chain.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import Config


def build_llm(cfg: Config) -> BaseChatModel:
    """Build the chat model for the configured backend.

    - ``huggingface``: free hosted model via the HF Inference API (needs a free
      HF token; the *recruiter* needs nothing - the token lives in the app's
      server-side secrets).
    - ``openai``: OpenAI chat model (requires ``openai_api_key``).
    """
    if cfg.llm_backend == "openai":
        from langchain_openai import ChatOpenAI

        if not cfg.openai_api_key:
            raise ValueError(
                "llm_backend='openai' requires an OpenAI API key "
                "(set OPENAI_API_KEY or paste it in the UI sidebar)."
            )
        return ChatOpenAI(
            model=cfg.openai_chat_model,
            temperature=cfg.temperature,
            api_key=cfg.openai_api_key,
            max_tokens=cfg.max_new_tokens,
        )

    if cfg.llm_backend == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        if not cfg.hf_token:
            raise ValueError(
                "llm_backend='huggingface' requires a HuggingFace token. "
                "Create a free one at https://huggingface.co/settings/tokens "
                "and set HF_TOKEN."
            )
        # ChatHuggingFace drives the provider's chat-completion API and expects
        # the underlying endpoint's task to be "text-generation" (it applies the
        # chat template itself). "conversational" is rejected by this version.
        endpoint = HuggingFaceEndpoint(
            repo_id=cfg.hf_chat_model,
            huggingfacehub_api_token=cfg.hf_token,
            task="text-generation",
            temperature=max(cfg.temperature, 0.01),  # some providers reject temp == 0
            max_new_tokens=cfg.max_new_tokens,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(f"Unknown llm_backend: {cfg.llm_backend!r}")
