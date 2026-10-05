"""Streamlit frontend - "document in, questions out".

Upload one or more PDFs, build a FAISS index over them, then chat with a
retrieval-augmented model that answers only from the document.

Runs on the free stack by default (local HuggingFace embeddings + a free hosted
HuggingFace LLM). Recruiters need no key: the app's HF token lives in server-side
secrets. An OpenAI backend is one toggle away for anyone with a key.
"""

from __future__ import annotations

import os
import time

import streamlit as st
from dotenv import load_dotenv

from ragchat import (
    Config,
    answer,
    build_embeddings,
    build_llm,
    build_qa_chain,
    build_retriever,
    ingest_pdfs,
)

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄", layout="centered")


# --------------------------------------------------------------------------- #
# Secrets / config helpers
# --------------------------------------------------------------------------- #
def _export_secrets_to_env() -> None:
    """Make root-level Streamlit secrets visible to ``Config.from_env``.

    Values already set in the environment win, so a real env var or ``.env``
    entry can still override a committed default.
    """
    try:
        items = dict(st.secrets)
    except Exception:
        return  # no secrets.toml - nothing to export
    for name, value in items.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(name, str(value))


# Local runs can keep settings in a .env file (see .env.example); on Streamlit
# Cloud they come from the app's secrets. Either way Config.from_env sees them.
load_dotenv()
_export_secrets_to_env()
BASE_CFG = Config.from_env()
DEFAULT_HF_TOKEN = BASE_CFG.hf_token


# --------------------------------------------------------------------------- #
# Cached heavy resources (keyed by their string args)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_embeddings(backend: str, hf_model: str, openai_model: str, openai_key: str):
    return build_embeddings(
        Config(
            embedding_backend=backend,  # type: ignore[arg-type]
            hf_embedding_model=hf_model,
            openai_embedding_model=openai_model,
            openai_api_key=openai_key,
        )
    )


@st.cache_resource(show_spinner=False)
def get_llm(
    backend: str,
    hf_model: str,
    openai_model: str,
    temperature: float,
    max_new_tokens: int,
    hf_token: str,
    openai_key: str,
):
    return build_llm(
        Config(
            llm_backend=backend,  # type: ignore[arg-type]
            hf_chat_model=hf_model,
            openai_chat_model=openai_model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            hf_token=hf_token,
            openai_api_key=openai_key,
        )
    )


# --------------------------------------------------------------------------- #
# Sidebar - backend + parameters
# --------------------------------------------------------------------------- #
def sidebar() -> Config:
    st.sidebar.header("⚙️ Settings")

    use_openai = st.sidebar.toggle(
        "Use OpenAI instead of the free stack",
        value=BASE_CFG.llm_backend == "openai",
        help="Off = free HuggingFace stack (no key needed). On = OpenAI (bring your own key).",
    )

    openai_key = ""
    hf_token = DEFAULT_HF_TOKEN
    if use_openai:
        backend = "openai"
        openai_key = st.sidebar.text_input(
            "OpenAI API key", type="password", value=BASE_CFG.openai_api_key
        ).strip()
    else:
        backend = "huggingface"
        if not DEFAULT_HF_TOKEN:
            hf_token = st.sidebar.text_input(
                "HuggingFace token (free)",
                type="password",
                help="The hosted demo already has one. Running locally? Paste a free "
                "token from huggingface.co/settings/tokens.",
            ).strip()

    with st.sidebar.expander("Advanced - chunking & retrieval"):
        # Start from the configured values (env / secrets), snapped onto the
        # slider grids so an odd value like CHUNK_SIZE=777 cannot break them.
        chunk_size = st.slider(
            "Chunk size", 300, 1500, _snap(BASE_CFG.chunk_size, 300, 1500, 100), 100
        )
        chunk_overlap = st.slider(
            "Chunk overlap", 0, 400, _snap(BASE_CFG.chunk_overlap, 0, 400, 50), 50
        )
        top_k = st.slider("Chunks retrieved (k)", 1, 8, _snap(BASE_CFG.top_k, 1, 8, 1), 1)
        temperature = st.slider(
            "Temperature", 0.0, 1.0, round(min(max(BASE_CFG.temperature, 0.0), 1.0), 1), 0.1
        )

    st.sidebar.caption(
        "Defaults (chunk 1000 / overlap 150 / k 4) are empirically tuned - "
        "see docs/EXPERIMENTS.md."
    )

    # Everything not exposed in the sidebar (model names, max_new_tokens, ...)
    # keeps the value from the environment / secrets.
    return BASE_CFG.with_overrides(
        embedding_backend=backend,
        llm_backend=backend,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        temperature=temperature,
        hf_token=hf_token,
        openai_api_key=openai_key,
    )


def _snap(value: float, lo: int, hi: int, step: int) -> int:
    """Clamp ``value`` into [lo, hi] and round it onto the slider's step grid."""
    value = min(max(value, lo), hi)
    return int(lo + round((value - lo) / step) * step)


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def index_signature(cfg: Config, filenames: list[str]) -> tuple:
    """Anything that, if changed, invalidates the current FAISS index."""
    return (
        cfg.embedding_backend,
        cfg.hf_embedding_model if cfg.embedding_backend == "huggingface" else cfg.openai_embedding_model,
        cfg.chunk_size,
        cfg.chunk_overlap,
        tuple(sorted(filenames)),
    )


def build_index(cfg: Config, uploads) -> None:
    sources = [(f.getvalue(), f.name) for f in uploads]
    embeddings = get_embeddings(
        cfg.embedding_backend,
        cfg.hf_embedding_model,
        cfg.openai_embedding_model,
        cfg.openai_api_key,
    )
    t0 = time.time()
    store, n_chunks = ingest_pdfs(sources, embeddings, cfg)
    st.session_state.store = store
    st.session_state.index_sig = index_signature(cfg, [f.name for f in uploads])
    st.session_state.n_chunks = n_chunks
    st.session_state.index_files = [f.name for f in uploads]
    st.session_state.build_secs = time.time() - t0
    # A fresh index means the old conversation no longer matches the corpus.
    st.session_state.messages = []


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📄 RAG Document Chatbot")
    st.write(
        "Upload a PDF (a report, a paper, a contract...) and ask questions about it. "
        "Answers are grounded in the document and cite the pages they came from."
    )

    cfg = sidebar()
    st.session_state.setdefault("messages", [])

    # ---- upload + index ---------------------------------------------------
    uploads = st.file_uploader(
        "PDF document(s)", type="pdf", accept_multiple_files=True
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        build_clicked = st.button("Build index", type="primary", disabled=not uploads)
    with col2:
        if "store" in st.session_state:
            st.caption(
                f"Indexed **{', '.join(st.session_state.index_files)}** -> "
                f"{st.session_state.n_chunks} chunks in {st.session_state.build_secs:.1f}s"
            )

    if build_clicked:
        # Only what indexing needs: the chunking and the embedding backend. LLM
        # problems (missing HF token, ...) surface when the first question is asked.
        problems = [p for p in cfg.validate() if "overlap" in p or "Embedding" in p]
        if problems:
            st.error(" ".join(problems))
        else:
            try:
                with st.spinner("Parsing, chunking and embedding..."):
                    build_index(cfg, uploads)
                st.success(f"Indexed {st.session_state.n_chunks} chunks. Ask away 👇")
            except Exception as e:  # noqa: BLE001 - surface any ingest error to the user
                st.error(f"Indexing failed: {e}")

    # Warn if parameters changed since the last build.
    if "store" in st.session_state and uploads:
        if st.session_state.index_sig != index_signature(cfg, [f.name for f in uploads]):
            st.info("Settings or files changed since the last build - click **Build index** to apply.")

    # ---- chat -------------------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    question = st.chat_input(
        "Ask a question about your document...",
        disabled="store" not in st.session_state,
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    llm = get_llm(
                        cfg.llm_backend,
                        cfg.hf_chat_model,
                        cfg.openai_chat_model,
                        cfg.temperature,
                        cfg.max_new_tokens,
                        cfg.hf_token,
                        cfg.openai_api_key,
                    )
                    retriever = build_retriever(st.session_state.store, cfg.top_k)
                    chain = build_qa_chain(llm, retriever)
                    result = answer(chain, question)
                st.markdown(result["answer"])
                _render_sources(result["sources"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                )
            except Exception as e:  # noqa: BLE001
                msg = _friendly_error(e, cfg)
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {msg}"})


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**{i}. {s['source']} - page {s['page']}**")
            st.caption(s["snippet"] + ("..." if len(s["snippet"]) >= 300 else ""))


def _friendly_error(e: Exception, cfg: Config) -> str:
    text = str(e)
    if cfg.llm_backend == "huggingface" and not cfg.hf_token:
        return (
            "No HuggingFace token configured. The hosted demo has one; running "
            "locally, paste a free token in the sidebar (huggingface.co/settings/tokens)."
        )
    if "rate" in text.lower() or "429" in text:
        return "The free HuggingFace inference tier is rate-limited right now - try again in a moment."
    if cfg.llm_backend == "openai" and not cfg.openai_api_key:
        return "OpenAI selected but no API key provided. Paste your key in the sidebar."
    return f"Could not generate an answer: {text}"


if __name__ == "__main__":
    main()
