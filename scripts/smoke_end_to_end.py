"""End-to-end smoke test of the FULL live stack (real embeddings + real LLM).

Unlike the pytest suite (which is offline and uses fakes), this actually calls
the configured LLM, so it needs a token. Use it to confirm the deployed demo
will work - run it once after setting your HuggingFace token:

    export HF_TOKEN=hf_xxx                 # free token: huggingface.co/settings/tokens
    python scripts/smoke_end_to_end.py

Or verify the OpenAI backend instead:

    export OPENAI_API_KEY=sk-xxx
    LLM_BACKEND=openai EMBEDDING_BACKEND=openai python scripts/smoke_end_to_end.py

If the HF model isn't available on your inference plan, swap it:

    HF_CHAT_MODEL=microsoft/Phi-3.5-mini-instruct python scripts/smoke_end_to_end.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragchat.chain import answer, build_qa_chain, build_retriever
from ragchat.config import Config
from ragchat.embeddings import build_embeddings
from ragchat.ingest import build_vectorstore, load_pdf, split_documents

PDF = Path(__file__).resolve().parents[1] / "sample_docs" / "acme_annual_report_2024.pdf"

QUESTIONS = [
    "Who is the CEO of the company?",
    "What was total revenue in 2024?",
    "What is the maximum payload of the AR-7 robot?",
    "What is the airspeed velocity of an unladen swallow?",  # not in the doc -> should decline
]


def main() -> int:
    cfg = Config.from_env()
    problems = cfg.validate()
    if problems:
        print("Configuration problems:\n  - " + "\n  - ".join(problems))
        return 2

    print(f"Backends: embeddings={cfg.embedding_backend}, llm={cfg.llm_backend}")
    print(f"LLM model: {cfg.openai_chat_model if cfg.llm_backend=='openai' else cfg.hf_chat_model}\n")

    from ragchat.llm import build_llm  # local import keeps config errors cheap

    pages = load_pdf(PDF)
    chunks = split_documents(pages, cfg.chunk_size, cfg.chunk_overlap)
    store = build_vectorstore(chunks, build_embeddings(cfg))
    chain = build_qa_chain(build_llm(cfg), build_retriever(store, cfg.top_k))

    for q in QUESTIONS:
        out = answer(chain, q)
        cites = ", ".join(f"{s['source']} p.{s['page']}" for s in out["sources"])
        print(f"Q: {q}")
        print(f"A: {out['answer']}")
        print(f"   sources: {cites}\n")
    print("END-TO-END SMOKE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
