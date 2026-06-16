"""Empirical tuning of chunk size, overlap and top-k.

Retrieval quality is what caps a RAG system's answers: if the right passage
isn't retrieved, no prompt can save the generation step. This script isolates
and measures *retrieval only* (no LLM, no API keys) so the chunking and
retrieval parameters can be chosen from data rather than by feel.

Metrics, over a hand-labeled question -> gold-substring set on the sample
report:
  - recall@k : fraction of questions whose gold fact appears in the top-k chunks
  - MRR      : mean reciprocal rank of the first chunk containing the gold fact
               (sensitive to ordering and chunk boundaries)

Run:
    python scripts/eval_retrieval.py
Results (on the committed sample PDF) are written up in docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/eval_retrieval.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragchat.config import Config
from ragchat.embeddings import build_embeddings
from ragchat.ingest import build_vectorstore, load_pdf, split_documents

PDF = Path(__file__).resolve().parents[1] / "sample_docs" / "acme_annual_report_2024.pdf"

# (question, gold substring that must appear in a retrieved chunk)
QA_SET: list[tuple[str, str]] = [
    ("What was total revenue in 2024?", "84.6 million"),
    ("What was revenue in 2023?", "68.8 million"),
    ("Who is the chief executive officer?", "Jane Doe"),
    ("Who is the chief technology officer?", "Miguel Santos"),
    ("Who is the chief financial officer?", "Priya Nair"),
    ("How many people were employed at the end of 2024?", "512"),
    ("What is the AR-7 battery life?", "11 hours"),
    ("What is the maximum payload of the AR-7?", "45 kilograms"),
    ("What share of revenue came from Germany?", "46 percent"),
    ("How much was spent on research and development?", "14.1 million"),
    ("When was the company founded?", "2011"),
    ("How much recurring revenue did AcmeOS generate?", "22 million"),
    ("What is the carbon neutral target year?", "2028"),
    ("What is the net revenue retention for AcmeOS?", "118 percent"),
    ("What revenue is expected in 2025?", "100 and 108 million"),
    ("Who chairs the supervisory board?", "Anke Vogel"),
]


def _rank_of_hit(retriever, question: str, gold: str) -> int | None:
    """1-indexed rank of the first retrieved chunk containing ``gold``."""
    docs = retriever.invoke(question)
    for i, d in enumerate(docs, start=1):
        if gold.lower() in d.page_content.lower():
            return i
    return None


def evaluate(embeddings, chunk_size: int, chunk_overlap: int, k: int) -> tuple[float, float, int]:
    """Return (recall@k, MRR, n_chunks) for one configuration."""
    pages = load_pdf(PDF)
    chunks = split_documents(pages, chunk_size, chunk_overlap)
    store = build_vectorstore(chunks, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": k})

    hits = 0
    rr_sum = 0.0
    for question, gold in QA_SET:
        rank = _rank_of_hit(retriever, question, gold)
        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank
    n = len(QA_SET)
    return hits / n, rr_sum / n, len(chunks)


def _table(header: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(header) + " |"
    sep = "| " + " | ".join("---" for _ in header) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def main() -> None:
    # HuggingFace embeddings, loaded once and reused across all configs.
    embeddings = build_embeddings(Config(embedding_backend="huggingface"))

    print(f"Corpus: {PDF.name} | {len(QA_SET)} labeled questions\n")

    # Experiment 1 - chunk size (overlap = 15% of size, k = 4)
    print("### Experiment 1 - chunk size (overlap 15%, k=4)\n")
    rows = []
    for cs in (300, 500, 800, 1000, 1500):
        recall, mrr, n = evaluate(embeddings, cs, int(cs * 0.15), k=4)
        rows.append([str(cs), str(n), f"{recall:.2f}", f"{mrr:.2f}"])
    print(_table(["chunk_size", "n_chunks", "recall@4", "MRR"], rows), "\n")

    # Experiment 2 - overlap (chunk size = 1000, k = 4)
    print("### Experiment 2 - overlap (chunk_size=1000, k=4)\n")
    rows = []
    for ov in (0, 50, 150, 300):
        recall, mrr, n = evaluate(embeddings, 1000, ov, k=4)
        rows.append([str(ov), str(n), f"{recall:.2f}", f"{mrr:.2f}"])
    print(_table(["overlap", "n_chunks", "recall@4", "MRR"], rows), "\n")

    # Experiment 3 - top-k (chunk size = 1000, overlap = 150)
    print("### Experiment 3 - top-k (chunk_size=1000, overlap=150)\n")
    rows = []
    for k in (1, 2, 4, 6, 8):
        recall, mrr, n = evaluate(embeddings, 1000, 150, k=k)
        rows.append([str(k), f"{recall:.2f}", f"{mrr:.2f}"])
    print(_table(["k", "recall@k", "MRR"], rows), "\n")


if __name__ == "__main__":
    main()
