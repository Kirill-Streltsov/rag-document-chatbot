# Retrieval tuning - how the defaults were chosen

> The resume line this backs: *"Chunk size and retrieval parameters were
> empirically tested and documented."* This is that documentation.

In a RAG system the generation step can only be as good as what retrieval hands
it: if the passage containing the answer isn't in the context, no prompt will
recover it. So the chunking and retrieval parameters were chosen by **measuring
retrieval in isolation** (no LLM, no API key) against a hand-labeled question
set, rather than by feel.

Reproduce everything below with:

```bash
python scripts/eval_retrieval.py
```

## Setup

- **Corpus:** `sample_docs/acme_annual_report_2024.pdf` (a fictional 3-page
  annual report).
- **Query set:** 16 questions, each paired with a gold substring that must
  appear in a retrieved chunk (e.g. *"Who is the CEO?" -> "Jane Doe"*). See
  `QA_SET` in `scripts/eval_retrieval.py`.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (the deployed
  default), cosine similarity on normalized vectors.
- **Metrics:**
  - **recall@k** - fraction of questions whose gold fact is somewhere in the
    top-k retrieved chunks. This is the ceiling on answer quality.
  - **MRR** - mean reciprocal rank of the first chunk containing the gold fact.
    Sensitive to *ordering* and chunk boundaries, so it separates configs that
    recall@k rates as tied.

> **Caveat, stated up front:** 16 questions over a 3-page document is a small
> benchmark. One question is worth ±0.06 recall, so small recall@k differences
> below are **within noise**. Where the signal is clear (overlap, k) I lean on
> it; where it isn't (chunk size on a tiny corpus) I say so and fall back to
> reasoning that generalizes to larger documents.

## Experiment 1 - chunk size

Overlap fixed at 15% of chunk size, k = 4.

| chunk_size | n_chunks | recall@4 | MRR  |
| ---------- | -------- | -------- | ---- |
| 300        | 17       | 0.94     | 0.82 |
| 500        | 11       | 0.94     | 0.74 |
| 800        | 6        | 0.94     | 0.83 |
| 1000       | 6        | 0.88     | 0.74 |
| 1500       | 3        | 0.94     | 0.88 |

**Read:** recall@4 is saturated (~0.9) across the board - expected, since the
document is small and k=4 already pulls a large share of it. There is **no
reliable chunk-size signal at this corpus size**; the recall dip at 1000 and the
high MRR at 1500 are both artifacts of a tiny document (at 1500 there are only 3
chunks, so "retrieval" is nearly the whole report).

**Decision:** chunk size is the **most corpus-dependent** knob, so it isn't
worth overfitting to 16 questions. The default is **1000 characters** - a
standard value for prose reports: large enough to keep a multi-sentence fact
(and its surrounding context) intact, small enough to keep the embedding focused
and the context budget reasonable. It is exposed in the UI and in `Config` so it
can be re-tuned per corpus. Very small chunks (300) fragment facts that span
sentences; very large chunks dilute the embedding and waste context on
irrelevant text.

## Experiment 2 - chunk overlap

Chunk size fixed at 1000, k = 4.

| overlap | n_chunks | recall@4 | MRR  |
| ------- | -------- | -------- | ---- |
| 0       | 6        | 0.88     | 0.64 |
| 50      | 6        | 0.88     | 0.64 |
| 150     | 6        | 0.88     | 0.74 |
| 300     | 6        | 0.94     | 0.86 |

**Read:** here the signal is clear and monotonic - **more overlap improves
ranking** (MRR 0.64 -> 0.74 -> 0.86) and eventually recall. Overlap keeps a fact
that straddles a chunk boundary fully present in at least one chunk instead of
being split across two, so the matching chunk scores higher.

**Decision:** default overlap **150 (15%)**. It captures most of the MRR gain
while avoiding the redundancy of 300 (30%), which re-embeds a large fraction of
the text and inflates index size and retrieval noise on larger corpora.

## Experiment 3 - top-k

Chunk size 1000, overlap 150.

| k | recall@k | MRR  |
| - | -------- | ---- |
| 1 | 0.62     | 0.62 |
| 2 | 0.81     | 0.72 |
| 4 | 0.88     | 0.74 |
| 6 | 0.94     | 0.75 |
| 8 | 0.94     | 0.75 |

**Read:** the clearest curve of the three. Recall climbs steeply from k=1 to k=4
(0.62 -> 0.88), then flattens - k=6 buys a little more recall, k=8 buys nothing.
MRR is essentially flat past k=4, meaning extra chunks are landing *below* the
already-retrieved answer, i.e. they add prompt size and distraction without
adding signal.

**Decision:** default **k = 4** - the knee of the curve. It captures most of the
achievable recall while keeping the prompt small (important for the free hosted
LLM's context window and latency). k can be raised in the UI for
recall-critical use.

## Chosen defaults

| Parameter      | Value | Why |
| -------------- | ----- | --- |
| `chunk_size`   | 1000  | Standard for prose; corpus-dependent, re-tunable |
| `chunk_overlap`| 150   | Clear MRR gain from overlap; 15% avoids redundancy |
| `top_k`        | 4     | Knee of the recall curve; larger k adds noise, not signal |

These live in [`ragchat/config.py`](../ragchat/config.py) and are overridable via
environment variables or the Streamlit sidebar.

## Honest limitations

- Retrieval recall ≠ answer correctness; it's an upper bound. End-to-end answer
  grading would need an LLM-as-judge pass, which is out of scope for a
  key-free, offline benchmark.
- The benchmark is small and single-document. The **method** (labeled set,
  recall@k + MRR, isolate retrieval) is what transfers; the exact numbers would
  be re-measured on a real target corpus.
- MiniLM is a small English embedder. For multilingual or domain-specific
  corpora a larger or specialized model would likely move recall more than any
  chunking tweak - which is exactly why the embedding model is swappable.
