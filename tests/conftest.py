"""Shared pytest fixtures.

All fixtures are offline: real PDF parsing (no network) plus fake embeddings and
a fake chat model, so the suite runs anywhere without API keys or model
downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "sample_docs" / "acme_annual_report_2024.pdf"


@pytest.fixture(scope="session")
def sample_pdf_path() -> Path:
    assert SAMPLE_PDF.exists(), (
        f"Sample PDF missing at {SAMPLE_PDF}. Generate it with "
        "`python scripts/make_sample_pdf.py`."
    )
    return SAMPLE_PDF


@pytest.fixture
def sample_pages() -> list[Document]:
    """Two synthetic pages with known, queryable facts."""
    return [
        Document(
            page_content=(
                "Acme Robotics GmbH reported total revenue of 84.6 million euros "
                "in 2024. The company is headquartered in Dortmund."
            ),
            metadata={"source": "report.pdf", "page": 0},
        ),
        Document(
            page_content=(
                "The Chief Executive Officer is Jane Doe. The company employed 512 "
                "people at the end of 2024."
            ),
            metadata={"source": "report.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Deterministic 64-dim fake embeddings - no model download."""
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def fake_llm() -> FakeListChatModel:
    """A chat model that always returns a fixed, canned answer."""
    return FakeListChatModel(responses=["The 2024 revenue was 84.6 million euros."])
