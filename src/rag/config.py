"""
Configuration for the SEC filing RAG pipeline.

Everything tunable lives here so the ingestion script and the agent agree on
paths, model names and the collection they are both pointing at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Project root is three levels up from this file: src/rag/config.py -> repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The SEC requires a descriptive User-Agent with contact details on every
# request and will return 403 without one. See
# https://www.sec.gov/os/accessing-edgar-data
DEFAULT_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "InvestmentAnalyst Capstone (nayseif13@gmail.com)",
)


@dataclass
class RAGConfig:
    """Settings shared by the ingestion pipeline and the retrieval store."""

    # --- storage ---------------------------------------------------------- #
    persist_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection: str = "sec_10q_filings"
    raw_dir: Path = PROJECT_ROOT / "data" / "filings" / "raw"

    # --- embeddings ------------------------------------------------------- #
    # "ollama" keeps everything local alongside the chat model; "default" uses
    # Chroma's bundled MiniLM ONNX model, which needs no Ollama embedding support.
    embed_backend: str = os.getenv("RAG_EMBED_BACKEND", "ollama")
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    embed_timeout: int = 120
    # Fall back to the bundled model instead of raising when Ollama cannot embed.
    allow_embed_fallback: bool = True

    # --- chunking --------------------------------------------------------- #
    chunk_target_words: int = 220
    chunk_overlap_words: int = 40
    min_chunk_words: int = 25

    # --- retrieval -------------------------------------------------------- #
    top_k: int = 12

    # --- EDGAR ------------------------------------------------------------ #
    user_agent: str = DEFAULT_USER_AGENT
    # SEC asks for no more than 10 requests/second; we stay well under.
    request_delay_sec: float = 0.2
    form_type: str = "10-Q"

    # Narrative sections worth scoring. The financial statement notes are mostly
    # numbers and boilerplate, which TextBlob cannot say anything useful about.
    sentiment_sections: List[str] = field(
        default_factory=lambda: ["mdna", "risk_factors", "market_risk", "controls"]
    )

    def __post_init__(self) -> None:
        self.persist_dir = Path(self.persist_dir)
        self.raw_dir = Path(self.raw_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embed_signature(self) -> str:
        """Identifies the vector space, so we never mix two models in one collection."""
        return f"{self.embed_backend}:{self.embed_model}"
