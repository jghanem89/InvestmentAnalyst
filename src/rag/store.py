"""
ChromaDB persistence and retrieval for 10-Q filings.

The one design point worth calling out: "the latest 10-Q" is a metadata
question, not a semantic one. Vector similarity has no notion of recency, so a
plain search for "how did the quarter go" will happily return a filing from
three years ago. Retrieval here therefore resolves the newest filing_date for a
ticker first, then searches semantically inside that filing only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb

from .config import RAGConfig
from .embeddings import get_embedding_function
from .schema import FilingChunk

_SIGNATURE_KEY = "embed_signature"


class FilingVectorStore:
    """Read/write access to the 10-Q collection."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.embedding_function, self.signature = get_embedding_function(self.config)
        self.client = chromadb.PersistentClient(path=str(self.config.persist_dir))
        self.collection = self._open_collection()

    def _open_collection(self):
        """
        Open the collection, refusing to mix embedding models.

        Querying vectors written by one model with another returns plausible-
        looking but meaningless neighbours, so this fails loudly instead.
        """
        collection = self.client.get_or_create_collection(
            name=self.config.collection,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", _SIGNATURE_KEY: self.signature},
        )
        stored = (collection.metadata or {}).get(_SIGNATURE_KEY)
        if stored and stored != self.signature and collection.count() > 0:
            raise RuntimeError(
                f"Collection {self.config.collection!r} was built with "
                f"embeddings {stored!r} but this process is using {self.signature!r}. "
                "Re-ingest with the original model, or delete "
                f"{self.config.persist_dir} and ingest again."
            )
        return collection

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def upsert(self, chunks: List[FilingChunk], batch_size: int = 64) -> int:
        """Add or replace chunks. Ids are deterministic, so re-ingest is idempotent."""
        if not chunks:
            return 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
            )
        return len(chunks)

    def has_filing(self, accession: str) -> bool:
        """True when this accession number is already stored."""
        found = self.collection.get(where={"accession": accession}, limit=1)
        return bool(found.get("ids"))

    def delete_ticker(self, ticker: str) -> None:
        self.collection.delete(where={"ticker": ticker.strip().upper()})

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def filings_for(self, ticker: str) -> List[Dict[str, Any]]:
        """Every stored filing for `ticker`, newest first."""
        ticker = ticker.strip().upper()
        found = self.collection.get(where={"ticker": ticker}, include=["metadatas"])

        filings: Dict[str, Dict[str, Any]] = {}
        for metadata in found.get("metadatas") or []:
            accession = metadata.get("accession")
            if not accession:
                continue
            entry = filings.setdefault(accession, {
                "accession": accession,
                "ticker": metadata.get("ticker"),
                "company": metadata.get("company"),
                "form": metadata.get("form"),
                "filing_date": metadata.get("filing_date"),
                "period_ending": metadata.get("period_ending"),
                "source_url": metadata.get("source_url"),
                "chunks": 0,
                "sections": set(),
            })
            entry["chunks"] += 1
            entry["sections"].add(metadata.get("section"))

        results = sorted(filings.values(), key=lambda f: f["filing_date"] or "", reverse=True)
        for entry in results:
            entry["sections"] = sorted(s for s in entry["sections"] if s)
        return results

    def latest_filing(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Metadata for the most recently filed 10-Q held for `ticker`."""
        filings = self.filings_for(ticker)
        return filings[0] if filings else None

    def query(
        self,
        query_text: str,
        ticker: str,
        top_k: Optional[int] = None,
        *,
        accession: Optional[str] = None,
        sections: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search within one ticker, optionally pinned to a filing/sections."""
        ticker = ticker.strip().upper()
        clauses: List[Dict[str, Any]] = [{"ticker": ticker}]
        if accession:
            clauses.append({"accession": accession})
        if sections:
            clauses.append({"section": {"$in": list(sections)}})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}

        found = self.collection.query(
            query_texts=[query_text],
            n_results=top_k or self.config.top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = (found.get("documents") or [[]])[0]
        metadatas = (found.get("metadatas") or [[]])[0]
        distances = (found.get("distances") or [[]])[0]

        return [
            {"text": document, "metadata": metadata, "distance": distance}
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]

    def latest_filing_chunks(
        self,
        ticker: str,
        query_text: str,
        top_k: Optional[int] = None,
        sections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve from the newest stored 10-Q for `ticker`.

        Resolves the newest filing by metadata first, then searches only inside
        it, so results can never drift into an older quarter.
        """
        ticker = ticker.strip().upper()
        latest = self.latest_filing(ticker)
        if not latest:
            return {
                "ticker": ticker,
                "filing": None,
                "chunks": [],
                "error": (
                    f"No 10-Q filings stored for {ticker}. Run "
                    f"`python scripts/ingest_filings.py --tickers {ticker}` first."
                ),
            }

        chunks = self.query(
            query_text=query_text,
            ticker=ticker,
            top_k=top_k,
            accession=latest["accession"],
            sections=sections,
        )
        return {"ticker": ticker, "filing": latest, "chunks": chunks}

    def stats(self) -> Dict[str, Any]:
        return {
            "collection": self.config.collection,
            "path": str(self.config.persist_dir),
            "embeddings": self.signature,
            "chunks": self.collection.count(),
        }
