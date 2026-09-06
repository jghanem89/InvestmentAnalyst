"""Wires EDGAR -> extraction -> chunking -> ChromaDB into one batch pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .chunker import FilingChunker
from .config import RAGConfig
from .edgar import EdgarClient
from .extract import extract_filing
from .store import FilingVectorStore


class FilingIngestionPipeline:
    """Downloads and indexes 10-Q filings. Run offline, before the agent starts."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        store: Optional[FilingVectorStore] = None,
    ):
        self.config = config or RAGConfig()
        self.client = EdgarClient(self.config)
        self.chunker = FilingChunker(self.config)
        self.store = store or FilingVectorStore(self.config)

    def ingest_ticker(
        self,
        ticker: str,
        limit: int = 2,
        refresh: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Index the `limit` most recent 10-Q filings for one ticker."""
        ticker = ticker.strip().upper()
        report: Dict[str, Any] = {"ticker": ticker, "filings": [], "chunks": 0}

        try:
            refs = self.client.find_10q_filings(ticker, limit=limit)
        except Exception as exc:
            report["error"] = f"EDGAR lookup failed: {exc}"
            return report

        if not refs:
            report["error"] = f"EDGAR full-text search returned no 10-Q filings for {ticker}."
            return report

        for ref in refs:
            if not refresh and self.store.has_filing(ref.accession):
                if verbose:
                    print(f"  [skip] {ref.accession} ({ref.filing_date}) already indexed")
                report["filings"].append({
                    "accession": ref.accession,
                    "filing_date": ref.filing_date.isoformat(),
                    "status": "already indexed",
                    "chunks": 0,
                })
                continue

            try:
                path = self.client.download(ref, refresh=refresh)
                document = extract_filing(path, ref)
                chunks = self.chunker.chunk(document)
                written = self.store.upsert(chunks)
            except Exception as exc:
                if verbose:
                    print(f"  [fail] {ref.accession}: {exc}")
                report["filings"].append({
                    "accession": ref.accession,
                    "filing_date": ref.filing_date.isoformat(),
                    "status": f"failed: {exc}",
                    "chunks": 0,
                })
                continue

            if verbose:
                sections = ", ".join(sorted({s.key for s in document.sections}))
                print(f"  [ok]   {ref.accession} ({ref.filing_date}) "
                      f"-> {written} chunks from {len(document.sections)} sections")
                print(f"         sections: {sections}")

            report["chunks"] += written
            report["filings"].append({
                "accession": ref.accession,
                "filing_date": ref.filing_date.isoformat(),
                "period_ending": ref.period_ending.isoformat() if ref.period_ending else None,
                "status": "indexed",
                "chunks": written,
                "url": ref.doc_url,
            })

        return report

    def run(
        self,
        tickers: List[str],
        limit: int = 2,
        refresh: bool = False,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        reports = []
        for ticker in tickers:
            if verbose:
                print(f"\n{ticker}")
            reports.append(self.ingest_ticker(ticker, limit=limit, refresh=refresh, verbose=verbose))
        return reports
