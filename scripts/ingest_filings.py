"""
Download and index the latest 10-Q filings into ChromaDB.

    python scripts/ingest_filings.py --tickers AAPL MSFT --limit 2

Run this before asking the sentiment agent for a filing-aware analysis; the
agent reads from the vector store and never ingests at query time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.config import RAGConfig
from src.rag.ingest import FilingIngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest SEC 10-Q filings into ChromaDB.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols, e.g. AAPL MSFT")
    parser.add_argument("--limit", type=int, default=2, help="Filings per ticker (default 2)")
    parser.add_argument("--refresh", action="store_true", help="Re-download and re-index existing filings")
    args = parser.parse_args()

    pipeline = FilingIngestionPipeline(RAGConfig())
    reports = pipeline.run([t.upper() for t in args.tickers], limit=args.limit, refresh=args.refresh)

    print("\n" + "-" * 60)
    failures = 0
    for report in reports:
        if report.get("error"):
            print(f"{report['ticker']}: ERROR {report['error']}")
            failures += 1
            continue
        print(f"{report['ticker']}: {report['chunks']} new chunks "
              f"across {len(report['filings'])} filings")
    print(f"store: {pipeline.store.stats()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
