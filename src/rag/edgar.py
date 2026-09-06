"""
EDGAR access: ticker -> CIK resolution, full-text search for 10-Q filings, and
document download with an on-disk cache.

The SEC rate-limits aggressively and blocks requests without a descriptive
User-Agent, so every call goes through one session that sets the header once.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import RAGConfig
from .schema import FilingRef

FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


class EdgarClient:
    """Thin EDGAR wrapper scoped to what the 10-Q pipeline needs."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._ticker_map: Optional[Dict[str, Dict[str, str]]] = None

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """One rate-limited GET. SEC asks for <10 req/s; we sleep between calls."""
        time.sleep(self.config.request_delay_sec)
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------ #
    # Ticker -> CIK
    # ------------------------------------------------------------------ #

    def _load_ticker_map(self) -> Dict[str, Dict[str, str]]:
        """SEC's ticker->CIK file, cached on disk so repeat runs stay offline."""
        if self._ticker_map is not None:
            return self._ticker_map

        cache_path = self.config.raw_dir.parent / "company_tickers.json"
        raw: Optional[Dict[str, Any]] = None
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = None

        if raw is None:
            raw = self._get(COMPANY_TICKERS_URL).json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(raw), encoding="utf-8")

        self._ticker_map = {
            str(entry["ticker"]).upper(): {
                "cik": str(entry["cik_str"]).zfill(10),
                "title": entry.get("title", ""),
            }
            for entry in raw.values()
        }
        return self._ticker_map

    def resolve_cik(self, ticker: str) -> Dict[str, str]:
        """Map a ticker to its zero-padded CIK and registered company name."""
        ticker = ticker.strip().upper()
        entry = self._load_ticker_map().get(ticker)
        if not entry:
            raise ValueError(f"No SEC CIK registered for ticker {ticker!r}.")
        return entry

    # ------------------------------------------------------------------ #
    # Full-text search
    # ------------------------------------------------------------------ #

    def find_10q_filings(self, ticker: str, limit: int = 4) -> List[FilingRef]:
        """
        Latest 10-Q filings for `ticker`, newest first.

        Uses EDGAR full-text search with an empty `q`. That matters: supplying a
        query term makes the endpoint rank by relevance and surface exhibits
        (EX-99.1 and friends) instead of the 10-Q itself, whereas an empty query
        filtered by cik + form returns primary documents in filing-date order.
        """
        ticker = ticker.strip().upper()
        entry = self.resolve_cik(ticker)

        params = {"q": "", "forms": self.config.form_type, "ciks": entry["cik"]}
        payload = self._get(FULL_TEXT_SEARCH_URL, params=params).json()
        hits = payload.get("hits", {}).get("hits", [])

        refs: List[FilingRef] = []
        for hit in hits:
            source = hit.get("_source", {})
            # `_id` is "<accession>:<primary document filename>".
            hit_id = str(hit.get("_id", ""))
            if ":" not in hit_id:
                continue
            accession, primary_doc = hit_id.split(":", 1)

            # Keep only the filing itself, not the exhibits attached to it.
            if source.get("file_type") != self.config.form_type:
                continue

            filing_date = _parse_date(source.get("file_date"))
            if filing_date is None:
                continue

            refs.append(FilingRef(
                ticker=ticker,
                cik=entry["cik"],
                company=entry["title"] or ticker,
                form=source.get("form", self.config.form_type),
                accession=accession,
                primary_doc=primary_doc,
                filing_date=filing_date,
                period_ending=_parse_date(source.get("period_ending")),
            ))

        refs.sort(key=lambda r: r.filing_date, reverse=True)
        return refs[:limit]

    # ------------------------------------------------------------------ #
    # Document download
    # ------------------------------------------------------------------ #

    def download(self, ref: FilingRef, refresh: bool = False) -> Path:
        """
        Fetch a filing's primary document, caching it under `raw_dir`.

        EDGAR serves 10-Q primary documents as inline-XBRL HTML, not PDF, so the
        suffix follows whatever the document actually is.
        """
        suffix = Path(ref.primary_doc).suffix or ".htm"
        cache_path = self.config.raw_dir / f"{ref.filing_id}{suffix}"

        if cache_path.exists() and not refresh and cache_path.stat().st_size > 0:
            return cache_path

        response = self._get(ref.doc_url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        return cache_path
