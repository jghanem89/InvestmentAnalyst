"""Pydantic models passed between the EDGAR client, the chunker and the store."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FilingRef(BaseModel):
    """A 10-Q hit from EDGAR full-text search, before the document is downloaded."""

    ticker: str
    cik: str
    company: str
    form: str
    accession: str
    primary_doc: str
    filing_date: date
    period_ending: Optional[date] = None

    @property
    def doc_url(self) -> str:
        """Archives URL of the primary document."""
        cik_int = str(int(self.cik))
        accession_flat = self.accession.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{accession_flat}/{self.primary_doc}"
        )

    @property
    def filing_id(self) -> str:
        """Stable identifier used for on-disk caching and chunk ids."""
        return f"{self.ticker}_{self.accession}"


class FilingSection(BaseModel):
    """One Item-level section of a 10-Q after text extraction."""

    key: str          # mdna | risk_factors | financial_statements | ...
    title: str        # the heading as it appeared in the document
    text: str
    order: int


class FilingDocument(BaseModel):
    """A downloaded and parsed 10-Q."""

    ref: FilingRef
    sections: List[FilingSection] = Field(default_factory=list)
    char_count: int = 0


class FilingChunk(BaseModel):
    """An embeddable unit of filing text plus the metadata we filter on."""

    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
