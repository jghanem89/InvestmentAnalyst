"""
Split filing sections into embeddable chunks.

Two rules drive the design:

* Chunks break on sentence boundaries. TextBlob scores these chunks later, and
  polarity computed over a truncated clause is meaningless.
* Chunks that are mostly digits are dropped. A condensed balance sheet flattens
  into runs of numbers that pollute retrieval and carry no sentiment.
"""

from __future__ import annotations

import re
from typing import List

from .config import RAGConfig
from .schema import FilingChunk, FilingDocument, FilingSection

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
# A chunk this numeric is a financial-statement table, not prose.
_MAX_DIGIT_RATIO = 0.2


def _sentences(text: str) -> List[str]:
    parts = [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _digit_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 1.0
    return sum(ch.isdigit() for ch in non_space) / len(non_space)


class FilingChunker:
    """Turns a parsed 10-Q into chunks ready for embedding."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()

    def _chunk_section(self, section: FilingSection) -> List[str]:
        """Greedy sentence packing up to the target size, with a sentence of overlap."""
        target = self.config.chunk_target_words
        overlap = self.config.chunk_overlap_words

        chunks: List[str] = []
        current: List[str] = []
        current_words = 0

        for sentence in _sentences(section.text):
            words = len(sentence.split())
            if current and current_words + words > target:
                chunks.append(" ".join(current))
                # Carry trailing sentences forward so a thought split across a
                # boundary is still retrievable from either side.
                carried: List[str] = []
                carried_words = 0
                for previous in reversed(current):
                    previous_words = len(previous.split())
                    if carried_words + previous_words > overlap:
                        break
                    carried.insert(0, previous)
                    carried_words += previous_words
                current = carried
                current_words = carried_words
            current.append(sentence)
            current_words += words

        if current:
            chunks.append(" ".join(current))
        return chunks

    def chunk(self, document: FilingDocument) -> List[FilingChunk]:
        """Chunk every section of `document`, tagging each chunk with its metadata."""
        ref = document.ref
        results: List[FilingChunk] = []

        for section in document.sections:
            for index, text in enumerate(self._chunk_section(section)):
                word_count = len(text.split())
                if word_count < self.config.min_chunk_words:
                    continue
                if _digit_ratio(text) > _MAX_DIGIT_RATIO:
                    continue

                results.append(FilingChunk(
                    # Deterministic, so re-ingesting a filing updates in place
                    # instead of duplicating it.
                    id=f"{ref.filing_id}_{section.key}_{index:04d}",
                    # The heading rides along in the embedded text so a query
                    # like "risk factors" matches on more than body wording.
                    text=f"[{ref.ticker} {ref.form} {ref.filing_date} | {section.title}]\n{text}",
                    metadata={
                        "ticker": ref.ticker,
                        "cik": ref.cik,
                        "company": ref.company,
                        "form": ref.form,
                        "accession": ref.accession,
                        "filing_date": ref.filing_date.isoformat(),
                        "period_ending": ref.period_ending.isoformat() if ref.period_ending else "",
                        "section": section.key,
                        "section_title": section.title,
                        "chunk_index": index,
                        "word_count": word_count,
                        "source_url": ref.doc_url,
                    },
                ))
        return results
