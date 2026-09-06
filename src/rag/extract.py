"""
Turn a downloaded 10-Q into clean, section-labelled text.

EDGAR serves 10-Q primary documents as inline-XBRL HTML (there is no PDF
rendition of a 10-Q on EDGAR), so the HTML path is the one that matters. A PDF
reader is wired in as well for filings supplied from elsewhere.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline-XBRL filings open with an XML declaration, so bs4 warns that an HTML
# parser is reading XML. The HTML parser is the right choice here -- these are
# XHTML documents meant to render in a browser -- so the warning is just noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Filings use typographic punctuation; folding it to ASCII keeps the stored text
# and anything printed from it readable on a Windows console.
_PUNCTUATION = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}

from .schema import FilingDocument, FilingRef, FilingSection

# Canonical section keys, in the order a 10-Q presents them. The regex matches
# the Item heading; `part` disambiguates Item 1/Item 2, which appear in both
# Part I (financials, MD&A) and Part II (legal proceedings, share repurchases).
SECTION_PATTERNS: List[Tuple[str, str, str]] = [
    ("financial_statements", "I",  r"item\s*1\s*[\.\:\-–—]?\s*(?:financial\s+statements|condensed)"),
    ("mdna", "I",               r"item\s*2\s*[\.\:\-–—]?\s*management.{0,3}s\s+discussion"),
    ("market_risk", "I",        r"item\s*3\s*[\.\:\-–—]?\s*quantitative\s+and\s+qualitative"),
    ("controls", "I",           r"item\s*4\s*[\.\:\-–—]?\s*controls\s+and\s+procedures"),
    ("legal_proceedings", "II", r"item\s*1\s*[\.\:\-–—]?\s*legal\s+proceedings"),
    ("risk_factors", "II",      r"item\s*1a\s*[\.\:\-–—]?\s*risk\s+factors"),
    ("unregistered_sales", "II",r"item\s*2\s*[\.\:\-–—]?\s*unregistered\s+sales"),
    ("other_information", "II", r"item\s*5\s*[\.\:\-–—]?\s*other\s+information"),
    ("exhibits", "II",          r"item\s*6\s*[\.\:\-–—]?\s*exhibits"),
]

SECTION_TITLES: Dict[str, str] = {
    "financial_statements": "Item 1. Financial Statements",
    "mdna": "Item 2. Management's Discussion and Analysis",
    "market_risk": "Item 3. Quantitative and Qualitative Disclosures About Market Risk",
    "controls": "Item 4. Controls and Procedures",
    "legal_proceedings": "Part II Item 1. Legal Proceedings",
    "risk_factors": "Part II Item 1A. Risk Factors",
    "unregistered_sales": "Part II Item 2. Unregistered Sales of Equity Securities",
    "other_information": "Part II Item 5. Other Information",
    "exhibits": "Part II Item 6. Exhibits",
}

# Lines that carry no analytical content and would otherwise skew sentiment.
_BOILERPLATE = re.compile(
    r"^(table of contents|form 10-?q|page \d+|\d+|"
    r"united states securities and exchange commission|"
    r"washington,?\s*d\.?c\.?\s*20549|"
    r"\(exact name of .*\)|\(state or other jurisdiction.*\))$",
    re.IGNORECASE,
)


def _html_to_text(html: str) -> str:
    """Flatten inline-XBRL HTML to plain text, one logical block per line."""
    soup = BeautifulSoup(html, "lxml")

    # Inline XBRL hides a metadata header and a pile of non-displayed facts in
    # the document; both are noise and neither is meant to be read.
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    text = soup.get_text("\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")

    lines: List[str] = []
    for raw_line in text.split("\n"):
        line = " ".join(raw_line.split())
        if not line or _BOILERPLATE.match(line):
            continue
        lines.append(line)

    # Inline XBRL splits sentences across many tags, so single words often land
    # on their own line. Re-join a line with the previous one unless it starts a
    # new sentence or looks like a heading.
    merged: List[str] = []
    for line in lines:
        if (
            merged
            and not re.match(r"^(item|part)\s", line, re.I)
            and not merged[-1].endswith((".", ":", ";", "?", "!"))
            and len(merged[-1]) < 900
        ):
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    return "\n".join(merged)


def _pdf_to_text(path: Path) -> str:
    """Read a PDF filing, if one was supplied from outside EDGAR."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Reading PDF filings needs `pypdf`. Install it with `pip install pypdf`, "
            "or use the EDGAR HTML documents, which is what this pipeline downloads."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(" ".join(p.split()) for p in pages if p.strip())


def _split_sections(text: str) -> List[FilingSection]:
    """
    Cut the flattened text into Item-level sections.

    Every Item heading appears at least twice in a 10-Q: once in the table of
    contents and once at the section itself. Matching naively would slice the
    table of contents into nine one-line sections, so for each key we keep the
    occurrence that yields the most text.
    """
    candidates: Dict[str, List[int]] = {}
    for key, _part, pattern in SECTION_PATTERNS:
        offsets = [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)]
        if offsets:
            candidates[key] = offsets

    if not candidates:
        return [FilingSection(key="full_text", title="Full filing", text=text, order=0)]

    # All heading offsets, so a section can end at whichever heading follows it.
    all_offsets = sorted({offset for offsets in candidates.values() for offset in offsets})

    best: Dict[str, Tuple[int, int]] = {}
    for key, offsets in candidates.items():
        for start in offsets:
            following = [o for o in all_offsets if o > start]
            end = following[0] if following else len(text)
            if key not in best or (end - start) > (best[key][1] - best[key][0]):
                best[key] = (start, end)

    sections: List[FilingSection] = []
    order_lookup = {key: i for i, (key, _p, _r) in enumerate(SECTION_PATTERNS)}
    for key, (start, end) in sorted(best.items(), key=lambda kv: kv[1][0]):
        body = text[start:end].strip()
        if len(body.split()) < 40:  # a table-of-contents remnant, not a section
            continue
        sections.append(FilingSection(
            key=key,
            title=SECTION_TITLES.get(key, key),
            text=body,
            order=order_lookup.get(key, 99),
        ))

    if not sections:
        return [FilingSection(key="full_text", title="Full filing", text=text, order=0)]
    return sections


def extract_filing(path: Path, ref: FilingRef) -> FilingDocument:
    """Read a cached filing off disk and return it split into sections."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        text = _pdf_to_text(path)
    else:
        text = _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))

    sections = _split_sections(text)
    return FilingDocument(ref=ref, sections=sections, char_count=len(text))
