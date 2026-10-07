"""Extract the MD&A section (10-K Item 7) from a filing's primary document as clean text.

Method:
1. HTML -> text with one paragraph per line. Hidden inline-XBRL data, scripts,
   numeric tables and page-number lines are dropped.
2. Scan short lines (headings, not prose) for "Item N" headings in order. An
   "Item 7 ... Management's Discussion" heading opens a section, "Item 7A" or
   "Item 8" closes it, and any other item heading cancels an open one. That
   cancel rule is what skips the table of contents.
3. Keep the longest closed section. If no Item 7 heading exists (some banks
   use a cross-reference layout), fall back to the "Management's Discussion
   and Analysis" title itself.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

log = logging.getLogger(__name__)

BLOCK_TAGS = [
    "p", "div", "br", "tr", "li", "ul", "ol", "table", "section", "center", "pre",
    "hr", "blockquote", "dl", "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6",
]  # fmt: skip
NUMERIC_TABLE_MIN_CHARS = 200    # tables shorter than this can be headings; keep them
NUMERIC_TABLE_DIGIT_SHARE = 0.25  # above this share of digits, a table is financial data

PAGE_NOISE = re.compile(
    r"^(?:[-–—\s]*\d{1,3}[-–—\s]*|page\s+\d+(?:\s+of\s+\d+)?|table\s+of\s+contents|index)$",
    re.IGNORECASE,
)

# "Item 7." / "ITEM 7 -" / "Items 7 and 7A." / "Part II, Item 7" / "Item 7(a)"
ITEM_HEADING = re.compile(
    r"(?:part\s+[iv]+\W{0,3})?items?\s*(\d{1,2})\s*(?:\(?([a-c])\)?)?(?![0-9a-z])",
    re.IGNORECASE,
)
# A real heading is followed by its capitalized title ("Item 8. Financial..."); a
# cross-reference is followed by prose ("Item 8 of this Report", "Item 8, Note 4").
# "Items 7 and 7A. Management's..." combines two items in one heading.
HEADING_TITLE_START = re.compile(
    r"[\s.:\-–—]*(?:(?:and|&)\s*\d{1,2}\s*\(?[a-c]?\)?[\s.:\-–—]*)?(?:[A-Z\[(]|$)"
)
MDA_TITLE = re.compile(r"discussion|md\s*&\s*a", re.IGNORECASE)
# End-heading titles must start right after the number, so "Item 8 - Note 7 to the
# consolidated financial statements" (a cross-reference) is not mistaken for Item 8.
END_TITLE = re.compile(
    r"[\s.:\-–—]*(?:quantitative|qualitative|market\s+risk"
    r"|(?:consolidated\s+)?financial\s+(?:statements|information))",
    re.IGNORECASE,
)
LOOKAHEAD_CHARS = 300  # heading title may be on the next line ("Item 7." / "Management's...")

FALLBACK_START = re.compile(r"management['’`]?s\s+discussion\s+and\s+analysis", re.IGNORECASE)
FALLBACK_END = re.compile(
    r"^(?:quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk"
    r"|financial\s+statements\s+and\s+supplementary\s+data"
    r"|management['’`]?s\s+report\s+on\s+internal\s+control"
    r"|report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm)",
    re.IGNORECASE,
)
# Asset-backed issuers file 10-Ks under Regulation AB, which omits Item 7 entirely.
ABS_ISSUER = re.compile(r"regulation\s+ab\b", re.IGNORECASE)
INCORPORATED = re.compile(r"incorporated\s+(?:herein\s+)?by\s+reference|exhibit\s+13", re.I)


@dataclass(frozen=True)
class Extraction:
    text: str | None
    n_words: int
    method: str | None      # "item_heading" or "title_fallback"
    failure: str | None     # None on success


# ---------------------------------------------------------------- HTML -> text


def _is_hidden(tag: Tag) -> bool:
    style = tag.get("style") or ""
    return bool(re.search(r"display\s*:\s*none", style, re.IGNORECASE))


def _is_numeric_table(table: Tag) -> bool:
    text = table.get_text(" ")
    if len(text.strip()) < NUMERIC_TABLE_MIN_CHARS:
        return False
    digits = sum(c.isdigit() for c in text)
    letters = sum(c.isalpha() for c in text)
    return digits / max(digits + letters, 1) > NUMERIC_TABLE_DIGIT_SHARE


def html_to_text(raw: bytes) -> str:
    """Visible text, one block (usually a paragraph) per line."""
    if not re.search(rb"<(html|body|div|p|table)\b", raw[:20000], re.IGNORECASE):
        return _clean_lines(raw.decode("utf-8", errors="replace"))  # plain-text filing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)  # XHTML filings parse fine
        soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name.startswith("ix:header") or _is_hidden(t)):
        tag.decompose()
    for table in soup.find_all("table"):
        if not table.decomposed and _is_numeric_table(table):
            table.decompose()
    for cell in soup.find_all(["td", "th"]):
        cell.append(" ")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    return _clean_lines(soup.get_text())


def _clean_lines(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "").replace("\r", "\n")
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line and not PAGE_NOISE.match(line))


# ---------------------------------------------------------------- locating Item 7


def _line_spans(text: str, max_heading_chars: int) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for line in text.split("\n"):
        if len(line) <= max_heading_chars:
            spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def _classify_item(text: str, start: int) -> str | None:
    # Match against the text, not the line: "Item" and "7." are often separate blocks.
    m = ITEM_HEADING.match(text, start, start + LOOKAHEAD_CHARS)
    if m is None or not HEADING_TITLE_START.match(text, m.end()):
        return None
    number, letter = int(m.group(1)), (m.group(2) or "").lower()
    ahead = text[m.end() : m.end() + LOOKAHEAD_CHARS]
    if number == 7 and not letter:
        return "start" if MDA_TITLE.search(ahead) else None
    if (number == 7 and letter == "a") or number == 8:
        return "end" if END_TITLE.match(ahead) else None
    return "other"


def _best_section(events: list[tuple[int, int, str]]) -> tuple[int, int] | None:
    """events are (heading_start, heading_end, kind) in document order."""
    best: tuple[int, int] | None = None
    open_at: int | None = None
    for h_start, h_end, kind in events:
        if kind == "start":
            if open_at is None:  # repeated running headers inside the section are ignored
                open_at = h_end
        elif kind == "end":
            if open_at is not None and (best is None or h_start - open_at > best[1] - best[0]):
                best = (open_at, h_start)
            open_at = None
        else:
            open_at = None  # another item heading: the open "Item 7" was a TOC entry
    return best


def find_item7(text: str, max_heading_chars: int) -> tuple[int, int] | None:
    events = []
    for start, end in _line_spans(text, max_heading_chars):
        kind = _classify_item(text, start)
        if kind:
            events.append((start, end, kind))
    return _best_section(events)


def find_mda_by_title(text: str, max_heading_chars: int) -> tuple[int, int] | None:
    """For filings whose Item 7 heading lacks the number, or that use no item headings."""
    events = []
    for start, end in _line_spans(text, max_heading_chars):
        if FALLBACK_START.match(text, start, start + LOOKAHEAD_CHARS):  # title may wrap lines
            events.append((start, end, "start"))
        elif FALLBACK_END.match(text[start:end]) or _classify_item(text, start) == "end":
            events.append((start, end, "end"))
    return _best_section(events)


def extract_mda(raw: bytes, min_words: int, max_heading_chars: int) -> Extraction:
    text = html_to_text(raw)
    if not text:
        return Extraction(None, 0, None, "empty_document")
    found = [
        ("item_heading", find_item7(text, max_heading_chars)),
        ("title_fallback", find_mda_by_title(text, max_heading_chars)),
    ]
    best_short: tuple[str, int] | None = None
    for method, span in found:
        if span is None:
            continue
        section = text[span[0] : span[1]].strip()
        n_words = len(section.split())
        if n_words >= min_words:
            return Extraction(section, n_words, method, None)
        if best_short is None:
            best_short = (section, n_words)
    section, n_words = best_short or ("", 0)
    if ABS_ISSUER.search(text):
        reason = "abs_issuer_no_mda"
    elif best_short is None:
        reason = "mda_heading_not_found"
    elif INCORPORATED.search(section):
        reason = "incorporated_by_reference"
    else:
        reason = "section_too_short"
    return Extraction(None, n_words, None, reason)


# ---------------------------------------------------------------- pipeline stage


def extract_sections(filings: pd.DataFrame, min_words: int, max_heading_chars: int) -> pd.DataFrame:
    """One row per filing: MD&A text on success, a failure reason otherwise."""
    rows = []
    for _, f in filings.iterrows():
        if pd.notna(f["fetch_error"]) or pd.isna(f["doc_path"]):  # NaN, not None, after Parquet
            result = Extraction(None, 0, None, f"fetch_failed: {f['fetch_error']}")
        else:
            try:
                result = extract_mda(Path(f["doc_path"]).read_bytes(), min_words, max_heading_chars)
            except Exception as exc:  # noqa: BLE001 - one bad document must not stop the run
                result = Extraction(None, 0, None, f"parse_error: {type(exc).__name__}: {exc}")
        if result.failure:
            log.debug("MD&A extraction failed for %s: %s", f["accession"], result.failure)
        rows.append(
            {
                "accession": f["accession"],
                "cik": f["cik"],
                "form_type": f["form_type"],
                "acceptance_datetime": f["acceptance_datetime"],
                "section": "mda",
                "text": result.text,
                "n_words": result.n_words,
                "method": result.method,
                "status": "ok" if result.failure is None else "failed",
                "failure_reason": result.failure,
            }
        )
    return pd.DataFrame(rows)
