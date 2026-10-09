"""Mask identifying details in MD&A excerpts for the memorization test.

If the model has memorized what happened to a company after a filing, it can only use
that knowledge if it recognizes the company and the year. Masking removes both cues:

- The company's own names (every EDGAR name the CIK has filed under, with and without
  legal suffixes, plus the distinctive first word, e.g. "Skyworks") -> [COMPANY].
- Its tickers -> [TICKER] (case-sensitive; 1-2 letter tickers only after an exchange
  name, since "IT" or "ON" are ordinary words).
- People -> [PERSON], products -> [PRODUCT], other organizations (customers, competitors,
  subsidiaries) -> [ORGANIZATION], found by a named-entity recognizer. The recognizer also
  tags plenty of ordinary vocabulary ("Consolidated Financial Statements", "Goodwill",
  "Notes"), and masking that would degrade the text for reasons unrelated to memorization.
  So an entity is kept when every word in it is a common word (in the Loughran-McDonald
  master dictionary, built from 10-K vocabulary), and regulators and standard setters
  (SEC, FASB, Federal Reserve, ...) are always kept.
- The recognizer misses many names (acquired brands such as "Worldpay", drug names, unit
  acronyms), so any remaining capitalized word that is not a common word is masked as
  [NAME], unless the recognizer tagged it as a place, nationality or law ("Brazil",
  "European", "Hart-Scott-Rodino Act"). And a name masked once is masked everywhere it
  appears in the excerpt, since the recognizer is not consistent across sentences.
- Masking costs some meaning (a customer's name can matter), but it costs the same before
  and after the training cutoff, which is what the comparison relies on.
- Explicit dates -> [DATE] and years -> [YEAR] ("fiscal 2023" -> "fiscal [YEAR]").

The excerpt is selected from the original text first and then masked, so the original and
anonymized scores read the same paragraphs and differ only in the masked details.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# (start, end, label) spans from a named-entity recognizer.
EntityFinder = Callable[[str], list[tuple[int, int, str]]]

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd",
    "limited", "llc", "plc", "lp", "l p", "nv", "n v", "sa", "s a", "ag", "se", "holdings",
    "holding", "group", "the", "trust", "bancorp", "de", "md", "ny", "nj", "ca", "pa",
}  # fmt: skip
# First words too generic to mask on their own ("American" Express, "General" Mills).
GENERIC_FIRST_WORDS = {
    "american", "general", "united", "first", "national", "international", "southern",
    "northern", "eastern", "western", "pacific", "atlantic", "global", "public", "royal",
    "state", "consolidated", "federal", "central", "new", "republic", "universal", "applied",
    "advanced", "public service", "texas", "dominion", "duke", "principal", "progressive",
    "the", "air", "home", "best", "regions", "realty", "digital", "health", "edison",
}  # fmt: skip
KEEP_ORGANIZATIONS = re.compile(
    r"^(?:the\s+)?(?:SEC|Securities and Exchange Commission|FASB|Financial Accounting "
    r"Standards Board|IASB|PCAOB|Federal Reserve(?: Board| Bank)?|Fed|FDIC|OCC|CFPB|FDA|"
    r"Food and Drug Administration|EPA|IRS|Internal Revenue Service|Treasury|U\.?S\.? "
    r"Treasury|Department of \w+(?: \w+)?|Congress|Senate|House|FERC|FCC|FTC|DOJ|OPEC|"
    r"NYSE|New York Stock Exchange|Nasdaq|NASDAQ|S&P|Moody's|Fitch|LIBOR|SOFR|GAAP|"
    r"U\.?S\.? GAAP|IFRS|ASC|ASU|COVID(?:-19)?|CARES Act|WHO|World Health Organization|"
    r"European Union|EU|OECD|IMF|Medicare|Medicaid|CMS|NAIC|FINRA|CFTC|OSHA|DOE|DOD|"
    r"MD&A|PSLRA|ESG|SKU|PPE|LLC|ARRC|SG&A|R&D|EBITDA|EPS|GHG|Non-GAAP|SaaS|EMEA|APAC|"
    r"LATAM|UK|USA|"
    r"Company|Board|Board of Directors|Management|[\w\-\s.&'’]+ Act(?: of \d{4})?)$",
    re.I,
)
EXCHANGE_TICKER = re.compile(r"\b(?:NYSE|Nasdaq|NASDAQ|NYSE American)\s*:\s*([A-Z.]{1,5})\b")
MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
         r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")  # fmt: skip
DATE_PATTERNS = [
    re.compile(rf"\b{MONTH}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b"),  # December 31, 2023
    re.compile(rf"\b{MONTH}\s+\d{{1,2}}\b"),  # December 31
    re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)?\d{2}\b"),  # 12/31/2023
]
YEAR_PATTERNS = [
    re.compile(r"\b(?:19[5-9]\d|20[0-4]\d)(?:s\b)?(?![\d,.]\d)"),  # 2023, 1990s
    re.compile(r"(?<=\bFY)(?:20)?\d{2}\b"),  # FY23, FY2023
    re.compile(r"(?<=\bfiscal )'?\d{2}\b", re.I),  # fiscal '23
]
NER_LABELS = {"PERSON": "[PERSON]", "PRODUCT": "[PRODUCT]", "ORG": "[ORGANIZATION]"}
# Entity types whose words are never masked as unknown names.
# (Not LAW: the recognizer tags "the Worldpay Merchant Solutions" as one; laws named
# "... Act" are kept by KEEP_ORGANIZATIONS. All-caps words are never protected: "DMED"
# tagged as a place is a business unit.)
PROTECTED_LABELS = {"GPE", "LOC", "NORP", "LANGUAGE", "MONEY", "PERCENT", "QUANTITY",
                    "CARDINAL", "ORDINAL"}  # fmt: skip
CALENDAR_WORDS = {
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER",
    "OCTOBER", "NOVEMBER", "DECEMBER", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
    "FRIDAY", "SATURDAY", "SUNDAY",
}  # fmt: skip
CAPITALIZED_WORD = re.compile(r"(?<![\w\[])[A-Za-z][A-Za-z&'’\-]*[A-Za-z](?![\w\]])")


@dataclass(frozen=True)
class Identity:
    """What identifies one company: its names over time and its tickers."""

    names: tuple[str, ...]
    tickers: tuple[str, ...]


def _clean(name: str) -> str:
    name = re.sub(r"/[A-Z]{2,3}/?$", "", name.strip())  # EDGAR state tags: "INTEL CORP /DE/"
    return re.sub(r"\s+", " ", name.replace(",", " ")).strip()


def name_variants(names: Iterable[str]) -> list[str]:
    """Strings to mask for a company, longest first.

    From "SKYWORKS SOLUTIONS, INC.": "SKYWORKS SOLUTIONS INC.", "SKYWORKS SOLUTIONS",
    "SKYWORKS". The first word is used only if it is distinctive (4+ letters, not generic).
    """
    out: set[str] = set()
    for raw in names:
        full = _clean(raw)
        if not full:
            continue
        out.add(full)
        words = full.split()
        while words and re.sub(r"[^a-z ]", "", words[-1].lower()).strip() in LEGAL_SUFFIXES:
            words = words[:-1]
        if words and words[0].lower() == "the":
            words = words[1:]
        if words:
            out.add(" ".join(words))
            first = words[0].strip("&.")
            if len(first) >= 4 and first.isalpha() and first.lower() not in GENERIC_FIRST_WORDS:
                out.add(first)
    return sorted(out, key=len, reverse=True)


def _name_pattern(variant: str) -> re.Pattern[str]:
    # Case-insensitive; words may be separated by any whitespace or punctuation such as
    # "." or "-" ("Coca-Cola" vs "COCA COLA"); "&" may appear as "and".
    parts = [re.escape(w.strip(".")) for w in variant.split()]
    sep = r"[\s.\-,]*(?:&|and)?[\s.\-,]*"
    body = sep.join(p if p not in ("\\&",) else "(?:&|and)" for p in parts)
    return re.compile(rf"(?<![\w&])(?:the\s+)?{body}\.?(?!\w)", re.I)


def _is_unknown_name(word: str, common_words: frozenset[str]) -> bool:
    """A capitalized word with a part that is neither a common word nor a calendar word:
    "Worldpay", "Seepex", "DMED"; not "Forward-looking", "Non-GAAP" or "December"."""
    if not any(ch.isupper() for ch in word) or KEEP_ORGANIZATIONS.match(word):
        return False
    for part in re.split(r"[-&]", re.sub(r"['’]s?$", "", word)):
        p = part.upper()
        if len(p) < 3 or p in common_words or p in CALENDAR_WORDS:
            continue
        if p.endswith("S") and p[:-1] in common_words:  # plurals: "NOLs"
            continue
        return True
    return False


def is_common(entity: str, common_words: frozenset[str]) -> bool:
    """True if every word of the entity is a common (dictionary) word, e.g. "Related
    Payables" or "COVID-19"; False for anything containing a name, e.g. "Cigna Healthcare"."""
    words = re.findall(r"[A-Za-z]+", re.sub(r"['’]s\b", "", entity))
    return all(w.upper() in common_words for w in words)


def find_spans(
    text: str,
    identity: Identity,
    entities: EntityFinder | None,
    common_words: frozenset[str] = frozenset(),
) -> list[tuple[int, int, str]]:
    """Every span to mask as (start, end, placeholder); may overlap."""
    spans: list[tuple[int, int, str]] = []
    for variant in name_variants(identity.names):
        spans += [(m.start(), m.end(), "[COMPANY]") for m in _name_pattern(variant).finditer(text)]
    for ticker in identity.tickers:
        if len(ticker) >= 3:
            pat = re.compile(rf"(?<![\w$]){re.escape(ticker)}(?!\w)")
            spans += [(m.start(), m.end(), "[TICKER]") for m in pat.finditer(text)]
    spans += [(m.start(1), m.end(1), "[TICKER]") for m in EXCHANGE_TICKER.finditer(text)
              if m.group(1) in identity.tickers]  # fmt: skip
    found = entities(text) if entities is not None else []
    named: list[tuple[int, int, str]] = []
    for start, end, label in found:
        span = text[start:end].strip()
        if label not in NER_LABELS or KEEP_ORGANIZATIONS.match(span):
            continue
        if common_words and is_common(span, common_words):
            continue
        named.append((start, end, NER_LABELS[label]))
    if common_words:
        protected = [(s, e) for s, e, label in found if label in PROTECTED_LABELS]
        for m in CAPITALIZED_WORD.finditer(text):
            word = m.group(0)
            inside = not word.isupper() and any(s <= m.start() and m.end() <= e
                                                for s, e in protected)  # fmt: skip
            if not inside and _is_unknown_name(word, common_words):
                named.append((m.start(), m.end(), "[NAME]"))
    # A name masked once is masked at every occurrence.
    for surface, label in {(text[s:e], lab) for s, e, lab in named if e - s >= 3}:
        pat = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)")
        named += [(m.start(), m.end(), label) for m in pat.finditer(text)]
    spans += named
    for pat in DATE_PATTERNS:
        spans += [(m.start(), m.end(), "[DATE]") for m in pat.finditer(text)]
    for pat in YEAR_PATTERNS:
        spans += [(m.start(), m.end(), "[YEAR]") for m in pat.finditer(text)]
    return spans


# When spans overlap, the more specific type wins ("Fastenal" is [COMPANY] even inside the
# recognizer's "Fastenal Company"), then the longer span (a full date beats its year).
PRIORITY = {"[COMPANY]": 0, "[TICKER]": 1, "[DATE]": 2, "[PERSON]": 3, "[PRODUCT]": 4,
            "[ORGANIZATION]": 5, "[NAME]": 6, "[YEAR]": 7}  # fmt: skip


def _resolve(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Non-overlapping spans: by type priority, then longest first."""
    chosen: list[tuple[int, int, str]] = []
    for s in sorted(spans, key=lambda s: (PRIORITY[s[2]], -(s[1] - s[0]), s[0])):
        if all(s[1] <= c[0] or s[0] >= c[1] for c in chosen):
            chosen.append(s)
    return sorted(chosen)


def anonymize(text: str, identity: Identity, entities: EntityFinder | None = None,
              common_words: frozenset[str] = frozenset()) -> tuple[str, Counter]:  # fmt: skip
    """(masked text, count of each placeholder used)."""
    spans = find_spans(text, identity, entities, common_words)
    out, last, counts = [], 0, Counter()
    for start, end, label in _resolve(spans):
        out.append(text[last:start])
        out.append(label)
        counts[label] += 1
        last = end
    out.append(text[last:])
    return "".join(out), counts


def leaks(text: str, identity: Identity) -> dict[str, int]:
    """Identifying strings still present after masking (an automatic spot check)."""
    variants = name_variants(identity.names)
    return {
        "company_name": sum(len(_name_pattern(v).findall(text)) for v in variants),
        "ticker": sum(len(re.findall(rf"(?<![\w$]){re.escape(t)}(?!\w)", text))
                      for t in identity.tickers if len(t) >= 3),  # fmt: skip
        "year": len(re.findall(r"\b(?:19[5-9]\d|20[0-4]\d)\b", text)),
    }


def spacy_entities(model: str = "en_core_web_sm") -> EntityFinder:
    """Named-entity finder backed by spaCy (loaded once)."""
    import spacy

    nlp = spacy.load(model, disable=["lemmatizer"])

    def find(text: str) -> list[tuple[int, int, str]]:
        return [(e.start_char, e.end_char, e.label_) for e in nlp(text).ents]

    return find
