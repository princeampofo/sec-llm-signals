"""Loughran-McDonald (LM) dictionary scores: share of negative and uncertainty words.

Point-in-time word lists. LM's master dictionary records, per category, the year a
word joined the list (2009 = original list; later additions such as CYBERATTACK in
2014) or, as a negative number, the year it was dropped (-2020: CLOSED, BREAKING...).
Scoring a 2012 filing with a 2014 word is a small lookahead, and scoring last year's
filing with a different list than this year's turns dictionary edits into fake
"tone changes". So words are counted per cohort, i.e. per (category, year added or
removed), and a share can then be computed for any dictionary version:
    share(category, as of year Y) = sum of counts of cohorts active in Y / n_words
A version is in force for a filing year Y if it was published before Y began:
words added in year A count when A < Y; words removed in year R count when Y <= R.

Word count. Following LM, the denominator is the number of tokens found in the master
dictionary (a near-complete English word list), which drops numbers, table debris and
other non-words that would otherwise dilute the shares.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

TOKEN = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Lexicon:
    vocabulary: frozenset[str]          # every master-dictionary word (the denominator)
    cohorts: dict[str, tuple[str, ...]]  # word -> cohort columns it counts toward

    @property
    def columns(self) -> list[str]:
        return sorted({c for cols in self.cohorts.values() for c in cols})


def parse_master_dictionary(raw: bytes) -> pd.DataFrame:
    # keep_default_na=False: the dictionary contains the words NA and NAN.
    df = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
    df["Word"] = df["Word"].astype(str).str.upper()
    return df


def cohort_column(category: str, code: int) -> str:
    """Negative 2014 -> 'negative__a2014'; Negative -2020 -> 'negative__r2020'."""
    return f"{category}__a{code}" if code > 0 else f"{category}__r{-code}"


def build_lexicon(master: pd.DataFrame, categories: list[str]) -> Lexicon:
    cohorts: dict[str, list[str]] = {}
    for category in categories:
        codes = master[category.capitalize()].astype(int)
        for word, code in zip(master.loc[codes != 0, "Word"], codes[codes != 0], strict=True):
            cohorts.setdefault(word, []).append(cohort_column(category, int(code)))
    return Lexicon(frozenset(master["Word"]), {w: tuple(c) for w, c in cohorts.items()})


def score_text(text: str, lexicon: Lexicon) -> dict[str, int]:
    """n_words plus a count per cohort column (all columns present, zeros included)."""
    counts: Counter[str] = Counter()
    n_words = 0
    for token in TOKEN.findall(text):
        word = token.upper()
        if word in lexicon.vocabulary:
            n_words += 1
            for column in lexicon.cohorts.get(word, ()):
                counts[column] += 1
    return {"n_words": n_words, **{c: counts.get(c, 0) for c in lexicon.columns}}


def score_sections(sections: pd.DataFrame, lexicon: Lexicon) -> pd.DataFrame:
    """Cohort counts for every successfully extracted section."""
    ok = sections[sections["status"] == "ok"]
    scores = pd.DataFrame([score_text(t, lexicon) for t in ok["text"]], index=ok.index)
    return pd.concat([ok[["accession", "cik"]], scores], axis=1).reset_index(drop=True)


def _active(column: str, year: int) -> bool:
    kind, cohort_year = column.split("__")[1][0], int(column.split("__")[1][1:])
    return cohort_year < year if kind == "a" else year <= cohort_year


def shares(counts: pd.DataFrame, category: str, years: pd.Series | int,
           point_in_time: bool = True) -> pd.Series:  # fmt: skip
    """Share of `category` words under the dictionary version in force in `years`.

    years is one year per row (or a single year). point_in_time=False uses the latest
    version for every row: all additions counted, all removals applied.
    """
    years = pd.Series(years, index=counts.index) if isinstance(years, int) else years
    columns = [c for c in counts.columns if c.startswith(f"{category}__")]
    hits = pd.Series(0, index=counts.index, dtype="int64")
    for column in columns:
        if point_in_time:
            active = years.map(lambda y, c=column: _active(c, int(y)))
        else:
            active = pd.Series(column.split("__")[1].startswith("a"), index=counts.index)
        hits += counts[column].where(active, 0)
    return hits / counts["n_words"].where(counts["n_words"] > 0)
