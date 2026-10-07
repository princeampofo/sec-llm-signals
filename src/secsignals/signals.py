"""Signals as the change from the same company's previous 10-K.

Writing style differs hugely between companies (a bank vs. a biotech), so the level
of negative words mostly measures style. The change against the company's own last
annual report measures news.

Point-in-time: a filing's previous 10-K is the latest one from the same CIK accepted
at least min_gap_days earlier, so it was public before the current filing. The signal
becomes known at the current filing's acceptance time (column known_at).
"""

from __future__ import annotations

import pandas as pd

from secsignals import dictionary


def previous_filings(filings: pd.DataFrame, min_gap_days: int, max_gap_days: int) -> pd.DataFrame:
    """Add prev_accession / prev_acceptance: the same CIK's latest 10-K accepted at least
    min_gap_days before this one. No match if that filing is over max_gap_days old
    (a stale comparison, e.g. after years without a 10-K).

    min_gap_days skips a re-filed or transition-period 10-K a few weeks earlier, which
    would compare a report with near-copy of itself.
    """
    f = filings[["accession", "cik", "acceptance_datetime"]].dropna()
    f = f.drop_duplicates(["accession", "cik"]).sort_values("acceptance_datetime")
    left = f.assign(cutoff=f["acceptance_datetime"] - pd.Timedelta(days=min_gap_days))
    right = f.rename(columns={"accession": "prev_accession",
                              "acceptance_datetime": "prev_acceptance"})  # fmt: skip
    out = pd.merge_asof(
        left.sort_values("cutoff"), right.sort_values("prev_acceptance"),
        left_on="cutoff", right_on="prev_acceptance", by="cik", direction="backward",
    )  # fmt: skip
    stale = out["acceptance_datetime"] - out["prev_acceptance"] > pd.Timedelta(days=max_gap_days)
    out.loc[stale, ["prev_accession", "prev_acceptance"]] = [None, pd.NaT]
    return out.drop(columns="cutoff")


def lm_change_signals(
    pairs: pd.DataFrame,
    counts: pd.DataFrame,
    categories: list[str],
    point_in_time: bool,
    max_length_ratio: float | None = None,
) -> pd.DataFrame:
    """Per filing: LM shares now and at the previous 10-K, and their changes.

    Both filings of a pair are scored with the dictionary version in force in the
    current filing's year, so a change never reflects an edit to the word list.
    d_<category> = share now - share before; lm_change = sum of those changes
    (higher = language got more negative/uncertain than the company's own last 10-K).

    max_length_ratio drops pairs whose sections differ in length by more than that
    factor: one year's extraction then almost always caught a stub or a different
    span, and the "change" compares unlike texts.
    """
    cur = pairs.merge(counts, on=["accession", "cik"], how="left")
    prev = pairs[["prev_accession", "cik"]].merge(
        counts.rename(columns={"accession": "prev_accession"}),
        on=["prev_accession", "cik"], how="left",
    )  # fmt: skip
    year = cur["acceptance_datetime"].dt.year
    out = pairs[["accession", "cik", "acceptance_datetime", "prev_accession",
                 "prev_acceptance"]].copy()  # fmt: skip
    out["known_at"] = out["acceptance_datetime"]
    out["dictionary_year"] = year
    out["lm_change"] = 0.0
    for category in categories:
        now = dictionary.shares(cur, category, year, point_in_time)
        before = dictionary.shares(prev, category, year, point_in_time)
        out[f"{category}_share"] = now.to_numpy()
        out[f"prev_{category}_share"] = before.to_numpy()
        out[f"d_{category}"] = (now - before).to_numpy()
        out["lm_change"] += out[f"d_{category}"]
    out["length_ratio"] = (cur["n_words"] / prev["n_words"]).to_numpy()
    if max_length_ratio is not None:
        r = out["length_ratio"]
        bad = (r > max_length_ratio) | (r < 1 / max_length_ratio)
        out.loc[bad, "lm_change"] = float("nan")
    return out


def missing_reasons(universe: pd.DataFrame, pairs: pd.DataFrame, ok: set[str],
                    signals: pd.DataFrame, signal: str) -> dict[str, int]:  # fmt: skip
    """Why universe filings lack a signal: the first failing step for each."""
    u = universe.drop_duplicates("accession")[["accession", "cik"]]
    u = u.merge(pairs[["accession", "cik", "prev_accession"]], on=["accession", "cik"], how="left")
    u = u.merge(signals[["accession", "cik", signal]], on=["accession", "cik"], how="left")
    u = u[u[signal].isna()]
    reason = pd.Series("pair_rejected_length_mismatch", index=u.index)
    reason[u["prev_accession"].notna() & ~u["prev_accession"].isin(ok)] = "previous_mda_failed"
    reason[u["prev_accession"].isna()] = "no_previous_10k"
    reason[~u["accession"].isin(ok)] = "current_mda_failed"
    return {k: int(v) for k, v in reason.value_counts().items()}


def signal_coverage(signals: pd.DataFrame, universe: pd.DataFrame, signal: str) -> dict:
    """Share of universe filings with a usable signal, overall and by year."""
    u = universe.drop_duplicates("accession")[["accession", "cik", "date_filed"]].merge(
        signals[["accession", "cik", signal]], on=["accession", "cik"], how="left"
    )
    has = u[signal].notna()
    years = pd.to_datetime(u["date_filed"]).dt.year
    by_year = has.groupby(years.to_numpy()).mean().round(4)
    return {
        "signal": signal,
        "universe_filings": int(len(u)),
        "with_signal": int(has.sum()),
        "coverage": round(float(has.mean()), 4),
        "by_year": {int(y): float(c) for y, c in by_year.items()},
    }
