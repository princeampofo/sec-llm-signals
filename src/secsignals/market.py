"""S&P 500 point-in-time universe, CIK<->ticker map, prices, forward returns, coverage.

Point-in-time rules (every function that answers "as of" a date takes it explicitly):
- universe_as_of(d) uses only membership snapshots dated on or before d.
- forward_returns() enters at the close of the first trading day strictly after the
  calendar date on which the signal became known (US Eastern). A filing accepted at
  9:00 am is therefore traded at the next day's close, not the same day's: we give
  up a few hours of edge in exchange for never trading on a price that might
  predate the news.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from secsignals.edgar import EASTERN

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- membership


def parse_membership(raw: bytes) -> pd.DataFrame:
    """fja05680 format: one row per change date, all member tickers comma-joined.

    Returns long snapshots (snapshot_date, ticker): the members from that date
    until the next snapshot.
    """
    wide = pd.read_csv(io.BytesIO(raw), parse_dates=["date"])
    long = wide.assign(ticker=wide["tickers"].str.split(",")).explode("ticker")
    long["ticker"] = long["ticker"].str.strip()
    out = long.rename(columns={"date": "snapshot_date"})[["snapshot_date", "ticker"]]
    return out.sort_values(["snapshot_date", "ticker"], ignore_index=True)


def universe_as_of(membership: pd.DataFrame, as_of: pd.Timestamp | date) -> list[str]:
    """Index members on as_of, from the latest snapshot dated on or before it."""
    as_of = pd.Timestamp(as_of)
    past = membership.loc[membership["snapshot_date"] <= as_of, "snapshot_date"]
    if past.empty:
        raise ValueError(f"no membership snapshot on or before {as_of.date()}")
    latest = past.max()
    return sorted(membership.loc[membership["snapshot_date"] == latest, "ticker"])


def membership_in_period(membership: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Snapshots in force at any time in [start, end], including the one live on start."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    dates = membership["snapshot_date"]
    first = dates[dates <= start].max()
    return membership[(dates >= first) & (dates <= end)].reset_index(drop=True)


def ticker_windows(membership: pd.DataFrame, end: date) -> pd.DataFrame:
    """First and last day each ticker was a member (within the given snapshots)."""
    dates = np.sort(membership["snapshot_date"].unique())
    next_date = dict(zip(dates[:-1], dates[1:], strict=True))
    last_day = membership["snapshot_date"].map(
        lambda d: next_date[d] - pd.Timedelta(days=1) if d in next_date else pd.Timestamp(end)
    )
    return (
        membership.assign(last_day=last_day.clip(upper=pd.Timestamp(end)))
        .groupby("ticker")
        .agg(first_day=("snapshot_date", "min"), last_day=("last_day", "max"))
        .reset_index()
    )


# ---------------------------------------------------------------- reference data parsers


def parse_sec_tickers(raw: bytes) -> pd.DataFrame:
    """SEC company_tickers.json -> (ticker, cik, title, rank). Rank 0 = primary listing."""
    rows = json.loads(raw).values()
    df = pd.DataFrame(rows).rename(columns={"cik_str": "cik"})
    df["ticker"] = df["ticker"].str.upper().str.replace("-", ".", regex=False)
    df["rank"] = df.groupby("cik").cumcount()  # file lists a company's main class first
    return df[["ticker", "cik", "title", "rank"]]


def parse_wikipedia(html: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(current constituents with CIK, change log with removed/added names)."""
    tables = pd.read_html(io.BytesIO(html))
    current = tables[0][["Symbol", "Security", "CIK"]].rename(
        columns={"Symbol": "ticker", "Security": "name", "CIK": "cik"}
    )
    current["ticker"] = current["ticker"].str.upper()
    changes = tables[1].copy()
    changes.columns = ["date", "added_ticker", "added_name", "removed_ticker", "removed_name",
                       "reason"]  # fmt: skip
    names = pd.concat(
        [
            changes[["added_ticker", "added_name"]].set_axis(["ticker", "name"], axis=1),
            changes[["removed_ticker", "removed_name"]].set_axis(["ticker", "name"], axis=1),
        ]
    ).dropna()
    names["ticker"] = names["ticker"].str.upper()
    return current, names.drop_duplicates(ignore_index=True)


# ---------------------------------------------------------------- CIK <-> ticker map

_SUFFIXES = (
    r"\b(?:the|inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|llc"
    r"|lp|l\s?p|n\s?v|s\s?a|ag|se|holdings?|group|international|intl|trust|bancorp)\b"
)


def normalize_name(name: str) -> str:
    """'Anadarko Petroleum Corp /DE/' and 'Anadarko Petroleum' -> 'anadarko petroleum'."""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"/[a-z]{2,4}/?", " ", s)   # EDGAR state tags: /DE/, /NEW/
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(_SUFFIXES, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _filer_activity(filing_index: pd.DataFrame) -> pd.DataFrame:
    df = filing_index[["cik", "company", "date_filed"]].copy()
    df["date_filed"] = pd.to_datetime(df["date_filed"])
    df["norm_name"] = df["company"].map(normalize_name)
    return df


def _filed_near(activity: pd.DataFrame, cik: int, first: pd.Timestamp, last: pd.Timestamp,
                window_days: int) -> bool:  # fmt: skip
    """Did this CIK file a 10-K while (or close to when) the ticker was a member?

    Guards against ticker reuse: SEC's ticker file says who owns a ticker today,
    which may not be the company that held it in the index years ago.
    """
    pad = pd.Timedelta(days=window_days)
    dates = activity.loc[activity["cik"] == cik, "date_filed"]
    return bool(((dates >= first - pad) & (dates <= last + pad)).any())


def build_ticker_map(
    windows: pd.DataFrame,
    sec_tickers: pd.DataFrame,
    wiki_current: pd.DataFrame,
    wiki_names: pd.DataFrame,
    filing_index: pd.DataFrame,
    overrides: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """Which company (CIK) a universe ticker meant, and over which dates.

    One row per (ticker, validity range). Usually a ticker has one row covering its
    whole membership window; a ticker that passed between companies while in the
    index (AGN: Allergan Inc, then Actavis renamed Allergan plc) has one row per
    holder, which only a hand-checked override can supply.

    Match order, most to least trusted:
      override   hand-checked rows in reference/ticker_overrides.csv
      wikipedia  current constituents table (has a CIK column)
      sec        SEC company_tickers.json (today's owner of the ticker)
      name       Wikipedia change-log company name == a 10-K filer's EDGAR name
    Every automatic match must have filed a 10-K near the ticker's membership window.
    """
    activity = _filer_activity(filing_index)
    by_name = activity.groupby("norm_name")["cik"].unique()
    wiki_cik = dict(zip(wiki_current["ticker"], wiki_current["cik"], strict=True))
    sec_cik = dict(zip(sec_tickers["ticker"], sec_tickers["cik"], strict=True))
    wiki_name = wiki_names.groupby("ticker")["name"].first()

    rows = []
    for w in windows.itertuples(index=False):
        base = {"ticker": w.ticker, "valid_from": w.first_day, "valid_to": w.last_day}
        manual = overrides[overrides["ticker"] == w.ticker]
        if not manual.empty:
            for o in manual.itertuples(index=False):
                valid_from = pd.Timestamp(o.valid_from) if pd.notna(o.valid_from) else w.first_day
                valid_to = pd.Timestamp(o.valid_to) if pd.notna(o.valid_to) else w.last_day
                rows.append({**base, "valid_from": valid_from, "valid_to": valid_to,
                             "cik": int(o.cik), "match_source": "override",
                             "yahoo_override": o.yahoo_symbol, "note": o.note})  # fmt: skip
            continue
        cik, source, note = None, None, ""

        def near(c: object, first: pd.Timestamp = w.first_day,
                 last: pd.Timestamp = w.last_day) -> bool:  # fmt: skip
            return _filed_near(activity, int(c), first, last, window_days)

        for src, candidate in (("wikipedia", wiki_cik.get(w.ticker)),
                               ("sec", sec_cik.get(w.ticker))):  # fmt: skip
            if candidate is None:
                continue
            if near(candidate):
                cik, source = int(candidate), src
                break
            note += f"{src} CIK {candidate} filed no 10-K near membership; "
        if cik is None and w.ticker in wiki_name.index:
            hits = [int(c) for c in by_name.get(normalize_name(wiki_name[w.ticker]), []) if near(c)]
            if len(hits) == 1:
                cik, source = hits[0], "name"
            else:
                note += f"name '{wiki_name[w.ticker]}' matched {len(hits)} filers {hits}; "
        rows.append({**base, "cik": cik, "match_source": source, "yahoo_override": None,
                     "note": note.strip()})  # fmt: skip

    out = pd.DataFrame(rows)
    out["cik"] = out["cik"].astype("Int64")
    out["yahoo_symbol"] = [
        _yahoo_symbol(r.ticker, r.cik, r.yahoo_override, sec_tickers)
        for r in out.itertuples(index=False)
    ]
    names = activity.sort_values("date_filed").groupby("cik")["company"].last()
    out["company"] = out["cik"].map(names)
    return out.drop(columns="yahoo_override")


def review_ticker_map(
    ticker_map: pd.DataFrame, filing_index: pd.DataFrame, window_days: int
) -> pd.DataFrame:
    """Rows whose CIK's 10-K history does not span its validity range.

    A gap usually means a predecessor CIK filed the earlier reports (Google Inc.
    before Alphabet) or the ticker belonged to a different company then (CEG).
    Each flag should end up either fixed by an override or explained in its note.
    """
    span = _filer_activity(filing_index).groupby("cik")["date_filed"].agg(["min", "max"])
    tm = ticker_map.dropna(subset=["cik"]).join(span, on="cik")
    pad = pd.Timedelta(days=window_days)
    starts_late = tm["min"] > tm["valid_from"] + pad
    ends_early = tm["max"] < tm["valid_to"] - pad
    flags = tm[starts_late | ends_early | tm["min"].isna()].copy()
    flags["issue"] = np.select(
        [flags["min"].isna(), starts_late[flags.index], ends_early[flags.index]],
        ["no 10-K in index", "first 10-K long after membership starts",
         "last 10-K long before membership ends"],  # fmt: skip
        default="",
    )
    cols = ["ticker", "cik", "valid_from", "valid_to", "min", "max", "match_source", "issue",
            "company", "note"]  # fmt: skip
    return flags.rename(columns={"min": "first_10k", "max": "last_10k"})[
        [c if c not in ("min", "max") else {"min": "first_10k", "max": "last_10k"}[c] for c in cols]
    ]


class TickerLookup:
    """Resolve (ticker, date) -> ticker-map row, honoring validity ranges."""

    def __init__(self, ticker_map: pd.DataFrame) -> None:
        self._rows: dict[str, list[dict]] = {}
        for r in ticker_map.to_dict("records"):
            self._rows.setdefault(r["ticker"], []).append(r)

    def resolve(self, ticker: str, day: pd.Timestamp) -> dict | None:
        rows = self._rows.get(ticker, [])
        for r in rows:
            if r["valid_from"] <= day <= r["valid_to"]:
                return r
        return rows[0] if len(rows) == 1 else None  # single holder: valid across the window

    def member_ciks(self, membership: pd.DataFrame, day: pd.Timestamp) -> dict[int, list[dict]]:
        """CIK -> map rows of its member tickers, for the index as of `day`."""
        out: dict[int, list[dict]] = {}
        for ticker in universe_as_of(membership, day):
            r = self.resolve(ticker, day)
            if r is not None and pd.notna(r["cik"]):
                out.setdefault(int(r["cik"]), []).append(r)
        return out


def _yahoo_symbol(ticker: str, cik: object, override: object, sec: pd.DataFrame) -> str | None:
    """Where Yahoo keeps this company's history today.

    Yahoo files a renamed company's full history under its new ticker (FB -> META),
    so a CIK that SEC still lists is priced under its current symbol. A CIK SEC no
    longer lists (acquired, delisted) is tried under its old ticker, unless another
    company now owns that ticker, in which case Yahoo's data would be someone else's.
    """
    if isinstance(override, str) and override:
        return None if override == "-" else override  # "-": checked, Yahoo has no usable series
    if pd.isna(cik):
        return None
    to_yahoo = lambda t: t.replace(".", "-")  # noqa: E731 - BRK.B is BRK-B on Yahoo
    same = sec[(sec["ticker"] == ticker) & (sec["cik"] == cik)]
    if not same.empty:
        return to_yahoo(ticker)
    current = sec[sec["cik"] == cik].sort_values("rank")
    if not current.empty:
        return to_yahoo(current["ticker"].iloc[0])
    if ticker in set(sec["ticker"]):
        return None
    return to_yahoo(ticker)


# ---------------------------------------------------------------- prices


class PriceSource(Protocol):
    """Anything that returns long (date, symbol, adj_close) daily prices.

    yfinance implements it; CRSP (with delisting returns) can replace it via config.
    """

    def adjusted_close(self, symbols: Iterable[str], start: date, end: date) -> pd.DataFrame: ...


class YFinanceSource:
    """Yahoo adjusted closes (split- and dividend-adjusted), cached one file per symbol.

    A symbol Yahoo has nothing for is cached as an empty file, so reruns do not
    hammer Yahoo for the same delisted tickers.
    """

    def __init__(self, cache_dir: Path, batch_size: int, pause_seconds: float,
                 download: Callable[..., pd.DataFrame] | None = None) -> None:  # fmt: skip
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.pause_seconds = pause_seconds
        self._download = download

    def _path(self, symbol: str, start: date, end: date) -> Path:
        return self.cache_dir / f"{start}_{end}" / f"{symbol}.parquet"

    def adjusted_close(self, symbols: Iterable[str], start: date, end: date) -> pd.DataFrame:
        symbols = sorted(set(symbols))
        todo = [s for s in symbols if not self._path(s, start, end).exists()]
        for i in range(0, len(todo), self.batch_size):
            self._fetch_batch(todo[i : i + self.batch_size], start, end)
            if i + self.batch_size < len(todo):
                time.sleep(self.pause_seconds)
        frames = [pd.read_parquet(p) for s in symbols if (p := self._path(s, start, end)).exists()]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["date", "symbol", "adj_close"])
        return pd.concat(frames, ignore_index=True)

    def _fetch_batch(self, batch: list[str], start: date, end: date) -> None:
        download = self._download
        if download is None:
            import yfinance as yf

            download = yf.download
        # yfinance's end is exclusive; add a day so `end` itself is included.
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        wide = download(batch, start=start_str, end=end_excl, auto_adjust=True,
                        progress=False, threads=True, group_by="column")  # fmt: skip
        close = wide["Close"] if "Close" in wide else pd.DataFrame()
        if isinstance(close, pd.Series):
            close = close.to_frame(batch[0])
        if close.dropna(how="all").empty and len(batch) > 1:
            # Every symbol empty is a failed request, not a batch of delisted stocks.
            # Cache nothing so the next run retries.
            log.warning("prices: whole batch empty, not cached: %s", batch)
            return
        for symbol in batch:
            path = self._path(symbol, start, end)
            path.parent.mkdir(parents=True, exist_ok=True)
            series = close[symbol].dropna() if symbol in close else pd.Series(dtype=float)
            df = pd.DataFrame({"date": pd.to_datetime(series.index).tz_localize(None).normalize(),
                               "symbol": symbol, "adj_close": series.to_numpy(float)})  # fmt: skip
            df.to_parquet(path, index=False)
        log.info("prices: %d symbols, %d empty", len(batch),
                 sum(pd.read_parquet(self._path(s, start, end)).empty for s in batch))  # fmt: skip


def price_source_from_config(cfg: dict) -> PriceSource:
    m = cfg["market"]
    if m["price_source"] == "yfinance":
        return YFinanceSource(Path(cfg["paths"]["raw"]) / "prices" / "yfinance",
                              m["download_batch_size"], m["pause_seconds"])  # fmt: skip
    raise ValueError(f"unknown price_source {m['price_source']!r}")


def vet_price_symbols(ticker_map: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame,
                                                                               pd.DataFrame]:
    """Drop Yahoo series that cannot be the mapped company's.

    Yahoo reuses symbols: "EMC" today is not EMC Corp (acquired 2016). A series
    that does not overlap the company's membership window at all belongs to some
    other security. Returns (prices without those series, rejected map rows).
    """
    span = prices.groupby("symbol")["date"].agg(["min", "max"])
    tm = ticker_map.dropna(subset=["yahoo_symbol"]).join(span, on="yahoo_symbol")
    bad = (tm["min"] > tm["valid_to"]) | (tm["max"] < tm["valid_from"])
    rejected = tm[bad].rename(columns={"min": "yahoo_first", "max": "yahoo_last"})
    still_used = set(tm.loc[~bad, "yahoo_symbol"])
    drop = set(rejected["yahoo_symbol"]) - still_used
    cols = ["ticker", "cik", "company", "valid_from", "valid_to", "yahoo_symbol", "yahoo_first",
            "yahoo_last"]  # fmt: skip
    return prices[~prices["symbol"].isin(drop)].reset_index(drop=True), rejected[cols]


# ---------------------------------------------------------------- forward returns


def market_close(day: pd.Timestamp, close_time: str) -> pd.Timestamp:
    """The tz-aware moment a trading day's closing price is set."""
    hh, mm = (int(x) for x in close_time.split(":"))
    return pd.Timestamp(datetime(day.year, day.month, day.day, hh, mm)).tz_localize(EASTERN)


def forward_returns(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    horizon: int,
    max_entry_lag_days: int,
    close_time: str,
) -> pd.DataFrame:
    """Return from entry close to the close `horizon` trading days later, per event.

    events: columns symbol and known_at (tz-aware: when the signal became public).
    Entry is the symbol's first trading day whose date is strictly after known_at's
    Eastern calendar date. Prices from on or before that date are never read.
    Missing data gives NaN, never a substituted price: if the stock stops trading
    before the exit day (e.g. delisted), the return is unknown.
    """
    out = events.reset_index(drop=True)
    # Mixed offsets (EST/EDT) arrive as object dtype; go through UTC to get one tz.
    out["known_at"] = pd.to_datetime(out["known_at"], utc=True).dt.tz_convert(EASTERN)
    out["entry_date"] = pd.NaT
    out["exit_date"] = pd.NaT
    out["fwd_return"] = np.nan
    known_date = out["known_at"].dt.tz_convert(EASTERN).dt.tz_localize(None).dt.normalize()
    by_symbol = {s: g.sort_values("date") for s, g in prices.groupby("symbol")}
    for idx in out.index:
        g = by_symbol.get(out.at[idx, "symbol"])
        if g is None:
            continue
        dates = g["date"].to_numpy()
        i = int(np.searchsorted(dates, known_date[idx].to_datetime64(), side="right"))
        if i >= len(dates):
            continue
        entry = pd.Timestamp(dates[i])
        if (entry - known_date[idx]).days > max_entry_lag_days:
            continue  # stock was not trading around the signal date
        out.at[idx, "entry_date"] = entry
        if i + horizon < len(dates):
            closes = g["adj_close"].to_numpy()
            out.at[idx, "exit_date"] = pd.Timestamp(dates[i + horizon])
            out.at[idx, "fwd_return"] = closes[i + horizon] / closes[i] - 1.0
    entered = out["entry_date"].notna()
    entry_ts = out.loc[entered, "entry_date"].map(lambda d: market_close(d, close_time))
    if not (entry_ts > out.loc[entered, "known_at"]).all():  # the no-lookahead invariant
        raise AssertionError("a forward-return window starts before its signal is known")
    return out


# ---------------------------------------------------------------- universe filings


def universe_filings(
    filing_index: pd.DataFrame, membership: pd.DataFrame, ticker_map: pd.DataFrame
) -> pd.DataFrame:
    """10-Ks whose filer was an index member on the filing date, with the member's
    Yahoo symbol. Share classes (GOOG/GOOGL) share a CIK: the alphabetically first
    member class is used. One row per document."""
    lookup = TickerLookup(ticker_map)
    days = pd.to_datetime(filing_index["date_filed"])
    members = {d: lookup.member_ciks(membership, d) for d in days.unique()}
    symbols = []
    for day, cik in zip(days, filing_index["cik"], strict=True):
        rows = members[day].get(int(cik))
        symbols.append(min((r["ticker"], r["yahoo_symbol"]) for r in rows)[1] if rows else "")
    out = filing_index.assign(symbol=symbols)
    out = out[out["symbol"] != ""].copy()
    out["symbol"] = out["symbol"].replace({None: pd.NA})
    return out.drop_duplicates("accession").reset_index(drop=True)


# ---------------------------------------------------------------- coverage


def month_ends(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")


def coverage(
    membership: pd.DataFrame,
    ticker_map: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    months: pd.DatetimeIndex,
) -> pd.DataFrame:
    """One row per universe-month: does the member have a price on the month's last
    trading day? (That is the price a month-end rebalance would trade on.)"""
    have = set(zip(prices["symbol"], prices["date"], strict=True))
    lookup = TickerLookup(ticker_map)
    rows = []
    for month_end in months:
        last_day = calendar[calendar <= month_end].max()
        for ticker in universe_as_of(membership, month_end):
            m = lookup.resolve(ticker, month_end)
            symbol = m["yahoo_symbol"] if m is not None else None
            if m is None or pd.isna(m["cik"]):
                reason = "no_cik"
            elif symbol is None or pd.isna(symbol):
                reason = "no_yahoo_symbol"
            elif (symbol, last_day) in have:
                reason = None
            else:
                reason = "no_price"
            rows.append({"month": month_end, "ticker": ticker,
                         "cik": None if m is None else m["cik"],
                         "company": None if m is None else m["company"],
                         "yahoo_symbol": symbol, "has_price": reason is None,
                         "missing_reason": reason})  # fmt: skip
    out = pd.DataFrame(rows)
    out["cik"] = out["cik"].astype("Int64")
    return out


def coverage_summary(cov: pd.DataFrame, periods: dict[str, tuple[date, date]]) -> dict:
    summary = {}
    for label, (start, end) in periods.items():
        c = cov[(cov["month"] >= pd.Timestamp(start)) & (cov["month"] <= pd.Timestamp(end))]
        summary[label] = {
            "universe_months": int(len(c)),
            "with_price": int(c["has_price"].sum()),
            "coverage": round(float(c["has_price"].mean()), 4),
            "missing_by_reason": c.loc[~c["has_price"], "missing_reason"].value_counts().to_dict(),
        }
    return summary


def coverage_by_year(cov: pd.DataFrame) -> pd.Series:
    return cov.groupby(cov["month"].dt.year)["has_price"].mean()


def coverage_report_md(summary: dict, by_year: pd.Series, missing: pd.DataFrame,
                       warn_below: float, top: int = 30) -> str:  # fmt: skip
    """Human-readable survivorship report (results/coverage_report.md)."""
    lines = ["# Price coverage of the S&P 500 universe", "",
             "Share of universe-months (index members at each month-end) with a Yahoo "
             "adjusted close on the month's last trading day.", "",
             "| Period | Universe-months | With price | Coverage | Missing: no price | "
             "Missing: no Yahoo symbol | Missing: no CIK |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for label, s in summary.items():
        r = s["missing_by_reason"]
        lines.append(f"| {label} | {s['universe_months']:,} | {s['with_price']:,} | "
                     f"{s['coverage']:.1%} | {r.get('no_price', 0):,} | "
                     f"{r.get('no_yahoo_symbol', 0):,} | {r.get('no_cik', 0):,} |")  # fmt: skip
    low = [k for k, s in summary.items() if s["coverage"] < warn_below]
    lines += [""]
    if low:
        lines += [f"**Coverage is below {warn_below:.0%} for: {', '.join(low)}.** The missing "
                  "companies are mostly ones later acquired or delisted, whose history Yahoo "
                  "no longer serves. Backtest results on this data are likely optimistic "
                  "(survivorship bias); CRSP with delisting returns would close the gap.",
                  ""]  # fmt: skip
    lines += ["## By year", "", "| Year | Coverage |", "|---|---|"]
    lines += [f"| {y} | {c:.1%} |" for y, c in by_year.items()]
    lines += ["", "## Reasons", "",
              "- **no_price**: mapped to a Yahoo symbol, but Yahoo has no close for that day "
              "(usually: the company was acquired or delisted and Yahoo dropped its history).",
              "- **no_yahoo_symbol**: the company's old ticker now belongs to a different "
              "security, so Yahoo's series under it would be someone else's prices.",
              "- **no_cik**: no SEC filer ID (banks that file with the FDIC, not the SEC).", "",
              f"## Most-missing companies (top {top}; full list in missing_companies.csv)", "",
              "| Ticker | Company | Reason | Months missing | First | Last |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for r in missing.head(top).itertuples(index=False):
        company = r.company if isinstance(r.company, str) else ""
        lines.append(f"| {r.ticker} | {company} | {r.missing_reason} | {r.months_missing} | "
                     f"{r.first_month:%Y-%m} | {r.last_month:%Y-%m} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def missing_companies(cov: pd.DataFrame) -> pd.DataFrame:
    miss = cov[~cov["has_price"]]
    return (
        miss.groupby(["ticker", "cik", "company", "yahoo_symbol", "missing_reason"], dropna=False)
        .agg(months_missing=("month", "size"), first_month=("month", "min"),
             last_month=("month", "max"))  # fmt: skip
        .reset_index()
        .sort_values("months_missing", ascending=False, ignore_index=True)
    )
