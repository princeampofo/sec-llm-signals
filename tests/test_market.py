from datetime import date

import numpy as np
import pandas as pd
import pytest

from secsignals import market
from secsignals.edgar import EASTERN

# ---------------------------------------------------------------- membership

MEMBERSHIP_CSV = b'''date,tickers
2019-12-20,"AAA,BBB,CCC"
2020-03-02,"AAA,BBB,DDD"
2020-06-22,"AAA,DDD,EEE"
'''


@pytest.fixture
def membership():
    return market.parse_membership(MEMBERSHIP_CSV)


def test_parse_membership_is_long(membership):
    assert list(membership.columns) == ["snapshot_date", "ticker"]
    assert len(membership) == 9


def test_universe_as_of_uses_latest_snapshot_on_or_before(membership):
    assert market.universe_as_of(membership, date(2020, 3, 1)) == ["AAA", "BBB", "CCC"]
    assert market.universe_as_of(membership, date(2020, 3, 2)) == ["AAA", "BBB", "DDD"]
    assert market.universe_as_of(membership, date(2020, 12, 31)) == ["AAA", "DDD", "EEE"]
    with pytest.raises(ValueError):
        market.universe_as_of(membership, date(2019, 1, 1))


def test_universe_as_of_never_reads_future_rows(membership):
    # No-lookahead: adding snapshots after the as-of date must not change the answer.
    future = pd.DataFrame({"snapshot_date": pd.Timestamp("2020-03-03"), "ticker": ["ZZZ"]})
    poisoned = pd.concat([membership, future], ignore_index=True)
    for d in [date(2020, 1, 15), date(2020, 3, 2)]:
        assert market.universe_as_of(poisoned, d) == market.universe_as_of(membership, d)


def test_membership_in_period_keeps_snapshot_live_on_start(membership):
    m = market.membership_in_period(membership, date(2020, 1, 1), date(2020, 5, 1))
    assert sorted(m["snapshot_date"].dt.date.unique()) == [date(2019, 12, 20), date(2020, 3, 2)]


def test_ticker_windows(membership):
    w = market.ticker_windows(membership, date(2020, 12, 31)).set_index("ticker")
    assert w.loc["CCC", "first_day"] == pd.Timestamp("2019-12-20")
    assert w.loc["CCC", "last_day"] == pd.Timestamp("2020-03-01")  # gone from 2020-03-02
    assert w.loc["BBB", "last_day"] == pd.Timestamp("2020-06-21")
    assert w.loc["AAA", "last_day"] == pd.Timestamp("2020-12-31")


# ---------------------------------------------------------------- ticker map


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("ANADARKO PETROLEUM CORP /DE/", "anadarko petroleum"),
        ("Anadarko Petroleum", "anadarko petroleum"),
        ("Twitter, Inc.", "twitter"),
        ("AT&T Inc.", "at and t"),
        ("The Hershey Company", "hershey"),
    ],
)
def test_normalize_name(raw, norm):
    assert market.normalize_name(raw) == norm


def filings(*rows):
    return pd.DataFrame(rows, columns=["cik", "company", "date_filed"]).assign(
        date_filed=lambda d: pd.to_datetime(d["date_filed"]).dt.date
    )


def windows(*rows):
    return pd.DataFrame(
        [(t, pd.Timestamp(a), pd.Timestamp(b)) for t, a, b in rows],
        columns=["ticker", "first_day", "last_day"],
    )


def sec(*rows):
    df = pd.DataFrame(rows, columns=["ticker", "cik", "title"])
    df["rank"] = df.groupby("cik").cumcount()
    return df


EMPTY_WIKI = pd.DataFrame(columns=["ticker", "name", "cik"])
EMPTY_NAMES = pd.DataFrame(columns=["ticker", "name"])
NO_OVERRIDES = pd.DataFrame(columns=["ticker", "cik", "valid_from", "valid_to",
                                     "yahoo_symbol", "note"])  # fmt: skip


def build(win, sec_t=None, wiki=EMPTY_WIKI, names=EMPTY_NAMES, fil=None, overrides=NO_OVERRIDES):
    return market.build_ticker_map(
        win,
        sec_t if sec_t is not None else sec(),
        wiki,
        names,
        fil if fil is not None else filings(),
        overrides,
        window_days=400,
    ).set_index("ticker")


def test_ticker_map_sources_and_renamed_company_priced_under_new_symbol():
    tm = build(
        windows(("AAA", "2015-01-01", "2020-12-31"), ("FB", "2013-01-01", "2022-06-08"),
                ("OLD", "2012-01-01", "2014-01-01")),  # fmt: skip
        sec_t=sec(("AAA", 1, "Triple A Inc"), ("META", 2, "Meta Platforms")),
        names=pd.DataFrame({"ticker": ["FB", "OLD"], "name": ["Facebook", "Old Co"]}),
        fil=filings((1, "TRIPLE A INC", "2018-02-01"), (2, "FACEBOOK INC", "2016-02-01"),
                    (3, "OLD CO", "2013-02-01")),  # fmt: skip
    )
    assert (tm.loc["AAA", "match_source"], tm.loc["AAA", "yahoo_symbol"]) == ("sec", "AAA")
    # FB matched by name; Yahoo keeps its history under the current ticker.
    assert (tm.loc["FB", "cik"], tm.loc["FB", "yahoo_symbol"]) == (2, "META")
    # Delisted, ticker not reused: try the old ticker on Yahoo.
    assert (tm.loc["OLD", "cik"], tm.loc["OLD", "yahoo_symbol"]) == (3, "OLD")


def test_ticker_map_rejects_reused_ticker():
    # SEC says SUN belongs to CIK 9 today, but CIK 9 filed nothing while SUN was a member.
    tm = build(
        windows(("SUN", "2011-12-30", "2012-10-04")),
        sec_t=sec(("SUN", 9, "Sunoco LP")),
        fil=filings((9, "SUNOCO LP", "2015-02-27")),
    )
    assert pd.isna(tm.loc["SUN", "cik"])
    assert "filed no 10-K near membership" in tm.loc["SUN", "note"]


def test_ticker_map_ambiguous_name_is_left_unmatched():
    tm = build(
        windows(("AGN", "2012-01-01", "2020-01-01")),
        names=pd.DataFrame({"ticker": ["AGN"], "name": ["Allergan"]}),
        fil=filings((1, "ALLERGAN INC", "2013-01-01"), (2, "Allergan plc", "2017-01-01")),
    )
    assert pd.isna(tm.loc["AGN", "cik"])
    assert "matched 2 filers" in tm.loc["AGN", "note"]


def test_ticker_map_override_with_date_ranges_and_no_yahoo_marker():
    overrides = pd.DataFrame(
        [("AGN", 1, None, "2015-03-17", None, "Allergan Inc"),
         ("AGN", 2, "2015-03-18", None, "-", "Allergan plc")],
        columns=NO_OVERRIDES.columns,
    )  # fmt: skip
    tm = market.build_ticker_map(
        windows(("AGN", "2012-01-01", "2020-01-01")), sec(), EMPTY_WIKI, EMPTY_NAMES,
        filings(), overrides, 400,
    )  # fmt: skip
    assert list(tm["cik"]) == [1, 2]
    assert list(tm["valid_from"]) == [pd.Timestamp("2012-01-01"), pd.Timestamp("2015-03-18")]
    assert list(tm["valid_to"]) == [pd.Timestamp("2015-03-17"), pd.Timestamp("2020-01-01")]
    assert pd.isna(tm["yahoo_symbol"].iloc[1])

    lookup = market.TickerLookup(tm)
    assert lookup.resolve("AGN", pd.Timestamp("2014-06-30"))["cik"] == 1
    assert lookup.resolve("AGN", pd.Timestamp("2016-06-30"))["cik"] == 2


def test_yahoo_symbol_none_when_old_ticker_now_belongs_to_someone_else():
    tm = build(
        windows(("CEG", "2011-12-30", "2012-03-12")),
        sec_t=sec(("CEG", 99, "Constellation Energy Corp")),
        names=pd.DataFrame({"ticker": ["CEG"], "name": ["Constellation Energy Group"]}),
        fil=filings((5, "CONSTELLATION ENERGY GROUP INC", "2012-02-29"),
                    (99, "Constellation Energy Corp", "2023-02-01")),  # fmt: skip
    )
    assert tm.loc["CEG", "cik"] == 5
    assert pd.isna(tm.loc["CEG", "yahoo_symbol"])  # Yahoo's CEG is the 2022 company


def test_review_flags_predecessor_gap():
    tm = market.build_ticker_map(
        windows(("GOOGL", "2012-01-01", "2020-12-31")), sec(("GOOGL", 2, "Alphabet")),
        EMPTY_WIKI, EMPTY_NAMES, filings((2, "Alphabet Inc", "2016-02-11"),
                                         (2, "Alphabet Inc", "2020-02-04")),
        NO_OVERRIDES, 400,
    )  # fmt: skip
    fil = filings((2, "Alphabet Inc", "2016-02-11"), (2, "Alphabet Inc", "2020-02-04"))
    review = market.review_ticker_map(tm, fil, 400)
    assert list(review["issue"]) == ["first 10-K long after membership starts"]


# ---------------------------------------------------------------- universe filings


def test_universe_filings_point_in_time_share_classes_and_co_registrants(membership):
    tmap = pd.DataFrame(
        [("AAA", 1, "2019-12-20", "2020-12-31", "AAA"),
         ("BBB", 2, "2019-12-20", "2020-06-21", "BBB"),
         ("CCC", 3, "2019-12-20", "2020-03-01", "CCC"),
         ("DDD", 4, "2020-03-02", "2020-12-31", "DDD"),
         ("EEE", 4, "2020-06-22", "2020-12-31", "EEE")],
        columns=["ticker", "cik", "valid_from", "valid_to", "yahoo_symbol"],
    ).astype({"valid_from": "datetime64[ns]", "valid_to": "datetime64[ns]"})  # fmt: skip
    index = pd.DataFrame(
        {
            "accession": ["a1", "a2", "a3", "a4", "a4", "a5"],
            "cik": [1, 3, 3, 4, 77, 99],
            "date_filed": [date(2020, 2, 1), date(2020, 2, 1), date(2020, 4, 1), date(2020, 7, 1),
                           date(2020, 7, 1), date(2020, 2, 1)],
        }
    )  # fmt: skip
    out = market.universe_filings(index, membership, tmap).set_index("accession")
    assert sorted(out.index) == ["a1", "a2", "a4"]  # a3: CCC left before filing; a5: never member
    assert out.loc["a4", "cik"] == 4  # combined filing kept once, under the member CIK
    assert out.loc["a4", "symbol"] == "DDD"  # two member classes: alphabetical first


# ---------------------------------------------------------------- forward returns

# A small trading calendar: Mon 2021-03-01 .. with a holiday on Fri 2021-04-02 (Good Friday).
CAL = pd.bdate_range("2021-03-01", "2021-05-31").drop(pd.Timestamp("2021-04-02"))


def prices_for(symbol="XYZ", calendar=CAL, start=100.0):
    closes = start * np.cumprod(np.full(len(calendar), 1.01))
    return pd.DataFrame({"date": calendar, "symbol": symbol, "adj_close": closes})


def events(*known_at, symbol="XYZ"):
    return pd.DataFrame({"symbol": symbol, "known_at": [pd.Timestamp(k) for k in known_at]})


def fwd(prices, ev, horizon=21, lag=7):
    return market.forward_returns(prices, ev, horizon, lag, "16:00")


@pytest.mark.parametrize(
    ("known_at", "entry"),
    [
        ("2021-03-01 17:30:00-05:00", "2021-03-02"),  # after the close -> next day
        ("2021-03-01 08:00:00-05:00", "2021-03-02"),  # before the open -> still next day
        ("2021-03-01 15:59:59-05:00", "2021-03-02"),  # during the session -> next day
        ("2021-03-05 18:00:00-05:00", "2021-03-08"),  # Friday evening -> Monday
        ("2021-03-06 12:00:00-05:00", "2021-03-08"),  # Saturday -> Monday
        ("2021-04-01 17:00:00-04:00", "2021-04-05"),  # skips the Good Friday holiday
        ("2021-03-02 04:30:00+00:00", "2021-03-02"),  # = Mar 1 23:30 Eastern; UTC date misleads
    ],
)
def test_entry_is_first_trading_day_after_signal_date(known_at, entry):
    out = fwd(prices_for(), events(known_at))
    assert out["entry_date"].iloc[0] == pd.Timestamp(entry)


def test_forward_return_value_and_exit_day():
    out = fwd(prices_for(), events("2021-03-01 17:30:00-05:00"))
    assert out["exit_date"].iloc[0] == CAL[1 + 21]
    assert out["fwd_return"].iloc[0] == pytest.approx(1.01**21 - 1)


def test_no_forward_window_starts_on_or_before_its_signal():
    """The core no-lookahead test: for many random signal times, the entry close
    is strictly after the signal and no trading day is skipped in between."""
    rng = np.random.default_rng(0)
    seconds = rng.integers(0, 60 * 24 * 3600, size=500)
    known = pd.Timestamp("2021-03-01", tz=EASTERN) + pd.to_timedelta(seconds, unit="s")
    out = fwd(prices_for(), pd.DataFrame({"symbol": "XYZ", "known_at": known}), horizon=5)
    assert out["entry_date"].notna().all()
    entry_close = out["entry_date"].map(lambda d: market.market_close(d, "16:00"))
    assert (entry_close > out["known_at"]).all()
    known_day = out["known_at"].dt.tz_convert(EASTERN).dt.tz_localize(None).dt.normalize()
    assert (out["entry_date"] > known_day).all()  # never the signal's own day
    for k, e in zip(known_day, out["entry_date"], strict=True):
        assert not ((CAL > k) & (CAL < e)).any()  # and the very next session, not later


def test_forward_returns_ignore_prices_on_or_before_signal_day():
    # No-lookahead: scrambling every price up to the signal day changes nothing.
    p = prices_for()
    ev = events("2021-03-10 17:00:00-05:00")
    poisoned = p.copy()
    poisoned.loc[poisoned["date"] <= "2021-03-10", "adj_close"] = 1e9
    pd.testing.assert_frame_equal(fwd(p, ev), fwd(poisoned, ev))


def test_forward_return_missing_cases_are_nan():
    out = fwd(
        prices_for(),
        pd.concat([events("2021-05-20 17:00:00-04:00"),  # not 21 days of data left
                   events("2021-03-01 17:00:00-05:00", symbol="NOPE"),  # no prices
                   events("2021-01-04 17:00:00-05:00")]),  # stock not trading near the signal
    )  # fmt: skip
    assert out["fwd_return"].isna().all()
    assert out["entry_date"].notna().tolist() == [True, False, False]


def test_forward_returns_raise_if_invariant_is_violated(monkeypatch):
    # If a bug ever made entry == signal day, the function must fail loudly.
    monkeypatch.setattr(market, "market_close", lambda d, t: pd.Timestamp("2000-01-01", tz=EASTERN))
    with pytest.raises(AssertionError, match="before its signal"):
        fwd(prices_for(), events("2021-03-01 17:30:00-05:00"))


# ---------------------------------------------------------------- coverage


def test_coverage_reasons_and_summary(membership):
    tmap = pd.DataFrame(
        [("AAA", 1, "AAA", "A Inc"), ("BBB", 2, None, "B Inc"), ("CCC", pd.NA, None, None),
         ("DDD", 4, "DDD", "D Inc")],
        columns=["ticker", "cik", "yahoo_symbol", "company"],
    ).assign(valid_from=pd.Timestamp("2019-01-01"),
             valid_to=pd.Timestamp("2020-12-31"))  # fmt: skip
    tmap["cik"] = tmap["cik"].astype("Int64")
    cal = pd.bdate_range("2020-01-01", "2020-04-30")
    prices = pd.concat([prices_for("AAA", cal), prices_for("DDD", cal[cal < "2020-03-15"])])
    cov = market.coverage(membership, tmap, prices, cal, market.month_ends(date(2020, 1, 1),
                                                                            date(2020, 3, 31)))
    reasons = cov.set_index(["month", "ticker"])["missing_reason"]
    assert pd.isna(reasons[(pd.Timestamp("2020-01-31"), "AAA")])
    assert reasons[(pd.Timestamp("2020-01-31"), "BBB")] == "no_yahoo_symbol"
    assert reasons[(pd.Timestamp("2020-01-31"), "CCC")] == "no_cik"
    assert reasons[(pd.Timestamp("2020-03-31"), "DDD")] == "no_price"  # data stops mid-March
    s = market.coverage_summary(cov, {"all": (date(2020, 1, 1), date(2020, 3, 31))})["all"]
    assert s["universe_months"] == 9 and s["with_price"] == 3
    missing = market.missing_companies(cov)
    assert set(missing["ticker"]) == {"BBB", "CCC", "DDD"}


def test_vet_price_symbols_rejects_reused_yahoo_symbols():
    tmap = pd.DataFrame(
        [("EMC", 1, "EMC", "2012-01-01", "2016-09-06"),   # Yahoo's EMC starts 2023: not EMC Corp
         ("GOOGL", 2, "GOOGL", "2012-01-01", "2015-10-01"),  # predecessor on a continuing series
         ("GOOGL", 3, "GOOGL", "2015-10-02", "2020-12-31")],
        columns=["ticker", "cik", "yahoo_symbol", "valid_from", "valid_to"],
    ).assign(company="x").astype({"valid_from": "datetime64[ns]",
                                  "valid_to": "datetime64[ns]"})  # fmt: skip
    emc = prices_for("EMC", pd.bdate_range("2023-05-15", "2023-06-30"))
    googl = prices_for("GOOGL", pd.bdate_range("2012-01-02", "2020-12-31"))
    prices = pd.concat([emc, googl])
    kept, rejected = market.vet_price_symbols(tmap, prices)
    assert rejected["ticker"].tolist() == ["EMC"]
    assert set(kept["symbol"]) == {"GOOGL"}


# ---------------------------------------------------------------- price cache


def test_yfinance_source_caches_and_marks_empty_symbols(tmp_path):
    calls = []

    def fake_download(batch, start, end, **kw):
        calls.append((tuple(batch), start, end))
        idx = pd.DatetimeIndex(["2021-03-01", "2021-03-02"], name="Date")
        close = pd.DataFrame({"AAA": [10.0, 11.0], "GONE": [np.nan, np.nan]}, index=idx)
        return pd.concat({"Close": close}, axis=1)

    src = market.YFinanceSource(tmp_path, batch_size=10, pause_seconds=0, download=fake_download)
    out = src.adjusted_close(["AAA", "GONE"], date(2021, 3, 1), date(2021, 3, 2))
    assert out["symbol"].unique().tolist() == ["AAA"]
    assert calls[0][2] == "2021-03-03"  # yfinance end is exclusive: asked one day more
    src.adjusted_close(["AAA", "GONE"], date(2021, 3, 1), date(2021, 3, 2))
    assert len(calls) == 1  # second call served from cache, including the empty GONE marker


def test_yfinance_source_does_not_cache_a_failed_batch(tmp_path):
    def failed_download(batch, start, end, **kw):
        idx = pd.DatetimeIndex([], name="Date")
        return pd.concat({"Close": pd.DataFrame(columns=batch, index=idx, dtype=float)}, axis=1)

    src = market.YFinanceSource(tmp_path, batch_size=10, pause_seconds=0, download=failed_download)
    out = src.adjusted_close(["AAA", "BBB"], date(2021, 3, 1), date(2021, 3, 2))
    assert out.empty
    assert not list(tmp_path.rglob("*.parquet"))  # nothing cached: the next run retries
