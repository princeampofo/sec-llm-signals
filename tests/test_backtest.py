import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from secsignals import backtest, factors

ET = "America/New_York"
CAL = pd.bdate_range("2019-12-02", "2022-01-31")
DATES = backtest.rebalance_dates(CAL, "2019-12-01", "2022-01-31")
N = 100
TICKERS = [f"T{i:03d}" for i in range(N)]


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(7)
    daily = rng.normal(0.0003, 0.02, size=(len(CAL), N))
    closes = 100 * np.cumprod(1 + daily, axis=0)
    prices = pd.DataFrame(closes, index=CAL, columns=TICKERS).stack().reset_index()
    prices.columns = ["date", "symbol", "adj_close"]
    membership = pd.DataFrame({"snapshot_date": pd.Timestamp("2019-01-01"), "ticker": TICKERS})
    tmap = pd.DataFrame({"ticker": TICKERS, "cik": range(N), "yahoo_symbol": TICKERS,
                         "valid_from": pd.Timestamp("2019-01-01"),
                         "valid_to": pd.Timestamp("2022-12-31")})  # fmt: skip
    returns = backtest.period_returns(prices, DATES)
    return prices, membership, tmap, returns


def monthly_signals(returns, score_fn):
    """One signal per company per rebalance date, accepted the evening before t."""
    rows = []
    for t, g in returns.groupby("date"):
        known = (t - pd.Timedelta(days=1)).tz_localize(ET) + pd.Timedelta(hours=17)
        for sym, ret in zip(g["symbol"], g["ret"], strict=True):
            rows.append((int(sym[1:]), known, score_fn(ret)))
    return pd.DataFrame(rows, columns=["cik", "known_at", "score"])


def run(market, signals, **kw):
    _, membership, tmap, returns = market
    panel = backtest.build_panel(signals, membership, tmap, returns, DATES, max_age_days=365)
    monthly = backtest.long_short(panel, kw.get("q", 5), kw.get("cost", 10), min_stocks=20)
    return panel, monthly, backtest.summarize(monthly)


# ---------------------------------------------------------------- the two sanity checks


def test_planted_future_signal_shows_huge_performance(market):
    # A "signal" that is next month's return: pure lookahead. The engine must turn it
    # into absurd performance; if it did not, the engine itself would be broken.
    _, monthly, s = run(market, monthly_signals(market[3], lambda r: r))
    assert s["mean_ic"] > 0.95
    assert s["gross"]["sharpe"] > 10
    assert (monthly["ret_long"] > monthly["ret_short"]).all()
    q = s["quantile_annual_returns"]
    assert q["q1"] < q["q2"] < q["q3"] < q["q4"] < q["q5"]


def test_random_noise_has_roughly_zero_ic(market):
    # One noise draw can land anywhere (|t| > 2 happens 5% of the time by construction),
    # so test the distribution over many draws: an engine that leaked future returns into
    # the ranking would shift it away from zero.
    _, membership, tmap, returns = market
    panel = backtest.build_panel(monthly_signals(returns, lambda r: 0.0), membership, tmap,
                                 returns, DATES, max_age_days=365)  # fmt: skip
    ics, ts = [], []
    for seed in range(100):
        noisy = panel.assign(score=np.random.default_rng(seed).normal(size=len(panel)))
        s = backtest.summarize(backtest.long_short(noisy, 5, 10, min_stocks=20))
        ics.append(s["mean_ic"])
        ts.append(s["ic_t"])
    ics, ts = np.array(ics), np.array(ts)
    assert abs(ics.mean()) < 0.005  # roughly zero IC on average
    assert abs(ts.mean()) < 0.3 and 0.8 < ts.std() < 1.3  # t-stats ~ N(0, 1)
    assert (np.abs(ts) > 2).mean() < 0.12


# ---------------------------------------------------------------- point-in-time rules


def one_signal(cik, known_at, score):
    return pd.DataFrame({"cik": [cik], "known_at": [pd.Timestamp(known_at, tz=ET)],
                         "score": [score]})  # fmt: skip


def test_signal_accepted_on_rebalance_day_waits_for_next_month():
    t = pd.Timestamp("2021-03-31")
    sig = one_signal(1, "2021-03-31 09:00", 1.0)  # before the close, but same day
    assert backtest.signals_as_of(sig, t, 365).empty
    sig = one_signal(1, "2021-03-30 17:30", 1.0)  # the evening before
    assert backtest.signals_as_of(sig, t, 365).loc[1] == 1.0


def test_signal_expires_after_max_age():
    sig = one_signal(1, "2020-03-01 17:00", 1.0)
    assert backtest.signals_as_of(sig, pd.Timestamp("2021-02-26"), 365).loc[1] == 1.0
    assert backtest.signals_as_of(sig, pd.Timestamp("2021-03-31"), 365).empty


def test_latest_signal_wins():
    sig = pd.concat([one_signal(1, "2020-03-01 17:00", -1.0),
                     one_signal(1, "2021-03-01 17:00", 2.0)])  # fmt: skip
    assert backtest.signals_as_of(sig, pd.Timestamp("2021-03-31"), 365).loc[1] == 2.0


def test_period_returns_read_only_the_holding_window():
    cal = pd.bdate_range("2021-01-01", "2021-03-31")
    dates = backtest.rebalance_dates(cal, "2021-01-01", "2021-03-31")
    p = pd.DataFrame({"date": cal, "symbol": "A", "adj_close": np.linspace(100, 130, len(cal))})
    base = backtest.period_returns(p, dates)
    poisoned = p.copy()
    outside = (poisoned["date"] < dates[0]) | (poisoned["date"] > dates[-1])
    poisoned.loc[outside, "adj_close"] = 1e9
    pd.testing.assert_frame_equal(base, backtest.period_returns(poisoned, dates))
    jan, feb = dates[0], dates[1]
    expected = p.set_index("date").loc[feb, "adj_close"] / p.set_index("date").loc[jan, "adj_close"]
    assert base.loc[base["date"] == jan, "ret"].iloc[0] == pytest.approx(expected - 1)


def test_stock_that_stops_trading_is_held_to_last_close():
    cal = pd.bdate_range("2021-01-01", "2021-02-26")
    dates = backtest.rebalance_dates(cal, "2021-01-01", "2021-02-26")
    p = pd.DataFrame({"date": cal, "symbol": "GONE", "adj_close": 100.0})
    p.loc[p["date"] == "2021-02-10", "adj_close"] = 80.0
    p = p[p["date"] <= "2021-02-10"]  # acquired / delisted on Feb 10
    r = backtest.period_returns(p, dates)
    assert r["ret"].iloc[0] == pytest.approx(-0.20)


# ---------------------------------------------------------------- turnover and costs


def test_turnover_and_costs_on_hand_example():
    # 10 stocks, quintiles of 2. Month 1 builds the book from cash; month 2 keeps the same
    # ranking (zero returns, so no drift): nothing to trade; month 3 flips the ranking.
    dates = pd.DatetimeIndex(["2021-01-29", "2021-02-26", "2021-03-31"])
    rows = []
    for k, t in enumerate(dates):
        for i in range(10):
            score = i if k < 2 else -i
            rows.append((t, i, f"T{i}", f"T{i}", score, 0.0))
    panel = pd.DataFrame(rows, columns=["date", "cik", "ticker", "symbol", "score", "ret"])
    m = backtest.long_short(panel, n_quantiles=5, cost_bps=10, min_stocks=5)
    assert m["turnover"].tolist() == pytest.approx([0.5, 0.0, 1.0])
    assert m["cost"].tolist() == pytest.approx([0.002, 0.0, 0.004])  # 10bp x traded value
    assert m["n_long"].tolist() == [2, 2, 2]
    assert m["ic"].isna().all()  # all returns are 0: rank correlation is undefined


@pytest.mark.parametrize(
    ("rebalance", "held"),
    [("2012-02-29", "2012-03-31"),  # rebalance on the calendar month-end
     ("2012-03-30", "2012-04-30"),  # last trading day before the 31st (a Friday)
     ("2021-12-31", "2022-01-31")],
)  # fmt: skip
def test_holding_month_label(rebalance, held):
    assert backtest.holding_month_end(pd.Timestamp(rebalance)) == pd.Timestamp(held)


def test_holding_months_are_unique_and_consecutive(market):
    _, monthly, _ = run(market, monthly_signals(market[3], lambda r: r))
    ends = monthly["period_end"]
    assert ends.is_unique and ends.is_monotonic_increasing
    assert (ends.diff().dropna().dt.days.between(28, 31)).all()


def test_share_classes_count_once(market):
    _, membership, tmap, returns = market
    extra = tmap.iloc[[0]].assign(ticker="T000B")  # second class of company 0, same symbol
    extra_member = pd.DataFrame({"snapshot_date": [pd.Timestamp("2019-01-01")],
                                 "ticker": ["T000B"]})  # fmt: skip
    membership2 = pd.concat([membership, extra_member])
    sig = monthly_signals(returns, lambda r: 0.0)
    panel = backtest.build_panel(sig, membership2, pd.concat([tmap, extra]), returns, DATES, 365)
    assert not panel.duplicated(["date", "cik"]).any()


# ---------------------------------------------------------------- factor regression


def french_zip(header: str, rows: list[str]) -> bytes:
    text = "This file was created using the 202608 CRSP database.\n\n" + header + "\n"
    text += "\n".join(rows) + "\n\n Annual Factors: January-December \n" + header + "\n2020, 1, 2\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F.csv", text)
    return buf.getvalue()


def test_parse_french_csv_keeps_monthly_block_in_decimals():
    raw = french_zip(",Mkt-RF,RF", ["202001,   1.50,    0.10", "202002,  -99.99,    0.12"])
    df, vintage = factors.parse_french_csv(raw)
    assert vintage == "202608"
    assert list(df.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    assert df.loc["2020-01-31", "Mkt-RF"] == pytest.approx(0.015)
    assert pd.isna(df.loc["2020-02-29", "Mkt-RF"])  # -99.99 is French's missing marker


def test_parse_french_csv_single_column_file():
    df, _ = factors.parse_french_csv(french_zip(",Mom", ["202001,   0.57", "202002,  -1.51"]))
    assert df["Mom"].tolist() == pytest.approx([0.0057, -0.0151])


def test_factor_regression_recovers_alpha_and_beta():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2012-01-31", periods=240, freq="ME")
    f = pd.DataFrame(rng.normal(0, 0.04, size=(240, 6)), index=idx, columns=factors.FACTOR_COLUMNS)
    r = 0.01 + 0.5 * f["Mkt-RF"] - 0.3 * f["Mom"] + rng.normal(0, 0.005, 240)
    out = factors.factor_regression(pd.Series(r, index=idx), f)
    assert out["alpha_monthly"] == pytest.approx(0.01, abs=0.001)
    assert out["betas"]["Mkt-RF"] == pytest.approx(0.5, abs=0.02)
    assert out["betas"]["Mom"] == pytest.approx(-0.3, abs=0.02)
    assert out["alpha_t"] > 10
    assert out["newey_west_lags"] == factors.newey_west_lags(240) == 4
