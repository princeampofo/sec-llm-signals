"""Monthly cross-sectional backtest: quintile long-short, IC, turnover, costs.

Timeline for each month-end rebalance date t (the month's last trading day):
- Universe: S&P 500 members as of t, one stock per company (share classes collapse).
- Signal: each company's most recent signal whose filing was accepted on a calendar day
  before t (US Eastern), and at most max_age_days old. A filing accepted on t itself
  waits for the next rebalance, the same rule as the forward returns.
- Portfolio: rank stocks by score, long the top quantile and short the bottom one,
  equal-weighted, entered at t's close.
- Return: from t's close to the next rebalance date's close (about 21 trading days).
  A stock that stops trading inside the period is held to its last close, then cash;
  dropping it instead would hide exactly the losers that survivorship bias hides.

Turnover and costs. Each leg's weights sum to 1. Before rebalancing, last month's
weights have drifted with returns; trading back to target costs cost_bps per side on
the traded value: cost_t = cost * (sum|dw_long| + sum|dw_short|). Reported turnover is
the share of each leg replaced, averaged over the two legs (100% = all new names).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from secsignals.edgar import EASTERN
from secsignals.market import TickerLookup, universe_as_of


def rebalance_dates(calendar: pd.DatetimeIndex, start: object, end: object) -> pd.DatetimeIndex:
    """Last trading day of each month in [start, end]."""
    cal = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    last = cal.to_series().groupby(cal.to_period("M")).max()
    return pd.DatetimeIndex(last.to_numpy())


def period_returns(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return of every symbol priced at t, from t's close to the next date's close.

    Rows: (date=t, symbol, ret). Only prices after t and up to the next date are read.
    """
    wide = prices.pivot_table(index="date", columns="symbol", values="adj_close").sort_index()
    rows = []
    for t0, t1 in zip(dates[:-1], dates[1:], strict=True):
        if t0 not in wide.index:
            continue
        p0 = wide.loc[t0]
        window = wide.loc[(wide.index > t0) & (wide.index <= t1)]
        last = window.ffill().iloc[-1] if len(window) else pd.Series(np.nan, index=wide.columns)
        ret = (last / p0 - 1).where(p0.notna())
        # Priced at t but never again in the period: delisted at t; treat as cash.
        ret = ret.where(~(p0.notna() & last.isna()), 0.0)
        held = p0.notna()
        rows.append(pd.DataFrame({"date": t0, "symbol": ret.index[held], "ret": ret[held].values}))
    return pd.concat(rows, ignore_index=True)


def signals_as_of(signals: pd.DataFrame, t: pd.Timestamp, max_age_days: int) -> pd.Series:
    """Latest score per CIK known strictly before t's calendar day, at most max_age_days old."""
    known_day = signals["known_at"].dt.tz_convert(EASTERN).dt.tz_localize(None).dt.normalize()
    ok = (known_day < t.normalize()) & (known_day >= t - pd.Timedelta(days=max_age_days))
    latest = signals[ok].sort_values("known_at").drop_duplicates("cik", keep="last")
    return latest.set_index("cik")["score"]


def build_panel(
    signals: pd.DataFrame,
    membership: pd.DataFrame,
    ticker_map: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    max_age_days: int,
) -> pd.DataFrame:
    """One row per (rebalance date, company): score known at t and the next-period return.

    signals: columns cik, known_at (tz-aware), score (higher = expected to outperform).
    """
    lookup = TickerLookup(ticker_map)
    rets = returns.set_index(["date", "symbol"])["ret"]
    priced = returns.groupby("date")["symbol"].agg(set)
    rows = []
    for t in dates[:-1]:
        avail = priced.get(t, set())
        members = {}
        for ticker in universe_as_of(membership, t):
            m = lookup.resolve(ticker, t)
            if m is None or pd.isna(m["cik"]) or m["yahoo_symbol"] not in avail:
                continue
            cik = int(m["cik"])
            # Share classes: keep the alphabetically first priced ticker per company.
            if cik not in members or ticker < members[cik][0]:
                members[cik] = (ticker, m["yahoo_symbol"])
        scores = signals_as_of(signals, t, max_age_days)
        for cik, (ticker, symbol) in members.items():
            if cik in scores.index:
                rows.append((t, cik, ticker, symbol, scores[cik], rets[(t, symbol)]))
    return pd.DataFrame(rows, columns=["date", "cik", "ticker", "symbol", "score", "ret"])


def information_coefficient(score: pd.Series, ret: pd.Series) -> float:
    """Spearman rank correlation of scores with next-period returns.

    Undefined (NaN) when either side has no variation, e.g. every return is identical;
    such months are left out of the mean IC instead of counting as zero.
    """
    if score.nunique() < 2 or ret.nunique() < 2:
        return float("nan")
    return float(score.corr(ret, method="spearman"))


def holding_month_end(t: pd.Timestamp) -> pd.Timestamp:
    """Calendar month-end of the month held after rebalancing on t (factor-data label).

    t + MonthEnd(1) would be wrong: from Friday 2012-03-30 it lands on 2012-03-31, the
    month the portfolio was formed in, not the month it was held.
    """
    return t + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)


def _drifted(weights: pd.Series, rets: pd.Series) -> pd.Series:
    grown = weights * (1 + rets.reindex(weights.index).fillna(0.0))
    return grown / grown.sum() if grown.sum() != 0 else grown


def long_short(panel: pd.DataFrame, n_quantiles: int, cost_bps: float,
               min_stocks: int) -> pd.DataFrame:  # fmt: skip
    """Monthly quantile long-short results, one row per holding period."""
    cost = cost_bps / 1e4
    prev_long = prev_short = pd.Series(dtype=float)
    prev_rets = pd.Series(dtype=float)
    rows = []
    for t, g in panel.groupby("date", sort=True):
        g = g.dropna(subset=["score", "ret"]).set_index("cik")
        if len(g) < min_stocks:
            continue
        q = pd.qcut(g["score"].rank(method="first"), n_quantiles, labels=False) + 1
        long_ = pd.Series(1.0, index=q.index[q == n_quantiles])
        short = pd.Series(1.0, index=q.index[q == 1])
        long_, short = long_ / len(long_), short / len(short)
        # Trades from drifted old weights to new targets, leg by leg.
        trades = []
        for new, old in ((long_, prev_long), (short, prev_short)):
            old = _drifted(old, prev_rets)
            trades.append(new.sub(old, fill_value=0).abs().sum())
        ret_long = float((long_ * g["ret"]).sum())
        ret_short = float((short * g["ret"]).sum())
        gross = ret_long - ret_short
        row = {
            "rebalance_date": t,
            "period_end": holding_month_end(t),
            "n_stocks": len(g),
            "n_long": len(long_),
            "n_short": len(short),
            "ret_long": ret_long,
            "ret_short": ret_short,
            "ret_ls": gross,
            "turnover": (trades[0] + trades[1]) / 4,
            "cost": cost * (trades[0] + trades[1]),
            "ic": information_coefficient(g["score"], g["ret"]),
        }
        row["ret_ls_net"] = gross - row["cost"]
        for k in range(1, n_quantiles + 1):
            row[f"ret_q{k}"] = float(g.loc[q == k, "ret"].mean())
        rows.append(row)
        prev_long, prev_short, prev_rets = long_, short, g["ret"]
    return pd.DataFrame(rows)


def _t_stat(x: pd.Series) -> float:
    """mean / (sd / sqrt(T)); infinite when every month is identical (sd = 0)."""
    sd = x.std()
    if sd == 0 or np.isnan(sd):
        return float(np.sign(x.mean()) * np.inf) if x.mean() != 0 else float("nan")
    return float(x.mean() / (sd / np.sqrt(len(x))))


def summarize(monthly: pd.DataFrame) -> dict:
    """Headline metrics. Long-short returns are self-financing: Sharpe uses them as is."""
    def ann(series: pd.Series) -> dict:
        mean, std = series.mean(), series.std()
        return {"annual_return": float(mean * 12), "annual_vol": float(std * np.sqrt(12)),
                "sharpe": float(mean / std * np.sqrt(12)) if std > 0 else float("nan")}  # fmt: skip

    ic = monthly["ic"].dropna()
    cum = (1 + monthly["ret_ls_net"]).cumprod()
    q_cols = [c for c in monthly.columns if c.startswith("ret_q")]
    return {
        "months": int(len(monthly)),
        "first_period": str(monthly["period_end"].min().date()),
        "last_period": str(monthly["period_end"].max().date()),
        "avg_stocks": float(monthly["n_stocks"].mean()),
        "mean_ic": float(ic.mean()),
        "ic_t": _t_stat(ic),
        "ic_positive_share": float((ic > 0).mean()),
        "gross": ann(monthly["ret_ls"]),
        "net": ann(monthly["ret_ls_net"]),
        "avg_turnover": float(monthly["turnover"].iloc[1:].mean()),  # month 1 is the initial build
        "max_drawdown_net": float((cum / cum.cummax() - 1).min()),
        "quantile_annual_returns": {c[4:]: float(monthly[c].mean() * 12) for c in q_cols},
    }


def report_markdown(name: str, summary: dict, regressions: dict, settings: dict,
                    vintage: str | None) -> str:  # fmt: skip
    """Backtest report for one signal (results/backtest_<name>.md)."""
    g, n = summary["gross"], summary["net"]
    reg_g, reg_n = regressions["gross"], regressions["net"]
    lines = [
        f"# Backtest: `{name}`", "",
        f"Monthly rebalance, {settings['quantiles']} quantiles, long top / short bottom, "
        f"equal-weighted; {settings['cost_bps_per_side']} bp per side. "
        f"{summary['months']} months, {summary['first_period']} to {summary['last_period']}, "
        f"{summary['avg_stocks']:.0f} stocks per month on average.", "",
        "| Metric | Gross | After costs |", "|---|---|---|",
        f"| Mean monthly IC (t-stat) | {summary['mean_ic']:.4f} ({summary['ic_t']:.2f}) | |",
        f"| Annualized return | {g['annual_return']:.2%} | {n['annual_return']:.2%} |",
        f"| Annualized volatility | {g['annual_vol']:.2%} | {n['annual_vol']:.2%} |",
        f"| Sharpe ratio | {g['sharpe']:.2f} | {n['sharpe']:.2f} |",
        f"| Factor alpha, annualized (t-stat) | {reg_g['alpha_annualized']:.2%} "
        f"({reg_g['alpha_t']:.2f}) | {reg_n['alpha_annualized']:.2%} ({reg_n['alpha_t']:.2f}) |",
        f"| Monthly turnover | {summary['avg_turnover']:.1%} | |",
        f"| Max drawdown | | {summary['max_drawdown_net']:.1%} |",
        f"| Months with positive IC | {summary['ic_positive_share']:.0%} | |", "",
        "## Quantile returns (annualized, equal-weighted)", "",
        "| " + " | ".join(summary["quantile_annual_returns"]) + " |",
        "|" + "---|" * len(summary["quantile_annual_returns"]),
        "| " + " | ".join(f"{v:.2%}" for v in summary["quantile_annual_returns"].values()) + " |",
        "", "## Factor exposures (gross long-short)", "",
        f"Fama-French 5 factors + momentum, Newey-West standard errors with "
        f"{reg_g['newey_west_lags']} lags, {reg_g['months']} months, R² = "
        f"{reg_g['r_squared']:.2f}. Factor data vintage: {vintage or 'unknown'} CRSP.", "",
        "| Factor | Beta | t-stat |", "|---|---|---|",
    ]  # fmt: skip
    lines += [f"| {k} | {v:.3f} | {reg_g['beta_t'][k]:.2f} |" for k, v in reg_g["betas"].items()]
    return "\n".join(lines) + "\n"
