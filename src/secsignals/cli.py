"""Command-line entry points for each pipeline stage (see Makefile)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from secsignals import (
    anonymize,
    backtest,
    dictionary,
    edgar,
    factors,
    llm_scorer,
    market,
    report,
    sections,
    signals,
    stats,
)
from secsignals.config import load_config


def _interim(cfg: dict, name: str) -> Path:
    path = Path(cfg["paths"]["interim"]) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cmd_index(cfg: dict, args: argparse.Namespace) -> None:
    e = cfg["edgar"]
    years = range(e["index_start_year"], e["index_end_year"] + 1)
    df = edgar.build_filing_index(
        edgar.client_from_config(cfg), e["archives_url"], years, e["forms"]
    )
    df.to_parquet(_interim(cfg, "filing_index.parquet"), index=False)
    print(f"filing index: {len(df)} filings")


def cmd_sample(cfg: dict, args: argparse.Namespace) -> None:
    s = cfg["sample"]
    source = {"all": "filing_index.parquet", "universe": "universe_index.parquet"}[s["population"]]
    index = pd.read_parquet(_interim(cfg, source))
    seed = cfg["project"]["seed"]
    df = edgar.sample_filings(index, s["size"], s["start_year"], s["end_year"], seed)
    df.to_parquet(_interim(cfg, "sample_index.parquet"), index=False)
    print(f"sample: {len(df)} filings from {s['population']} 10-Ks")


def cmd_fetch(cfg: dict, args: argparse.Namespace) -> None:
    index = pd.read_parquet(_interim(cfg, f"{args.subset}_index.parquet"))
    df = edgar.fetch_filings(
        edgar.client_from_config(cfg), cfg["edgar"]["archives_url"], index,
        cfg["edgar"]["workers"], with_documents=not args.meta_only,
    )  # fmt: skip
    df.to_parquet(_interim(cfg, f"filings_{args.subset}.parquet"), index=False)
    print(f"fetched: {df['fetch_error'].isna().sum()}/{len(df)} ok")


def cmd_sections(cfg: dict, args: argparse.Namespace) -> None:
    s = cfg["sections"]
    filings = pd.read_parquet(_interim(cfg, f"filings_{args.subset}.parquet"))
    df = sections.extract_sections(filings, s["min_words"], s["max_heading_chars"], s["workers"])
    df.to_parquet(_interim(cfg, f"sections_{args.subset}.parquet"), index=False)
    failures = df[df["status"] == "failed"].merge(
        filings[["accession", "company", "primary_doc_url"]], on="accession"
    )
    failures[["accession", "cik", "company", "failure_reason", "primary_doc_url"]].to_csv(
        _interim(cfg, f"section_failures_{args.subset}.csv"), index=False
    )
    print(f"sections: {(df['status'] == 'ok').sum()}/{len(df)} ok")


def cmd_check(cfg: dict, args: argparse.Namespace) -> None:
    df = pd.read_parquet(_interim(cfg, f"sections_{args.subset}.parquet"))
    ok = df["status"] == "ok"
    population = cfg["sample"]["population"] if args.subset == "sample" else args.subset
    report = {
        "subset": args.subset,
        "population": population,
        "filings": int(len(df)),
        "succeeded": int(ok.sum()),
        "success_rate": round(float(ok.mean()), 4),
        "min_success_rate": cfg["sample"]["min_success_rate"],
        "median_words": int(df.loc[ok, "n_words"].median()),
        "methods": df.loc[ok, "method"].value_counts().to_dict(),
        "failure_reasons": df.loc[~ok, "failure_reason"]
        .str.split(":").str[0].value_counts().to_dict(),
    }
    out = Path(cfg["paths"]["results"]) / f"extraction_check_{population}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["success_rate"] < report["min_success_rate"]:
        sys.exit(f"FAIL: success rate {report['success_rate']:.1%} below target")
    print("PASS")


def _results(cfg: dict, name: str) -> Path:
    path = Path(cfg["paths"]["results"]) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cmd_universe(cfg: dict, args: argparse.Namespace) -> None:
    u = cfg["universe"]
    client = edgar.client_from_config(cfg)
    membership = market.parse_membership(client.get(u["membership_url"]))
    membership = market.membership_in_period(membership, u["start"], u["end"])
    wiki_current, wiki_names = market.parse_wikipedia(client.get(u["wikipedia_url"]))
    sec_tickers = market.parse_sec_tickers(client.get(u["sec_tickers_url"]))
    filing_index = pd.read_parquet(_interim(cfg, "filing_index.parquet"))
    overrides = pd.read_csv(u["overrides"], dtype={"ticker": str, "yahoo_symbol": str,
                                                   "note": str})  # fmt: skip

    windows = market.ticker_windows(membership, u["end"])
    tmap = market.build_ticker_map(windows, sec_tickers, wiki_current, wiki_names,
                                   filing_index, overrides, u["match_window_days"])  # fmt: skip
    membership.to_parquet(_interim(cfg, "membership.parquet"), index=False)
    tmap.to_parquet(_interim(cfg, "ticker_map.parquet"), index=False)
    unmatched = tmap[tmap["cik"].isna()]
    unmatched.to_csv(_results(cfg, "unmatched_tickers.csv"), index=False)
    review = market.review_ticker_map(tmap, filing_index, u["match_window_days"])
    review.to_csv(_results(cfg, "ticker_map_review.csv"), index=False)

    in_period = pd.to_datetime(filing_index["date_filed"]) >= pd.Timestamp(u["start"])
    filings = market.universe_filings(filing_index[in_period], membership, tmap)
    filings.to_parquet(_interim(cfg, "universe_index.parquet"), index=False)
    print(f"universe: {tmap['ticker'].nunique()} tickers, {len(unmatched)} without a CIK "
          f"(see results/unmatched_tickers.csv); {len(filings)} universe 10-Ks")  # fmt: skip
    print(tmap["match_source"].value_counts(dropna=False).to_string())
    print(f"{len(review)} ticker-map rows flagged for review (results/ticker_map_review.csv)")


def cmd_prices(cfg: dict, args: argparse.Namespace) -> None:
    m = cfg["market"]
    tmap = pd.read_parquet(_interim(cfg, "ticker_map.parquet"))
    symbols = set(tmap["yahoo_symbol"].dropna()) | {m["calendar_symbol"]}
    source = market.price_source_from_config(cfg)
    prices = source.adjusted_close(symbols, m["price_start"], m["price_end"])
    prices, rejected = market.vet_price_symbols(tmap, prices)
    rejected.to_csv(_results(cfg, "rejected_price_symbols.csv"), index=False)
    prices.to_parquet(_interim(cfg, "prices.parquet"), index=False)
    print(f"prices: {prices['symbol'].nunique()}/{len(symbols)} symbols with data; "
          f"{len(rejected)} reused-symbol series rejected (results/rejected_price_symbols.csv)")


def cmd_returns(cfg: dict, args: argparse.Namespace) -> None:
    m = cfg["market"]
    filings = pd.read_parquet(_interim(cfg, "filings_universe.parquet"))
    prices = pd.read_parquet(_interim(cfg, "prices.parquet"))
    events = filings.dropna(subset=["acceptance_datetime", "symbol"])
    events = events[["accession", "cik", "symbol"]].assign(
        known_at=events["acceptance_datetime"]
    )
    out = market.forward_returns(prices, events, m["forward_horizon_days"],
                                 m["max_entry_lag_days"], m["market_close"])  # fmt: skip
    out.to_parquet(_interim(cfg, "filing_returns.parquet"), index=False)
    print(f"forward returns: {out['fwd_return'].notna().sum()}/{len(out)} filings")


def cmd_coverage(cfg: dict, args: argparse.Namespace) -> None:
    u, m = cfg["universe"], cfg["market"]
    membership = pd.read_parquet(_interim(cfg, "membership.parquet"))
    tmap = pd.read_parquet(_interim(cfg, "ticker_map.parquet"))
    prices = pd.read_parquet(_interim(cfg, "prices.parquet"))
    calendar = pd.DatetimeIndex(
        prices.loc[prices["symbol"] == m["calendar_symbol"], "date"]
    ).sort_values()
    cov = market.coverage(membership, tmap, prices, calendar,
                          market.month_ends(u["start"], u["end"]))  # fmt: skip
    cov.to_parquet(_interim(cfg, "coverage.parquet"), index=False)
    periods = {"full": (u["start"], u["end"]), "llm": (u["llm_start"], u["end"])}
    summary = market.coverage_summary(cov, periods)
    missing = market.missing_companies(cov)
    missing.to_csv(_results(cfg, "missing_companies.csv"), index=False)
    _results(cfg, "coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _results(cfg, "coverage_report.md").write_text(market.coverage_report_md(
        summary, market.coverage_by_year(cov), missing, m["coverage_warn_below"]))
    print(json.dumps(summary, indent=2))
    for label, s in summary.items():
        if s["coverage"] < m["coverage_warn_below"]:
            print(f"WARNING: {label} coverage {s['coverage']:.1%} < "
                  f"{m['coverage_warn_below']:.0%}: results are likely optimistic")  # fmt: skip
    print(f"{len(missing)} missing companies listed in results/missing_companies.csv")


def _processed(cfg: dict, name: str) -> Path:
    path = Path(cfg["paths"]["processed"]) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cmd_documents(cfg: dict, args: argparse.Namespace) -> None:
    """Universe 10-Ks plus each universe company's 10-Ks from up to max_gap_days before
    it joined: a company's first universe filing still needs last year's to compare."""
    index = pd.read_parquet(_interim(cfg, "filing_index.parquet"))
    universe = pd.read_parquet(_interim(cfg, "universe_index.parquet"))
    index["filed"] = pd.to_datetime(index["date_filed"])
    span = (universe.assign(filed=pd.to_datetime(universe["date_filed"]))
            .groupby("cik")["filed"].agg(["min", "max"]))  # fmt: skip
    pad = pd.Timedelta(days=cfg["signals"]["max_gap_days"])
    cand = index.join(span, on="cik", how="inner")
    history = cand[(cand["filed"] >= cand["min"] - pad) & (cand["filed"] <= cand["max"])]
    docs = pd.concat([universe[index.columns.drop("filed")], history[index.columns.drop("filed")]])
    docs = docs.drop_duplicates(["accession", "cik"]).reset_index(drop=True)
    docs.to_parquet(_interim(cfg, "documents_index.parquet"), index=False)
    print(f"documents: {len(docs)} 10-Ks ({len(docs) - len(universe)} from before index entry)")


def _lm_master(cfg: dict) -> tuple[pd.DataFrame, str]:
    d = cfg["dictionary"]
    raw = edgar.client_from_config(cfg).get(d["url"])
    digest = hashlib.sha256(raw).hexdigest()
    if d["sha256"] and digest != d["sha256"]:
        sys.exit(f"dictionary checksum {digest} != pinned {d['sha256']}: the file changed")
    return dictionary.parse_master_dictionary(raw), digest


def cmd_dictionary(cfg: dict, args: argparse.Namespace) -> None:
    d = cfg["dictionary"]
    master, digest = _lm_master(cfg)
    lexicon = dictionary.build_lexicon(master, d["categories"])
    secs = pd.read_parquet(_interim(cfg, "sections_documents.parquet"))
    counts = dictionary.score_sections(secs, lexicon)
    counts.to_parquet(_processed(cfg, "lm_counts.parquet"), index=False)
    print(f"dictionary sha256 {digest}; scored {len(counts)} sections, "
          f"cohorts: {', '.join(lexicon.columns)}")  # fmt: skip


def cmd_signals(cfg: dict, args: argparse.Namespace) -> None:
    s, d = cfg["signals"], cfg["dictionary"]
    filings = pd.read_parquet(_interim(cfg, "filings_documents.parquet"))
    secs = pd.read_parquet(_interim(cfg, "sections_documents.parquet"), columns=["accession",
                                                                                "status"])
    counts = pd.read_parquet(_processed(cfg, "lm_counts.parquet"))
    universe = pd.read_parquet(_interim(cfg, "universe_index.parquet"))

    pairs = signals.previous_filings(filings, s["min_gap_days"], s["max_gap_days"])
    sig = signals.lm_change_signals(pairs, counts, d["categories"], d["point_in_time"],
                                    s["max_length_ratio"])  # fmt: skip
    sig = sig.dropna(subset=["lm_change"])
    sig.to_parquet(_processed(cfg, "signals_lm.parquet"), index=False)

    report = signals.signal_coverage(sig, universe, "lm_change")
    ok = set(secs.loc[secs["status"] == "ok", "accession"])
    report["missing_reasons"] = signals.missing_reasons(universe, pairs, ok, sig, "lm_change")
    report["min_coverage"] = s["min_coverage"]
    _results(cfg, "signal_coverage_lm.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["coverage"] < s["min_coverage"]:
        sys.exit(f"FAIL: signal coverage {report['coverage']:.1%} below {s['min_coverage']:.0%}")
    print("PASS")


def _load_factors(cfg: dict) -> tuple[pd.DataFrame, str | None]:
    client = edgar.client_from_config(cfg)
    f = cfg["factors"]
    return factors.load_factors(client.get(f["five_factor_url"]), client.get(f["momentum_url"]))


def _backtest_inputs(cfg: dict, every_months: int = 1) -> tuple:
    """(period returns, membership, ticker map, rebalance dates); every_months > 1 for
    multi-month holding periods."""
    b, m = cfg["backtest"], cfg["market"]
    prices = pd.read_parquet(_interim(cfg, "prices.parquet"))
    calendar = pd.DatetimeIndex(
        prices.loc[prices["symbol"] == m["calendar_symbol"], "date"]
    ).sort_values()
    # One extra holding period so the last rebalance has returns.
    end = pd.Timestamp(b["end"]) + pd.offsets.MonthEnd(every_months)
    dates = backtest.rebalance_dates(calendar, b["start"], end, every_months)
    returns = backtest.period_returns(prices, dates)
    membership = pd.read_parquet(_interim(cfg, "membership.parquet"))
    tmap = pd.read_parquet(_interim(cfg, "ticker_map.parquet"))
    return returns, membership, tmap, dates


def _panel(cfg: dict, signals_df: pd.DataFrame, inputs: tuple) -> pd.DataFrame:
    returns, membership, tmap, dates = inputs
    return backtest.build_panel(signals_df, membership, tmap, returns, dates,
                                cfg["backtest"]["signal_max_age_days"])  # fmt: skip


def _run(cfg: dict, panel: pd.DataFrame) -> pd.DataFrame:
    b = cfg["backtest"]
    return backtest.long_short(panel, b["quantiles"], b["cost_bps_per_side"], b["min_stocks"])


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _log_trial(cfg: dict, name: str, settings: dict, summary: dict, regressions: dict) -> None:
    """Append one row to results/trials.csv. config_hash identifies the variant, so the
    deflated Sharpe ratio can count distinct trials rather than reruns."""
    key = json.dumps({"signal": name, **settings}, sort_keys=True, default=str)
    row = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "signal": name,
        "config_hash": hashlib.sha256(key.encode()).hexdigest()[:12],
        "git_commit": _git_commit(),
        "settings": key,
        "months": summary["months"],
        "mean_ic": summary["mean_ic"],
        "ic_t": summary["ic_t"],
        "annual_return_gross": summary["gross"]["annual_return"],
        "sharpe_gross": summary["gross"]["sharpe"],
        "annual_return_net": summary["net"]["annual_return"],
        "sharpe_net": summary["net"]["sharpe"],
        "alpha_annualized": regressions["gross"]["alpha_annualized"],
        "alpha_t": regressions["gross"]["alpha_t"],
        "avg_turnover": summary["avg_turnover"],
    }
    path = _results(cfg, "trials.csv")
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def cmd_factors(cfg: dict, args: argparse.Namespace) -> None:
    f, vintage = _load_factors(cfg)
    print(f"factors: {f.index.min():%Y-%m} to {f.index.max():%Y-%m}, CRSP vintage {vintage}")


def cmd_backtest(cfg: dict, args: argparse.Namespace) -> None:
    b = cfg["backtest"]
    inputs = _backtest_inputs(cfg)
    factor_data, vintage = _load_factors(cfg)
    names = [args.signal] if args.signal else list(b["signals"])
    for name in names:
        spec = b["signals"][name]
        sig = pd.read_parquet(_processed(cfg, spec["file"])).dropna(subset=[spec["column"]])
        sig = sig[["cik", "known_at"]].assign(score=spec["sign"] * sig[spec["column"]])
        panel = _panel(cfg, sig, inputs)
        panel.to_parquet(_processed(cfg, f"panel_{name}.parquet"), index=False)
        monthly = _run(cfg, panel)
        summary = backtest.summarize(monthly)
        returns = monthly.set_index("period_end")
        regressions = {k: factors.factor_regression(returns[col], factor_data, b["newey_west_lags"])
                       for k, col in (("gross", "ret_ls"), ("net", "ret_ls_net"))}  # fmt: skip
        keys = ("start", "end", "quantiles", "signal_max_age_days", "cost_bps_per_side",
                "min_stocks")  # fmt: skip
        settings = {k: b[k] for k in keys} | {"sign": spec["sign"]}
        out = {"signal": name, "settings": settings, "summary": summary,
               "factor_regression": regressions, "factor_vintage": vintage}  # fmt: skip
        _results(cfg, f"backtest_{name}.json").write_text(json.dumps(out, indent=2, default=str))
        monthly.to_csv(_results(cfg, f"backtest_{name}_monthly.csv"), index=False)
        _results(cfg, f"backtest_{name}.md").write_text(
            backtest.report_markdown(name, summary, regressions, settings, vintage))
        _log_trial(cfg, name, settings, summary, regressions)
        g, n, r = summary["gross"], summary["net"], regressions["gross"]
        print(f"{name}: IC {summary['mean_ic']:.4f} (t {summary['ic_t']:.2f}), "
              f"Sharpe {g['sharpe']:.2f} gross / {n['sharpe']:.2f} net, "
              f"alpha {r['alpha_annualized']:.2%} (t {r['alpha_t']:.2f}); "
              f"report results/backtest_{name}.md")  # fmt: skip


def cmd_sanity(cfg: dict, args: argparse.Namespace) -> None:
    """Engine checks on the real universe: a lookahead signal (next period's return) must
    look spectacular, and random signals must average zero IC."""
    b = cfg["backtest"]
    inputs = _backtest_inputs(cfg)
    returns, _, _, _ = inputs
    known = returns[["date"]].drop_duplicates()
    # Every company gets a dummy score known the day before each rebalance; the planted and
    # noise scores then replace it inside the panel.
    tmap = pd.read_parquet(_interim(cfg, "ticker_map.parquet"))
    shell = tmap["cik"].dropna().astype(int).unique()
    sig = pd.DataFrame([(c, (d - pd.Timedelta(days=1)).tz_localize(edgar.EASTERN), 0.0)
                        for d in known["date"] for c in shell],
                       columns=["cik", "known_at", "score"])  # fmt: skip
    panel = _panel(cfg, sig, inputs)
    planted = backtest.summarize(_run(cfg, panel.assign(score=panel["ret"])))
    noise = []
    for seed in range(b["noise_seeds"]):
        rng = np.random.default_rng(cfg["project"]["seed"] + seed)
        noise.append(backtest.summarize(_run(cfg, panel.assign(score=rng.normal(size=len(panel))))))
    ics = np.array([s["mean_ic"] for s in noise])
    ts = np.array([s["ic_t"] for s in noise])
    report = {
        "planted_future_return": {"mean_ic": planted["mean_ic"],
                                  "sharpe_gross": planted["gross"]["sharpe"],
                                  "annual_return_gross": planted["gross"]["annual_return"]},
        "random_noise": {"draws": len(noise), "mean_ic_avg": float(ics.mean()),
                         "mean_ic_max_abs": float(np.abs(ics).max()),
                         "ic_t_avg": float(ts.mean()), "ic_t_sd": float(ts.std()),
                         "share_abs_t_above_2": float((np.abs(ts) > 2).mean())},
    }  # fmt: skip
    passed = planted["mean_ic"] > 0.5 and planted["gross"]["sharpe"] > 3 and abs(ics.mean()) < 0.005
    report["passed"] = bool(passed)
    _results(cfg, "backtest_sanity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        sys.exit("FAIL: backtest engine sanity checks")
    print("PASS")


def _llm_model_id(cfg: dict) -> str:
    c = cfg["llm"]
    name = c["model"] if c["backend"] == "mlx" else c["served_model_name"]
    return f"{name}@{c['model_revision'][:7]}"


def _llm_scope(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sections to score, filing pairs). Scope: universe 10-Ks accepted in the LLM period,
    plus each one's previous 10-K, which the change signal needs."""
    c, s = cfg["llm"], cfg["signals"]
    filings = pd.read_parquet(_interim(cfg, "filings_documents.parquet"))
    universe = pd.read_parquet(_interim(cfg, "universe_index.parquet"))
    pairs = signals.previous_filings(filings, s["min_gap_days"], s["max_gap_days"])
    acc = pairs["acceptance_datetime"].dt.tz_convert(edgar.EASTERN).dt.tz_localize(None)
    end = pd.Timestamp(c["end"]) + pd.Timedelta(days=1)
    in_period = (acc >= pd.Timestamp(c["start"])) & (acc < end)
    current = pairs[in_period & pairs["accession"].isin(universe["accession"])]
    scope = set(current["accession"]) | set(current["prev_accession"].dropna())
    secs = pd.read_parquet(_interim(cfg, "sections_documents.parquet"),
                           columns=["accession", "status", "text"])  # fmt: skip
    secs = secs[secs["accession"].isin(scope) & (secs["status"] == "ok")]
    secs = secs.drop_duplicates("accession").sort_values("accession").reset_index(drop=True)
    return secs, current


def _llm_cache(cfg: dict) -> llm_scorer.ScoreCache:
    return llm_scorer.ScoreCache(_processed(cfg, "llm_scores.sqlite"))


def cmd_llm_score(cfg: dict, args: argparse.Namespace) -> None:
    c = cfg["llm"]
    secs, _ = _llm_scope(cfg)
    if args.justify:
        n = min(c["justification_sample"], len(secs))
        secs = secs.sample(n=n, random_state=cfg["project"]["seed"]).sort_values("accession")
    backend = llm_scorer.backend_from_config(cfg)
    stats = llm_scorer.score_sections(
        secs, backend, _llm_cache(cfg), c["prompt_version"], c["excerpt_tokens"],
        c["min_forward_tokens"], c["max_new_tokens_justify" if args.justify else "max_new_tokens"],
        c["max_retries"], justify=args.justify, limit=args.limit,
    )  # fmt: skip
    per = stats.seconds / max(stats.valid + stats.invalid, 1)
    left = stats.to_score - stats.cached - stats.valid - stats.invalid
    print(f"{backend.model_id}: {stats.to_score} filings in scope, {stats.cached} already cached, "
          f"{stats.valid + stats.invalid} scored now ({stats.invalid} invalid), "
          f"new model calls: {stats.model_calls}, {per:.1f} s/filing"
          + (f"; {left} left (~{left * per / 3600:.1f} h)" if left else ""))  # fmt: skip


def cmd_llm_signals(cfg: dict, args: argparse.Namespace) -> None:
    c = cfg["llm"]
    secs, current = _llm_scope(cfg)
    cached = _llm_cache(cfg).frame(c["prompt_version"], _llm_model_id(cfg))
    cached = cached[cached["accession"].isin(secs["accession"])]
    valid = cached[cached["status"] == "valid"]
    filings = pd.read_parquet(_interim(cfg, "filings_documents.parquet"))
    s = cfg["signals"]
    pairs = signals.previous_filings(filings, s["min_gap_days"], s["max_gap_days"])
    sig = signals.llm_change_signals(pairs[pairs["accession"].isin(current["accession"])], valid)
    sig = sig.dropna(subset=["llm_change"])
    sig.to_parquet(_processed(cfg, "signals_llm.parquet"), index=False)
    invalid = cached[cached["status"] == "invalid"]
    invalid[["accession", "attempts", "error", "raw_output"]].to_csv(
        _results(cfg, "llm_invalid_outputs.csv"), index=False)
    universe = pd.read_parquet(_interim(cfg, "universe_index.parquet"))
    period_universe = universe[universe["accession"].isin(current["accession"])]
    report = {
        "model": _llm_model_id(cfg), "prompt_version": c["prompt_version"],
        "training_cutoff": str(c["training_cutoff"]),
        "filings_in_scope": int(len(secs)), "scored": int(len(cached)),
        "unscored": int(len(secs) - len(cached)), "valid": int(len(valid)),
        "invalid": int(len(invalid)),
        "valid_share": round(len(valid) / max(len(secs), 1), 4),
        "needed_retry": int((valid["attempts"].astype(int) > 1).sum()),
        "excerpt_methods": valid["excerpt_method"].value_counts().to_dict(),
        "median_seconds_per_filing": float(cached["seconds"].median()) if len(cached) else None,
        "signal_coverage": signals.signal_coverage(sig, period_universe, "llm_change"),
        "min_valid_share": c["min_valid_share"],
    }  # fmt: skip
    # Hand-check sheet: the justified sample, with a link to read each filing.
    justified = _llm_cache(cfg).frame(c["prompt_version"] + "-justify", _llm_model_id(cfg))
    docs = filings.drop_duplicates("accession")[["accession", "company", "primary_doc_url"]]
    justified.merge(docs, on="accession", how="left")[
        ["accession", "company", "status", "tone", "hedging", "risk_severity", "justification",
         "excerpt_method", "primary_doc_url"]
    ].to_csv(_results(cfg, "llm_hand_check.csv"), index=False)  # fmt: skip
    _results(cfg, "llm_scoring_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["valid_share"] < c["min_valid_share"]:
        sys.exit(f"FAIL: valid LLM scores for {report['valid_share']:.1%} of filings in scope")
    print("PASS")


# ---------------------------------------------------------------- robustness


def cmd_size(cfg: dict, args: argparse.Namespace) -> None:
    """Public float of every universe company, from its 10-K cover pages (value weights)."""
    tmap = pd.read_parquet(_interim(cfg, "ticker_map.parquet")).dropna(subset=["cik"])
    ciks = tmap["cik"].astype(int).unique()
    floats = edgar.fetch_public_float(edgar.client_from_config(cfg), ciks)
    floats.to_parquet(_interim(cfg, "public_float.parquet"), index=False)
    print(f"public float: {floats['cik'].nunique()}/{len(ciks)} companies, {len(floats)} values")


def _evaluate(cfg: dict, monthly: pd.DataFrame, factor_data: pd.DataFrame,
              months_held: int) -> tuple[dict, dict]:  # fmt: skip
    """(summary, factor regressions) for one backtest's holding-period returns."""
    b = cfg["backtest"]
    summary = backtest.summarize(monthly, 12 // months_held)
    f = factor_data if months_held == 1 else factors.to_holding_periods(
        factor_data, pd.DatetimeIndex(monthly["period_end"]), months_held)  # fmt: skip
    returns = monthly.set_index("period_end")
    regs = {k: factors.factor_regression(returns[col], f, b["newey_west_lags"], 12 // months_held)
            for k, col in (("gross", "ret_ls"), ("net", "ret_ls_net"))}  # fmt: skip
    return summary, regs


def _robustness_row(name: str, variant: str, summary: dict, regs: dict) -> dict:
    g, n = summary["gross"], summary["net"]
    return {
        "signal": name, "variant": variant, "periods": summary["months"],
        "first_period": summary["first_period"], "last_period": summary["last_period"],
        "mean_ic": summary["mean_ic"], "ic_t": summary["ic_t"],
        "annual_return_gross": g["annual_return"], "sharpe_gross": g["sharpe"],
        "annual_return_net": n["annual_return"], "sharpe_net": n["sharpe"],
        "alpha_annualized": regs["gross"]["alpha_annualized"], "alpha_t": regs["gross"]["alpha_t"],
        "avg_turnover": summary["avg_turnover"], "avg_stocks": summary["avg_stocks"],
    }  # fmt: skip


def cmd_robustness(cfg: dict, args: argparse.Namespace) -> None:
    """Baseline variations (horizon, costs, weighting, subperiods), each logged as a trial,
    then the deflated Sharpe ratio of each baseline given every trial on record."""
    b, r = cfg["backtest"], cfg["robustness"]
    keys = ("start", "end", "quantiles", "signal_max_age_days", "cost_bps_per_side", "min_stocks")
    factor_data, _ = _load_factors(cfg)
    prices = pd.read_parquet(_interim(cfg, "prices.parquet"))
    floats = pd.read_parquet(_interim(cfg, "public_float.parquet"))
    monthly_inputs = _backtest_inputs(cfg)
    long_inputs = _backtest_inputs(cfg, r["horizon_months"])
    rows, baselines = [], {}
    for name in r["signals"]:
        spec = b["signals"][name]
        sig = pd.read_parquet(_processed(cfg, spec["file"])).dropna(subset=[spec["column"]])
        sig = sig[["cik", "known_at"]].assign(score=spec["sign"] * sig[spec["column"]])
        panel = backtest.add_size(_panel(cfg, sig, monthly_inputs), floats, prices,
                                  r["size_max_age_days"])  # fmt: skip
        base = {k: b[k] for k in keys} | {"sign": spec["sign"]}
        q, cost, min_n = b["quantiles"], b["cost_bps_per_side"], b["min_stocks"]
        k = r["horizon_months"]
        monthly = backtest.long_short(panel, q, cost, min_n)
        baselines[name] = monthly
        variants = [("baseline", monthly, 1, {})]
        variants += [(f"costs_{c}bp", backtest.long_short(panel, q, c, min_n), 1,
                      {"cost_bps_per_side": c}) for c in r["costs_bps"]]  # fmt: skip
        variants.append(("value_weighted", backtest.long_short(panel, q, cost, min_n, "value"),
                         1, {"weighting": "value"}))  # fmt: skip
        held = backtest.long_short(_panel(cfg, sig, long_inputs), q, cost, min_n, months_held=k)
        variants.append((f"horizon_{k}m", held, k, {"months_held": k}))
        for start, end in r["subperiods"]:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            sub = monthly[(monthly["rebalance_date"] >= s) & (monthly["rebalance_date"] <= e)]
            if len(sub) >= 12:
                variants.append((f"{s.year}_{e.year}", sub.reset_index(drop=True), 1,
                                 {"start": start, "end": end}))  # fmt: skip
        for variant, m, months_held, changes in variants:
            summary, regs = _evaluate(cfg, m, factor_data, months_held)
            if variant != "baseline":  # the baseline is already logged by the backtest stage
                _log_trial(cfg, name, base | changes, summary, regs)
            rows.append(_robustness_row(name, variant, summary, regs))
    table = pd.DataFrame(rows)
    table.to_csv(_results(cfg, "robustness.csv"), index=False)
    # Deflated Sharpe ratio: every distinct trial on record counts.
    trials = pd.read_csv(_results(cfg, "trials.csv"))
    sharpes = stats.trial_sharpes(trials)
    dsr = {name: stats.deflated_sharpe(m["ret_ls_net"], len(sharpes), float(sharpes.var()))
           for name, m in baselines.items()}  # fmt: skip
    out = {"variants": rows, "deflated_sharpe": dsr, "distinct_trials": int(len(sharpes))}
    _results(cfg, "robustness.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    show = table[["signal", "variant", "periods", "mean_ic", "ic_t", "sharpe_net", "alpha_t"]]
    print(show.round(3).to_string(index=False))
    for name, d in dsr.items():
        print(f"{name}: deflated Sharpe {d['dsr']:.3f} ({d['n_trials']} trials, "
              f"SR0 {d['expected_max_sharpe']:.3f}/month vs SR {d['sharpe_per_period']:.3f})")


def cmd_report(cfg: dict, args: argparse.Namespace) -> None:
    path = Path(cfg["paths"]["report"])
    report.build_report(Path(cfg["paths"]["results"]), path)
    print(f"report: {path} (charts in {Path(cfg['paths']['results']) / 'figures'})")


# ---------------------------------------------------------------- memorization experiment


def _memorization_pairs(cfg: dict) -> pd.DataFrame:
    """Filing pairs in the experiment: every LLM-signal 10-K accepted after the model's
    training cutoff, plus a seeded random sample of the earlier ones."""
    c, mm = cfg["llm"], cfg["memorization"]
    sig = pd.read_parquet(_processed(cfg, "signals_llm.parquet"))
    day = sig["known_at"].dt.tz_convert(edgar.EASTERN).dt.tz_localize(None).dt.normalize()
    after = day > pd.Timestamp(c["training_cutoff"])
    pre = sig[~after].sample(n=min(mm["pre_cutoff_sample"], int((~after).sum())),
                             random_state=cfg["project"]["seed"])  # fmt: skip
    pairs = pd.concat([pre.assign(period="pre_cutoff"), sig[after].assign(period="post_cutoff")])
    cols = ["accession", "cik", "acceptance_datetime", "prev_accession", "prev_acceptance",
            "known_at", "period"]  # fmt: skip
    return pairs[cols].sort_values("accession").reset_index(drop=True)


def _identities(cfg: dict, accessions: set[str]) -> dict[str, anonymize.Identity]:
    """Names (every EDGAR name the company filed under) and tickers per filing; a combined
    filing gets the names of all its co-registrants."""
    filings = pd.read_parquet(_interim(cfg, "filings_documents.parquet"),
                              columns=["accession", "cik"])  # fmt: skip
    index = pd.read_parquet(_interim(cfg, "filing_index.parquet"), columns=["cik", "company"])
    names = index.groupby("cik")["company"].agg(set)
    tmap = pd.read_parquet(_interim(cfg, "ticker_map.parquet")).dropna(subset=["cik"])
    tmap["cik"] = tmap["cik"].astype(int)
    tickers = tmap.groupby("cik").apply(
        lambda g: (set(g["ticker"]) | set(g["yahoo_symbol"].dropna())) - {"-"},
        include_groups=False)  # fmt: skip
    out = {}
    for acc, g in filings[filings["accession"].isin(accessions)].groupby("accession"):
        ciks = [int(x) for x in g["cik"]]
        n = set().union(*(names.get(k, set()) for k in ciks))
        t = set().union(*(tickers.get(k, set()) for k in ciks))
        out[acc] = anonymize.Identity(tuple(sorted(n)), tuple(sorted(t)))
    return out


def cmd_anonymize(cfg: dict, args: argparse.Namespace) -> None:
    """Same excerpt as the original scoring, then masked; plus leak checks and a sheet of
    side-by-side examples for the manual spot check."""
    c, mm = cfg["llm"], cfg["memorization"]
    pairs = _memorization_pairs(cfg)
    needed = set(pairs["accession"]) | set(pairs["prev_accession"])
    secs = pd.read_parquet(_interim(cfg, "sections_documents.parquet"),
                           columns=["accession", "status", "text"])  # fmt: skip
    secs = secs[secs["accession"].isin(needed) & (secs["status"] == "ok")]
    secs = secs.drop_duplicates("accession").sort_values("accession")
    identities = _identities(cfg, needed)
    count_tokens = llm_scorer.token_counter_from_config(cfg)
    entities = anonymize.spacy_entities(mm["ner_model"])
    common_words = frozenset(_lm_master(cfg)[0]["Word"])
    rows = []
    for acc, text in zip(secs["accession"], secs["text"], strict=True):
        ident = identities[acc]
        excerpt, method, tokens = llm_scorer.select_excerpt(
            text, c["excerpt_tokens"], c["min_forward_tokens"], count_tokens)  # fmt: skip
        masked, counts = anonymize.anonymize(excerpt, ident, entities, common_words)
        masks = {f"masked_{k.strip('[]').lower()}": counts.get(k, 0) for k in anonymize.PRIORITY}
        leaks = {f"leak_{k}": v for k, v in anonymize.leaks(masked, ident).items()}
        rows.append({"accession": acc, "excerpt_method": method, "original_tokens": tokens,
                     "original": excerpt, "excerpt": masked, "company": "; ".join(ident.names),
                     **masks, **leaks})  # fmt: skip
    out = pd.DataFrame(rows)
    out.to_parquet(_processed(cfg, "anonymized_excerpts.parquet"), index=False)
    # The excerpt must be the one the original score read: compare token counts.
    orig = _llm_cache(cfg).frame(c["prompt_version"], _llm_model_id(cfg))
    same = out.merge(orig[["accession", "excerpt_tokens"]], on="accession")
    masked_cols = [col for col in out if col.startswith("masked_")]
    leak_cols = [col for col in out if col.startswith("leak_")]
    report = {
        "excerpts": len(out),
        "pairs": {k: int(v) for k, v in pairs["period"].value_counts().items()},
        "same_excerpt_as_original_score": round(float(
            (same["original_tokens"] == same["excerpt_tokens"].astype(float)).mean()), 4),
        "mean_masks_per_excerpt": out[masked_cols].mean().round(2).to_dict(),
        "share_with_masks": (out[masked_cols] > 0).mean().round(3).to_dict(),
        "share_with_leaks": (out[leak_cols] > 0).mean().round(4).to_dict(),
    }  # fmt: skip
    _results(cfg, "anonymization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    check = out.sample(n=min(mm["spot_check"], len(out)), random_state=cfg["project"]["seed"])
    check[["accession", "company", "original", "excerpt", *masked_cols, *leak_cols]].rename(
        columns={"excerpt": "anonymized"}).sort_values("accession").to_csv(
        _results(cfg, "anonymization_spot_check.csv"), index=False)  # fmt: skip
    print(json.dumps(report, indent=2))


def _anonymized_section(cfg: dict) -> str:
    """Cache section name for anonymized scores; versioned, so changed masking rules
    never reuse scores of differently masked text."""
    return f"mda_anonymized_v{cfg['memorization']['anonymizer_version']}"


def cmd_llm_score_anonymized(cfg: dict, args: argparse.Namespace) -> None:
    c = cfg["llm"]
    excerpts = pd.read_parquet(_processed(cfg, "anonymized_excerpts.parquet"),
                               columns=["accession", "excerpt", "excerpt_method"])  # fmt: skip
    backend = llm_scorer.backend_from_config(cfg)
    stats = llm_scorer.score_sections(
        excerpts, backend, _llm_cache(cfg), c["prompt_version"], c["excerpt_tokens"],
        c["min_forward_tokens"], c["max_new_tokens"], c["max_retries"], limit=args.limit,
        section=_anonymized_section(cfg),
    )  # fmt: skip
    print(f"{backend.model_id} (anonymized): {stats.to_score} excerpts, {stats.cached} already "
          f"cached, {stats.valid + stats.invalid} scored now ({stats.invalid} invalid), "
          f"new model calls: {stats.model_calls}")  # fmt: skip


def cmd_memorization(cfg: dict, args: argparse.Namespace) -> None:
    """2 x 2: original vs. anonymized text, filings before vs. after the training cutoff."""
    c, mm, b = cfg["llm"], cfg["memorization"], cfg["backtest"]
    pairs = _memorization_pairs(cfg)
    cache, model = _llm_cache(cfg), _llm_model_id(cfg)

    def change(section: str) -> pd.DataFrame:
        f = cache.frame(c["prompt_version"], model, section)
        sig = signals.llm_change_signals(pairs, f[f["status"] == "valid"])
        return sig[["accession", "cik", "known_at", "pessimism", "llm_change"]]

    data = change("mda").merge(change(_anonymized_section(cfg)).drop(columns=["cik", "known_at"]),
                               on="accession", suffixes=("_original", "_anonymized"))  # fmt: skip
    data = data.dropna(subset=["llm_change_original", "llm_change_anonymized"])
    rets = pd.read_parquet(_interim(cfg, "filing_returns.parquet"))
    data = data.merge(pairs[["accession", "period"]], on="accession").merge(
        rets[["accession", "fwd_return"]].drop_duplicates("accession"), on="accession", how="left")
    sign = b["signals"]["llm_change"]["sign"]
    inputs = _backtest_inputs(cfg)
    factor_data, vintage = _load_factors(cfg)
    rng = np.random.default_rng(cfg["project"]["seed"])
    cells, gaps, agreement = {}, {}, {}
    for period in ("pre_cutoff", "post_cutoff"):
        d = data[data["period"] == period]
        for version in ("original", "anonymized"):
            score = sign * d[f"llm_change_{version}"]
            panel = _panel(cfg, d[["cik", "known_at"]].assign(score=score), inputs)
            monthly = backtest.long_short(panel, b["quantiles"], b["cost_bps_per_side"],
                                          mm["min_stocks"])  # fmt: skip
            summary = backtest.summarize(monthly)
            reg = factors.factor_regression(monthly.set_index("period_end")["ret_ls"], factor_data,
                                            b["newey_west_lags"])  # fmt: skip
            cells[f"{period}/{version}"] = {"filing_ic": backtest.filing_ic(score, d["fwd_return"]),
                                            "monthly": summary, "alpha": reg}  # fmt: skip
            settings = {k: b[k] for k in ("quantiles", "signal_max_age_days", "cost_bps_per_side")}
            settings |= {"min_stocks": mm["min_stocks"], "sign": sign, "period": period,
                         "text": version, "filings": len(d),
                         "seed": cfg["project"]["seed"]}  # fmt: skip
            _log_trial(cfg, f"llm_change_{period}_{version}", settings, summary,
                       {"gross": reg})  # fmt: skip
        draws = backtest.bootstrap_ic_gap(sign * d["llm_change_original"],
                                          sign * d["llm_change_anonymized"], d["fwd_return"],
                                          mm["bootstrap"], rng)  # fmt: skip
        gaps[period] = draws
        po, pa = d["pessimism_original"], d["pessimism_anonymized"]
        agreement[period] = {
            "filings": len(d),
            "pessimism_spearman": float(po.corr(pa, method="spearman")),
            "change_spearman": float(d["llm_change_original"].corr(d["llm_change_anonymized"],
                                                                   method="spearman")),
            "mean_abs_pessimism_shift": float((po - pa).abs().mean()),
        }  # fmt: skip

    def ci(x: np.ndarray) -> dict:
        lo, hi = np.nanpercentile(x, [2.5, 97.5])
        return {"low": float(lo), "high": float(hi)}

    gap = {p: cells[f"{p}/original"]["filing_ic"]["ic"]
           - cells[f"{p}/anonymized"]["filing_ic"]["ic"]
           for p in ("pre_cutoff", "post_cutoff")}  # fmt: skip
    tests = {f"{p}_gap": {"value": gap[p], "ci95": ci(gaps[p])} for p in gap}
    tests["difference_in_differences"] = {
        "value": gap["pre_cutoff"] - gap["post_cutoff"],
        "ci95": ci(gaps["pre_cutoff"] - gaps["post_cutoff"]),
    }
    out = {"model": model, "training_cutoff": str(c["training_cutoff"]), "cells": cells,
           "ic_gap_tests": tests, "agreement": agreement, "factor_vintage": vintage}  # fmt: skip
    _results(cfg, "memorization.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    _results(cfg, "memorization.md").write_text(_memorization_md(out, mm))
    print(_memorization_md(out, mm))


def _memorization_md(out: dict, mm: dict) -> str:
    cells, tests, agree = out["cells"], out["ic_gap_tests"], out["agreement"]

    def row(label: str, get: Callable[[dict], str]) -> str:  # both periods x both texts
        vals = []
        for p in ("pre_cutoff", "post_cutoff"):
            vals += [get(cells[f"{p}/original"]), get(cells[f"{p}/anonymized"])]
        return f"| {label} | " + " | ".join(vals) + " |"

    def fic(cell: dict) -> str:
        return f"{cell['filing_ic']['ic']:.4f} ({cell['filing_ic']['t']:.2f})"

    def mic(cell: dict) -> str:
        return f"{cell['monthly']['mean_ic']:.4f} ({cell['monthly']['ic_t']:.2f})"

    def alpha(cell: dict) -> str:
        return f"{cell['alpha']['alpha_annualized']:.2%} ({cell['alpha']['alpha_t']:.2f})"

    def ci(t: dict) -> str:
        return f"{t['value']:+.4f} [{t['ci95']['low']:+.4f}, {t['ci95']['high']:+.4f}]"

    lines = [
        "# Memorization test", "",
        f"Model `{out['model']}`, stated training cutoff {out['training_cutoff']}. The LLM change "
        "signal is rebuilt from anonymized excerpts (company names, tickers, people, products, "
        "other organizations, dates and years masked) and compared with the original, for "
        f"filings accepted before the cutoff ({agree['pre_cutoff']['filings']} randomly sampled "
        f"pairs) and after it ({agree['post_cutoff']['filings']} pairs, all of them).", "",
        "If the model used memorized knowledge of what happened next, the original text would "
        "beat the anonymized text before the cutoff but not after it.", "",
        "| | Before cutoff, original | Before cutoff, anonymized | After cutoff, original "
        "| After cutoff, anonymized |", "|---|---|---|---|---|",
        row("Filings with a 21-day return", lambda x: str(x["filing_ic"]["n"])),
        row("Filing-level IC with the 21-day return (t)", fic),
        row("Mean monthly IC, quintile portfolios (t)", mic),
        row("Months", lambda x: str(x["monthly"]["months"])),
        row("Factor alpha, annualized (Newey-West t)", alpha), "",
        "## Original minus anonymized filing-level IC (95% bootstrap interval)", "",
        f"- Before cutoff: {ci(tests['pre_cutoff_gap'])}",
        f"- After cutoff: {ci(tests['post_cutoff_gap'])}",
        f"- Difference in differences: {ci(tests['difference_in_differences'])}", "",
        "## How much anonymization changes the reading", "",
        "| | Before cutoff | After cutoff |", "|---|---|---|",
        *(f"| {label} | {agree['pre_cutoff'][key]:{fmt}} | {agree['post_cutoff'][key]:{fmt}} |"
          for label, key, fmt in (
              ("Spearman, original vs. anonymized pessimism", "pessimism_spearman", ".2f"),
              ("Spearman, original vs. anonymized change", "change_spearman", ".2f"),
              ("Mean absolute pessimism shift", "mean_abs_pessimism_shift", ".3f"))), "",
        f"Monthly portfolios need at least {mm['min_stocks']} scored stocks; the before-cutoff "
        "sample is thin, so its monthly numbers are noisy. With 500 to 1,000 filings per "
        "cell, only a large memorization effect would be detectable.",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="secsignals")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "sample", "universe", "prices", "returns", "coverage", "documents",
                 "dictionary", "signals", "factors", "sanity"):
        sub.add_parser(name)
    sub.add_parser("backtest").add_argument("--signal", help="one signal from config; default all")
    p = sub.add_parser("llm-score")
    p.add_argument("--limit", type=int, help="score at most this many new filings (timing runs)")
    p.add_argument("--justify", action="store_true",
                   help="score the hand-check sample with written justifications")
    sub.add_parser("llm-signals")
    sub.add_parser("anonymize")
    sub.add_parser("llm-score-anonymized").add_argument("--limit", type=int)
    sub.add_parser("memorization")
    sub.add_parser("size")
    sub.add_parser("robustness")
    sub.add_parser("report")
    for name in ("fetch", "sections", "check"):
        p = sub.add_parser(name)
        p.add_argument("--subset", default="sample")
        if name == "fetch":
            p.add_argument("--meta-only", action="store_true",
                           help="index pages only (acceptance times), no documents")
    args = parser.parse_args(argv)
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    commands = {"index": cmd_index, "sample": cmd_sample, "fetch": cmd_fetch,
                "sections": cmd_sections, "check": cmd_check, "universe": cmd_universe,
                "prices": cmd_prices, "returns": cmd_returns, "coverage": cmd_coverage,
                "documents": cmd_documents, "dictionary": cmd_dictionary,
                "signals": cmd_signals, "factors": cmd_factors, "backtest": cmd_backtest,
                "sanity": cmd_sanity, "llm-score": cmd_llm_score,
                "llm-signals": cmd_llm_signals, "anonymize": cmd_anonymize,
                "llm-score-anonymized": cmd_llm_score_anonymized,
                "memorization": cmd_memorization, "size": cmd_size,
                "robustness": cmd_robustness, "report": cmd_report}  # fmt: skip
    commands[args.command](load_config(args.config), args)


if __name__ == "__main__":
    main()
