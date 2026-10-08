"""Command-line entry points for each pipeline stage (see Makefile)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from secsignals import backtest, dictionary, edgar, factors, llm_scorer, market, sections, signals
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


def cmd_dictionary(cfg: dict, args: argparse.Namespace) -> None:
    d = cfg["dictionary"]
    raw = edgar.client_from_config(cfg).get(d["url"])
    digest = hashlib.sha256(raw).hexdigest()
    if d["sha256"] and digest != d["sha256"]:
        sys.exit(f"dictionary checksum {digest} != pinned {d['sha256']}: the file changed")
    lexicon = dictionary.build_lexicon(dictionary.parse_master_dictionary(raw), d["categories"])
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


def _backtest_inputs(cfg: dict) -> tuple:
    """(period returns, membership, ticker map, rebalance dates)."""
    b, m = cfg["backtest"], cfg["market"]
    prices = pd.read_parquet(_interim(cfg, "prices.parquet"))
    calendar = pd.DatetimeIndex(
        prices.loc[prices["symbol"] == m["calendar_symbol"], "date"]
    ).sort_values()
    # One extra month-end so the last rebalance has a holding period.
    end = pd.Timestamp(b["end"]) + pd.offsets.MonthEnd(1)
    dates = backtest.rebalance_dates(calendar, b["start"], end)
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
                "llm-signals": cmd_llm_signals}  # fmt: skip
    commands[args.command](load_config(args.config), args)


if __name__ == "__main__":
    main()
