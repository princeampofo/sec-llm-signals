"""Command-line entry points for each pipeline stage (see Makefile)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from secsignals import dictionary, edgar, market, sections, signals
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="secsignals")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "sample", "universe", "prices", "returns", "coverage", "documents",
                 "dictionary", "signals"):
        sub.add_parser(name)
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
                "signals": cmd_signals}  # fmt: skip
    commands[args.command](load_config(args.config), args)


if __name__ == "__main__":
    main()
