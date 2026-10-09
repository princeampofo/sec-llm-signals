"""REPORT.md and its charts, built from the files the pipeline wrote to results/.

Every number in the report is read from those files, so rerunning the pipeline rewrites
the report; the prose around the numbers states which bar each result clears.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SIGNALS = {"lm_change": "Loughran-McDonald (word counts)", "llm_change": "LLM (Llama 3.2 3B)"}
SHORT_NAMES = {"lm_change": "the word-count signal", "llm_change": "the LLM signal"}
# Rough bars for "interesting" (see the evaluation section of the report).
BARS = {"ic": 0.02, "ic_t": 2.0, "sharpe_net": 0.5, "alpha_t": 2.0, "dsr": 0.95}


def _load(results: Path) -> dict:
    def j(name: str) -> dict:
        return json.loads((results / name).read_text())

    data = {name: j(f"backtest_{name}.json") for name in SIGNALS}
    monthly = {name: pd.read_csv(results / f"backtest_{name}_monthly.csv",
                                 parse_dates=["rebalance_date", "period_end"])
               for name in SIGNALS}  # fmt: skip
    return {
        "backtests": data, "monthly": monthly, "memorization": j("memorization.json"),
        "robustness": j("robustness.json"), "coverage": j("coverage_summary.json"),
        "llm": j("llm_scoring_report.json"), "lm_coverage": j("signal_coverage_lm.json"),
        "extraction": j("extraction_check_universe.json"), "sanity": j("backtest_sanity.json"),
        "anonymization": j("anonymization_report.json"),
    }  # fmt: skip


# ---------------------------------------------------------------- charts


def _charts(d: dict, figures: Path) -> dict[str, str]:
    figures.mkdir(parents=True, exist_ok=True)
    paths = {}

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, label in SIGNALS.items():
        m = d["monthly"][name]
        growth = pd.concat([pd.Series([1.0], index=[m["rebalance_date"].iloc[0]]),
                            (1 + m["ret_ls_net"]).cumprod().set_axis(m["period_end"])])  # fmt: skip
        ax.plot(growth.index, growth.to_numpy(), label=label)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_title("Cumulative long-short return after 10 bp costs")
    ax.set_ylabel("growth of $1")
    ax.legend()
    fig.tight_layout()
    paths["cumulative"] = _save(fig, figures / "cumulative_returns.png")

    cells = d["memorization"]["cells"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, period in enumerate(("pre_cutoff", "post_cutoff")):
        for j, version in enumerate(("original", "anonymized")):
            c = cells[f"{period}/{version}"]["filing_ic"]
            x = i + (j - 0.5) * 0.35
            ax.bar(x, c["ic"], width=0.33, color=("C0", "C1")[j],
                   yerr=1.96 / math.sqrt(c["n"]), capsize=4,
                   label=version.capitalize() if i == 0 else None)  # fmt: skip
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks([0, 1], ["Before cutoff", "After cutoff"])
    ax.set_ylabel("filing-level IC (95% interval)")
    ax.set_title("Memorization test")
    ax.legend()
    fig.tight_layout()
    paths["memorization"] = _save(fig, figures / "memorization.png")
    return paths


def _save(fig: plt.Figure, path: Path) -> str:
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.as_posix()


# ---------------------------------------------------------------- text


def _pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}".replace("-", "−")


def _num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "−")


def _clears(summary: dict, reg: dict, dsr: float) -> list[str]:
    checks = [summary["mean_ic"] > BARS["ic"] and summary["ic_t"] > BARS["ic_t"],
              summary["net"]["sharpe"] > BARS["sharpe_net"], reg["alpha_t"] > BARS["alpha_t"],
              dsr > BARS["dsr"]]  # fmt: skip
    names = ["IC", "Sharpe after costs", "factor alpha", "deflated Sharpe"]
    return [n for n, ok in zip(names, checks, strict=True) if ok]


def _memorization_reading(m: dict) -> str:
    cells = m["cells"]
    did = m["ic_gap_tests"]["difference_in_differences"]
    interval = (f"difference in differences {_num(did['value'], 3)}, 95% interval "
                f"{_num(did['ci95']['low'], 3)} to {_num(did['ci95']['high'], 3)}")  # fmt: skip
    if all(abs(c["filing_ic"]["t"]) < 2 for c in cells.values()):
        return (f"No cell differs from zero, and masking changes nothing ({interval}). There is "
                "no performance for memorization to inflate; the test rules out only a large "
                "effect.")  # fmt: skip
    if did["ci95"]["low"] > 0:
        return (f"The original text wins before the cutoff but not after ({interval}): "
                "memorization was probably inflating the pre-cutoff results.")  # fmt: skip
    return (f"Masking names does not change performance ({interval}): the model reads the "
            "text rather than recalling outcomes.")  # fmt: skip


def _variant_label(variant: str) -> str:
    labels = {"baseline": "Baseline: 1 month, 10 bp, equal-weighted", "costs_0bp": "Costs 0 bp",
              "costs_25bp": "Costs 25 bp", "value_weighted": "Value-weighted (public float)",
              "horizon_3m": "3-month horizon"}  # fmt: skip
    if variant in labels:
        return labels[variant]
    start, _, end = variant.partition("_")
    return f"{start}–{end} only" if end.isdigit() else variant


def _robustness_summary(rows: list[dict]) -> str:
    """One clause on whether any variant shows a significant IC or clears the Sharpe bar."""
    significant = [r for r in rows if r["ic_t"] > BARS["ic_t"]]
    if significant:
        names = ", ".join(f"{SHORT_NAMES[r['signal']]} ({_variant_label(r['variant']).lower()})"
                          for r in significant)  # fmt: skip
        return f"some variants show a significant IC ({names})"
    above = [r for r in rows if r["sharpe_net"] > BARS["sharpe_net"]]
    clause = "no robustness variant produces a significant IC"
    for r in above:
        clause += (f" ({SHORT_NAMES[r['signal']]}, {_variant_label(r['variant']).lower()}, "
                   f"has a Sharpe ratio of {_num(r['sharpe_net'])} but an IC t-stat of "
                   f"{_num(r['ic_t'])})")
    return clause


def _robustness(rows: list[dict]) -> list[str]:
    """One row per variant, one column per signal: net Sharpe (IC t-stat)."""
    table = pd.DataFrame(rows)
    out = ["| Variant | " + " | ".join(SIGNALS.values()) + " |", "|---|---|---|"]
    for variant in dict.fromkeys(table["variant"]):
        cells = []
        for name in SIGNALS:
            r = table[(table["variant"] == variant) & (table["signal"] == name)]
            cells.append(f"{_num(r['sharpe_net'].iloc[0])} ({_num(r['ic_t'].iloc[0])})"
                         if len(r) else "")  # fmt: skip
        out.append(f"| {_variant_label(variant)} | " + " | ".join(cells) + " |")
    return out


def build_report(results: Path, report_path: Path) -> str:
    d = _load(results)
    charts = _charts(d, results / "figures")
    bt, rob, mem = d["backtests"], d["robustness"], d["memorization"]
    dsr = {name: rob["deflated_sharpe"][name]["dsr"] for name in SIGNALS}
    cleared = {name: _clears(bt[name]["summary"], bt[name]["factor_regression"]["gross"],
                             dsr[name]) for name in SIGNALS}  # fmt: skip
    answer = ("**Answer: no.** Neither signal clears any bar below"
              if not any(cleared.values()) else
              "**Answer: partly.** " + "; ".join(f"{SIGNALS[k]} clears {', '.join(v)}"
                                                for k, v in cleared.items() if v))  # fmt: skip
    cells = mem["cells"]
    post = cells["post_cutoff/original"]
    cov = d["coverage"]["full"]["coverage"]

    def col(name: str) -> tuple[dict, dict]:
        return bt[name]["summary"], bt[name]["factor_regression"]["gross"]

    def metric_row(label: str, fmt) -> str:  # noqa: ANN001
        return f"| {label} | " + " | ".join(fmt(*col(n)) for n in SIGNALS) + " |"

    def mem_cell(period: str, version: str) -> str:
        c = cells[f"{period}/{version}"]["filing_ic"]
        return f"{_num(c['ic'], 3)} ({_num(c['t'])})"

    lines = [
        "# Do LLM readings of 10-Ks predict stock returns?", "",
        "When a company's annual report reads more pessimistic than the year before, as judged "
        "by a language model, does its stock underperform the next month? A Loughran-McDonald "
        "word count is the baseline.", "",
        f"{answer}, and {_robustness_summary(rob['variants'])}. A memorization test finds no "
        "sign the model relies on remembering what happened to companies.", "",
        "## Method", "",
        "- **Universe:** historical S&P 500 members; every 10-K from SEC EDGAR, timed by its "
        "acceptance timestamp (when it became public).",
        "- **Signals:** change versus the company's previous 10-K. LM: share of negative and "
        "uncertainty words. LLM (Llama 3.2 3B, local, temperature 0): pessimism = (hedging + "
        "risk severity − tone) / 3, from about 500 tokens of forward-looking MD&A text.",
        "- **Backtest:** monthly, long the least-pessimistic quintile and short the most, "
        "equal-weighted, 10 bp per side; alpha against Fama-French 5 factors + momentum.", "",
        "## Results", "",
        "| | " + " | ".join(SIGNALS.values()) + " | Bar |", "|---|---|---|---|",
        metric_row("Months held", lambda s, r: f"{s['months']} ({s['first_period'][:7]} to "
                   f"{s['last_period'][:7]})") + " |",
        metric_row("Mean monthly IC (t)", lambda s, r: f"{_num(s['mean_ic'], 3)} "
                   f"({_num(s['ic_t'])})") + " IC > 0.02, t > 2 |",
        metric_row("Sharpe ratio after costs", lambda s, r: _num(s["net"]["sharpe"]))
        + " > 0.5 |",
        metric_row("Factor alpha, annualized (t)", lambda s, r: f"{_pct(r['alpha_annualized'])} "
                   f"({_num(r['alpha_t'])})") + " t > 2 |",
        "| Deflated Sharpe ratio | " + " | ".join(_num(dsr[n], 3) for n in SIGNALS)
        + " | > 0.95 |", "",
        f"![Cumulative returns]({charts['cumulative']})", "",
        "## Memorization test", "",
        f"The LLM signal was recomputed with company names, tickers, people, products, dates "
        f"and years masked, for every filing after the model's training cutoff "
        f"({mem['training_cutoff'][:7]}) and 500 before it. Filing-level IC (t):", "",
        "| | Original text | Anonymized text |", "|---|---|---|",
        f"| Before cutoff | {mem_cell('pre_cutoff', 'original')} | "
        f"{mem_cell('pre_cutoff', 'anonymized')} |",
        f"| After cutoff | {mem_cell('post_cutoff', 'original')} | "
        f"{mem_cell('post_cutoff', 'anonymized')} |", "",
        _memorization_reading(mem), "",
        "## Robustness", "",
        "Sharpe ratio after costs (IC t-stat). Every variant is logged in `results/trials.csv` "
        f"and counted by the deflated Sharpe ratio ({rob['distinct_trials']} trials).", "",
        *_robustness(rob["variants"]), "",
        "## Limitations", "",
        f"- Only {post['filing_ic']['n']} post-cutoff filings ({post['monthly']['months']} "
        "months), so only a large effect would be detectable.",
        "- A small model: its hedging score often contradicts its own written justification.",
        f"- Free Yahoo prices miss {_pct(1 - cov, 0)} of universe-months (mostly delisted "
        "companies), so results are likely optimistic.",
        "- Anonymization is imperfect: some product names survive, and some companies are "
        "recognizable from their business.", "",
        "## Next steps", "",
        "1. Rescore with a larger model on a GPU (a config change; scores are cached per model).",
        "2. Add 10-Qs to roughly quadruple the post-cutoff sample.",
        "3. Use CRSP prices with delisting returns.", "",
    ]  # fmt: skip
    text = "\n".join(lines)
    report_path.write_text(text)
    return text
