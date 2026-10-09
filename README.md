# LLM-derived signals from SEC filings

**Question:** when a company's 10-K reads more pessimistic than last year's, as judged by a
language model, does its stock underperform the next month?

**Answer: no.** Neither the LLM signal nor a word-count baseline predicts S&P 500 returns,
before or after costs and known risk factors, and no robustness variant produces a
significant IC. A memorization test finds no sign the model relies on remembering what
happened to companies.
Details: [REPORT.md](REPORT.md).

| | Word-count baseline, 2012–2025 | LLM (Llama 3.2 3B), 2020–2025 |
|---|---|---|
| Mean monthly IC (t-stat) | −0.005 (−1.02) | −0.001 (−0.19) |
| Long-short Sharpe after costs | −0.22 | −0.30 |
| Factor alpha, annualized (t-stat) | −2.7% (−2.14) | 0.2% (0.17) |
| Deflated Sharpe ratio (bar: 0.95) | 0.000 | 0.003 |

## How it works

One command goes from raw SEC EDGAR filings to the report:

1. Download every 10-K of historical S&P 500 members and extract the MD&A section.
2. Score each MD&A: Loughran-McDonald word counts, and a local LLM rating tone, hedging and
   risk severity from about 500 tokens of forward-looking text.
3. Signal = change versus the same company's previous 10-K (measures news, not writing style).
4. Monthly backtest: long the least-pessimistic quintile, short the most, with 10 bp costs,
   Fama-French 5 + momentum alpha, and a deflated Sharpe ratio over every variant tried.
5. Memorization test: rescore with names and dates masked, before vs. after the model's
   training cutoff.

**Safeguards against lookahead:** filings are timed by their acceptance timestamp, not the
filing date; the universe and the dictionary are point-in-time; a planted future-return
signal must look absurd and random signals must give zero IC, both checked by tests.

**Known weaknesses:** free Yahoo prices miss 14% of universe-months (delisted companies, so
results are likely optimistic); a small 3B model, whose hedging score is unreliable; only
two post-cutoff years. See the limitations in [REPORT.md](REPORT.md).

## Running

```
make install
export SEC_USER_AGENT="Your Name your.email@example.edu"   # SEC requires a contact
make all      # raw data -> results/ and REPORT.md; cached, so reruns only fetch what is new
make test     # no network needed
```

LLM scoring runs locally on Apple silicon (MLX); `llm.backend: openai_compatible` in
`config.yaml` points it at a GPU server instead. All parameters live in `config.yaml`.

## Layout

```
src/secsignals/   pipeline code: edgar, sections, market, dictionary, llm_scorer, anonymize,
                  signals, backtest, factors, stats, report, cli
tests/            unit tests + 5 saved 10-K fixtures
reference/        hand-checked ticker overrides
results/          reports, logs, figures and trials.csv (every backtest variant run)
REPORT.md         the write-up, generated from results/
data/             downloads and intermediate files (not committed; rebuilt by `make all`)
```
