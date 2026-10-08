# LLM-derived signals from SEC filings

Does a change in the tone of a company's 10-K, as read by a language model, predict its stock
returns over the following month, once lookahead bias, LLM memorization and known risk factors
are removed?

The pipeline goes from raw SEC EDGAR filings to a tone-change signal and forward returns with
one command, and every stage is built to be point-in-time: nothing uses information that was not
public at the time. A local language model scores each MD&A for tone, hedging and risk
severity, and a Loughran-McDonald word-count signal serves as the baseline it has to beat.

## Pipeline

```
EDGAR quarterly indexes ─▶ 10-K list ─▶ S&P 500 universe filter ─▶ acceptance timestamps
                                              │                          │
                                              ▼                          ▼
          10-K documents ─▶ MD&A (Item 7) ─┬▶ LM word counts ──────┬▶ change vs. previous 10-K
                                           └▶ LLM scores (cached) ─┘
                                              │
                         CIK ↔ ticker map ─▶ Yahoo prices ─▶ forward returns, coverage report
                                                      │
                     monthly quintile backtest ◀──────┘──▶ IC, turnover, costs, factor alpha
```

| Stage | Module | Output |
|---|---|---|
| Filing index | `edgar.py` | Every 10-K filed 2012–2025 (including co-registrants on combined filings) |
| Universe | `market.py` | S&P 500 members as of any date; date-ranged CIK ↔ ticker map |
| Filings | `edgar.py` | Acceptance timestamp for each universe 10-K |
| Documents | `edgar.py` | Full 10-Ks for universe companies, plus each one's prior-year 10-K (gzipped cache) |
| Section extraction | `sections.py` | Clean MD&A text, with a logged reason for every failure |
| Dictionary scores | `dictionary.py` | Loughran-McDonald negative and uncertainty word counts per MD&A |
| Baseline signal | `signals.py` | Change in each share vs. the same company's previous 10-K |
| LLM scores | `llm_scorer.py` | Tone, hedging and risk severity in [−1, 1] per MD&A, validated JSON, cached |
| LLM signal | `signals.py` | Change in LLM pessimism vs. the same company's previous 10-K |
| Prices | `market.py` | Split- and dividend-adjusted daily closes, behind a swappable price-source interface |
| Forward returns | `market.py` | 21-trading-day return from the first trading day after acceptance |
| Coverage | `market.py` | Share of universe-months with prices; list of missing companies |
| Backtest | `backtest.py` | Monthly quintile long-short, IC, turnover, returns after costs |
| Factor regression | `factors.py` | Alpha vs. Fama-French 5 factors + momentum, Newey-West errors |

## Design choices

- **Acceptance timestamp, not filing date.** 58% of these 10-Ks were accepted after the 4 pm
  close. Returns start at the close of the first trading day *after* the acceptance date, so no
  return window can begin before the filing was public. A test checks this on 500 random
  timestamps, and the return function refuses to produce a window that violates it.
- **Point-in-time universe.** Index membership comes from a historical S&P 500 list, so
  companies that were later acquired or delisted are included in the months they were members.
- **Tickers are mapped through SEC company IDs (CIKs) over date ranges.** Tickers get reused
  (CEG, SNDK), move between companies (AGN, IR), and survive holding-company reorganizations
  (Google → Alphabet). Automatic matches must have filed a 10-K during the ticker's membership;
  the remaining cases are hand-checked in `reference/ticker_overrides.csv`, each with a note.
- **Signals are changes, not levels.** A bank and a biotech write very differently, so the
  share of negative words mostly measures style. The signal is the change against the company's
  own previous 10-K (accepted 9–18 months earlier), so it measures news.
- **Point-in-time word lists.** Loughran-McDonald added words over time (e.g. CYBERATTACK in
  2014) and removed others (CLOSED, 2020). Both filings of a pair are scored with the list in
  force at the later filing, so a change never reflects an edit to the dictionary, and no
  filing is scored with words that were added after it. The dictionary file is pinned by checksum.
- **Mismatched pairs are dropped.** If a company's two MD&As differ in length by more than 3×,
  one extraction almost always caught a stub; those 20 pairs average 6× the typical change and
  are excluded.
- **The backtest only trades on what was public.** At each month-end rebalance, a signal counts
  only if its 10-K was accepted on an earlier calendar day, and stays valid for 12 months.
  Stocks that stop trading mid-month are held to their last price, not dropped.
- **The engine is tested against known answers.** A planted signal built from next month's
  returns must produce absurd performance (it does: IC 1.0, Sharpe 14.7 on the real universe),
  and random signals must average zero IC (20 draws: mean IC −0.0002). See
  `results/backtest_sanity.json`.
- **The LLM sees a short, forward-looking excerpt.** About 500 tokens of the paragraphs that
  say *expect*, *anticipate*, *outlook* or *believe* (the section's opening if there are too
  few), at temperature 0. It returns only three numbers as JSON; replies are checked against a
  strict schema (exact keys, finite values in [−1, 1]) and get one corrective retry, and any
  reply still invalid is logged, not guessed. Written justifications are requested only for a
  200-filing sample kept for hand-checking (`results/llm_hand_check.csv`).
- **Model: Llama 3.2 3B Instruct, run locally** (8-bit MLX weights, pinned revision), chosen for
  compute: about 3 s per filing on a laptop, 3× faster than Llama 3.1 8B. Both have the same
  stated knowledge cutoff (December 2023), so filings from 2024 on cannot have been memorized.
  On a 50-filing comparison the 3B's pessimism ranking agreed only moderately with the 8B's
  (Spearman 0.63), so a larger model may read the text better; the backend is a config switch
  (`llm.backend: openai_compatible` points at e.g. vLLM on a GPU) and scores are cached per
  model, so the comparison is one config change away.
- **LLM scores are cached by (filing, section, prompt version, model).** A rerun makes no
  model calls, and changing the prompt or the model never mixes old and new scores.
- **Every backtest run is logged** to `results/trials.csv` with a hash of its settings, so the
  number of variants tried is on record for multiple-testing adjustments.
- **Reused Yahoo symbols are rejected.** A price series that does not overlap the company's
  membership dates belongs to another security and is dropped (`results/rejected_price_symbols.csv`).

## Data coverage (survivorship bias)

Yahoo Finance drops companies after they are acquired or delisted. Because the universe comes
from historical membership, the gap can be measured rather than assumed:

| Period | Universe-months with a price |
|---|---|
| 2012–2025 | **86.2%** |
| 2020–2025 | **94.5%** |

Both are below 95%, so **backtest results on this data are likely optimistic**. Coverage falls
from 97.9% in 2025 to 71.9% in 2012: the further back, the more of that year's index has since
disappeared. CRSP data with delisting returns would close the gap; the price loader is behind
one interface so it can be swapped by config. Details: `results/coverage_report.md`.

## Results

Loughran-McDonald baseline (`lm_change`, long the stocks whose language became *less* negative
and uncertain), 167 months from March 2012 to January 2026, about 390 stocks a month:

| Metric | Value |
|---|---|
| Mean monthly IC (t-stat) | −0.005 (−1.02) |
| Long-short Sharpe, gross / after 10 bp costs | −0.13 / −0.22 |
| Factor alpha, annualized (Newey-West t) | −2.7% (−2.14) |
| Monthly turnover | 10.4% |

The word-count signal does not predict returns in large caps; after controlling for known
factors it leans slightly the wrong way, mostly before 2017. Full report:
`results/backtest_lm_change.md`.

LLM signal (`llm_change`, long the stocks whose MD&A became *less* pessimistic, where pessimism
= (hedging + risk severity − tone) / 3), 10-Ks accepted 2020–2025, 71 months from March 2020 to
January 2026. The baseline over the same months is shown for comparison:

| Metric | LLM | LM baseline, same months |
|---|---|---|
| Mean monthly IC (t-stat) | −0.001 (−0.19) | 0.004 (0.51) |
| Long-short Sharpe, gross / after 10 bp costs | −0.19 / −0.30 | 0.47 / 0.38 |
| Factor alpha, annualized (Newey-West t) | 0.2% (0.17) | |
| Monthly turnover | 10.7% | |

Neither signal predicts returns over this period. The two are nearly unrelated (Spearman 0.11
between the changes), so the model is not just counting negative words. 100% of the 3,458
filings in scope got valid scores and 96.1% of universe 10-Ks have an LLM signal; scores are
coarse (29% of changes are exactly zero). In the 200-filing hand-check sample, 21 of the 54 justifications that
state a hedging level come with a score of the opposite sign (4 of 5 "heavily hedged" texts
scored below zero), so the 3B model's hedging score is unreliable; that is the first thing to
fix, with a larger model or a revised prompt. Details: `results/backtest_llm_change.md`,
`results/llm_scoring_report.json`.

## Signal coverage

| | |
|---|---|
| MD&A extracted (all 7,732 universe-company 10-Ks) | 94.8% |
| Universe 10-Ks with a baseline signal | **92.9%** (6,483 of 6,977) |
| Missing: own MD&A not extractable (mostly incorporated by reference to Exhibit 13) | 346 |
| Missing: no previous 10-K within 9–18 months (IPOs, spin-offs) | 71 |
| Missing: previous MD&A not extractable | 57 |
| Missing: length-mismatched pair | 20 |

MD&A extraction succeeds on 99.5% of a random sample of S&P 500 10-Ks
(`results/extraction_check_universe.json`); per-year coverage is in
`results/signal_coverage_lm.json`.

## Running

```
make install  # LLM scoring runs locally on Apple silicon; elsewhere use an OpenAI-compatible server
export SEC_USER_AGENT="Your Name your.email@example.edu"   # SEC requires a contact
make all      # full pipeline; downloads are cached, so reruns only fetch what is new
make test     # no network needed
```

All parameters (dates, rate limits, horizons, thresholds) live in `config.yaml`.

## Repository layout

```
src/secsignals/   pipeline code (edgar, sections, market, dictionary, llm_scorer, signals,
                  backtest, factors, cli)
tests/            unit tests + 5 saved 10-K fixtures (no network needed)
reference/        hand-checked ticker overrides
results/          reports and logs produced by the pipeline
data/             downloads and intermediate Parquet files (not committed; rebuilt by `make all`)
```
