# LLM-derived signals from SEC filings

Does a change in the tone of a company's 10-K, as read by a language model, predict its stock
returns over the following month, once lookahead bias, LLM memorization and known risk factors
are removed?

The pipeline goes from raw SEC EDGAR filings to forward returns with one command, and every
stage is built to be point-in-time: nothing uses information that was not public at the time.

## Pipeline

```
EDGAR quarterly indexes ─▶ 10-K list ─▶ S&P 500 universe filter ─▶ acceptance timestamps
                                              │
                                              ▼
              MD&A (Item 7) extraction   CIK ↔ ticker map ─▶ Yahoo prices ─▶ forward returns
                                                                         └─▶ coverage report
```

| Stage | Module | Output |
|---|---|---|
| Filing index | `edgar.py` | Every 10-K filed 2012–2025 (including co-registrants on combined filings) |
| Universe | `market.py` | S&P 500 members as of any date; date-ranged CIK ↔ ticker map |
| Filings | `edgar.py` | Acceptance timestamp for each universe 10-K |
| Section extraction | `sections.py` | Clean MD&A text, with a logged reason for every failure |
| Prices | `market.py` | Split- and dividend-adjusted daily closes, behind a swappable price-source interface |
| Forward returns | `market.py` | 21-trading-day return from the first trading day after acceptance |
| Coverage | `market.py` | Share of universe-months with prices; list of missing companies |

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

MD&A extraction succeeds on 98% of a random sample of S&P 500 10-Ks
(`results/extraction_check_universe.json`).

## Running

```
make install
export SEC_USER_AGENT="Your Name your.email@example.edu"   # SEC requires a contact
make all      # full pipeline; downloads are cached, so reruns only fetch what is new
make test     # no network needed
```

All parameters (dates, rate limits, horizons, thresholds) live in `config.yaml`.

## Repository layout

```
src/secsignals/   pipeline code (edgar, sections, market, cli)
tests/            unit tests + 5 saved 10-K fixtures
reference/        hand-checked ticker overrides
results/          reports and logs produced by the pipeline
data/             downloads and intermediate Parquet files (not committed; rebuilt by `make all`)
```
