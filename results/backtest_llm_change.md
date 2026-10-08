# Backtest: `llm_change`

Monthly rebalance, 5 quantiles, long top / short bottom, equal-weighted; 10 bp per side. 71 months, 2020-03-31 to 2026-01-31, 436 stocks per month on average.

| Metric | Gross | After costs |
|---|---|---|
| Mean monthly IC (t-stat) | -0.0012 (-0.19) | |
| Annualized return | -0.87% | -1.41% |
| Annualized volatility | 4.60% | 4.64% |
| Sharpe ratio | -0.19 | -0.30 |
| Factor alpha, annualized (t-stat) | 0.24% (0.17) | -0.31% (-0.21) |
| Monthly turnover | 10.7% | |
| Max drawdown | | -16.0% |
| Months with positive IC | 46% | |

## Quantile returns (annualized, equal-weighted)

| q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|
| 14.75% | 15.81% | 13.39% | 15.69% | 13.88% |

## Factor exposures (gross long-short)

Fama-French 5 factors + momentum, Newey-West standard errors with 3 lags, 71 months, R² = 0.08. Factor data vintage: 202608 CRSP.

| Factor | Beta | t-stat |
|---|---|---|
| Mkt-RF | -0.058 | -1.25 |
| SMB | -0.030 | -0.45 |
| HML | -0.056 | -1.15 |
| RMW | -0.012 | -0.17 |
| CMA | 0.011 | 0.13 |
| Mom | -0.036 | -0.81 |
