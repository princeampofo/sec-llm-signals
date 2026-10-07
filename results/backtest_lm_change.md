# Backtest: `lm_change`

Monthly rebalance, 5 quantiles, long top / short bottom, equal-weighted; 10 bp per side. 167 months, 2012-03-31 to 2026-01-31, 386 stocks per month on average.

| Metric | Gross | After costs |
|---|---|---|
| Mean monthly IC (t-stat) | -0.0052 (-1.02) | |
| Annualized return | -0.66% | -1.17% |
| Annualized volatility | 5.28% | 5.29% |
| Sharpe ratio | -0.13 | -0.22 |
| Factor alpha, annualized (t-stat) | -2.73% (-2.14) | -3.25% (-2.54) |
| Monthly turnover | 10.4% | |
| Max drawdown | | -33.3% |
| Months with positive IC | 51% | |

## Quantile returns (annualized, equal-weighted)

| q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|
| 13.05% | 15.02% | 13.94% | 13.24% | 12.38% |

## Factor exposures (gross long-short)

Fama-French 5 factors + momentum, Newey-West standard errors with 4 lags, 167 months, R² = 0.17. Factor data vintage: 202608 CRSP.

| Factor | Beta | t-stat |
|---|---|---|
| Mkt-RF | 0.108 | 3.74 |
| SMB | -0.017 | -0.32 |
| HML | -0.076 | -1.55 |
| RMW | 0.198 | 2.73 |
| CMA | 0.006 | 0.08 |
| Mom | 0.052 | 1.27 |
