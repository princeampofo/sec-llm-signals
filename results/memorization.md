# Memorization test

Model `mlx-community/Llama-3.2-3B-Instruct-8bit@ff05489`, stated training cutoff 2023-12-31. The LLM change signal is rebuilt from anonymized excerpts (company names, tickers, people, products, other organizations, dates and years masked) and compared with the original, for filings accepted before the cutoff (500 randomly sampled pairs) and after it (971 pairs, all of them).

If the model used memorized knowledge of what happened next, the original text would beat the anonymized text before the cutoff but not after it.

| | Before cutoff, original | Before cutoff, anonymized | After cutoff, original | After cutoff, anonymized |
|---|---|---|---|---|
| Filings with a 21-day return | 470 | 470 | 940 | 940 |
| Filing-level IC with the 21-day return (t) | -0.0194 (-0.42) | -0.0175 (-0.38) | 0.0140 (0.43) | 0.0092 (0.28) |
| Mean monthly IC, quintile portfolios (t) | -0.0092 (-0.49) | 0.0036 (0.23) | 0.0157 (1.84) | 0.0176 (1.75) |
| Months | 49 | 49 | 23 | 23 |
| Factor alpha, annualized (Newey-West t) | 0.07% (0.01) | 0.55% (0.16) | 1.02% (0.53) | 2.75% (1.15) |

## Original minus anonymized filing-level IC (95% bootstrap interval)

- Before cutoff: -0.0018 [-0.0697, +0.0603]
- After cutoff: +0.0048 [-0.0495, +0.0590]
- Difference in differences: -0.0066 [-0.0895, +0.0753]

## How much anonymization changes the reading

| | Before cutoff | After cutoff |
|---|---|---|
| Spearman, original vs. anonymized pessimism | 0.82 | 0.84 |
| Spearman, original vs. anonymized change | 0.72 | 0.66 |
| Mean absolute pessimism shift | 0.037 | 0.035 |

Monthly portfolios need at least 30 scored stocks; the before-cutoff sample is thin, so its monthly numbers are noisy. With 500 to 1,000 filings per cell, only a large memorization effect would be detectable.
