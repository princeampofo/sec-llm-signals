# Price coverage of the S&P 500 universe

Share of universe-months (index members at each month-end) with a Yahoo adjusted close on the month's last trading day.

| Period | Universe-months | With price | Coverage | Missing: no price | Missing: no Yahoo symbol | Missing: no CIK |
|---|---|---|---|---|---|---|
| full | 84,449 | 72,750 | 86.2% | 10,663 | 969 | 67 |
| llm | 36,273 | 34,286 | 94.5% | 1,932 | 0 | 55 |

**Coverage is below 95% for: full, llm.** The missing companies are mostly ones later acquired or delisted, whose history Yahoo no longer serves. Backtest results on this data are likely optimistic (survivorship bias); CRSP with delisting returns would close the gap.

## By year

| Year | Coverage |
|---|---|
| 2012 | 71.9% |
| 2013 | 73.7% |
| 2014 | 75.3% |
| 2015 | 77.2% |
| 2016 | 80.7% |
| 2017 | 84.3% |
| 2018 | 86.4% |
| 2019 | 88.7% |
| 2020 | 90.7% |
| 2021 | 92.2% |
| 2022 | 94.0% |
| 2023 | 95.7% |
| 2024 | 96.6% |
| 2025 | 97.9% |

## Reasons

- **no_price**: mapped to a Yahoo symbol, but Yahoo has no close for that day (usually: the company was acquired or delisted and Yahoo dropped its history).
- **no_yahoo_symbol**: the company's old ticker now belongs to a different security, so Yahoo's series under it would be someone else's prices.
- **no_cik**: no SEC filer ID (banks that file with the FDIC, not the SEC).

## Most-missing companies (top 30; full list in missing_companies.csv)

| Ticker | Company | Reason | Months missing | First | Last |
|---|---|---|---|---|---|
| AVB | AVALONBAY COMMUNITIES INC | no_price | 168 | 2012-01 | 2025-12 |
| EA | ELECTRONIC ARTS INC. | no_price | 168 | 2012-01 | 2025-12 |
| K | KELLANOVA | no_price | 167 | 2012-01 | 2025-11 |
| IPG | INTERPUBLIC GROUP OF COMPANIES, INC. | no_price | 166 | 2012-01 | 2025-10 |
| HES | HESS CORP | no_price | 162 | 2012-01 | 2025-06 |
| JNPR | JUNIPER NETWORKS INC | no_price | 162 | 2012-01 | 2025-06 |
| DFS | Discover Financial Services | no_price | 160 | 2012-01 | 2025-04 |
| MRO | MARATHON OIL CORP | no_price | 154 | 2012-01 | 2024-10 |
| CMA | COMERICA INC /NEW/ | no_price | 149 | 2012-01 | 2024-05 |
| PXD | PIONEER NATURAL RESOURCES CO | no_price | 148 | 2012-01 | 2024-04 |
| SEE | SEALED AIR CORP/DE | no_price | 143 | 2012-01 | 2023-11 |
| CTXS | CITRIX SYSTEMS INC | no_price | 129 | 2012-01 | 2022-09 |
| WBA | Walgreens Boots Alliance, Inc. | no_price | 128 | 2014-12 | 2025-07 |
| CERN | CERNER Corp | no_price | 125 | 2012-01 | 2022-05 |
| DISCA | Warner Bros. Discovery, Inc. | no_price | 123 | 2012-01 | 2022-03 |
| PBCT | People's United Financial, Inc. | no_price | 123 | 2012-01 | 2022-03 |
| XLNX | XILINX INC | no_price | 121 | 2012-01 | 2022-01 |
| LEG | LEGGETT & PLATT INC | no_price | 119 | 2012-01 | 2021-11 |
| HOLX | HOLOGIC INC | no_price | 118 | 2016-03 | 2025-12 |
| COG | Coterra Energy Inc. | no_price | 117 | 2012-01 | 2021-09 |
| QRVO | Qorvo, Inc. | no_price | 114 | 2015-06 | 2024-11 |
| FLIR | FLIR SYSTEMS INC | no_price | 112 | 2012-01 | 2021-04 |
| VAR | VARIAN MEDICAL SYSTEMS INC | no_price | 111 | 2012-01 | 2021-03 |
| NLSN | Nielsen Holdings plc | no_price | 111 | 2013-07 | 2022-09 |
| ALXN | ALEXION PHARMACEUTICALS, INC. | no_price | 110 | 2012-05 | 2021-06 |
| TIF | TIFFANY & CO | no_price | 108 | 2012-01 | 2020-12 |
| NBL | NOBLE ENERGY INC | no_price | 105 | 2012-01 | 2020-09 |
| ETFC | E TRADE FINANCIAL CORP | no_price | 105 | 2012-01 | 2020-09 |
| KSU | KANSAS CITY SOUTHERN | no_price | 103 | 2013-05 | 2021-11 |
| JWN | NORDSTROM INC | no_price | 101 | 2012-01 | 2020-05 |
