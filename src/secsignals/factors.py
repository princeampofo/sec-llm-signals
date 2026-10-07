"""Fama-French 5 factors + momentum, and the factor regression with Newey-West errors.

Data: Ken French Data Library monthly CSVs (percent returns). The library revises its
files monthly as CRSP updates, so the downloaded zip is cached and its vintage
("created using the YYYYMM CRSP database") is reported alongside the results.

Regression: R_t = alpha + b1 MKT + b2 SMB + b3 HML + b4 RMW + b5 CMA + b6 MOM + e_t.
R_t is a long-short return, already self-financing, so the risk-free rate is not
subtracted. Alpha is what the factors leave unexplained. Newey-West (HAC) standard
errors allow e_t to be autocorrelated (persistent signals produce persistent
portfolios), which plain OLS errors would understate.
"""

from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def parse_french_csv(raw_zip: bytes) -> tuple[pd.DataFrame, str | None]:
    """Monthly table from a Ken French zip -> (DataFrame indexed by month-end, vintage).

    The CSV holds a monthly block, then an annual block, separated by text lines;
    only the monthly block (YYYYMM rows) is kept. Values become decimals.
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    vintage = re.search(r"created using the (\d{6}) CRSP database", text)
    lines = text.splitlines()
    # Header row: ",Mkt-RF,SMB,..." or, in the momentum file, just ",Mom".
    header_at = next(i for i, line in enumerate(lines) if re.match(r",\s*[A-Za-z]", line))
    header = [c.strip() for c in lines[header_at].split(",")]
    rows = []
    for line in lines[header_at + 1 :]:
        cells = [c.strip() for c in line.split(",")]
        if not re.fullmatch(r"\d{6}", cells[0]):
            break  # end of the monthly block
        rows.append(cells)
    df = pd.DataFrame(rows, columns=["month"] + header[1:])
    df.index = pd.to_datetime(df.pop("month"), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.astype(float).replace([-99.99, -999.0], np.nan) / 100.0
    return df, vintage.group(1) if vintage else None


def load_factors(five_factor_zip: bytes, momentum_zip: bytes) -> tuple[pd.DataFrame, str | None]:
    ff5, vintage = parse_french_csv(five_factor_zip)
    mom, _ = parse_french_csv(momentum_zip)
    return ff5.join(mom, how="inner"), vintage


def newey_west_lags(n_obs: int) -> int:
    """Common rule of thumb: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def factor_regression(returns: pd.Series, factors: pd.DataFrame,
                      lags: int | None = None) -> dict:  # fmt: skip
    """OLS of monthly returns on the factors with Newey-West standard errors.

    returns is indexed by month-end; months missing from either side are dropped.
    """
    data = factors[FACTOR_COLUMNS].join(returns.rename("ret"), how="inner").dropna()
    lags = newey_west_lags(len(data)) if lags is None else lags
    x = sm.add_constant(data[FACTOR_COLUMNS])
    fit = sm.OLS(data["ret"], x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "months": int(len(data)),
        "newey_west_lags": lags,
        "alpha_monthly": float(fit.params["const"]),
        "alpha_annualized": float(fit.params["const"] * 12),
        "alpha_t": float(fit.tvalues["const"]),
        "betas": {k: round(float(fit.params[k]), 4) for k in FACTOR_COLUMNS},
        "beta_t": {k: round(float(fit.tvalues[k]), 2) for k in FACTOR_COLUMNS},
        "r_squared": float(fit.rsquared),
    }
