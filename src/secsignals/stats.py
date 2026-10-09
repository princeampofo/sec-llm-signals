"""Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014).

Trying many variants and reporting the best inflates the Sharpe ratio: among N strategies
with no skill, the best one's Sharpe is expected to be well above zero. The deflated Sharpe
ratio is the probability that the true Sharpe exceeds that expected maximum, SR0:

    SR0 = sqrt(V) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))
    DSR = Z( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2) )

where V is the variance of the Sharpe ratios across the N trials, g the Euler-Mascheroni
constant, T the number of return periods, skew and kurt (not excess) the moments of the
strategy's returns. All Sharpe ratios are per period (not annualized). With one trial,
SR0 = 0 and the DSR reduces to the probabilistic Sharpe ratio against zero.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """SR0: expected best per-period Sharpe among n_trials skill-less strategies."""
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    z = st.norm.ppf
    spread = (1 - EULER_GAMMA) * z(1 - 1 / n_trials) + EULER_GAMMA * z(1 - 1 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * spread


def deflated_sharpe(returns: pd.Series, n_trials: int, sharpe_variance: float) -> dict:
    """DSR of one strategy's per-period returns, given the trials it was chosen from."""
    r = returns.dropna().to_numpy(float)
    t = len(r)
    sr = float(r.mean() / r.std(ddof=1))
    skew = float(st.skew(r))
    kurt = float(st.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    dsr = float(st.norm.cdf((sr - sr0) * math.sqrt(t - 1) / denom))
    return {"periods": t, "sharpe_per_period": sr, "skew": skew, "kurtosis": kurt,
            "n_trials": n_trials, "sharpe_variance": sharpe_variance,
            "expected_max_sharpe": sr0, "dsr": dsr}  # fmt: skip


def trial_sharpes(trials: pd.DataFrame, column: str = "sharpe_net") -> pd.Series:
    """Per-period Sharpe of each distinct trial (latest run per config hash).

    trials.csv stores annualized Sharpe ratios; a trial's settings record its holding
    period ("months_held", 1 if absent), which sets the annualization to undo.
    """
    latest = trials.sort_values("run_at").drop_duplicates("config_hash", keep="last")
    months = latest["settings"].map(lambda s: json.loads(s).get("months_held", 1))
    per_period = latest[column] / np.sqrt(12 / months)
    return per_period.set_axis(latest["config_hash"]).dropna()
