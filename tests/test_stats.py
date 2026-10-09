import json

import numpy as np
import pandas as pd
import pytest

from secsignals import stats


def test_expected_max_sharpe_grows_with_trials():
    assert stats.expected_max_sharpe(1, 0.01) == 0
    few, many = stats.expected_max_sharpe(10, 0.01), stats.expected_max_sharpe(1000, 0.01)
    assert 0 < few < many
    # Bailey & Lopez de Prado: about 1.57 sd for 10 trials, about 3.26 sd for 1,000.
    assert few == pytest.approx(0.1 * 1.575, abs=0.01)
    assert many == pytest.approx(0.1 * 3.255, abs=0.01)


def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.01, 0.04, size=240))  # Sharpe about 0.25 per month
    alone = stats.deflated_sharpe(r, 1, 0.0)
    crowded = stats.deflated_sharpe(r, 200, 0.02)
    assert alone["dsr"] > 0.99 and crowded["dsr"] < alone["dsr"]
    noise = stats.deflated_sharpe(pd.Series(rng.normal(0, 0.04, 240)), 50, 0.01)
    assert noise["dsr"] < 0.5


def test_trial_sharpes_undo_annualization_and_count_each_config_once():
    trials = pd.DataFrame({
        "run_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "config_hash": ["a", "a", "b"],
        "settings": [json.dumps({}), json.dumps({}), json.dumps({"months_held": 3})],
        "sharpe_net": [1.0, np.sqrt(12), 2.0],
    })  # fmt: skip
    s = stats.trial_sharpes(trials)
    assert len(s) == 2
    assert s["a"] == pytest.approx(1.0)  # latest run of "a", annual sqrt(12) -> 1 per month
    assert s["b"] == pytest.approx(1.0)  # 2.0 annual over quarters -> 2 / sqrt(4)
