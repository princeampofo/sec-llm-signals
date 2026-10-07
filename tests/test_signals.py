import pandas as pd
import pytest

from secsignals import dictionary, signals

ET = "America/New_York"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=ET)


def filings(*rows):
    return pd.DataFrame(
        [(a, c, ts(t)) for a, c, t in rows], columns=["accession", "cik", "acceptance_datetime"]
    )


def prev_of(df):
    out = signals.previous_filings(df, min_gap_days=270, max_gap_days=550)
    return dict(zip(out["accession"], out["prev_accession"], strict=True))


# ---------------------------------------------------------------- previous 10-K


def test_previous_is_same_company_last_year():
    df = filings(("a20", 1, "2020-02-20 16:30"), ("a21", 1, "2021-02-19 17:05"),
                 ("a22", 1, "2022-02-18 16:10"), ("b21", 2, "2021-03-01 08:00"))  # fmt: skip
    p = prev_of(df)
    assert pd.isna(p["a20"]) and pd.isna(p["b21"])  # first filings: nothing to compare with
    assert p["a21"] == "a20" and p["a22"] == "a21"  # never another company's filing


def test_previous_skips_a_refiled_report_weeks_earlier():
    # A re-filed 10-K two months before is not "last year's" report.
    df = filings(("a20", 1, "2020-02-20 16:30"), ("a20x", 1, "2020-12-15 16:30"),
                 ("a21", 1, "2021-02-19 17:05"))  # fmt: skip
    assert prev_of(df)["a21"] == "a20"


def test_previous_too_old_is_not_compared():
    df = filings(("a17", 1, "2017-02-20 16:30"), ("a21", 1, "2021-02-19 17:05"))
    assert pd.isna(prev_of(df)["a21"])


def test_previous_is_always_accepted_before_current():
    # No lookahead: across many filings, the comparison is always to an older report.
    rows = [(f"x{i}", i % 7, f"{2012 + i // 7}-0{1 + i % 9}-1{i % 10} 16:30") for i in range(90)]
    out = signals.previous_filings(filings(*rows), 270, 550).dropna(subset=["prev_accession"])
    assert len(out) > 0
    gap = out["acceptance_datetime"] - out["prev_acceptance"]
    assert (gap >= pd.Timedelta(days=270)).all() and (gap <= pd.Timedelta(days=550)).all()


# ---------------------------------------------------------------- change calculation


MASTER = dictionary.parse_master_dictionary(b"""Word,Seq_num,Negative,Uncertainty
LOSS,1,2009,0
DECLINE,2,2009,0
CYBERATTACK,3,2014,0
MAY,4,0,2009
REVENUE,5,0,0
GREW,6,0,0
""")
LEX = dictionary.build_lexicon(MASTER, ["negative", "uncertainty"])


def run(pairs_df, texts):
    counts = pd.DataFrame([{"accession": a, "cik": c, **dictionary.score_text(t, LEX)}
                           for (a, c), t in texts.items()])  # fmt: skip
    pairs = signals.previous_filings(pairs_df, 270, 550)
    return signals.lm_change_signals(pairs, counts, ["negative", "uncertainty"], True).set_index(
        "accession"
    )


def test_change_on_hand_made_example():
    # 2020: 10 words, 2 negative, 1 uncertain -> 20% / 10%
    # 2021: 20 words, 6 negative, 1 uncertain -> 30% / 5%
    t20 = "loss decline may " + "revenue grew " * 3 + "revenue"
    t21 = "loss loss decline decline loss decline may " + "revenue grew " * 6 + "revenue"
    out = run(filings(("a20", 1, "2020-02-20 16:30"), ("a21", 1, "2021-02-19 17:05")),
              {("a20", 1): t20, ("a21", 1): t21})  # fmt: skip
    r = out.loc["a21"]
    assert r["negative_share"] == pytest.approx(0.30)
    assert r["prev_negative_share"] == pytest.approx(0.20)
    assert r["d_negative"] == pytest.approx(0.10)
    assert r["d_uncertainty"] == pytest.approx(0.05 - 0.10)
    assert r["lm_change"] == pytest.approx(0.10 - 0.05)
    assert r["known_at"] == ts("2021-02-19 17:05")  # known when the newer 10-K is public
    assert pd.isna(out.loc["a20", "lm_change"])  # no previous 10-K -> no signal


def test_identical_text_gives_zero_change_across_a_dictionary_update():
    # CYBERATTACK joined the list in 2014. Scoring 2014's filing with the old list and
    # 2015's with the new one would show a fake jump; both use the 2015 list instead.
    text = "cyberattack loss revenue grew"
    out = run(filings(("a14", 1, "2014-02-20 16:30"), ("a15", 1, "2015-02-19 17:05")),
              {("a14", 1): text, ("a15", 1): text})  # fmt: skip
    assert out.loc["a15", "d_negative"] == 0
    assert out.loc["a15", "negative_share"] == pytest.approx(0.5)  # 2015 list: both words


def test_style_differences_between_companies_cancel_out():
    # A gloomy-styled company with steady tone gets the same signal as a cheerful one.
    gloomy = "loss decline loss revenue"
    cheerful = "revenue grew revenue grew"
    out = run(filings(("g20", 1, "2020-02-20 16:30"), ("g21", 1, "2021-02-19 17:05"),
                      ("c20", 2, "2020-02-20 16:30"), ("c21", 2, "2021-02-19 17:05")),
              {("g20", 1): gloomy, ("g21", 1): gloomy,
               ("c20", 2): cheerful, ("c21", 2): cheerful})  # fmt: skip
    assert out.loc["g21", "negative_share"] == 0.75 and out.loc["c21", "negative_share"] == 0
    assert out.loc["g21", "lm_change"] == out.loc["c21", "lm_change"] == 0


def test_pairs_with_very_different_lengths_are_dropped():
    short = "loss revenue"
    long = "revenue grew " * 20
    pairs = filings(("a20", 1, "2020-02-20 16:30"), ("a21", 1, "2021-02-19 17:05"))
    counts = pd.DataFrame([{"accession": a, "cik": 1, **dictionary.score_text(t, LEX)}
                           for a, t in [("a20", short), ("a21", long)]])  # fmt: skip
    p = signals.previous_filings(pairs, 270, 550)
    kept = signals.lm_change_signals(p, counts, ["negative"], True, max_length_ratio=None)
    dropped = signals.lm_change_signals(p, counts, ["negative"], True, max_length_ratio=3)
    assert kept.set_index("accession").loc["a21", "length_ratio"] == 20
    assert pd.notna(kept.set_index("accession").loc["a21", "lm_change"])
    assert pd.isna(dropped.set_index("accession").loc["a21", "lm_change"])


def test_missing_previous_section_gives_no_signal():
    out = run(filings(("a20", 1, "2020-02-20 16:30"), ("a21", 1, "2021-02-19 17:05")),
              {("a21", 1): "loss revenue"})  # 2020 section failed to extract
    assert pd.isna(out.loc["a21", "lm_change"])


# ---------------------------------------------------------------- coverage


def test_coverage_and_missing_reasons():
    universe = pd.DataFrame({"accession": ["u1", "u2", "u3", "u4"], "cik": [1, 2, 3, 4],
                             "date_filed": ["2020-02-20", "2020-03-01", "2021-02-01",
                                            "2021-02-02"]})  # fmt: skip
    sig = pd.DataFrame({"accession": ["u1"], "cik": [1], "lm_change": [0.01]})
    pairs = pd.DataFrame({"accession": ["u1", "u2", "u3", "u4"], "cik": [1, 2, 3, 4],
                          "prev_accession": ["p1", None, "p3", "p4"]})  # fmt: skip
    ok = {"u1", "p1", "u2", "u3", "p4"}  # u4's own section failed; p3 failed
    cov = signals.signal_coverage(sig, universe, "lm_change")
    assert cov["coverage"] == 0.25 and cov["by_year"] == {2020: 0.5, 2021: 0.0}
    reasons = signals.missing_reasons(universe, pairs, ok, sig, "lm_change")
    assert reasons == {"no_previous_10k": 1, "previous_mda_failed": 1, "current_mda_failed": 1}
