import pandas as pd
import pytest

from secsignals import dictionary

# A tiny master dictionary in LM's format: category columns hold the year a word was
# added, a negative year if it was removed, 0 if not in the category.
MASTER_CSV = b"""Word,Seq_num,Negative,Positive,Uncertainty
LOSS,1,2009,0,0
DECLINE,2,2009,0,0
CYBERATTACK,3,2014,0,0
CLOSED,4,-2020,0,0
MAY,5,0,0,2009
UNCERTAIN,6,2009,0,2009
THE,7,0,0,0
REVENUE,8,0,0,0
INCREASED,9,0,0,0
NA,10,0,0,0
"""


@pytest.fixture
def lexicon():
    master = dictionary.parse_master_dictionary(MASTER_CSV)
    return dictionary.build_lexicon(master, ["negative", "uncertainty"])


def test_parse_keeps_the_word_na():
    master = dictionary.parse_master_dictionary(MASTER_CSV)
    assert "NA" in set(master["Word"])  # pandas would read it as missing by default


def test_lexicon_cohorts(lexicon):
    assert lexicon.cohorts["LOSS"] == ("negative__a2009",)
    assert lexicon.cohorts["CLOSED"] == ("negative__r2020",)
    assert set(lexicon.cohorts["UNCERTAIN"]) == {"negative__a2009", "uncertainty__a2009"}
    assert "THE" not in lexicon.cohorts and "THE" in lexicon.vocabulary


def test_score_text_counts_only_dictionary_words(lexicon):
    text = "The revenue DECLINE may 2023 be uncertain; loss, loss. $4.5bn xyzzy"
    s = dictionary.score_text(text, lexicon)
    # Words in the master dictionary: THE REVENUE DECLINE MAY UNCERTAIN LOSS LOSS = 7
    # ("be" is not in this tiny dictionary; numbers and junk never count).
    assert s["n_words"] == 7
    assert s["negative__a2009"] == 4  # DECLINE, UNCERTAIN, LOSS, LOSS
    assert s["uncertainty__a2009"] == 2  # MAY, UNCERTAIN
    assert s["negative__a2014"] == 0 and s["negative__r2020"] == 0


def counts_frame(lexicon, *texts):
    rows = [dictionary.score_text(t, lexicon) for t in texts]
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2014, 2 / 4),  # LOSS + CLOSED; CYBERATTACK (added 2014) not in force until 2015
        (2015, 3 / 4),  # + CYBERATTACK
        (2020, 3 / 4),  # CLOSED was removed in 2020: still counted for 2020 filings
        (2021, 2 / 4),  # ...but not after: LOSS + CYBERATTACK
    ],
)
def test_point_in_time_shares(lexicon, year, expected):
    counts = counts_frame(lexicon, "loss cyberattack closed revenue")
    assert dictionary.shares(counts, "negative", year).iloc[0] == pytest.approx(expected)


def test_latest_version_ignores_dates(lexicon):
    counts = counts_frame(lexicon, "loss cyberattack closed revenue")
    # Latest list: additions counted, removals applied -> LOSS + CYBERATTACK.
    assert dictionary.shares(counts, "negative", 2012, point_in_time=False).iloc[0] == 0.5


def test_share_is_nan_for_empty_text(lexicon):
    counts = counts_frame(lexicon, "1234 $$$")
    assert counts["n_words"].iloc[0] == 0
    assert dictionary.shares(counts, "negative", 2020).isna().all()


def test_score_sections_skips_failed_extractions(lexicon):
    secs = pd.DataFrame({"accession": ["a", "b"], "cik": [1, 2], "status": ["ok", "failed"],
                         "text": ["loss revenue", None]})  # fmt: skip
    out = dictionary.score_sections(secs, lexicon)
    assert out["accession"].tolist() == ["a"]
    assert out["negative__a2009"].tolist() == [1]
