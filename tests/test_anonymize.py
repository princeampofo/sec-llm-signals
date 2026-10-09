import pytest

from secsignals import anonymize

ACME = anonymize.Identity(("ACME WIDGETS, INC.", "ACME HOLDINGS CORP /DE/"), ("ACMW",))
COMMON = frozenset({
    "WE", "EXPECT", "THE", "CONSOLIDATED", "FINANCIAL", "STATEMENTS", "GOODWILL", "NOTES",
    "SALES", "TO", "GROW", "IN", "FISCAL", "OUR", "CUSTOMER", "SAID", "CHIEF", "EXECUTIVE",
    "REVENUE", "FROM", "ROSE", "AND", "MARKETS", "FORWARD", "LOOKING", "NON", "GAAP",
})  # fmt: skip


def fake_entities(spans_by_text: dict[str, str]):
    """Stand-in recognizer: tags every occurrence of the given strings."""

    def find(text: str) -> list[tuple[int, int, str]]:
        out = []
        for surface, label in spans_by_text.items():
            start = text.find(surface)
            if start >= 0:  # like a real recognizer: only the first mention
                out.append((start, start + len(surface), label))
        return out

    return find


def test_name_variants_strip_suffixes_and_state_tags():
    v = anonymize.name_variants(ACME.names)
    assert {"ACME WIDGETS INC.", "ACME WIDGETS", "ACME HOLDINGS CORP", "ACME"} <= set(v)
    assert v == sorted(v, key=len, reverse=True)
    # Generic first words are not masked on their own.
    assert "AMERICAN" not in anonymize.name_variants(["AMERICAN EXPRESS CO"])
    assert "AMERICAN EXPRESS" in anonymize.name_variants(["AMERICAN EXPRESS CO"])


def test_own_name_ticker_years_and_dates_are_masked():
    text = ("Acme Widgets, Inc. (NYSE: ACMW) expects Acme's sales to grow in fiscal 2024. "
            "As of December 31, 2023, ACME had cash. FY23 results and the 1990s.")  # fmt: skip
    out, counts = anonymize.anonymize(text, ACME)
    assert out == ("[COMPANY] (NYSE: [TICKER]) expects [COMPANY]'s sales to grow in fiscal "
                   "[YEAR]. As of [DATE], [COMPANY] had cash. FY[YEAR] results and the "
                   "[YEAR].")  # fmt: skip
    assert counts["[COMPANY]"] == 3 and counts["[YEAR]"] == 3
    assert anonymize.leaks(out, ACME) == {"company_name": 0, "ticker": 0, "year": 0}


def test_numbers_that_are_not_years_survive():
    out, _ = anonymize.anonymize("Revenue was $2,023 million, 2023.5 tons and 12,000 units.", ACME)
    assert out == "Revenue was $2,023 million, 2023.5 tons and 12,000 units."


def test_short_ticker_masked_only_after_an_exchange_name():
    ident = anonymize.Identity(("GARTNER INC",), ("IT",))
    out, _ = anonymize.anonymize("IT spending rose. Our stock trades on NYSE: IT.", ident)
    assert out == "IT spending rose. Our stock trades on NYSE: [TICKER]."


def test_entities_masked_but_common_vocabulary_and_regulators_kept():
    text = ("Our customer Boeing said Jane Doe, chief executive, expects Consolidated Financial "
            "Statements and Goodwill notes reviewed by the SEC and FASB.")  # fmt: skip
    ner = fake_entities({"Boeing": "ORG", "Jane Doe": "PERSON", "SEC": "ORG", "FASB": "ORG",
                         "Consolidated Financial Statements": "ORG",
                         "Goodwill": "PERSON"})  # fmt: skip
    out, counts = anonymize.anonymize(text, ACME, ner, COMMON)
    assert "[ORGANIZATION] said [PERSON]" in out
    assert "Consolidated Financial Statements and Goodwill" in out
    assert "the SEC and FASB" in out
    assert counts == {"[ORGANIZATION]": 1, "[PERSON]": 1}


def test_name_masked_once_is_masked_everywhere():
    text = "We acquired Frutarom in 2018. Frutarom revenue rose and Frutarom markets grew."
    out, _ = anonymize.anonymize(text, ACME, fake_entities({"Frutarom": "ORG"}), COMMON)
    assert "Frutarom" not in out and out.count("[ORGANIZATION]") == 3


def test_unknown_capitalized_words_masked_places_and_compounds_kept():
    text = "Revenue from Worldpay and DMED rose in Brazil. Forward-looking Non-GAAP sales."
    ner = fake_entities({"Brazil": "GPE", "DMED": "GPE"})  # a recognizer error on DMED
    out, counts = anonymize.anonymize(text, ACME, ner, COMMON)
    assert out == ("Revenue from [NAME] and [NAME] rose in Brazil. Forward-looking Non-GAAP "
                   "sales.")  # fmt: skip
    assert counts["[NAME]"] == 2


def test_own_company_label_wins_inside_a_longer_entity():
    text = "Acme Widgets Group expects sales to grow."
    out, _ = anonymize.anonymize(text, ACME, fake_entities({"Acme Widgets Group": "ORG"}), COMMON)
    assert out.startswith("[COMPANY]")


@pytest.mark.parametrize(
    ("entity", "common"),
    [("Related Payables", True), ("COVID-19", True), ("Cigna Healthcare", False),
     ("The Boeing Company's", False)],
)  # fmt: skip
def test_is_common(entity, common):
    words = COMMON | {"RELATED", "PAYABLES", "COVID", "HEALTHCARE", "COMPANY"}
    assert anonymize.is_common(entity, words) == common
