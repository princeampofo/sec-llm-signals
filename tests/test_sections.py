import gzip
from pathlib import Path

import pandas as pd
import pytest

from secsignals import sections

FILINGS = Path(__file__).parent / "fixtures" / "filings"
MIN_WORDS = 250
MAX_HEADING = 200


def load(accession: str) -> bytes:
    return gzip.decompress((FILINGS / f"{accession}.htm.gz").read_bytes())


def extract(raw: bytes) -> sections.Extraction:
    return sections.extract_mda(raw, MIN_WORDS, MAX_HEADING)


# ---------------------------------------------------------------- 5 saved filings

SAVED = [
    # accession, why it is here, method, expected start, expected end
    pytest.param(
        "0000064040-25-000052", "item_heading",
        "The following Management’s Discussion and Analysis",
        "liquidity in future periods.",
        id="spgi-standard-large-cap",
    ),
    pytest.param(
        "0001091818-23-000200", "item_heading",
        "7. Management’s Discussion and Analysis",
        "We\ndo not have any off-balance sheet arrangements.",
        id="cleartronic-item-and-number-on-separate-lines",
    ),
    pytest.param(
        "0001493152-22-008511", "title_fallback",
        "DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION",
        "currently do not have any off-balance sheet arrangements.",
        id="therapeutic-heading-missing-item-number",
    ),
    pytest.param(
        "0001140361-21-009786", "item_heading",
        "MANAGEMENT’S DISCUSSION AND ANALYSIS",
        "there is no assurance that the Company’s expectations will be realized.",
        id="ambase-item8-cross-reference-inside-mda",
    ),
]  # fmt: skip


@pytest.mark.parametrize(("accession", "method", "start", "end"), SAVED)
def test_saved_filing_extracts_mda(accession, method, start, end):
    result = extract(load(accession))
    assert result.failure is None
    assert result.method == method
    assert result.text.startswith(start)
    assert result.text.endswith(end)
    assert result.n_words >= MIN_WORDS
    # Nothing from Item 7A or Item 8 leaked in.
    assert "QUANTITATIVE AND QUALITATIVE DISCLOSURES" not in result.text.upper()
    assert "REPORT OF INDEPENDENT REGISTERED" not in result.text.upper()


def test_saved_large_cap_has_full_section():
    result = extract(load("0000064040-25-000052"))
    assert 15_000 < result.n_words < 30_000  # S&P Global's MD&A, not the TOC entry
    assert "Results of Operations" in result.text


def test_saved_abs_trust_is_logged_as_no_mda():
    # Asset-backed issuers omit Item 7 under Regulation AB; this must be a logged failure.
    result = extract(load("0001056404-21-002865"))
    assert result.text is None
    assert result.failure == "abs_issuer_no_mda"


# ---------------------------------------------------------------- HTML -> text


def test_html_to_text_drops_hidden_xbrl_scripts_and_numeric_tables():
    rows = "".join(f"<tr><td>Revenue</td><td>{i},123,456</td></tr>" for i in range(20))
    html = f"""<html><head><title>t</title><script>var x=1;</script></head><body>
    <ix:header><ix:hidden>HIDDEN FACT 123</ix:hidden></ix:header>
    <div style="display:none">invisible</div>
    <p>First&nbsp;paragraph.</p><div>Second <b>bold</b> paragraph.</div>
    <table>{rows}</table>
    <table><tr><td>Item 7.</td><td>Management's Discussion</td></tr></table>
    <p>12</p><p>Table of Contents</p>
    </body></html>""".encode()
    text = sections.html_to_text(html)
    assert text.split("\n") == [
        "First paragraph.",
        "Second bold paragraph.",
        "Item 7. Management's Discussion",  # short heading table kept, cells joined
    ]


def test_html_to_text_plain_text_filing():
    assert sections.html_to_text(b"ITEM 7.  MD&A\n\n  Revenue rose.  \n") == (
        "ITEM 7. MD&A\nRevenue rose."
    )


# ---------------------------------------------------------------- locating Item 7

BODY = " ".join(["Revenue increased because demand was strong."] * 60)  # 360 words


def doc(*lines: str) -> bytes:
    return ("<html><body>" + "".join(f"<p>{x}</p>" for x in lines) + "</body></html>").encode()


TOC = [
    "Item 1. Business 3", "Item 1A. Risk Factors 9",
    "Item 7. Management's Discussion and Analysis 30",
    "Item 7A. Quantitative and Qualitative Disclosures About Market Risk 45",
    "Item 8. Financial Statements and Supplementary Data 46",
]  # fmt: skip


def test_skips_table_of_contents():
    raw = doc(
        *TOC,
        "Item 1. Business", "We make widgets.",
        "Item 7. Management's Discussion and Analysis", BODY,
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk", "Rates.",
        "Item 8. Financial Statements and Supplementary Data", "Balance sheet.",
    )  # fmt: skip
    result = extract(raw)
    assert result.text == BODY
    assert result.method == "item_heading"


def test_toc_without_item_word_does_not_swallow_items_1_to_6():
    # TOC lists "7A." without "Item", so the TOC "Item 7" never closes; the
    # following "Item 1" heading must cancel it.
    raw = doc(
        "Item 7. Management's Discussion and Analysis 30", "7A. Quantitative 45",
        "Item 1. Business", "We make widgets. " * 400,
        "Item 7. Management's Discussion and Analysis", BODY,
        "Item 8. Financial Statements and Supplementary Data",
    )  # fmt: skip
    assert extract(raw).text == BODY


def test_running_page_headers_do_not_split_section():
    raw = doc(
        "Item 7. Management's Discussion and Analysis", BODY,
        "Item 7. Management's Discussion and Analysis (continued)", BODY,
        "Item 8. Financial Statements and Supplementary Data",
    )  # fmt: skip
    header = "Item 7. Management's Discussion and Analysis (continued)"
    assert extract(raw).text == f"{BODY}\n{header}\n{BODY}"  # header kept, not a split


def test_cross_reference_lines_are_not_headings():
    # Short lines that start with "Item N" but continue as prose (NI Holdings,
    # Great Southern, AmBase) must neither close nor cancel the section.
    cross_refs = [
        "Item 8 of this Report.",
        "Part II, Item 8, Note 4 “Investments”.",
        "Item 8 – Note 7 to the Company’s consolidated financial statements.",
        "Item 1A of our prior report describes risks.",
    ]
    raw = doc(
        "Item 7. Management's Discussion and Analysis", BODY, *cross_refs, BODY,
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
    )  # fmt: skip
    assert extract(raw).text == "\n".join([BODY, *cross_refs, BODY])


def test_item_and_number_on_separate_lines():
    raw = doc("ITEM", "7.", "MANAGEMENT'S DISCUSSION AND ANALYSIS", BODY, "ITEM", "8.",
              "FINANCIAL STATEMENTS")  # fmt: skip
    assert extract(raw).text.endswith(BODY)


def test_combined_items_7_and_7a():
    raw = doc("Items 7 and 7A. Management's Discussion and Analysis and Market Risk", BODY,
              "Item 8. Financial Statements")  # fmt: skip
    assert extract(raw).text == BODY


def test_title_fallback_for_cross_reference_layout():
    raw = doc("Management's Discussion and Analysis", BODY,
              "Report of Independent Registered Public Accounting Firm")  # fmt: skip
    result = extract(raw)
    assert result.method == "title_fallback"
    assert result.text == BODY


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        (["Item 1. Business", BODY], "mda_heading_not_found"),
        (["Item 7. Management's Discussion and Analysis",
          "The information required is incorporated herein by reference to Exhibit 13.",
          "Item 8. Financial Statements"], "incorporated_by_reference"),
        (["Item 7. Management's Discussion and Analysis", "Short.",
          "Item 8. Financial Statements"], "section_too_short"),
    ],
)  # fmt: skip
def test_failure_reasons(lines, reason):
    result = extract(doc(*lines))
    assert result.text is None
    assert result.failure == reason


# ---------------------------------------------------------------- pipeline stage


def test_extract_sections_logs_every_failure(tmp_path):
    good = tmp_path / "good.htm"
    good.write_bytes(doc("Item 7. Management's Discussion and Analysis", BODY,
                         "Item 8. Financial Statements"))  # fmt: skip
    broken = tmp_path / "broken.htm"
    broken.write_bytes(doc("Item 1. Business", BODY))
    ts = pd.Timestamp("2021-03-01 16:30", tz="America/New_York")
    filings = pd.DataFrame(
        {
            "accession": ["a", "b", "c"],
            "cik": [1, 2, 3],
            "form_type": ["10-K"] * 3,
            "acceptance_datetime": [ts] * 3,
            "doc_path": [str(good), str(broken), None],
            "fetch_error": [None, None, "HTTP 404"],
        }
    )
    out = sections.extract_sections(filings, MIN_WORDS, MAX_HEADING)
    assert list(out["status"]) == ["ok", "failed", "failed"]
    assert out.loc[1, "failure_reason"] == "mda_heading_not_found"
    assert out.loc[2, "failure_reason"].startswith("fetch_failed")
    assert out.loc[0, "acceptance_datetime"] == ts  # timestamp carried through untouched
