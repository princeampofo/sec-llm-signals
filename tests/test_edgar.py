import gzip
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from secsignals import edgar

FIXTURES = Path(__file__).parent / "fixtures"
ARCHIVES = "https://www.sec.gov/Archives"


# ---------------------------------------------------------------- quarterly index


def test_parse_master_index():
    df = edgar.parse_master_index((FIXTURES / "master_sample.idx").read_bytes())
    assert list(df.columns) == edgar.INDEX_COLUMNS
    assert len(df) == 5
    row = df[df["accession"] == "0001140361-21-009786"].iloc[0]
    assert row["cik"] == 20639
    assert row["form_type"] == "10-K"
    assert row["date_filed"] == date(2021, 3, 24)


def test_parse_master_index_keeps_pipes_in_company_names():
    df = edgar.parse_master_index((FIXTURES / "master_sample.idx").read_bytes())
    row = df[df["cik"] == 99999].iloc[0]
    assert row["company"] == "PIPE | NAME CO"
    assert row["accession"] == "0000099999-21-000001"


def test_parse_master_index_rejects_other_files():
    with pytest.raises(ValueError):
        edgar.parse_master_index(b"<html>not an index</html>")


def test_quarter_is_complete():
    assert edgar.quarter_is_complete(2025, 3, date(2025, 10, 1))
    assert not edgar.quarter_is_complete(2025, 3, date(2025, 9, 30))
    assert edgar.quarter_is_complete(2025, 4, date(2026, 1, 1))
    assert not edgar.quarter_is_complete(2025, 4, date(2025, 12, 31))


class FakeClient:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.calls: list[tuple[str, bool]] = []

    def get(self, url: str, use_cache: bool = True) -> bytes:
        self.calls.append((url, use_cache))
        return self.raw


def test_build_filing_index_filters_forms_and_skips_future_quarters():
    client = FakeClient((FIXTURES / "master_sample.idx").read_bytes())
    df = edgar.build_filing_index(client, ARCHIVES, [2021], ["10-K"], today=date(2021, 5, 15))
    # Q1 and Q2 requested; Q3/Q4 have not started. Q2 is in progress, so no cache.
    assert [c[1] for c in client.calls] == [True, False]
    assert set(df["form_type"]) == {"10-K"}  # 10-Q and the 10-K/A amendment dropped
    assert not df.duplicated(["accession", "cik"]).any()  # fixture served twice, deduplicated


def test_build_filing_index_keeps_co_registrants():
    # NextEra and its subsidiary FPL file one combined 10-K; both CIKs must survive.
    raw = (FIXTURES / "master_sample.idx").read_bytes() + (
        b"753308|NEXTERA ENERGY INC|10-K|2021-02-12|"
        b"edgar/data/753308/0000753308-21-000014.txt\n"
        b"37634|FLORIDA POWER & LIGHT CO|10-K|2021-02-12|"
        b"edgar/data/37634/0000753308-21-000014.txt\n"
    )
    df = edgar.build_filing_index(FakeClient(raw), ARCHIVES, [2021], ["10-K"],
                                  today=date(2021, 3, 1))  # fmt: skip
    assert set(df.loc[df["accession"] == "0000753308-21-000014", "cik"]) == {753308, 37634}


def test_sample_filings_is_seeded_and_respects_years():
    index = pd.DataFrame(
        {
            "accession": [f"a{i}" for i in range(100)],
            "date_filed": [date(2015 + i % 10, 3, 1) for i in range(100)],
        }
    )
    a = edgar.sample_filings(index, 20, 2020, 2022, seed=1)
    b = edgar.sample_filings(index, 20, 2020, 2022, seed=1)
    assert a.equals(b)
    assert pd.to_datetime(a["date_filed"]).dt.year.between(2020, 2022).all()


# ---------------------------------------------------------------- filing index page


def test_parse_filing_index_page():
    html = (FIXTURES / "0000064040-25-000052-index.htm").read_bytes()
    meta = edgar.parse_filing_index_page(html, "10-K", ARCHIVES)
    assert meta.acceptance_datetime == pd.Timestamp("2025-02-11 17:29:21", tz=edgar.EASTERN)
    assert meta.period_of_report == date(2024, 12, 31)
    # The inline-XBRL viewer prefix (/ix?doc=) is stripped to get the raw document.
    assert meta.primary_doc_url == (
        "https://www.sec.gov/Archives/edgar/data/64040/000006404025000052/spgi-20241231.htm"
    )


def test_acceptance_datetime_is_eastern_across_dst():
    winter = edgar.parse_acceptance_datetime("2025-02-11 17:29:21")
    summer = edgar.parse_acceptance_datetime("2021-03-17 17:33:08")
    assert winter.utcoffset().total_seconds() == -5 * 3600
    assert summer.utcoffset().total_seconds() == -4 * 3600
    # 5:29 pm Eastern is after the 4 pm close: the filing was public only after hours.
    assert winter.hour == 17


def test_parse_filing_index_page_requires_accepted_field():
    with pytest.raises(ValueError, match="Accepted"):
        edgar.parse_filing_index_page(b"<html><body></body></html>", "10-K", ARCHIVES)


def test_filing_urls():
    assert edgar.filing_index_page_url(ARCHIVES, 20639, "0001140361-21-009786") == (
        f"{ARCHIVES}/edgar/data/20639/000114036121009786/0001140361-21-009786-index.htm"
    )
    assert edgar.index_url(ARCHIVES, 2021, 1) == f"{ARCHIVES}/edgar/full-index/2021/QTR1/master.idx"


# ---------------------------------------------------------------- rate limit, retries, cache


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_rate_limiter_spaces_requests():
    clock = FakeClock()
    limiter = edgar.RateLimiter(10, clock=clock, sleep=clock.sleep)
    times = []
    for _ in range(5):
        limiter.wait()
        times.append(clock.now)
    assert times == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_rate_limiter_does_not_sleep_when_idle():
    clock = FakeClock()
    limiter = edgar.RateLimiter(10, clock=clock, sleep=clock.sleep)
    limiter.wait()
    clock.now = 5.0
    limiter.wait()
    assert clock.now == 5.0


class FakeResponse:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code = status
        self.content = content


class FakeSession:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.headers_seen: list[dict] = []

    def get(self, url: str, headers: dict, timeout: float) -> FakeResponse:
        self.headers_seen.append(headers)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_client(tmp_path: Path, responses: list, sleeps: list | None = None) -> edgar.EdgarClient:
    return edgar.EdgarClient(
        user_agent="Test Person test@university.edu",
        cache_dir=tmp_path,
        max_requests_per_second=1000,
        max_retries=3,
        backoff_seconds=1.0,
        timeout_seconds=5,
        session=FakeSession(responses),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


URL = "https://www.sec.gov/Archives/edgar/data/1/0001/doc.htm"


def test_client_caches_downloads(tmp_path):
    client = make_client(tmp_path, [FakeResponse(200, b"hello")])
    assert client.get(URL) == b"hello"
    assert client.get(URL) == b"hello"  # second call served from disk; session would fail
    assert client.cache_path(URL).read_bytes() == b"hello"
    assert client.session.headers_seen[0]["User-Agent"] == "Test Person test@university.edu"


def test_client_use_cache_false_redownloads(tmp_path):
    client = make_client(tmp_path, [FakeResponse(200, b"v1"), FakeResponse(200, b"v2")])
    client.get(URL)
    assert client.get(URL, use_cache=False) == b"v2"
    assert client.cache_path(URL).read_bytes() == b"v2"


def test_client_retries_with_backoff(tmp_path):
    sleeps: list[float] = []
    responses = [FakeResponse(429), requests.ConnectionError("reset"), FakeResponse(200, b"ok")]
    client = make_client(tmp_path, responses, sleeps)
    assert client.get(URL) == b"ok"
    assert [s for s in sleeps if s >= 1] == [1.0, 2.0]  # exponential backoff


def test_client_gives_up_and_caches_nothing(tmp_path):
    client = make_client(tmp_path, [FakeResponse(503)] * 4)
    with pytest.raises(edgar.EdgarError, match="gave up"):
        client.get(URL)
    assert not client.cache_path(URL).exists()


def test_client_does_not_retry_404(tmp_path):
    client = make_client(tmp_path, [FakeResponse(404)])
    with pytest.raises(edgar.EdgarError, match="404"):
        client.get(URL)


def test_fetch_one_records_errors_instead_of_raising(tmp_path):
    client = make_client(tmp_path, [FakeResponse(404)])
    row = pd.Series({"accession": "0000000001-21-000001", "cik": 1, "form_type": "10-K"})
    out = edgar.fetch_one(client, ARCHIVES, row)
    assert "404" in out["fetch_error"]
    assert out["doc_path"] is None


def test_fetch_one_downloads_primary_document(tmp_path):
    page = (FIXTURES / "0000064040-25-000052-index.htm").read_bytes()
    doc = gzip.decompress((FIXTURES / "filings" / "0000064040-25-000052.htm.gz").read_bytes())
    client = make_client(tmp_path, [FakeResponse(200, page), FakeResponse(200, doc)])
    row = pd.Series({"accession": "0000064040-25-000052", "cik": 64040, "form_type": "10-K"})
    out = edgar.fetch_one(client, ARCHIVES, row)
    assert out["fetch_error"] is None
    assert Path(out["doc_path"]).read_bytes() == doc
