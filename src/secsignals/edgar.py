"""EDGAR quarterly index download, filing fetch, rate limiting and raw cache.

Timestamps: EDGAR reports acceptance datetimes in US Eastern time with no zone
marker. They are stored tz-aware (America/New_York) so later stages can compare
them to market sessions without guessing.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

EASTERN = "America/New_York"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EdgarError(RuntimeError):
    """A request EDGAR answered with a non-retryable error, or retries ran out."""


class RateLimiter:
    """Spaces calls at least 1/max_per_second apart, shared across threads."""

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        # Reserve a slot under the lock, then sleep outside it so other threads
        # can reserve the following slots meanwhile.
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval
        if slot > now:
            self._sleep(slot - now)


class EdgarClient:
    """GET with a User-Agent, rate limit, retries and an on-disk cache.

    Cached files mirror the URL path under cache_dir, so a cached download is
    the exact bytes EDGAR served and can be inspected by hand.
    """

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        max_requests_per_second: float,
        max_retries: int,
        backoff_seconds: float,
        timeout_seconds: float,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if "example.com" in user_agent:
            log.warning("SEC_USER_AGENT not set; SEC asks for a real name and email")
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.cache_dir = cache_dir
        self.limiter = RateLimiter(max_requests_per_second, sleep=sleep)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self._sleep = sleep

    def cache_path(self, url: str) -> Path:
        parts = urlsplit(url)
        path = self.cache_dir / parts.netloc / parts.path.lstrip("/")
        if parts.query:  # e.g. a pinned Wikipedia revision: ?oldid=...
            path = path.with_name(f"{path.name}__{quote(parts.query, safe='')}")
        return path

    def get(self, url: str, use_cache: bool = True) -> bytes:
        path = self.cache_path(url)
        if use_cache and path.exists():
            return path.read_bytes()
        data = self._download(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic: an interrupted run never leaves a half file
        return data

    def _download(self, url: str) -> bytes:
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, headers=self.headers, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                reason = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code not in RETRYABLE_STATUS:
                    raise EdgarError(f"HTTP {resp.status_code} for {url}")
                reason = f"HTTP {resp.status_code}"
            if attempt < self.max_retries:
                delay = self.backoff_seconds * 2**attempt
                log.info("retry %d for %s (%s), sleeping %.1fs", attempt + 1, url, reason, delay)
                self._sleep(delay)
        raise EdgarError(f"gave up on {url} after {self.max_retries + 1} tries ({reason})")


# ---------------------------------------------------------------- quarterly index

INDEX_COLUMNS = ["cik", "company", "form_type", "date_filed", "accession", "filename"]


def index_url(archives_url: str, year: int, quarter: int) -> str:
    return f"{archives_url}/edgar/full-index/{year}/QTR{quarter}/master.idx"


def parse_master_index(raw: bytes) -> pd.DataFrame:
    """Parse master.idx: 'CIK|Company Name|Form Type|Date Filed|Filename' rows."""
    lines = raw.decode("latin-1").splitlines()
    try:
        header = next(i for i, line in enumerate(lines) if line.startswith("CIK|"))
    except StopIteration as exc:
        raise ValueError("no CIK| header row; not a master.idx file") from exc
    rows = []
    for line in lines[header + 2 :]:  # skip header and the dashed separator
        if line.count("|") < 4:
            continue
        cik, rest = line.split("|", 1)
        company, form_type, date_filed, filename = rest.rsplit("|", 3)  # names may hold '|'
        accession = filename.rsplit("/", 1)[-1].removesuffix(".txt")
        rows.append((int(cik), company.strip(), form_type.strip(), date_filed, accession, filename))
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    df["date_filed"] = pd.to_datetime(df["date_filed"]).dt.date
    return df


def quarter_is_complete(year: int, quarter: int, today: date) -> bool:
    end_month = quarter * 3
    first_after = date(year + (end_month == 12), end_month % 12 + 1, 1)
    return today >= first_after


def build_filing_index(
    client: EdgarClient,
    archives_url: str,
    years: Iterable[int],
    forms: list[str],
    today: date | None = None,
) -> pd.DataFrame:
    """All filings of the given form types across the given years' quarterly indexes."""
    today = today or date.today()
    frames = []
    for year in years:
        for quarter in range(1, 5):
            if date(year, quarter * 3 - 2, 1) > today:
                continue
            # An in-progress quarter's index still grows, so never trust its cache.
            complete = quarter_is_complete(year, quarter, today)
            raw = client.get(index_url(archives_url, year, quarter), use_cache=complete)
            df = parse_master_index(raw)
            frames.append(df[df["form_type"].isin(forms)])
            log.info("%d Q%d: %d %s filings", year, quarter, len(frames[-1]), "/".join(forms))
    out = pd.concat(frames, ignore_index=True)
    # A combined 10-K (parent + subsidiaries, common for utilities) has one row per
    # co-registrant. Keep them all: each CIK really did file it.
    out = out.drop_duplicates(["accession", "cik"])
    return out.sort_values(["date_filed", "cik"], ignore_index=True)


def sample_filings(
    index: pd.DataFrame, size: int, start_year: int, end_year: int, seed: int
) -> pd.DataFrame:
    index = index.drop_duplicates("accession")  # one draw per document, not per co-registrant
    years = pd.to_datetime(index["date_filed"]).dt.year
    pool = index[(years >= start_year) & (years <= end_year)]
    return pool.sample(n=min(size, len(pool)), random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- filing index page


@dataclass(frozen=True)
class FilingMeta:
    acceptance_datetime: pd.Timestamp
    period_of_report: date | None
    primary_doc_url: str | None


def filing_folder_url(archives_url: str, cik: int, accession: str) -> str:
    return f"{archives_url}/edgar/data/{cik}/{accession.replace('-', '')}"


def filing_index_page_url(archives_url: str, cik: int, accession: str) -> str:
    return f"{filing_folder_url(archives_url, cik, accession)}/{accession}-index.htm"


def parse_acceptance_datetime(text: str) -> pd.Timestamp:
    """'2023-11-02 18:08:27' (Eastern, as EDGAR shows it) -> tz-aware Timestamp."""
    ts = pd.Timestamp(text.strip())
    return ts.tz_localize(EASTERN)


def parse_filing_index_page(html: bytes, form_type: str, archives_url: str) -> FilingMeta:
    """Read acceptance time, period and the primary document link off -index.htm."""
    soup = BeautifulSoup(html, "lxml")
    info: dict[str, str] = {}
    for head in soup.find_all("div", class_="infoHead"):
        value = head.find_next_sibling("div", class_="info")
        if value is not None:
            info[head.get_text(strip=True)] = value.get_text(strip=True)
    if "Accepted" not in info:
        raise ValueError("filing index page has no 'Accepted' field")

    period = info.get("Period of Report")
    host = f"{urlsplit(archives_url).scheme}://{urlsplit(archives_url).netloc}"
    primary = None
    table = soup.find("table", class_="tableFile", summary="Document Format Files")
    if table is not None:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            link = row.find("a", href=True)
            if len(cells) < 4 or link is None:
                continue
            if cells[3].get_text(strip=True) == form_type:
                href = re.sub(r"^/ix\?doc=", "", link["href"])  # inline-XBRL viewer wrapper
                primary = host + href
                break
    return FilingMeta(
        acceptance_datetime=parse_acceptance_datetime(info["Accepted"]),
        period_of_report=date.fromisoformat(period) if period else None,
        primary_doc_url=primary,
    )


def fetch_one(
    client: EdgarClient, archives_url: str, row: pd.Series, with_document: bool = True
) -> dict[str, object]:
    out: dict[str, object] = {
        "accession": row["accession"],
        "acceptance_datetime": pd.NaT,
        "period_of_report": None,
        "primary_doc_url": None,
        "doc_path": None,
        "fetch_error": None,
    }
    try:
        page = client.get(filing_index_page_url(archives_url, row["cik"], row["accession"]))
        meta = parse_filing_index_page(page, row["form_type"], archives_url)
        out["acceptance_datetime"] = meta.acceptance_datetime
        out["period_of_report"] = meta.period_of_report
        out["primary_doc_url"] = meta.primary_doc_url
        if meta.primary_doc_url is None:
            raise ValueError(f"no document of type {row['form_type']} in filing index")
        if with_document:
            client.get(meta.primary_doc_url)
            out["doc_path"] = str(client.cache_path(meta.primary_doc_url))
    except (EdgarError, ValueError) as exc:
        out["fetch_error"] = str(exc)
        log.warning("fetch failed for %s: %s", row["accession"], exc)
    return out


def fetch_filings(
    client: EdgarClient,
    archives_url: str,
    index: pd.DataFrame,
    workers: int,
    with_documents: bool = True,
) -> pd.DataFrame:
    """Download each filing's index page (and primary document); returns metadata rows.

    with_documents=False reads only the small index pages, which is enough for
    acceptance timestamps; documents can be fetched later from the same cache.
    """
    rows = [r for _, r in index.drop_duplicates("accession").iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(lambda r: fetch_one(client, archives_url, r, with_documents), rows)
        )
    meta = pd.DataFrame(results)
    meta["acceptance_datetime"] = pd.to_datetime(meta["acceptance_datetime"]).dt.tz_convert(
        EASTERN
    )
    return index.merge(meta, on="accession", how="left", validate="many_to_one")


def client_from_config(cfg: dict) -> EdgarClient:
    e = cfg["edgar"]
    return EdgarClient(
        user_agent=e["user_agent"],
        cache_dir=Path(cfg["paths"]["raw"]),
        max_requests_per_second=e["max_requests_per_second"],
        max_retries=e["max_retries"],
        backoff_seconds=e["backoff_seconds"],
        timeout_seconds=e["timeout_seconds"],
    )
