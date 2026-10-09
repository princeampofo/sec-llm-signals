"""Score MD&A excerpts with a local LLM: tone, hedging and risk severity in [-1, 1].

Pipeline per filing:
1. Excerpt: about `budget` tokens of forward-looking paragraphs (ones that say expect,
   anticipate, outlook or believe), in document order. If the section has fewer than
   `min_forward` tokens of those, the section's opening is used instead.
2. Prompt (versioned): asks for a JSON object with exactly the three scores. Greedy
   decoding (temperature 0), so the same excerpt, prompt and model give the same answer.
3. Validation: the reply must parse as JSON with exactly the expected keys, each a
   number in [-1, 1]. An invalid reply gets one corrective follow-up; if that also
   fails, the filing is logged as invalid with the raw output.
4. Cache: every result (valid or not) is stored in SQLite under (accession, section,
   prompt version, model) and committed immediately, so a run can stop and resume and
   a rerun makes no model calls.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

log = logging.getLogger(__name__)

SCORE_KEYS = ("tone", "hedging", "risk_severity")
FORWARD_LOOKING = re.compile(r"\b(?:expect\w*|anticipat\w*|outlook\w*|believ\w*)\b", re.I)
MIN_PARAGRAPH_WORDS = 8  # shorter lines are headings or table debris

SYSTEM_PROMPT = (
    "You are a financial analyst who rates the language of annual-report excerpts. "
    "You reply with a JSON object only."
)
USER_PROMPT = """Below is an excerpt from the Management's Discussion and Analysis section of a \
company's annual report (Form 10-K).

Rate the excerpt on three scales from -1 to 1:
- "tone": -1 = very negative, 0 = neutral, 1 = very positive.
- "hedging": -1 = confident and definite, 0 = typical, 1 = heavily hedged and uncertain.
- "risk_severity": -1 = minor or routine risks, 0 = moderate, 1 = severe risks to the business.
{extra}
Reply with only a JSON object with exactly these keys, for example:
{example}

Excerpt:
\"\"\"
{excerpt}
\"\"\""""
JUSTIFY_EXTRA = (
    '- "justification": one or two sentences explaining the three ratings, '
    "quoting the excerpt where useful.\n"
)
RETRY_PROMPT = (
    "That reply was not a valid JSON object with exactly the keys {keys}, each a number "
    "from -1 to 1. Reply with only the JSON object."
)


# ---------------------------------------------------------------- excerpt


def _paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n") if len(p.split()) >= MIN_PARAGRAPH_WORDS]


def truncate_to_tokens(text: str, budget: int, count_tokens: Callable[[str], int]) -> str:
    """Longest word prefix of text within the token budget."""
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:  # binary search on the number of words
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def _fill(paragraphs: Iterable[str], budget: int,
          count_tokens: Callable[[str], int]) -> tuple[list[str], int]:  # fmt: skip
    chosen, used = [], 0
    for p in paragraphs:
        n = count_tokens(p)
        if used + n <= budget:
            chosen.append(p)
            used += n
        else:
            rest = truncate_to_tokens(p, budget - used, count_tokens)
            if rest:
                chosen.append(rest)
                used += count_tokens(rest)
            break
    return chosen, used


def select_excerpt(text: str, budget: int, min_forward: int,
                   count_tokens: Callable[[str], int]) -> tuple[str, str, int]:  # fmt: skip
    """(excerpt, method, tokens). method is "forward_looking" or "opening_fallback"."""
    paragraphs = _paragraphs(text)
    forward = [p for p in paragraphs if FORWARD_LOOKING.search(p)]
    chosen, used = _fill(forward, budget, count_tokens)
    method = "forward_looking"
    if used < min_forward:
        chosen, used = _fill(paragraphs, budget, count_tokens)
        method = "opening_fallback"
    return "\n\n".join(chosen), method, used


# ---------------------------------------------------------------- prompt and validation


def build_messages(excerpt: str, justify: bool) -> list[dict[str, str]]:
    example = {"tone": 0.2, "hedging": -0.1, "risk_severity": 0.3}
    if justify:
        example["justification"] = "..."
    user = USER_PROMPT.format(extra=JUSTIFY_EXTRA if justify else "",
                              example=json.dumps(example), excerpt=excerpt)  # fmt: skip
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_scores(raw: str, justify: bool) -> tuple[dict | None, str | None]:
    """(scores, None) for a valid reply, else (None, reason)."""
    match = re.search(r"\{.*\}", raw, re.S)
    if match is None:
        return None, "no JSON object"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(obj, dict):
        return None, "not an object"
    expected = set(SCORE_KEYS) | ({"justification"} if justify else set())
    if set(obj) != expected:
        return None, f"keys {sorted(obj)} != {sorted(expected)}"
    for key in SCORE_KEYS:
        v = obj[key]
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            return None, f"{key} is not a number"
        if not -1 <= v <= 1:
            return None, f"{key}={v} outside [-1, 1]"
    if justify and not isinstance(obj["justification"], str):
        return None, "justification is not text"
    return {k: (float(obj[k]) if k in SCORE_KEYS else obj[k]) for k in obj}, None


# ---------------------------------------------------------------- backends


class Backend(Protocol):
    model_id: str  # goes into the cache key: model + exact version (+ quantization)

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str: ...
    def count_tokens(self, text: str) -> int: ...


class MLXBackend:
    """Local Apple-silicon inference with mlx-lm; greedy decoding (temperature 0).

    Prefix reuse: the system prompt and instructions (~220 of ~720 prompt tokens) are the
    same for every filing. Their attention keys/values are computed once and kept; each
    call processes only the rest of its prompt, then the cache is trimmed back to the
    prefix. The prefix is the token sequence shared by the first two prompts, and it is
    reused only when a prompt starts with exactly those tokens.
    """

    def __init__(self, repo: str, revision: str, reuse_prefix: bool = True) -> None:
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        path = snapshot_download(repo, revision=revision)
        self.model, self.tokenizer = load(path)
        self.model_id = f"{repo}@{revision[:7]}"
        self.reuse_prefix = reuse_prefix
        self._first: list[int] | None = None
        self._prefix: list[int] = []
        self._prefix_cache: list | None = None

    def _ids(self, messages: list[dict[str, str]]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                 tokenize=True)  # fmt: skip
        return list(ids["input_ids"] if isinstance(ids, dict) else ids)

    def _build_prefix(self, a: list[int], b: list[int]) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        shared = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                      min(len(a), len(b)))  # fmt: skip
        n = shared - 1  # always leave at least one prompt token to process per call
        if n < 16:
            return
        cache = make_prompt_cache(self.model)
        self.model(mx.array(a[:n])[None], cache=cache)
        mx.eval([c.state for c in cache])
        self._prefix, self._prefix_cache = a[:n], cache

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.models.cache import trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        ids = self._ids(messages)
        kwargs = {"max_tokens": max_new_tokens, "sampler": make_sampler(temp=0.0),
                  "verbose": False}  # fmt: skip
        if self.reuse_prefix and self._prefix_cache is None:
            if self._first is None:
                self._first = ids
            else:
                self._build_prefix(self._first, ids)
        n = len(self._prefix)
        if self._prefix_cache is not None and ids[:n] == self._prefix:
            try:
                return generate(self.model, self.tokenizer, ids[n:],
                                prompt_cache=self._prefix_cache, **kwargs)  # fmt: skip
            finally:
                trim_prompt_cache(self._prefix_cache, self._prefix_cache[0].offset - n)
        return generate(self.model, self.tokenizer, ids, **kwargs)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class OpenAICompatibleBackend:
    """Any server with an OpenAI-style /chat/completions endpoint (e.g. vLLM on a rented
    GPU). Token counts use the same model's Hugging Face tokenizer."""

    def __init__(self, base_url: str, model: str, tokenizer_repo: str, revision: str,
                 timeout: float = 120) -> None:  # fmt: skip
        from transformers import AutoTokenizer

        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, revision=revision)
        self.model_id = f"{model}@{revision[:7]}"
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        resp = requests.post(self.url, timeout=self.timeout, json={
            "model": self.model, "messages": messages, "temperature": 0.0,
            "max_tokens": max_new_tokens})  # fmt: skip
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def backend_from_config(cfg: dict) -> Backend:
    c = cfg["llm"]
    if c["backend"] == "mlx":
        return MLXBackend(c["model"], c["model_revision"])
    if c["backend"] == "openai_compatible":
        return OpenAICompatibleBackend(c["base_url"], c["served_model_name"],
                                       c["tokenizer_repo"], c["model_revision"])  # fmt: skip
    raise ValueError(f"unknown llm backend {c['backend']!r}")


def token_counter_from_config(cfg: dict) -> Callable[[str], int]:
    """The scoring model's token counter, without loading the model weights."""
    from transformers import AutoTokenizer

    c = cfg["llm"]
    if c["backend"] == "mlx":
        from huggingface_hub import snapshot_download

        path = snapshot_download(c["model"], revision=c["model_revision"],
                                 allow_patterns=["*.json", "*.model", "*.txt"])  # fmt: skip
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(c["tokenizer_repo"], revision=c["model_revision"])
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------- cache

CACHE_COLUMNS = [
    "accession", "section", "prompt_version", "model", "status", "tone", "hedging",
    "risk_severity", "justification", "excerpt_method", "excerpt_tokens", "attempts",
    "error", "raw_output", "seconds", "created_at",
]  # fmt: skip


class ScoreCache:
    """SQLite store of every model result, keyed by (accession, section, prompt_version,
    model). Each insert is committed at once, so an interrupted run loses nothing."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        cols = ", ".join(f"{c} {'REAL' if c in SCORE_KEYS + ('seconds',) else 'TEXT'}"
                         for c in CACHE_COLUMNS)  # fmt: skip
        self.db.execute(f"CREATE TABLE IF NOT EXISTS scores ({cols}, PRIMARY KEY "
                        "(accession, section, prompt_version, model))")  # fmt: skip
        self.db.commit()

    def keys(self, prompt_version: str, model: str) -> set[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT accession, section FROM scores WHERE prompt_version=? AND model=?",
            (prompt_version, model),
        )
        return set(rows.fetchall())

    def put(self, record: dict) -> None:
        values = [record.get(c) for c in CACHE_COLUMNS]
        marks = ", ".join("?" * len(values))
        self.db.execute(f"INSERT OR REPLACE INTO scores VALUES ({marks})", values)
        self.db.commit()

    def frame(self, prompt_version: str, model: str, section: str = "mda") -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM scores WHERE prompt_version=? AND model=? AND section=?",
            self.db, params=(prompt_version, model, section))  # fmt: skip


# ---------------------------------------------------------------- scoring run


@dataclass
class RunStats:
    to_score: int = 0
    cached: int = 0
    model_calls: int = 0
    valid: int = 0
    invalid: int = 0
    seconds: float = 0.0


def score_one(backend: Backend, excerpt: str, justify: bool, max_new_tokens: int,
              max_retries: int) -> tuple[dict | None, str | None, str, int]:  # fmt: skip
    """(scores or None, error, last raw output, attempts)."""
    messages = build_messages(excerpt, justify)
    raw, error = "", None
    for attempt in range(1, max_retries + 2):
        raw = backend.generate(messages, max_new_tokens)
        scores, error = parse_scores(raw, justify)
        if scores is not None:
            return scores, None, raw, attempt
        keys = ", ".join(SCORE_KEYS + (("justification",) if justify else ()))
        follow_up = {"role": "user", "content": RETRY_PROMPT.format(keys=keys)}
        messages = [*messages, {"role": "assistant", "content": raw}, follow_up]
    return None, error, raw, max_retries + 1


def score_sections(
    sections: pd.DataFrame,
    backend: Backend,
    cache: ScoreCache,
    prompt_version: str,
    budget: int,
    min_forward: int,
    max_new_tokens: int,
    max_retries: int,
    justify: bool = False,
    limit: int | None = None,
    progress_every: int = 25,
    section: str = "mda",
) -> RunStats:
    """Score every section not already in the cache (one model call per new filing).

    sections has either a "text" column (the excerpt is selected here) or ready-made
    "excerpt" and "excerpt_method" columns (e.g. anonymized excerpts, cached under their
    own section name).
    """
    version = prompt_version + ("-justify" if justify else "")
    done = cache.keys(version, backend.model_id)
    todo = sections[[(a, section) not in done for a in sections["accession"]]]
    stats = RunStats(to_score=len(sections), cached=len(sections) - len(todo))
    if limit is not None:
        todo = todo.head(limit)
    started = time.monotonic()
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        if "excerpt" in sections:
            excerpt, method = row.excerpt, row.excerpt_method
            tokens = backend.count_tokens(excerpt)
        else:
            excerpt, method, tokens = select_excerpt(row.text, budget, min_forward,
                                                     backend.count_tokens)  # fmt: skip
        t0 = time.monotonic()
        scores, error, raw, attempts = score_one(backend, excerpt, justify, max_new_tokens,
                                                 max_retries)  # fmt: skip
        stats.model_calls += attempts
        record = {"accession": row.accession, "section": section, "prompt_version": version,
                  "model": backend.model_id, "status": "valid" if scores else "invalid",
                  "excerpt_method": method, "excerpt_tokens": tokens, "attempts": attempts,
                  "error": error, "raw_output": raw, "seconds": time.monotonic() - t0,
                  "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                  **(scores or {})}  # fmt: skip
        cache.put(record)
        stats.valid += scores is not None
        stats.invalid += scores is None
        if i % progress_every == 0 or i == len(todo):
            rate = (time.monotonic() - started) / i
            log.info("scored %d/%d new (%.1f s/filing, ~%.0f min left), %d invalid",
                     i, len(todo), rate, rate * (len(todo) - i) / 60, stats.invalid)  # fmt: skip
    stats.seconds = time.monotonic() - started
    return stats
