import json

import pandas as pd
import pytest

from secsignals import llm_scorer, signals


def words(text: str) -> int:
    """Stand-in tokenizer: one token per word."""
    return len(text.split())


FWD = "We expect revenue growth to continue next year as demand for our products rises."
OTHER = "Revenue increased by four percent compared with the prior fiscal year overall."


# ---------------------------------------------------------------- excerpt selection


def test_excerpt_takes_forward_looking_paragraphs_in_order():
    text = "\n".join(["Overview", OTHER, FWD, OTHER, FWD.replace("expect", "anticipate")])
    excerpt, method, tokens = llm_scorer.select_excerpt(text, budget=100, min_forward=20,
                                                        count_tokens=words)  # fmt: skip
    assert method == "forward_looking"
    assert excerpt.split("\n\n") == [FWD, FWD.replace("expect", "anticipate")]
    assert tokens == 2 * words(FWD)


def test_excerpt_respects_budget_and_truncates_last_paragraph():
    text = "\n".join([FWD] * 10)
    excerpt, _, tokens = llm_scorer.select_excerpt(text, budget=40, min_forward=10,
                                                   count_tokens=words)  # fmt: skip
    assert tokens == 40 and words(excerpt) == 40
    left = 40 - 2 * words(FWD)  # two whole paragraphs fit; the third is cut to what is left
    assert excerpt.split("\n\n")[-1] == " ".join(FWD.split()[:left])


def test_excerpt_falls_back_to_opening_without_enough_forward_text():
    text = "\n".join([OTHER, OTHER, FWD, OTHER])
    excerpt, method, _ = llm_scorer.select_excerpt(text, budget=30, min_forward=20,
                                                   count_tokens=words)  # fmt: skip
    assert method == "opening_fallback"
    assert excerpt.startswith(OTHER)


def test_forward_looking_keywords():
    for word in ["expects", "anticipated", "Outlook", "believe", "expectations"]:
        assert llm_scorer.FORWARD_LOOKING.search(f"Management {word} things.")
    assert not llm_scorer.FORWARD_LOOKING.search("Unexpected costs rose.")


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("raw", "ok"),
    [
        ('{"tone": 0.2, "hedging": -0.1, "risk_severity": 0.3}', True),
        ('Sure! {"tone": 1, "hedging": 0, "risk_severity": -1} Hope that helps.', True),
        ('{"tone": 0.2, "hedging": -0.1}', False),  # missing key
        ('{"tone": 0.2, "hedging": -0.1, "risk_severity": 0.3, "note": "x"}', False),  # extra key
        ('{"tone": 1.5, "hedging": 0, "risk_severity": 0}', False),  # out of range
        ('{"tone": true, "hedging": 0, "risk_severity": 0}', False),  # bool is not a score
        ('{"tone": "0.2", "hedging": 0, "risk_severity": 0}', False),  # string
        ('{"tone": NaN, "hedging": 0, "risk_severity": 0}', False),  # not finite
        ("The tone is positive.", False),  # no JSON
        ('{"tone": 0.2, "hedging": ', False),  # truncated
    ],
)
def test_parse_scores(raw, ok):
    scores, error = llm_scorer.parse_scores(raw, justify=False)
    assert (scores is not None) == ok
    assert (error is None) == ok


def test_parse_scores_with_justification():
    raw = '{"tone": -0.5, "hedging": 0.4, "risk_severity": 0.6, "justification": "Cites risk."}'
    scores, _ = llm_scorer.parse_scores(raw, justify=True)
    assert scores["justification"] == "Cites risk." and scores["tone"] == -0.5
    assert llm_scorer.parse_scores(raw, justify=False)[0] is None


def test_prompt_mentions_every_key_and_the_excerpt():
    msgs = llm_scorer.build_messages("EXCERPT TEXT", justify=False)
    user = msgs[-1]["content"]
    assert all(f'"{k}"' in user for k in llm_scorer.SCORE_KEYS) and "EXCERPT TEXT" in user
    assert "justification" not in user
    assert "justification" in llm_scorer.build_messages("x", justify=True)[-1]["content"]


# ---------------------------------------------------------------- scoring run and cache


class FakeBackend:
    def __init__(self, replies, model_id="fake@0000000"):
        self.replies = list(replies)
        self.model_id = model_id
        self.calls = []

    def generate(self, messages, max_new_tokens):
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else json.dumps(
            {"tone": 0.1, "hedging": 0.2, "risk_severity": 0.3})

    def count_tokens(self, text):
        return words(text)


def sections(n):
    return pd.DataFrame({"accession": [f"a{i}" for i in range(n)], "text": [FWD * 3] * n})


def run(backend, cache, secs, **kw):
    return llm_scorer.score_sections(secs, backend, cache, "v1", budget=100, min_forward=10,
                                     max_new_tokens=60, max_retries=1, **kw)  # fmt: skip


def test_rerun_hits_cache_with_no_model_calls(tmp_path):
    cache = llm_scorer.ScoreCache(tmp_path / "scores.sqlite")
    first = run(FakeBackend([]), cache, sections(3))
    assert (first.model_calls, first.valid, first.cached) == (3, 3, 0)
    backend = FakeBackend([])
    second = run(backend, cache, sections(3))
    assert second.model_calls == 0 and second.cached == 3 and backend.calls == []
    # A reopened cache (a new process) remembers everything too.
    third = run(FakeBackend([]), llm_scorer.ScoreCache(tmp_path / "scores.sqlite"), sections(3))
    assert third.model_calls == 0


def test_cache_key_includes_prompt_version_and_model(tmp_path):
    cache = llm_scorer.ScoreCache(tmp_path / "scores.sqlite")
    run(FakeBackend([]), cache, sections(2))
    other_model = FakeBackend([], model_id="other@1111111")
    assert run(other_model, cache, sections(2)).model_calls == 2
    new_prompt = llm_scorer.score_sections(sections(2), FakeBackend([]), cache, "v2",
                                           100, 10, 60, 1)  # fmt: skip
    assert new_prompt.model_calls == 2


def test_invalid_reply_gets_one_corrective_retry(tmp_path):
    cache = llm_scorer.ScoreCache(tmp_path / "scores.sqlite")
    backend = FakeBackend(["The tone is upbeat.",
                           '{"tone": 0.5, "hedging": 0.0, "risk_severity": -0.2}'])  # fmt: skip
    stats = run(backend, cache, sections(1))
    assert stats.valid == 1 and stats.model_calls == 2
    follow_up = backend.calls[1][-1]
    assert follow_up["role"] == "user" and "not a valid JSON" in follow_up["content"]
    row = cache.frame("v1", "fake@0000000").iloc[0]
    assert row["attempts"] == "2" or int(row["attempts"]) == 2
    assert row["tone"] == 0.5


def test_persistently_invalid_output_is_logged_not_retried_forever(tmp_path):
    cache = llm_scorer.ScoreCache(tmp_path / "scores.sqlite")
    stats = run(FakeBackend(["nope", "still nope"]), cache, sections(1))
    assert stats.invalid == 1 and stats.model_calls == 2
    row = cache.frame("v1", "fake@0000000").iloc[0]
    assert row["status"] == "invalid" and row["raw_output"] == "still nope" and row["error"]
    assert run(FakeBackend([]), cache, sections(1)).model_calls == 0  # cached as invalid


def test_limit_scores_only_some_new_filings(tmp_path):
    cache = llm_scorer.ScoreCache(tmp_path / "scores.sqlite")
    assert run(FakeBackend([]), cache, sections(5), limit=2).model_calls == 2
    assert run(FakeBackend([]), cache, sections(5)).model_calls == 3


# ---------------------------------------------------------------- change signal


def test_llm_change_on_hand_made_example():
    et = "America/New_York"
    pairs = pd.DataFrame({
        "accession": ["a21"], "cik": [1],
        "acceptance_datetime": [pd.Timestamp("2021-02-19 17:05", tz=et)],
        "prev_accession": ["a20"], "prev_acceptance": [pd.Timestamp("2020-02-20 16:30", tz=et)],
    })  # fmt: skip
    scores = pd.DataFrame({"accession": ["a20", "a21"], "tone": [0.6, 0.0],
                           "hedging": [0.0, 0.3], "risk_severity": [0.0, 0.3]})  # fmt: skip
    r = signals.llm_change_signals(pairs, scores).iloc[0]
    # pessimism 2020 = (0 + 0 - 0.6)/3 = -0.2 ; 2021 = (0.3 + 0.3 - 0)/3 = 0.2
    assert r["prev_pessimism"] == pytest.approx(-0.2)
    assert r["pessimism"] == pytest.approx(0.2)
    assert r["llm_change"] == pytest.approx(0.4)
    assert r["d_tone"] == pytest.approx(-0.6)
    assert r["known_at"] == pairs["acceptance_datetime"].iloc[0]


def test_ready_made_excerpts_are_cached_under_their_own_section(tmp_path):
    cache = llm_scorer.ScoreCache(tmp_path / "scores.sqlite")
    run(FakeBackend([]), cache, sections(2))  # original section scores
    masked = pd.DataFrame({"accession": ["a0", "a1"], "excerpt": ["[COMPANY] expects growth."] * 2,
                           "excerpt_method": ["forward_looking"] * 2})  # fmt: skip
    backend = FakeBackend([])
    stats = run(backend, cache, masked, section="mda_anonymized_v1")
    assert stats.model_calls == 2  # the original scores do not count for the masked text
    assert "[COMPANY] expects growth." in backend.calls[0][-1]["content"]
    assert len(cache.frame("v1", "fake@0000000")) == 2  # default section: originals only
    anon = cache.frame("v1", "fake@0000000", "mda_anonymized_v1")
    assert len(anon) == 2 and set(anon["section"]) == {"mda_anonymized_v1"}
    assert run(FakeBackend([]), cache, masked, section="mda_anonymized_v1").model_calls == 0
