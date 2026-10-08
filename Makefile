PY ?= .venv/bin/python
RUN = $(PY) -m secsignals.cli

.PHONY: all install lint test index universe filings prices returns coverage documents \
	sections dictionary signals llm-score llm-signals factors sanity backtest extraction-check

## Full pipeline, raw EDGAR/Yahoo data -> results/. Every stage caches its downloads,
## so a rerun only fetches what is new.
all: index universe filings prices returns coverage documents sections dictionary signals \
	llm-score llm-signals factors sanity backtest extraction-check

install:
	uv venv --python 3.13 .venv && uv pip install --python $(PY) -e ".[dev,llm]"

lint:
	$(PY) -m ruff check src tests

test:
	$(PY) -m pytest

index:             ## EDGAR quarterly indexes -> every 10-K filed
	$(RUN) index

universe:          ## point-in-time S&P 500 membership, CIK<->ticker map, universe 10-Ks
	$(RUN) universe

filings:           ## acceptance timestamps for universe 10-Ks
	$(RUN) fetch --subset universe --meta-only

prices:            ## Yahoo adjusted closes; reused symbols rejected
	$(RUN) prices

returns:           ## 21-day forward returns from the first trading day after acceptance
	$(RUN) returns

coverage:          ## survivorship check: universe-months with prices, missing companies
	$(RUN) coverage

documents:         ## full 10-K documents (gzipped) for universe companies, incl. prior-year 10-Ks
	$(RUN) documents
	$(RUN) fetch --subset documents

sections:          ## MD&A (Item 7) text for every document
	$(RUN) sections --subset documents

dictionary:        ## Loughran-McDonald negative/uncertainty word counts per section
	$(RUN) dictionary

signals:           ## change vs. the company's previous 10-K; fails below the coverage target
	$(RUN) signals

llm-score:         ## LLM tone/hedging/risk scores per MD&A excerpt; cached, reruns make no new calls
	$(RUN) llm-score
	$(RUN) llm-score --justify

llm-signals:       ## LLM pessimism change vs. the previous 10-K; fails below the valid-score target
	$(RUN) llm-signals

factors:           ## Fama-French 5 factors + momentum (Ken French Data Library)
	$(RUN) factors

sanity:            ## engine checks: planted lookahead signal must win big, noise must give ~0 IC
	$(RUN) sanity

backtest:          ## quintile long-short, IC, turnover, costs, factor alpha -> results/backtest_*.md
	$(RUN) backtest

extraction-check:  ## MD&A extraction quality on a random sample (fails below the configured rate)
	$(RUN) sample
	$(RUN) fetch --subset sample
	$(RUN) sections --subset sample
	$(RUN) check --subset sample
