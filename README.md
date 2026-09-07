## Objective
The financial analyst agent will provide an analysis summary for a prompted public stock/company.
The reasoning loop will be as follows:
1- Generate a thought/action belonging to the below tasks
    - Compare actual EPS vs estimates from earnings reports
    - Summarize fundamentals (income statement, balance sheet, P/E, gross margins, SEC filings)
    - Perform market/news sentiment analysis
2- Parse thought and action
3- Execute tools and get observations
4- Move to next task in (1)
5- Finish and return analysis summary

It is a multi-agent architecture allowing for specialized agents to perform their roles better. Instead of a single agent that will perform all research and return results, we will have:
1- The main agent that will break down the research into sub-tasks, receives results from the sub-agents and formulate the summary/recommendation
2- A sub-agent responsible for the fundamental analysis sub-task
3- A sub-agent responsible for the earnings analysis sub-task
4- A sub-agent responsible for the sentiment analysis sub-task
5- Potentially a sub-agent for RAG retrieval?

---

## Installation
1- Download repo
2- Install python>=3.14.6
3- Make sure to intall all requirement with the below command in terminal
pip install -r .\requirements.txt
4- Download Ollama
run the below command in Windows PowerShell as administrator
irm https://ollama.com/install.ps1 | iex
5- Pull the chat model and the embedding model
```
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

---

## Running an analysis

```
python main.py AAPL
```

`ResearchManagerAgent` orchestrates the whole run: it refreshes the 10-Q vector
store, delegates to the specialist sub-agents concurrently, reconciles their
findings and writes the conclusion to `reports/<SYMBOL>_<YYYY-MM-DD>.md`.

### Orchestration

```
                    +-- refresh 10-Q filings (EDGAR -> ChromaDB) --+
                    |                                             |
                    +-- fundamental analyst ----------------------+
analyze(symbol) ----+                                             +--> aggregate
                    +-- earnings analyst -------------------------+     -> detect
                    |                                             |        contradictions
                    +-- (waits for filings) sentiment analyst ----+     -> LLM synthesis
                                                                        -> report + file
```

The fan-out uses `asyncio.gather`, not LLM tool calls. A tool-calling model
issues one call at a time, so sub-agents exposed as tools would run one after
another; orchestrating in Python actually runs them at once. The fundamental
analysis starts immediately, while the sentiment analysis waits on ingestion
because it reads the filing store.

Sub-agent failures are isolated. A crash or a timeout degrades the report and
lowers confidence rather than sinking the run; the manager only fails outright
when every sub-agent fails.

### Contradiction detection

Disagreement between sub-agents is found in Python before the LLM is asked
anything, so the model is told where the tension is instead of being trusted to
notice it. The checks compare structured metrics, not prose:

| Check | Fires when |
| --- | --- |
| `valuation_vs_sentiment` | DCF upside and sentiment point opposite ways |
| `growth_vs_sentiment` | Revenue is shrinking while coverage is positive |
| `delivery_vs_quality` | Beating consensus on low-quality earnings |
| `earnings_vs_sentiment` | Missing consensus while the press stays positive |
| `quality_vs_valuation` | DCF shows upside but earnings quality is Low |
| `one_time_profit` | Net income exceeded operating income over the window |
| `news_vs_filing` | Management tone and press tone diverge |
| `filing_trend_vs_news` | Filing tone softened but news is positive |
| `stated_conclusions` | The two write-ups end on opposing verdicts |

The last one reads the verdict out of the sub-agents' text, so it is a
heuristic cross-check; the others run on figures.

The manager's tools (`list_findings`, `get_analysis`,
`get_contradiction_report`, `get_filing_status`) read findings that were already
collected. Keeping the long sub-agent write-ups behind a tool lets the model
pull only what it needs instead of carrying all of them in the prompt.

---

## Earnings analyst

`EarningsAnalystAgent` covers the six most recent reported quarters (about
eighteen months), using `yfinance` earnings dates for estimate-versus-actual EPS
and the quarterly income statement for the quality indicators.

### Earnings pattern

Beat rate over the window, where an in-line print counts as neither a beat nor a
miss and so dilutes the rate:

| Beat rate | Pattern |
| --- | --- |
| >= 80% | Consistent Beat |
| 60-80% | Regular Beat |
| 40-60% | Inconsistent |
| 20-40% | Regular Miss |
| < 20% | Consistent Miss |

### Earnings quality

Four indicators, each scored good / bad / neutral:

| Indicator | Good | Bad |
| --- | --- | --- |
| Revenue volatility | CV < 0.10 | CV > 0.30 |
| Gross margin volatility | std < 0.02 | std > 0.05 |
| Net vs operating income | *not reachable* | net exceeds operating by >30%, or falls short by >45% |
| Earnings vs revenue growth | both growing, earnings faster | earnings growing, revenue not |

Categories, checked in order:

| Category | Condition |
| --- | --- |
| Very Good | 3 good, 0 bad |
| Good | (3 good, 1 bad) or (2 good, 0 bad) |
| Low | 2 bad, 0 good |
| Very Low | 3 or more bad |
| Average | anything else |

Net income against operating income scores bad or neutral only: a wide gap in
either direction means profit is not tracking operations, so no reading of it
counts in the company's favour. Three good is therefore a clean sweep of every
indicator that can be won, which is why Very Good sits there rather than at
four.

Two notes on the implementation:

**Revenue volatility is a coefficient of variation** (std / mean), not a raw
standard deviation. The 0.1 / 0.3 thresholds are dimensionless, so they only
make sense against a normalised measure; a raw standard deviation would be in
currency units and would scale with company size. Gross margin is already a
ratio, so that one is a plain standard deviation.

**Negative operating income bypasses the ratio.** Comparing net income against a
negative operating income produces meaningless output (INTC reported "short by
570%" before this was handled), so operating losses are scored bad directly,
with the one-time signal raised when the company is net-profitable anyway.

`yfinance` publishes only about five quarters of income statement against
twenty-plus quarters of earnings dates, so the pattern uses the full six-quarter
window while the quality indicators use whatever quarters exist, and report the
count.

---

## SEC 10-Q RAG pipeline

`src/rag/` downloads 10-Q filings from EDGAR and indexes them in ChromaDB so the
sentiment agent can score what management actually wrote, alongside the news.

### Ingesting filings

Ingestion is run by the researchManager at the start of every run. 
However, the process is slightly faster if all required data is ingested in batch before the analysis.
Use the below query to load the vector db with as many 10-Q filings
```
python scripts/ingest_filings.py --tickers AAPL MSFT --limit 2
```

`--limit` is filings per ticker, newest first. Indexing two quarters per ticker
lets the agent report the quarter-over-quarter change in tone, which is more
informative than any single filing's score. `--refresh` re-downloads and
re-indexes filings that are already stored.

Filings are cached as HTML under `data/filings/raw/` and vectors under
`data/chroma/`; both are safe to delete and rebuild.

### Pipeline stages

| Module | Responsibility |
| --- | --- |
| `config.py` | Paths, model names, chunk sizes, EDGAR settings |
| `edgar.py` | Ticker to CIK, full-text search for 10-Qs, cached downloads |
| `extract.py` | Inline-XBRL HTML to clean text, split into Item-level sections |
| `chunker.py` | Sentence-aligned chunks, drops numeric tables |
| `embeddings.py` | Ollama embeddings, with a bundled-model fallback |
| `store.py` | ChromaDB persistence and retrieval |
| `ingest.py` | Runs the stages end to end |

### Notes on the implementation

**EDGAR serves 10-Qs as HTML**
The primary document is inline XBRL. `extract.py` parses that HTML. A
`pypdf` path is wired in for filings supplied from elsewhere, but the EDGAR
route never uses it.

**Full-text search is queried with an empty `q`.** Supplying a search term makes
the endpoint rank by relevance and return exhibits (`EX-99.1` and similar)
instead of the 10-Q itself. An empty query filtered by `ciks` and `forms`
returns primary documents in filing-date order, which is what "latest 10-Q"
needs.

**"Latest" is resolved by metadata, not by similarity.** Vector search has no
concept of recency, so `FilingVectorStore.latest_filing_chunks()` finds the
newest `filing_date` for the ticker first and only then searches inside that one
filing. Results can never drift into an older quarter.

**Retrieval uses a per-section quota.** MD&A is the longest narrative section
and matches almost any business-tone query, so a single top-k search returns
nothing but MD&A. Each section is queried separately with its own quota so the
breakdown covers every section the filing contains.

**The embedding model is recorded on the collection.** Querying vectors written
by one model using another returns plausible-looking nonsense, so the store
refuses to open a collection built with a different embedding model rather than
failing silently.

### Reading the sentiment output

The sentiment agent exposes three tools: `score_news_sentiment`,
`score_filing_sentiment` and `score_combined_sentiment`.

A 10-Q is written by management and reviewed by lawyers, so its absolute
polarity reflects drafting convention more than business performance. Item 1A
scores negative in essentially every filing ever written. The interpretable
signals are therefore:

- `polarity_change_vs_prior` — tone against the previous quarter's 10-Q
- `by_section` — compare a section against itself last quarter, not against
  other sections
- `divergence` — management tone against press tone; a large gap is the single
  most informative output, and the composite score is only a convenience

Not every 10-Q contains every section. Many companies incorporate risk factors
by reference to their 10-K and omit Item 1A from the 10-Q entirely, so the
section breakdown varies by filer.
