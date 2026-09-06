"""
Sentiment analysis sub-agent.

Combines two sources of tone for a ticker:

* recent headlines from yfinance, and
* the narrative sections of the company's latest 10-Q, retrieved from the
  ChromaDB store built by `scripts/ingest_filings.py`.

Both are scored with TextBlob here, so the LLM reads sentiment figures rather
than judging tone from raw text itself.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import yfinance as yf
from langchain_core.tools import BaseTool, tool
from textblob import TextBlob

try:  # running with `src` on the path
    from src.agents.baseAgent import AgentResult, BaseAgent
    from src.rag.store import FilingVectorStore
except ImportError:  # running from inside `src/agents`
    from baseAgent import AgentResult, BaseAgent
    from rag.store import FilingVectorStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _clean(text: Any) -> str:
    """Collapse whitespace and drop None so TextBlob always sees a real string."""
    if not text:
        return ""
    return " ".join(str(text).split())


def _timestamp(value: Any) -> Optional[str]:
    """yfinance returns either an ISO string or a unix epoch; normalize to ISO."""
    if value in (None, "", 0):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    return str(value)


def _article(item: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one yfinance news item; newer versions nest everything under "content"."""
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
    link = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
    return {
        "title": _clean(content.get("title")),
        "summary": _clean(content.get("summary") or content.get("description")),
        "publisher": _clean(provider.get("displayName") or item.get("publisher")),
        "published": _timestamp(
            content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
        ),
        "link": _clean(link.get("url") if isinstance(link, dict) else item.get("link")),
    }


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class SentimentAnalystAgent(BaseAgent):
    """
    Sub-agent that answers the "what is the press saying about this company?"
    half of the research loop.

    Headlines come from yfinance only. Every article is scored here with TextBlob
    so the LLM reads sentiment figures rather than judging tone from raw text.
    """

    DEFAULT_NAME = "sentiment_analyst"
    DEFAULT_DESCRIPTION = (
        "Analyzes news sentiment for a ticker: recent headlines from yfinance, "
        "scored with TextBlob for polarity and subjectivity."
    )

    # TextBlob polarity runs -1 to +1; anything inside this band reads as neutral.
    POSITIVE_THRESHOLD = 0.1
    NEGATIVE_THRESHOLD = -0.1

    DEFAULT_NEWS_LIMIT = 25

    # 10-Q chunks pulled per analysis, across all narrative sections.
    DEFAULT_FILING_TOP_K = 16

    # Sections worth scoring, each with the query used to rank inside it.
    # Financial statements and exhibit lists are numbers and boilerplate;
    # TextBlob has nothing to say about either.
    #
    # Retrieval runs once per section rather than once over the whole filing.
    # A single top-k search returns only MD&A: it is the longest narrative
    # section and its wording matches almost any business-tone query, so it
    # crowds out the others and leaves the section breakdown with one row.
    FILING_SECTION_QUERIES = {
        "mdna": (
            "management discussion of results of operations, revenue and gross margin "
            "trends, segment performance and outlook"
        ),
        "risk_factors": (
            "material risks and uncertainties facing the business, adverse effects on "
            "results of operations and financial condition"
        ),
        "market_risk": (
            "interest rate risk, foreign currency exchange rate exposure and market risk "
            "management"
        ),
        "legal_proceedings": (
            "legal proceedings, regulatory investigations, claims and their potential "
            "impact on the company"
        ),
    }
    FILING_SECTIONS = list(FILING_SECTION_QUERIES)
    # Floor per section, so a short section still contributes something.
    MIN_CHUNKS_PER_SECTION = 2

    # News is timelier but noisier; the filing is authoritative but quarterly.
    NEWS_WEIGHT = 0.6
    FILING_WEIGHT = 0.4
    # Polarity gap above which the two sources are telling different stories.
    DIVERGENCE_THRESHOLD = 0.25
    # A 10-Q older than this is stale enough to flag: the quarter has moved on.
    STALE_FILING_DAYS = 100

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
        news_limit: int = DEFAULT_NEWS_LIMIT,
        filing_store: Optional[FilingVectorStore] = None,
        filing_top_k: int = DEFAULT_FILING_TOP_K,
        use_filings: bool = True,
    ):
        # Set before super().__init__: BaseAgent calls _get_tools() and _get_prompt().
        self.news_limit = news_limit
        self.filing_top_k = filing_top_k
        self.use_filings = use_filings
        # Opened lazily: constructing the store loads an embedding model, and a
        # news-only run should not pay for that or fail when Ollama is down.
        self._filing_store = filing_store
        self._filing_store_error: Optional[str] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        super().__init__(name=name, description=description, tools=tools, verbose=verbose)

    # ------------------------------------------------------------------ #
    # yfinance access
    # ------------------------------------------------------------------ #

    def _slot(self, symbol: str) -> Dict[str, Any]:
        """Per-symbol cache bucket so one analysis hits the network once."""
        return self._cache.setdefault(symbol.strip().upper(), {})

    def _ticker(self, symbol: str) -> yf.Ticker:
        slot = self._slot(symbol)
        if "ticker" not in slot:
            slot["ticker"] = yf.Ticker(symbol.strip().upper())
        return slot["ticker"]

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Drop cached news responses so the next call refetches."""
        if symbol:
            self._cache.pop(symbol.strip().upper(), None)
        else:
            self._cache.clear()

    def _get_stock_news(self, symbol: str, limit: Optional[int] = DEFAULT_NEWS_LIMIT) -> Dict[str, Any]:
        """Latest news articles for `symbol` from yfinance."""
        # print("get_stock_news: START")
        symbol = symbol.strip().upper()
        limit = int(limit) if limit else self.news_limit
        # print(f"get_stock_news: Number of articles getting retrieved: {limit}")
        slot = self._slot(symbol)
        key = "news"

        if key not in slot:
            try:
                raw = self._ticker(symbol).get_news(count=limit)
                # print(f"Number of articles getting retrieved: {limit}")
            except Exception as exc:  # yfinance raises a wide range of errors
                # print("get_stock_news: Error getting news from yfinance: ", exc)
                return {"symbol": symbol, "articles": [], "error": f"Could not fetch news: {exc}"}
            articles = [_article(item) for item in (raw or []) if isinstance(item, dict)]
            slot[key] = [article for article in articles if article["title"]]

        articles = slot[key][:limit]
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "source": "yfinance news",
            "requested": limit,
            "returned": len(articles),
            "articles": articles,
        }

        # print(f"get_stock_news: {self._slot(symbol)["news"]}")
        if not articles:
            # print(f"get_stock_news: No news articles for {symbol}")
            payload["error"] = "yfinance returned no news articles for this symbol."
        # print("get_stock_news: END")
        return payload

    # ------------------------------------------------------------------ #
    # TextBlob scoring
    # ------------------------------------------------------------------ #

    @classmethod
    def _label(cls, polarity: float) -> str:
        """Turn a TextBlob polarity into a positive / neutral / negative label."""
        if polarity >= cls.POSITIVE_THRESHOLD:
            return "positive"
        if polarity <= cls.NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    def _score(self, symbol: str, source: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Score each document's `text` with TextBlob and average the results.

        A document may carry a `weight`; when present the averages are weighted
        by it. Headlines are all about the same length so they average evenly,
        but 10-Q chunks run from one sentence to a full page, and an unweighted
        mean would let a stray heading count as much as a page of MD&A.
        """
        # print(f"_score: START with {len(documents)} articles")
        scored: List[Dict[str, Any]] = []
        for document in documents:
            text = _clean(document.get("text"))
            if not text:
                continue
            sentiment = TextBlob(text).sentiment
            scored.append({
                **{key: value for key, value in document.items() if key not in ("text", "weight")},
                "weight": float(document.get("weight") or 1.0),
                "polarity": round(sentiment.polarity, 4),
                "subjectivity": round(sentiment.subjectivity, 4),
                "label": self._label(sentiment.polarity),
            })

        summary: Dict[str, Any] = {
            "symbol": symbol,
            "source": source,
            "documents_analyzed": len(scored),
            "polarity_scale": "-1.0 (negative) to +1.0 (positive)",
            "subjectivity_scale": "0.0 (objective) to 1.0 (opinionated)",
        }
        if not scored:
            summary.update({
                "average_polarity": None,
                "average_subjectivity": None,
                "sentiment_label": None,
                "error": f"No {source} text was available to analyze for {symbol}.",
            })
            # print(f"_score: Error no articles to score")
            return summary

        total_weight = sum(item["weight"] for item in scored) or float(len(scored))
        average_polarity = sum(item["polarity"] * item["weight"] for item in scored) / total_weight
        average_subjectivity = sum(item["subjectivity"] * item["weight"] for item in scored) / total_weight
        distribution = {"positive": 0, "neutral": 0, "negative": 0}
        for item in scored:
            distribution[item["label"]] += 1

        summary.update({
            "average_polarity": round(average_polarity, 4),
            "average_subjectivity": round(average_subjectivity, 4),
            "sentiment_label": self._label(average_polarity),
            "label_distribution": distribution,
            "article_count": len(scored),
            "documents": scored,
        })
        # print(f"_score: END")
        return summary

    def _score_sentiment(self, symbol: str) -> Dict[str, Any]:
        """TextBlob sentiment over the latest yfinance headlines and summaries."""
        # print("_score_sentiment: START")
        symbol = symbol.strip().upper()
        # print(f"_score_sentiment: {self._slot(symbol)["news"]}")
        # slot = self._slot(symbol)
        # key = "news"
        # articles = slot[key][:self.news_limit]
        # payload: Dict[str, Any] = {
        #     "symbol": symbol,
        #     "source": "yfinance news",
        #     "requested": self.news_limit,
        #     "returned": len(articles),
        #     "articles": articles,
        # }
        # news = payload
        news = self._get_stock_news(symbol)
        # print(f"_score_sentiment: Parsing {len(news["articles"])} news articles")
        if not news.get("articles"):
            # print(f"_score_sentiment: No news articles for {symbol}")
            return {
                "symbol": symbol,
                "source": "yfinance news",
                "documents_analyzed": 0,
                "average_polarity": None,
                "average_subjectivity": None,
                "sentiment_label": None,
                "error": news.get("error", "No news articles were available to analyze."),
            }

        documents = [
            {
                "title": article["title"],
                "publisher": article["publisher"],
                "published": article["published"],
                "text": _clean(f"{article['title']}. {article['summary']}"),
            }
            for article in news["articles"]
        ]
        # print("_score_sentiment: END")
        return self._score(symbol, "yfinance news", documents)

    # ------------------------------------------------------------------ #
    # 10-Q filing sentiment (RAG)
    # ------------------------------------------------------------------ #

    def _store(self) -> Optional[FilingVectorStore]:
        """Open the filing store on first use; remember the failure if it cannot."""
        if not self.use_filings:
            return None
        if self._filing_store is None and self._filing_store_error is None:
            try:
                self._filing_store = FilingVectorStore()
            except Exception as exc:  # no Ollama, no embedding model, bad path
                self._filing_store_error = str(exc)
        return self._filing_store

    @staticmethod
    def _chunk_body(text: str) -> str:
        """
        Drop the bracketed heading the chunker prepends for embedding.

        That prefix ("[AAPL 10-Q 2026-07-31 | Item 2. ...]") helps retrieval but
        is metadata, not prose, and would dilute the chunk's polarity.
        """
        if text.startswith("[") and "\n" in text:
            return text.split("\n", 1)[1]
        return text

    def _filing_documents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map retrieved chunks into the document shape `_score` expects."""
        documents: List[Dict[str, Any]] = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            body = _clean(self._chunk_body(chunk.get("text", "")))
            if not body:
                continue
            documents.append({
                "section": metadata.get("section"),
                "section_title": metadata.get("section_title"),
                "relevance": round(1.0 - float(chunk.get("distance", 0.0)), 4),
                "text": body,
                # Longer passages carry proportionally more of the filing's tone.
                "weight": float(metadata.get("word_count") or len(body.split())),
            })
        return documents

    def _section_breakdown(self, scored_documents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Weighted polarity per 10-Q section.

        This matters more than the headline number: Item 1A is a list of things
        that could go wrong, so it scores negative in essentially every filing
        ever written, while MD&A is management prose and skews positive. Only
        the split between them is interpretable.
        """
        buckets: Dict[str, Dict[str, float]] = {}
        for item in scored_documents:
            section = item.get("section") or "unknown"
            bucket = buckets.setdefault(section, {"weight": 0.0, "polarity": 0.0, "count": 0})
            bucket["weight"] += item["weight"]
            bucket["polarity"] += item["polarity"] * item["weight"]
            bucket["count"] += 1

        breakdown: Dict[str, Any] = {}
        for section, bucket in sorted(buckets.items()):
            average = bucket["polarity"] / bucket["weight"] if bucket["weight"] else None
            breakdown[section] = {
                "average_polarity": round(average, 4) if average is not None else None,
                "sentiment_label": self._label(average) if average is not None else None,
                "chunks_analyzed": int(bucket["count"]),
            }
        return breakdown

    def _score_one_filing(
        self,
        symbol: str,
        store: FilingVectorStore,
        accession: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve and score the narrative sections of one specific filing.

        Each section gets its own quota and its own query, so the breakdown
        always has every section the filing contains rather than whatever a
        single ranked search happened to favour.
        """
        per_section = max(
            self.MIN_CHUNKS_PER_SECTION,
            self.filing_top_k // max(len(self.FILING_SECTION_QUERIES), 1),
        )

        documents: List[Dict[str, Any]] = []
        for section, query_text in self.FILING_SECTION_QUERIES.items():
            chunks = store.query(
                query_text=query_text,
                ticker=symbol,
                top_k=per_section,
                accession=accession,
                sections=[section],
            )
            documents.extend(self._filing_documents(chunks))

        if not documents:
            return None
        return self._score(symbol, "SEC 10-Q filing", documents)

    @staticmethod
    def _filing_age_days(filing_date: Optional[str]) -> Optional[int]:
        """Days between the filing date and today, for staleness checks."""
        if not filing_date:
            return None
        try:
            filed = date.fromisoformat(str(filing_date)[:10])
        except ValueError:
            return None
        return (datetime.now(timezone.utc).date() - filed).days

    def _score_filing_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        TextBlob sentiment over the latest 10-Q's narrative sections.

        Retrieval is pinned to the newest filing on record, so this never mixes
        quarters. The previous filing is scored too: the absolute polarity of a
        10-Q is largely a matter of drafting convention, but the change between
        consecutive quarters is a real signal.
        """
        symbol = symbol.strip().upper()
        base: Dict[str, Any] = {
            "symbol": symbol,
            "source": "SEC 10-Q filing",
            "documents_analyzed": 0,
            "average_polarity": None,
            "average_subjectivity": None,
            "sentiment_label": None,
        }

        store = self._store()
        if store is None:
            base["error"] = (
                self._filing_store_error
                or "Filing retrieval is disabled for this agent (use_filings=False)."
            )
            return base

        filings = store.filings_for(symbol)
        if not filings:
            base["error"] = (
                "No 10-Q filings indexed for " + symbol + ". Run "
                "`python scripts/ingest_filings.py --tickers " + symbol + "` first."
            )
            return base

        latest = filings[0]
        summary = self._score_one_filing(symbol, store, latest["accession"])
        if summary is None:
            base["error"] = f"The indexed 10-Q for {symbol} held no scoreable narrative text."
            return base

        scored_documents = summary.pop("documents", [])
        summary["filing"] = {
            "accession": latest["accession"],
            "filing_date": latest["filing_date"],
            "period_ending": latest.get("period_ending") or None,
            "form": latest.get("form"),
            "company": latest.get("company"),
            "source_url": latest.get("source_url"),
        }
        summary["chunks_analyzed"] = len(scored_documents)
        summary["by_section"] = self._section_breakdown(scored_documents)
        summary["age_days"] = self._filing_age_days(latest.get("filing_date"))
        summary["is_stale"] = bool(
            summary["age_days"] is not None and summary["age_days"] > self.STALE_FILING_DAYS
        )

        # Quarter-over-quarter change, when an earlier filing is also indexed.
        if len(filings) > 1:
            prior = filings[1]
            prior_summary = self._score_one_filing(symbol, store, prior["accession"])
            if prior_summary and prior_summary.get("average_polarity") is not None:
                delta = summary["average_polarity"] - prior_summary["average_polarity"]
                summary["prior_filing"] = {
                    "accession": prior["accession"],
                    "filing_date": prior["filing_date"],
                    "average_polarity": prior_summary["average_polarity"],
                    "sentiment_label": prior_summary["sentiment_label"],
                }
                summary["polarity_change_vs_prior"] = round(delta, 4)
                summary["tone_trend"] = (
                    "more positive" if delta > 0.05
                    else "more negative" if delta < -0.05
                    else "little changed"
                )

        # Keep the strongest passages so the LLM can cite evidence.
        summary["top_passages"] = [
            {
                "section": item.get("section_title") or item.get("section"),
                "polarity": item["polarity"],
                "label": item["label"],
                "relevance": item.get("relevance"),
            }
            for item in sorted(
                scored_documents, key=lambda d: d.get("relevance") or 0.0, reverse=True
            )[:5]
        ]
        return summary

    # ------------------------------------------------------------------ #
    # Combined view
    # ------------------------------------------------------------------ #

    def _combined_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Merge news and 10-Q sentiment into one view.

        The two are reported side by side rather than collapsed into a single
        number. They measure different things: news is a week of outside
        opinion, a 10-Q is a legal document written by management. The composite
        is a convenience; the divergence between the two is the actual signal.
        """
        symbol = symbol.strip().upper()
        news = self._score_sentiment(symbol)
        filing = self._score_filing_sentiment(symbol)

        news_polarity = news.get("average_polarity")
        filing_polarity = filing.get("average_polarity")
        notes: List[str] = []

        result: Dict[str, Any] = {
            "symbol": symbol,
            "news": news,
            "filing": filing,
            "weights": {"news": self.NEWS_WEIGHT, "filing": self.FILING_WEIGHT},
            "polarity_scale": "-1.0 (negative) to +1.0 (positive)",
        }

        if news_polarity is not None and filing_polarity is not None:
            composite = news_polarity * self.NEWS_WEIGHT + filing_polarity * self.FILING_WEIGHT
            divergence = abs(news_polarity - filing_polarity)
            result["composite_polarity"] = round(composite, 4)
            result["composite_label"] = self._label(composite)
            result["divergence"] = round(divergence, 4)

            confidence = "high"
            if divergence >= self.DIVERGENCE_THRESHOLD:
                result["divergence_flag"] = (
                    "filing_more_positive_than_press"
                    if filing_polarity > news_polarity
                    else "press_more_positive_than_filing"
                )
                confidence = "medium"
                notes.append(
                    "News and filing tone disagree materially; treat the overall "
                    "label with caution and explain the gap."
                )
            else:
                result["divergence_flag"] = "aligned"

            if filing.get("is_stale"):
                confidence = "medium"
                notes.append(
                    f"The latest 10-Q is {filing.get('age_days')} days old; "
                    "the quarter it describes may no longer be current."
                )
            if news.get("documents_analyzed", 0) < 5:
                confidence = "low"
                notes.append("Fewer than five news articles were available.")
            result["confidence"] = confidence

        elif news_polarity is not None:
            result["composite_polarity"] = round(news_polarity, 4)
            result["composite_label"] = self._label(news_polarity)
            result["divergence"] = None
            result["divergence_flag"] = "filing_unavailable"
            result["confidence"] = "low"
            notes.append(
                "No 10-Q sentiment available, so this reflects news coverage only: "
                + str(filing.get("error", "filing retrieval returned nothing."))
            )

        elif filing_polarity is not None:
            result["composite_polarity"] = round(filing_polarity, 4)
            result["composite_label"] = self._label(filing_polarity)
            result["divergence"] = None
            result["divergence_flag"] = "news_unavailable"
            result["confidence"] = "low"
            notes.append(
                "No news articles available, so this reflects the 10-Q only: "
                + str(news.get("error", "news retrieval returned nothing."))
            )

        else:
            result["composite_polarity"] = None
            result["composite_label"] = None
            result["divergence"] = None
            result["divergence_flag"] = "no_data"
            result["confidence"] = "low"
            result["error"] = "Neither news nor 10-Q sentiment could be computed."

        result["notes"] = notes
        return result

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    def _get_tools(self) -> List[BaseTool]:
        """Expose the news fetcher and the TextBlob scorer as LLM tools."""

        def _dump(payload: Any) -> str:
            return json.dumps(payload, default=str, indent=2)

        # @tool("get_stock_news")
        # def get_stock_news_tool(symbol: str) -> str:
        #     """Get the latest news articles for a ticker symbol from yfinance, with the title, summary, publisher, publication time and link for each."""
        #     # print(f"Calling get_stock_news with limit = {self.news_limit}")
        #     return _dump(self.get_stock_news(symbol, limit=self.news_limit))

        @tool("score_news_sentiment")
        def score_news_sentiment_tool(symbol: str) -> str:
            """Run TextBlob sentiment analysis on the latest yfinance news articles for a ticker symbol and return the average polarity, average subjectivity, overall sentiment label, the positive/neutral/negative split and the per-article scores."""
            # print(f"Calling score_news_sentiment_tool")
            summary = self._score_sentiment(symbol)
            summary.pop("documents", None)
            return _dump(summary)

        @tool("score_filing_sentiment")
        def score_filing_sentiment_tool(symbol: str) -> str:
            """Run TextBlob sentiment analysis on the narrative sections (MD&A, risk factors, market risk, legal proceedings) of the company's most recent 10-Q filing, retrieved from the SEC filing vector database. Returns the average polarity and subjectivity, a per-section breakdown, the change versus the previous quarter's 10-Q, and which filing was used."""
            # print(f"Calling score_filing_sentiment_tool")
            return _dump(self._score_filing_sentiment(symbol))

        @tool("score_combined_sentiment")
        def score_combined_sentiment_tool(symbol: str) -> str:
            """Score both the latest news articles and the most recent 10-Q filing for a ticker symbol, and return them side by side with a weighted composite polarity, the divergence between the two sources and an overall confidence level. Use this tool for a full sentiment picture."""
            # print(f"Calling score_combined_sentiment_tool")
            combined = self._combined_sentiment(symbol)
            combined.get("news", {}).pop("documents", None)
            return _dump(combined)

        return [
            # get_stock_news_tool,
            score_news_sentiment_tool,
            score_filing_sentiment_tool,
            score_combined_sentiment_tool,
        ]

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #

    def _get_prompt(self) -> str:
        """Sentiment-analysis system prompt."""
        # print("Added sentiment system prompt")
        return """You are a sentiment analyst agent working as part of a multi-agent research team.
        You score two sources of tone: recent news coverage, and the narrative sections of the
        company's most recent 10-Q filing retrieved from the SEC filing database.
        Call score_combined_sentiment first - it covers both sources in one call.
        Never perform sentiment analysis on your own - only through a tool.

        ## Reading the filing scores
        A 10-Q is written by management and reviewed by lawyers, so its absolute polarity says
        more about drafting convention than about the business. Do not read a positive filing
        score as good news on its own. The interpretable signals are:
        - polarity_change_vs_prior: how the tone moved against last quarter's 10-Q
        - by_section: risk_factors is negative in every filing ever written, so compare it
          against the same section last quarter rather than against the other sections
        - divergence: management tone against press tone. A large gap is the most
          informative single output you have - always explain it.
        If the filing data is missing, say so plainly and report news sentiment only.

        ## Output Format
        ---SENTIMENT ANALYSIS---
        1- News Sentiment: score, trend, key themes
        2- 10-Q Filing Sentiment: score, which filing (date and period), section breakdown,
           and the change versus the prior quarter
        3- Divergence between management tone and press coverage, and what explains it
        4- Any notable conflict between articles
        5- Overall negative/neutral/positive sentiment with a confidence of low/medium/high
        """

    # ------------------------------------------------------------------ #
    # Main call
    # ------------------------------------------------------------------ #

    async def analyze_sentiment(
        self,
        symbol: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Analyze sentiment for `symbol` from yfinance news and the latest 10-Q.

        Args:
            symbol: Ticker symbol, for example "AAPL".
            context: Extra context merged into the payload sent to the model.

        Returns:
            The AgentResult from the inherited execute(). Which filing was used
            is attached under result.data["filing"] so the orchestrator can see
            whether the filing store actually contributed.
        """
        symbol = symbol.strip().upper()
        display_name = self._ticker(symbol).info.get("longName") \
            or self._ticker(symbol).info.get("shortName") \
                or symbol

        task = (
            f"Perform a sentiment analysis of {display_name} ({symbol}) covering both "
            f"recent news coverage and the most recent 10-Q filing."
        )

        result = await self.execute(task=task, context=context)
        if result.success:
            result.data["symbol"] = symbol
            result.data["analysis_type"] = "Sentiment"
            # Surface which filing backed the analysis, without re-running the LLM.
            store = self._store()
            latest = store.latest_filing(symbol) if store else None
            result.data["filing"] = latest
        return result