"""
Sentiment analysis sub-agent.

Pulls the latest headlines for a ticker from yfinance, scores them with TextBlob,
and exposes both the raw articles and the aggregate polarity/subjectivity as
LangChain tools the LLM can call on its own.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yfinance as yf
from langchain_core.tools import BaseTool, tool
from textblob import TextBlob

try:  # running with `src` on the path
    from src.agents.baseAgent import AgentResult, BaseAgent
except ImportError:  # running from inside `src/agents`
    from baseAgent import AgentResult, BaseAgent


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

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
        news_limit: int = DEFAULT_NEWS_LIMIT,
    ):
        # Set before super().__init__: BaseAgent calls _get_tools() and _get_prompt().
        self.news_limit = news_limit
        # print(f"sentiment analyst instantiated with news_limit = {self.news_limit}")
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
        """Score each document's `text` with TextBlob and average the results."""
        # print(f"_score: START with {len(documents)} articles")
        scored: List[Dict[str, Any]] = []
        for document in documents:
            text = _clean(document.get("text"))
            if not text:
                continue
            sentiment = TextBlob(text).sentiment
            scored.append({
                **{key: value for key, value in document.items() if key != "text"},
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

        average_polarity = sum(item["polarity"] for item in scored) / len(scored)
        average_subjectivity = sum(item["subjectivity"] for item in scored) / len(scored)
        distribution = {"positive": 0, "neutral": 0, "negative": 0}
        for item in scored:
            distribution[item["label"]] += 1

        summary.update({
            "average_polarity": round(average_polarity, 4),
            "average_subjectivity": round(average_subjectivity, 4),
            "sentiment_label": self._label(average_polarity),
            "label_distribution": distribution,
            "article_count": len(scored),
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

        @tool("score_sentiment")
        def score_sentiment_tool(symbol: str) -> str:
            """Run TextBlob sentiment analysis on the retrieved yfinance news articles for a ticker symbol and return the average polarity, average subjectivity, overall sentiment label, the positive/neutral/negative split and the per-article scores."""
            # print(f"Calling score_sentiment_tool")
            return _dump(self._score_sentiment(symbol))

        return [
            # get_stock_news_tool,
            score_sentiment_tool
        ]

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #

    def _get_prompt(self) -> str:
        """Sentiment-analysis system prompt."""
        # print("Added sentiment system prompt")
        return """You are a news sentiment analyst agent working as part of a multi-agent research team.
        Your job is to always fetch recent news articles and parse them for sentiment scoring.
        Never perform sentiment analysis on your own - only through a tool.

        ## Output Format
        ---SENTIMENT ANALYSIS---
        1- News Sentiment: Score, trend, key themes
        2- Any notable conflict between articles
        3- Overall negative/neutral/positive sentiment with a confidence of low/medium/high
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
        Analyze sentiment for `symbol` from yfinance news.

        Args:
            symbol: Ticker symbol, for example "AAPL".
            context: Extra context merged into the payload sent to the model.

        Returns:
            The AgentResult from the inherited execute(). The fetched data is also
            attached under result.data["sentiment"].
        """
        symbol = symbol.strip().upper()
        display_name = self._ticker(symbol).info.get("longName") \
            or self._ticker(symbol).info.get("shortName") \
                or symbol

        task = (
            f"""Perform a news sentiment analysis of {display_name} ({symbol})"""
            )

        result = await self.execute(task=task)
        if result.success:
            result.data["symbol"] = symbol
            result.data["analysis_type"] = "Sentiment"
        return result