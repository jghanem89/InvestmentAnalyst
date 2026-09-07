"""
Quarterly earnings analysis sub-agent.

Answers the "does this company deliver what it promises, and is the profit
real?" half of the research loop, over the six most recent reported quarters
(roughly eighteen months):

* earnings pattern - how often reported EPS beat the consensus estimate
* earnings quality - four indicators separating operating profit from one-offs

Every figure is computed here so the LLM reads finished numbers rather than
doing arithmetic on raw statements itself.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf
from langchain_core.tools import BaseTool, tool

try:  # running with `src` on the path
    from src.agents.baseAgent import AgentResult, BaseAgent
except ImportError:  # running from inside `src/agents`
    from baseAgent import AgentResult, BaseAgent


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_float(value: Any) -> Optional[float]:
    """yfinance mixes numpy scalars, None and NaN; normalize to float or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN != NaN


def _row_value(frame, candidates: List[str], column) -> Optional[float]:
    """First matching row label in `candidates`, read at `column`."""
    if frame is None or getattr(frame, "empty", True):
        return None
    labels = {str(label).strip().lower(): label for label in frame.index}
    for candidate in candidates:
        label = labels.get(candidate.strip().lower())
        if label is not None:
            return _to_float(frame.loc[label, column])
    return None


def _pct_change(latest: Optional[float], oldest: Optional[float]) -> Optional[float]:
    """Percent change from oldest to latest, guarding sign flips and zero bases."""
    if latest is None or oldest is None or oldest == 0:
        return None
    return (latest - oldest) / abs(oldest) * 100.0


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class EarningsAnalystAgent(BaseAgent):
    """
    Sub-agent that scores earnings delivery and earnings quality.

    Data comes from yfinance: `get_earnings_dates` for estimate-vs-actual EPS,
    and the quarterly income statement for the quality indicators.
    """

    DEFAULT_NAME = "earnings_analyst"
    DEFAULT_DESCRIPTION = (
        "Analyzes quarterly earnings: actual EPS versus consensus estimates, "
        "the beat/miss pattern, and earnings quality separating operating "
        "performance from one-time items."
    )

    # Six quarterly reports, i.e. about eighteen months.
    ANALYSIS_QUARTERS = 6
    ANALYSIS_MONTHS = 18

    # Beat-rate cut-offs for the pattern label.
    CONSISTENT_BEAT_RATE = 0.80
    REGULAR_BEAT_RATE = 0.60
    INCONSISTENT_RATE = 0.40
    REGULAR_MISS_RATE = 0.20

    # Quality indicator thresholds.
    REVENUE_CV_GOOD = 0.10       # coefficient of variation, std / mean
    REVENUE_CV_BAD = 0.30
    GROSS_MARGIN_STD_GOOD = 0.02  # margin is a ratio, so 0.02 = 2 percentage points
    GROSS_MARGIN_STD_BAD = 0.05
    # Net income against operating income. This indicator scores bad or neutral
    # only: a wide gap either way means profit is not tracking operations, so
    # there is no reading of it that counts in the company's favour.
    INCOME_RATIO_UPPER_BAND = 0.30   # net exceeds operating by >30% -> bad
    INCOME_RATIO_LOWER_BAND = 0.45   # net falls short by >45%       -> bad

    # A surprise this large either way is worth calling out on its own.
    NOTABLE_SURPRISE_PCT = 10.0

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
        quarters: int = ANALYSIS_QUARTERS,
    ):
        # Set before super().__init__: BaseAgent calls _get_tools() and _get_prompt().
        self.quarters = quarters
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
        """Drop cached yfinance responses so the next call refetches."""
        if symbol:
            self._cache.pop(symbol.strip().upper(), None)
        else:
            self._cache.clear()

    def _quarterly_income(self, symbol: str):
        """Quarterly income statement frame, newest column first."""
        slot = self._slot(symbol)
        if "quarterly_income" not in slot:
            try:
                slot["quarterly_income"] = self._ticker(symbol).quarterly_income_stmt
            except Exception:
                slot["quarterly_income"] = None
        return slot["quarterly_income"]

    # ------------------------------------------------------------------ #
    # Earnings history
    # ------------------------------------------------------------------ #

    def get_earnings_history(self, symbol: str) -> Dict[str, Any]:
        """
        The most recent reported quarters, with estimate, actual and surprise.

        Scheduled future dates come back from yfinance with an empty reported
        EPS; those rows are dropped so an upcoming announcement is never counted
        as a miss.
        """
        symbol = symbol.strip().upper()
        slot = self._slot(symbol)

        if "earnings_history" not in slot:
            try:
                # Ask for more than we need: the response includes future dates.
                frame = self._ticker(symbol).get_earnings_dates(limit=self.quarters * 4)
            except Exception as exc:
                return {"symbol": symbol, "quarters": [], "error": f"Could not fetch earnings dates: {exc}"}

            quarters: List[Dict[str, Any]] = []
            if frame is not None and not frame.empty:
                for index, row in frame.iterrows():
                    reported = _to_float(row.get("Reported EPS"))
                    estimate = _to_float(row.get("EPS Estimate"))
                    if reported is None or estimate is None:
                        continue  # not yet announced
                    surprise = _to_float(row.get("Surprise(%)"))
                    if surprise is None and estimate:
                        surprise = (reported - estimate) / abs(estimate) * 100.0
                    quarters.append({
                        "earnings_date": index.date().isoformat() if hasattr(index, "date") else str(index),
                        "eps_estimate": round(estimate, 4),
                        "eps_reported": round(reported, 4),
                        "eps_difference": round(reported - estimate, 4),
                        "surprise_pct": round(surprise, 2) if surprise is not None else None,
                        "result": "beat" if reported > estimate else "miss" if reported < estimate else "in line",
                    })
            slot["earnings_history"] = quarters

        quarters = slot["earnings_history"][:self.quarters]
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "source": "yfinance earnings dates",
            "window": f"{self.quarters} most recent reported quarters (~{self.ANALYSIS_MONTHS} months)",
            "quarters_requested": self.quarters,
            "quarters_returned": len(quarters),
            "quarters": quarters,
        }
        if quarters:
            payload["period_covered"] = {
                "from": quarters[-1]["earnings_date"],
                "to": quarters[0]["earnings_date"],
            }
        else:
            payload["error"] = f"yfinance returned no reported earnings for {symbol}."
        return payload

    # ------------------------------------------------------------------ #
    # Earnings pattern
    # ------------------------------------------------------------------ #

    @classmethod
    def _pattern_label(cls, beat_rate: float) -> str:
        """Map a beat rate in 0..1 onto the pattern bands."""
        if beat_rate >= cls.CONSISTENT_BEAT_RATE:
            return "Consistent Beat"
        if beat_rate >= cls.REGULAR_BEAT_RATE:
            return "Regular Beat"
        if beat_rate >= cls.INCONSISTENT_RATE:
            return "Inconsistent"
        if beat_rate >= cls.REGULAR_MISS_RATE:
            return "Regular Miss"
        return "Consistent Miss"

    def earnings_pattern(self, symbol: str) -> Dict[str, Any]:
        """Beat rate over the analysis window and the pattern it implies."""
        symbol = symbol.strip().upper()
        history = self.get_earnings_history(symbol)
        quarters = history.get("quarters", [])

        if not quarters:
            return {
                "symbol": symbol,
                "pattern": None,
                "beat_rate_pct": None,
                "quarters_analyzed": 0,
                "error": history.get("error", "No reported earnings were available."),
            }

        beats = sum(1 for q in quarters if q["result"] == "beat")
        misses = sum(1 for q in quarters if q["result"] == "miss")
        in_line = sum(1 for q in quarters if q["result"] == "in line")
        # An in-line print is not a beat, so it dilutes the rate rather than
        # counting toward it.
        beat_rate = beats / len(quarters)

        surprises = [q["surprise_pct"] for q in quarters if q["surprise_pct"] is not None]
        return {
            "symbol": symbol,
            "window": history.get("window"),
            "period_covered": history.get("period_covered"),
            "quarters_analyzed": len(quarters),
            "beats": beats,
            "misses": misses,
            "in_line": in_line,
            "beat_rate_pct": round(beat_rate * 100, 2),
            "pattern": self._pattern_label(beat_rate),
            "average_surprise_pct": round(statistics.fmean(surprises), 2) if surprises else None,
            "largest_beat_pct": round(max(surprises), 2) if surprises else None,
            "largest_miss_pct": round(min(surprises), 2) if surprises else None,
            "quarters": quarters,
        }

    # ------------------------------------------------------------------ #
    # Quarterly fundamentals behind the quality indicators
    # ------------------------------------------------------------------ #

    def quarterly_series(self, symbol: str) -> Dict[str, Any]:
        """
        Revenue, gross margin, operating income and net income per quarter.

        yfinance publishes fewer quarters of statements than of earnings dates
        (typically five), so this returns whatever is available and reports the
        count rather than padding the window.
        """
        symbol = symbol.strip().upper()
        frame = self._quarterly_income(symbol)
        if frame is None or getattr(frame, "empty", True):
            return {
                "symbol": symbol,
                "quarters": [],
                "error": "yfinance returned no quarterly income statement.",
            }

        quarters: List[Dict[str, Any]] = []
        for column in list(frame.columns)[:self.quarters]:
            revenue = _row_value(frame, ["Total Revenue", "Operating Revenue"], column)
            gross_profit = _row_value(frame, ["Gross Profit"], column)
            operating_income = _row_value(
                frame, ["Operating Income", "Total Operating Income As Reported"], column
            )
            net_income = _row_value(
                frame,
                ["Net Income", "Net Income Common Stockholders",
                 "Net Income From Continuing Operation Net Minority Interest"],
                column,
            )
            quarters.append({
                "period": str(column)[:10],
                "revenue": revenue,
                "gross_profit": gross_profit,
                "gross_margin": (gross_profit / revenue) if revenue and gross_profit is not None else None,
                "operating_income": operating_income,
                "net_income": net_income,
            })

        return {
            "symbol": symbol,
            "quarters_requested": self.quarters,
            "quarters_returned": len(quarters),
            "quarters": quarters,
        }

    # ------------------------------------------------------------------ #
    # Quality indicators
    # ------------------------------------------------------------------ #

    @staticmethod
    def _indicator(name: str, verdict: str, value: Optional[float], detail: str, **extra) -> Dict[str, Any]:
        """Uniform envelope so every indicator reads the same way."""
        payload = {"indicator": name, "verdict": verdict, "value": value, "detail": detail}
        payload.update({k: v for k, v in extra.items() if v is not None})
        return payload

    def _revenue_volatility(self, quarters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Revenue stability, as the coefficient of variation.

        The thresholds (0.1 / 0.3) are dimensionless, so the comparison is
        std / mean rather than the raw standard deviation, which would be in
        currency units and would scale with company size.
        """
        values = [q["revenue"] for q in quarters if q.get("revenue")]
        if len(values) < 2:
            return self._indicator(
                "revenue_volatility", "unknown", None,
                "Fewer than two quarters of revenue were available.",
            )

        mean = statistics.fmean(values)
        if not mean:
            return self._indicator("revenue_volatility", "unknown", None, "Mean revenue was zero.")
        cv = statistics.stdev(values) / abs(mean)

        if cv < self.REVENUE_CV_GOOD:
            verdict, detail = "good", f"Revenue is stable (CV {cv:.3f} < {self.REVENUE_CV_GOOD})."
        elif cv > self.REVENUE_CV_BAD:
            verdict, detail = "bad", f"Revenue swings widely (CV {cv:.3f} > {self.REVENUE_CV_BAD})."
        else:
            verdict, detail = "neutral", f"Revenue variability is moderate (CV {cv:.3f})."

        return self._indicator(
            "revenue_volatility", verdict, round(cv, 4), detail,
            quarters_used=len(values), mean_revenue=round(mean, 2),
        )

    def _gross_margin_volatility(self, quarters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gross margin stability, as the standard deviation of the margin ratio."""
        values = [q["gross_margin"] for q in quarters if q.get("gross_margin") is not None]
        if len(values) < 2:
            return self._indicator(
                "gross_margin_volatility", "unknown", None,
                "Fewer than two quarters of gross margin were available.",
            )

        deviation = statistics.stdev(values)
        if deviation < self.GROSS_MARGIN_STD_GOOD:
            verdict = "good"
            detail = f"Gross margin is steady (std {deviation:.4f} < {self.GROSS_MARGIN_STD_GOOD})."
        elif deviation > self.GROSS_MARGIN_STD_BAD:
            verdict = "bad"
            detail = f"Gross margin is erratic (std {deviation:.4f} > {self.GROSS_MARGIN_STD_BAD})."
        else:
            verdict = "neutral"
            detail = f"Gross margin variability is moderate (std {deviation:.4f})."

        return self._indicator(
            "gross_margin_volatility", verdict, round(deviation, 4), detail,
            quarters_used=len(values),
            average_gross_margin_pct=round(statistics.fmean(values) * 100, 2),
        )

    def _income_composition(self, quarters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Net income measured against operating income across the window.

        Scores bad or neutral only. A gap in either direction means reported
        profit is not tracking operations - non-operating gains above, heavy
        charges or tax below - so no reading of this ratio counts in the
        company's favour.

        Totals are summed over the window rather than averaged per quarter, so
        one loss-making quarter cannot distort the ratio.
        """
        operating = [q["operating_income"] for q in quarters if q.get("operating_income") is not None]
        net = [q["net_income"] for q in quarters if q.get("net_income") is not None]
        if not operating or not net:
            return self._indicator(
                "net_vs_operating_income", "unknown", None,
                "Operating or net income was not reported for these quarters.",
            )

        total_operating, total_net = sum(operating), sum(net)
        if total_operating == 0:
            return self._indicator(
                "net_vs_operating_income", "unknown", None, "Operating income summed to zero."
            )

        # A ratio against negative operating income is not interpretable: the
        # percentage band would report nonsense like "short by 570%". Operating
        # losses are a quality problem in their own right, so score them
        # directly instead of running them through the thresholds.
        if total_operating < 0:
            profitable = total_net > 0
            return self._indicator(
                "net_vs_operating_income", "bad", None,
                (
                    f"Operations lost {abs(total_operating):,.0f} over the window while net "
                    f"income was {total_net:,.0f}: profit is not coming from operations."
                    if profitable else
                    f"Both operating income ({total_operating:,.0f}) and net income "
                    f"({total_net:,.0f}) were negative over the window."
                ),
                total_operating_income=round(total_operating, 2),
                total_net_income=round(total_net, 2),
                one_time_signal=profitable,
            )

        ratio = total_net / total_operating
        upper = 1 + self.INCOME_RATIO_UPPER_BAND   # 1.30
        lower = 1 - self.INCOME_RATIO_LOWER_BAND   # 0.55

        # Profitable operations that still end in a net loss mean large charges
        # below the operating line - impairments, write-downs, settlements.
        # Expressing that as "short by 570%" is arithmetically true and useless,
        # so it gets its own wording.
        if total_net < 0:
            return self._indicator(
                "net_vs_operating_income", "bad", round(ratio, 4),
                (
                    f"Operations earned {total_operating:,.0f} over the window but net income "
                    f"was {total_net:,.0f}: charges below the operating line wiped out "
                    "operating profit and more."
                ),
                total_operating_income=round(total_operating, 2),
                total_net_income=round(total_net, 2),
                non_operating_contribution=round(total_net - total_operating, 2),
            )

        if ratio > upper:
            verdict = "bad"
            detail = (
                f"Net income exceeds operating income by {(ratio - 1) * 100:.1f}% "
                f"(ratio {ratio:.2f}): profit is arriving from outside operations."
            )
        elif ratio < lower:
            verdict = "bad"
            detail = (
                f"Net income falls short of operating income by {(1 - ratio) * 100:.1f}% "
                f"(ratio {ratio:.2f})."
            )
        else:
            verdict = "neutral"
            detail = f"Net income tracks operating income closely (ratio {ratio:.2f})."

        # Net income above operating income means profit is arriving from
        # outside operations. The indicator now scores that bad in its own
        # right, but the flag stays so red flags and the research manager can
        # name the one-time reading specifically.
        non_operating = round(total_net - total_operating, 2)
        return self._indicator(
            "net_vs_operating_income", verdict, round(ratio, 4), detail,
            total_operating_income=round(total_operating, 2),
            total_net_income=round(total_net, 2),
            non_operating_contribution=non_operating,
            one_time_signal=bool(ratio > upper),
        )

    def _growth_alignment(self, quarters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Earnings growth measured against revenue growth across the window.

        Earnings outgrowing revenue is operating leverage; earnings growing
        while revenue does not is the classic sign of profit coming from cost
        cuts or one-off items rather than from the business expanding.
        """
        usable = [q for q in quarters if q.get("revenue") and q.get("net_income") is not None]
        if len(usable) < 2:
            return self._indicator(
                "earnings_vs_revenue_growth", "unknown", None,
                "Fewer than two quarters with both revenue and net income.",
            )

        # `quarters` arrives newest-first.
        latest, oldest = usable[0], usable[-1]
        revenue_growth = _pct_change(latest["revenue"], oldest["revenue"])
        earnings_growth = _pct_change(latest["net_income"], oldest["net_income"])
        if revenue_growth is None or earnings_growth is None:
            return self._indicator(
                "earnings_vs_revenue_growth", "unknown", None,
                "Growth could not be computed from the reported figures.",
            )

        if revenue_growth > 0 and earnings_growth > 0 and earnings_growth > revenue_growth:
            verdict = "good"
            detail = (
                f"Earnings grew {earnings_growth:.1f}% against revenue growth of "
                f"{revenue_growth:.1f}%: operating leverage."
            )
        elif earnings_growth > 0 and revenue_growth <= 0:
            verdict = "bad"
            detail = (
                f"Earnings grew {earnings_growth:.1f}% while revenue fell "
                f"{revenue_growth:.1f}%: profit is not coming from growth."
            )
        else:
            verdict = "neutral"
            detail = (
                f"Earnings growth {earnings_growth:.1f}% against revenue growth "
                f"{revenue_growth:.1f}%."
            )

        return self._indicator(
            "earnings_vs_revenue_growth", verdict,
            round(earnings_growth - revenue_growth, 2), detail,
            revenue_growth_pct=round(revenue_growth, 2),
            earnings_growth_pct=round(earnings_growth, 2),
            from_period=oldest["period"], to_period=latest["period"],
        )

    @staticmethod
    def _quality_label(good: int, bad: int) -> str:
        """
        Category from the count of good and bad indicators.

        Only three of the four indicators can score good - net income against
        operating income is bad or neutral only - so three good is a clean
        sweep of everything that can be won.

        Very Low is `bad >= 3` rather than exactly three: all four indicators
        can score bad at once, and an exact test would drop that worst case
        through to Average.
        """
        label = "Average"
        if good == 3 and bad == 0:
            label = "Very Good"
        elif (good == 3 and bad == 1) or (good == 2 and bad == 0):
            label = "Good"
        elif (bad == 2 and good == 0) or (bad == 3 and good == 1):
            label = "Low"
        elif bad >= 3:
            label = "Very Low"
        return label

    def earnings_quality(self, symbol: str) -> Dict[str, Any]:
        """Score the four quality indicators and reduce them to a category."""
        symbol = symbol.strip().upper()
        series = self.quarterly_series(symbol)
        quarters = series.get("quarters", [])

        if not quarters:
            return {
                "symbol": symbol,
                "quality": None,
                "indicators": [],
                "error": series.get("error", "No quarterly financials were available."),
            }

        indicators = [
            self._revenue_volatility(quarters),
            self._gross_margin_volatility(quarters),
            self._income_composition(quarters),
            self._growth_alignment(quarters),
        ]
        good = sum(1 for i in indicators if i["verdict"] == "good")
        bad = sum(1 for i in indicators if i["verdict"] == "bad")
        neutral = sum(1 for i in indicators if i["verdict"] == "neutral")
        unknown = sum(1 for i in indicators if i["verdict"] == "unknown")

        result = {
            "symbol": symbol,
            "quarters_analyzed": len(quarters),
            "quality": self._quality_label(good, bad),
            "good_indicators": good,
            "bad_indicators": bad,
            "neutral_indicators": neutral,
            "unknown_indicators": unknown,
            "indicators": indicators,
            "quarterly_data": quarters,
        }
        if unknown:
            result["note"] = (
                f"{unknown} indicator(s) could not be computed, so the category rests "
                "on fewer than four signals."
            )
        return result

    # ------------------------------------------------------------------ #
    # Surprises and red flags
    # ------------------------------------------------------------------ #

    def earnings_red_flags(self, symbol: str) -> Dict[str, Any]:
        """Notable surprises and quality warnings, gathered in one place."""
        symbol = symbol.strip().upper()
        pattern = self.earnings_pattern(symbol)
        quality = self.earnings_quality(symbol)

        surprises: List[Dict[str, Any]] = []
        red_flags: List[Dict[str, str]] = []

        for quarter in pattern.get("quarters", []):
            surprise = quarter.get("surprise_pct")
            if surprise is None:
                continue
            if abs(surprise) >= self.NOTABLE_SURPRISE_PCT:
                surprises.append({
                    "earnings_date": quarter["earnings_date"],
                    "surprise_pct": surprise,
                    "direction": "beat" if surprise > 0 else "miss",
                    "detail": (
                        f"Reported {quarter['eps_reported']} against an estimate of "
                        f"{quarter['eps_estimate']} ({surprise:+.1f}%)."
                    ),
                })
            if quarter["result"] == "miss":
                red_flags.append({
                    "kind": "eps_miss",
                    "detail": (
                        f"Missed consensus on {quarter['earnings_date']}: "
                        f"{quarter['eps_reported']} against {quarter['eps_estimate']}."
                    ),
                })

        if pattern.get("pattern") in ("Regular Miss", "Consistent Miss"):
            red_flags.append({
                "kind": "delivery_pattern",
                "detail": (
                    f"Pattern is '{pattern['pattern']}' with a beat rate of "
                    f"{pattern.get('beat_rate_pct')}%: guidance is not being met."
                ),
            })

        for indicator in quality.get("indicators", []):
            if indicator["verdict"] == "bad":
                red_flags.append({"kind": indicator["indicator"], "detail": indicator["detail"]})
            # Surfaced separately: this reads as "good" under the scoring rule
            # but means profit is arriving from outside operations.
            if indicator.get("one_time_signal"):
                red_flags.append({
                    "kind": "one_time_gains",
                    "detail": (
                        "Net income exceeds operating income, so a material share of "
                        "profit came from outside normal operations."
                    ),
                })

        if quality.get("quality") in ("Low", "Very Low"):
            red_flags.append({
                "kind": "earnings_quality",
                "detail": (
                    f"Earnings quality is '{quality['quality']}' with "
                    f"{quality.get('bad_indicators')} negative indicator(s)."
                ),
            })

        return {
            "symbol": symbol,
            "pattern": pattern.get("pattern"),
            "quality": quality.get("quality"),
            "notable_surprises": surprises,
            "red_flags": red_flags,
            "red_flag_count": len(red_flags),
        }

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    def _get_tools(self) -> List[BaseTool]:
        """Expose the earnings history, pattern, quality and red flags as LLM tools."""

        def _dump(payload: Any) -> str:
            return json.dumps(payload, default=str, indent=2)

        @tool("get_earnings_history")
        def get_earnings_history_tool(symbol: str) -> str:
            """Get the six most recent reported quarters for a ticker symbol, each with the consensus EPS estimate, the reported EPS, the difference, the surprise percentage and whether it was a beat, a miss or in line."""
            return _dump(self.get_earnings_history(symbol))

        @tool("analyze_earnings_pattern")
        def analyze_earnings_pattern_tool(symbol: str) -> str:
            """Calculate the beat rate over the six most recent reported quarters and classify the earnings pattern as Consistent Beat, Regular Beat, Inconsistent, Regular Miss or Consistent Miss. Also returns the beat/miss counts and the average, largest and smallest surprise."""
            return _dump(self.earnings_pattern(symbol))

        @tool("assess_earnings_quality")
        def assess_earnings_quality_tool(symbol: str) -> str:
            """Assess earnings quality over the analysis window using four indicators: revenue volatility, gross margin volatility, net income versus operating income, and earnings growth versus revenue growth. Each is scored good, bad or neutral, and the counts give an overall category of Very Good, Good, Average, Low or Very Low quality."""
            return _dump(self.earnings_quality(symbol))

        @tool("get_earnings_red_flags")
        def get_earnings_red_flags_tool(symbol: str) -> str:
            """Get the notable EPS surprises and the red flags for a ticker symbol: individual misses, a weak delivery pattern, any negative quality indicator, and profit arriving from outside normal operations."""
            return _dump(self.earnings_red_flags(symbol))

        return [
            get_earnings_history_tool,
            analyze_earnings_pattern_tool,
            assess_earnings_quality_tool,
            get_earnings_red_flags_tool,
        ]

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #

    def _get_prompt(self) -> str:
        """Earnings-analysis system prompt."""
        return """You are an earnings analyst agent working as part of a multi-agent research team.
        You assess how reliably a company delivers against consensus estimates, and whether the
        profit it reports is operational or one-off. The window is the six most recent reported
        quarters, roughly eighteen months.
        Never compute a ratio or a growth rate yourself - only through a tool.

        ## How to work
        1- Call analyze_earnings_pattern for the beat rate and the pattern label.
        2- Call assess_earnings_quality for the four quality indicators and the category.
        3- Call get_earnings_red_flags for surprises and warnings.
        4- Call get_earnings_history if you need the individual quarters to explain a trend.

        ## Reading the numbers
        - The pattern label describes delivery against expectations, not growth. A Consistent
          Beat against steadily lowered estimates is not the same as a strong business, so check
          whether reported EPS is actually rising across the quarters.
        - Revenue volatility is a coefficient of variation, so it is comparable across company
          sizes. Seasonal businesses score high on it for reasons that are not quality problems;
          say so when the pattern looks seasonal rather than erratic.
        - Net income above operating income means profit came from outside operations. Treat that
          as a one-time signal when judging quality, whatever the indicator scored.
        - Earnings growing while revenue does not usually means cost cuts or one-off gains, not
          a healthier business.
        - If an indicator is unknown, say so and lower your confidence instead of guessing.

        ## Output Format
        ---EARNINGS ANALYSIS---
        1- Delivery Record: beat rate, pattern label, and the quarters behind it
        2- Notable Surprises: any large beat or miss and what it was
        3- Earnings Quality: the category, with each of the four indicators and its verdict
        4- Red Flags: anything that undermines the reported numbers, or "none identified"
        5- Overall earnings assessment with a confidence of low/medium/high
        """

    # ------------------------------------------------------------------ #
    # Main call
    # ------------------------------------------------------------------ #

    async def analyze_earnings(
        self,
        symbol: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Analyze quarterly earnings for `symbol`.

        Args:
            symbol: Ticker symbol, for example "AAPL".
            context: Extra context merged into the payload sent to the model.

        Returns:
            The AgentResult from the inherited execute(). The computed pattern
            and quality are also attached to result.data so the orchestrator can
            read them without parsing the write-up.
        """
        symbol = symbol.strip().upper()
        display_name = self._ticker(symbol).info.get("longName") \
            or self._ticker(symbol).info.get("shortName") \
                or symbol

        task = (
            f"Perform a quarterly earnings analysis of {display_name} ({symbol}) over the "
            f"{self.quarters} most recent reported quarters."
        )

        result = await self.execute(task=task, context=context)
        if result.success:
            result.data["symbol"] = symbol
            result.data["analysis_type"] = "Earnings"
            pattern = self.earnings_pattern(symbol)
            quality = self.earnings_quality(symbol)
            result.data["earnings_pattern"] = pattern.get("pattern")
            result.data["beat_rate_pct"] = pattern.get("beat_rate_pct")
            result.data["earnings_quality"] = quality.get("quality")
        return result
