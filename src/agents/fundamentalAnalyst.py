"""
Fundamental analysis sub-agent.

Wraps yfinance as a data source and exposes the classic fundamental ratios
(P/E, EV/EBITDA, margins, ROE, liquidity, quick, cash, revenue growth) both as
plain Python methods and as LangChain tools the LLM can call on its own.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import yfinance as yf
from langchain_core.tools import BaseTool, tool

try:  # running with `src` on the path
    from src.agents.baseAgent import AgentResult, BaseAgent
except ImportError:  # running from inside `src/agents`
    from baseAgent import AgentResult, BaseAgent


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _norm(label: Any) -> str:
    """Normalize a statement row label so lookups survive yfinance renames."""
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def _to_float(value: Any) -> Optional[float]:
    """Best-effort float conversion; returns None for NaN/None/non-numeric."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Division that returns None instead of raising on missing/zero inputs."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _row_value(df, candidates: List[str], col: int = 0) -> Optional[float]:
    """Pull a value from a yfinance statement frame, trying several row labels."""
    if df is None or getattr(df, "empty", True):
        return None
    if col >= len(df.columns):
        return None
    index_map = {_norm(idx): idx for idx in df.index}
    for candidate in candidates:
        key = _norm(candidate)
        if key in index_map:
            value = _to_float(df.loc[index_map[key]].iloc[col])
            if value is not None:
                return value
    return None


def _col_label(df, col: int = 0) -> Optional[str]:
    """Human readable period label for a statement column."""
    if df is None or getattr(df, "empty", True) or col >= len(df.columns):
        return None
    label = df.columns[col]
    date_fn = getattr(label, "date", None)
    return str(date_fn()) if callable(date_fn) else str(label)


def _frame_to_dict(df, max_periods: int = 4, keep_rows: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convert a statement frame to {period: {line_item: value}} for prompting."""
    if df is None or getattr(df, "empty", True):
        return {}
    frame = df.iloc[:, :max_periods]
    if keep_rows:
        wanted = {_norm(r) for r in keep_rows}
        rows = [idx for idx in frame.index if _norm(idx) in wanted]
        if rows:
            frame = frame.loc[rows]
    else:
        frame = frame.head(40)

    out: Dict[str, Any] = {}
    for col in frame.columns:
        date_fn = getattr(col, "date", None)
        period = str(date_fn()) if callable(date_fn) else str(col)
        out[period] = {str(idx): _to_float(frame.loc[idx, col]) for idx in frame.index}
    return out


# Statement rows kept when trimming the frames sent to the model.
_INCOME_ROWS = [
    "Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income",
    "Operating Expense", "EBITDA", "EBIT", "Net Income", "Diluted EPS", "Basic EPS",
]
_BALANCE_ROWS = [
    "Total Assets", "Total Liabilities Net Minority Interest", "Current Assets",
    "Current Liabilities", "Inventory", "Cash And Cash Equivalents",
    "Other Short Term Investments", "Cash Cash Equivalents And Short Term Investments",
    "Stockholders Equity", "Total Debt", "Retained Earnings",
]
_CASHFLOW_ROWS = [
    "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
    "Free Cash Flow", "Capital Expenditure", "Depreciation And Amortization",
]

# Row-label candidates for each figure the ratios need.
_REVENUE = ["Total Revenue", "Operating Revenue", "Revenue"]
_GROSS_PROFIT = ["Gross Profit"]
_COST_OF_REVENUE = ["Cost Of Revenue", "Cost Of Goods Sold"]
_OPERATING_INCOME = ["Operating Income", "EBIT", "Total Operating Income As Reported"]
_NET_INCOME = ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"]
_EBITDA = ["EBITDA", "Normalized EBITDA"]
_EQUITY = [
    "Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity",
    "Total Equity Gross Minority Interest",
]
_TOTAL_ASSETS = ["Total Assets"]
_TOTAL_LIABILITIES = ["Total Liabilities Net Minority Interest", "Total Liabilities"]
_CURRENT_ASSETS = ["Current Assets", "Total Current Assets"]
_CURRENT_LIABILITIES = ["Current Liabilities", "Total Current Liabilities"]
_INVENTORY = ["Inventory", "Inventories"]
_CASH = ["Cash And Cash Equivalents", "Cash Financial", "Cash And Cash Equivalents At Carrying Value"]
_SHORT_TERM_INV = ["Other Short Term Investments", "Short Term Investments"]
_CASH_AND_STI = ["Cash Cash Equivalents And Short Term Investments"]
_DEP_AMORT = ["Depreciation And Amortization", "Depreciation Amortization Depletion"]


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class FundamentalAnalystAgent(BaseAgent):
    """
    Sub-agent that answers the "is this company financially sound and fairly
    priced?" half of the research loop.

    Data comes from yfinance only. Every ratio is computed here from the reported
    statements so the LLM never has to do arithmetic on raw figures itself.
    """

    DEFAULT_NAME = "fundamental_analyst"
    DEFAULT_DESCRIPTION = (
        "Analyzes company fundamentals: income statement, balance sheet, "
        "valuation multiples, margins, returns, liquidity and revenue growth."
    )

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
        history_period: str = "1y",
    ):
        # Set before super().__init__: BaseAgent calls _get_tools() and _get_prompt().
        self.history_period = history_period
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

    def _info(self, symbol: str) -> Dict[str, Any]:
        slot = self._slot(symbol)
        if "info" not in slot:
            try:
                slot["info"] = dict(self._ticker(symbol).info or {})
            except Exception as exc:  # yfinance raises a wide range of errors
                slot["info"] = {"error": f"Could not fetch info: {exc}"}
        return slot["info"]

    def _statement(self, symbol: str, kind: str):
        """Fetch and cache an annual statement frame. kind: income | balance | cashflow."""
        slot = self._slot(symbol)
        key = f"stmt_{kind}"
        if key not in slot:
            attr = {"income": "income_stmt", "balance": "balance_sheet", "cashflow": "cashflow"}[kind]
            try:
                slot[key] = getattr(self._ticker(symbol), attr)
            except Exception:
                slot[key] = None
        return slot[key]

    # -------------------------- data fetchers -------------------------- #

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Company profile, sector, and headline valuation figures."""
        symbol = symbol.strip().upper()
        info = self._info(symbol)
        if len(info) == 1 and "error" in info:
            return {"symbol": symbol, "error": info["error"]}

        fields = [
            "shortName", "longName", "sector", "industry", "country", "website",
            "fullTimeEmployees", "marketCap", "enterpriseValue", "currency",
            "trailingPE", "forwardPE", "priceToBook", "trailingEps", "forwardEps",
            "beta", "dividendYield", "payoutRatio", "sharesOutstanding",
            "totalDebt", "totalCash", "recommendationKey", "targetMeanPrice",
        ]
        profile: Dict[str, Any] = {f: info.get(f) for f in fields if info.get(f) is not None}
        profile["symbol"] = symbol
        summary = info.get("longBusinessSummary")
        if summary:
            profile["businessSummary"] = str(summary)[:800]
        return profile

    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Latest quote plus the 52-week context around it."""
        symbol = symbol.strip().upper()
        info = self._info(symbol)
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if price is None:
            try:
                hist = self._ticker(symbol).history(period="5d")
                if hist is not None and not hist.empty:
                    price = _to_float(hist["Close"].iloc[-1])
            except Exception as exc:
                return {"symbol": symbol, "error": f"Could not fetch price: {exc}"}

        price = _to_float(price)
        prev_close = _to_float(info.get("previousClose"))
        change = (price - prev_close) if (price is not None and prev_close is not None) else None
        change_pct = _safe_div(change, prev_close)

        return {
            "symbol": symbol,
            "price": price,
            "previous_close": prev_close,
            "change_pct": round(change_pct * 100, 2) if change_pct is not None else None,
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_day_average": info.get("fiftyDayAverage"),
            "two_hundred_day_average": info.get("twoHundredDayAverage"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency"),
        }

    def get_historical_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """Summarize the price history over `period` (defaults to history_period)."""
        symbol = symbol.strip().upper()
        period = period or self.history_period
        try:
            hist = self._ticker(symbol).history(period=period, interval=interval)
        except Exception as exc:
            return {"symbol": symbol, "error": f"Could not fetch history: {exc}"}
        if hist is None or hist.empty:
            return {"symbol": symbol, "error": f"No history returned for period={period}"}

        closes = hist["Close"].dropna()
        if closes.empty:
            return {"symbol": symbol, "error": "History contained no closing prices."}

        first = _to_float(closes.iloc[0])
        last = _to_float(closes.iloc[-1])
        period_return = _safe_div((last - first) if (first is not None and last is not None) else None, first)

        returns = closes.pct_change().dropna()
        annualized_vol = _to_float(returns.std() * (252 ** 0.5)) if len(returns) > 1 else None

        sampled = closes.resample("ME").last().tail(12) if len(closes) > 20 else closes.tail(12)

        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "start_date": str(closes.index[0].date()),
            "end_date": str(closes.index[-1].date()),
            "start_close": first,
            "end_close": last,
            "period_return_pct": round(period_return * 100, 2) if period_return is not None else None,
            "period_high": _to_float(hist["High"].max()),
            "period_low": _to_float(hist["Low"].min()),
            "average_volume": _to_float(hist["Volume"].mean()) if "Volume" in hist else None,
            "annualized_volatility_pct": round(annualized_vol * 100, 2) if annualized_vol is not None else None,
            "closes_by_period": {str(idx.date()): _to_float(val) for idx, val in sampled.items()},
        }

    def get_financial_statements(self, symbol: str, max_periods: int = 4) -> Dict[str, Any]:
        """Annual income statement, balance sheet and cash flow, trimmed to key rows."""
        symbol = symbol.strip().upper()
        statements = {
            "symbol": symbol,
            "income_statement": _frame_to_dict(self._statement(symbol, "income"), max_periods, _INCOME_ROWS),
            "balance_sheet": _frame_to_dict(self._statement(symbol, "balance"), max_periods, _BALANCE_ROWS),
            "cash_flow": _frame_to_dict(self._statement(symbol, "cashflow"), max_periods, _CASHFLOW_ROWS),
        }
        if not any(statements[k] for k in ("income_statement", "balance_sheet", "cash_flow")):
            statements["error"] = "No financial statements available from yfinance for this symbol."
        return statements

    # ------------------------------------------------------------------ #
    # Ratio calculations
    # ------------------------------------------------------------------ #

    @staticmethod
    def _metric(name: str, value: Optional[float], unit: str, **extra) -> Dict[str, Any]:
        """Uniform envelope so the model sees value, unit, inputs and period together."""
        payload: Dict[str, Any] = {"metric": name, "value": value, "unit": unit}
        payload.update({k: v for k, v in extra.items() if v is not None})
        if value is None and "error" not in payload:
            payload["error"] = "Required inputs were not reported by yfinance."
        return payload

    def pe_ratio(self, symbol: str) -> Dict[str, Any]:
        """Price / trailing earnings per share."""
        info = self._info(symbol)
        price = _to_float(
            info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        )
        eps = _to_float(info.get("trailingEps"))
        value = _safe_div(price, eps)
        if value is None:
            value = _to_float(info.get("trailingPE"))
        note = "Negative or zero EPS makes P/E meaningless." if (eps is not None and eps <= 0) else None
        return self._metric(
            "P/E (trailing)",
            round(value, 2) if value is not None else None,
            "x",
            inputs={"price": price, "trailing_eps": eps},
            forward_pe=_to_float(info.get("forwardPE")),
            note=note,
        )

    def ev_ebitda(self, symbol: str) -> Dict[str, Any]:
        """Enterprise value / EBITDA."""
        info = self._info(symbol)
        income = self._statement(symbol, "income")
        cashflow = self._statement(symbol, "cashflow")

        ev = _to_float(info.get("enterpriseValue"))
        if ev is None:
            market_cap = _to_float(info.get("marketCap"))
            debt = _to_float(info.get("totalDebt")) or 0.0
            cash = _to_float(info.get("totalCash")) or 0.0
            ev = (market_cap + debt - cash) if market_cap is not None else None

        ebitda = _row_value(income, _EBITDA) or _to_float(info.get("ebitda"))
        if ebitda is None:
            operating_income = _row_value(income, _OPERATING_INCOME)
            dep_amort = _row_value(cashflow, _DEP_AMORT)
            if operating_income is not None and dep_amort is not None:
                ebitda = operating_income + dep_amort

        value = _safe_div(ev, ebitda)
        note = "Negative EBITDA makes this multiple meaningless." if (ebitda is not None and ebitda <= 0) else None
        return self._metric(
            "EV/EBITDA",
            round(value, 2) if value is not None else None,
            "x",
            inputs={"enterprise_value": ev, "ebitda": ebitda},
            period=_col_label(income),
            note=note,
        )

    def gross_margin(self, symbol: str) -> Dict[str, Any]:
        """Gross profit / revenue for the most recent fiscal year."""
        income = self._statement(symbol, "income")
        revenue = _row_value(income, _REVENUE)
        gross = _row_value(income, _GROSS_PROFIT)
        if gross is None:
            cost = _row_value(income, _COST_OF_REVENUE)
            if revenue is not None and cost is not None:
                gross = revenue - cost
        value = _safe_div(gross, revenue)
        return self._metric(
            "Gross margin",
            round(value * 100, 2) if value is not None else None,
            "%",
            inputs={"gross_profit": gross, "revenue": revenue},
            period=_col_label(income),
        )

    def operating_margin(self, symbol: str) -> Dict[str, Any]:
        """Operating income / revenue for the most recent fiscal year."""
        income = self._statement(symbol, "income")
        revenue = _row_value(income, _REVENUE)
        operating_income = _row_value(income, _OPERATING_INCOME)
        value = _safe_div(operating_income, revenue)
        return self._metric(
            "Operating margin",
            round(value * 100, 2) if value is not None else None,
            "%",
            inputs={"operating_income": operating_income, "revenue": revenue},
            period=_col_label(income),
        )

    def roe(self, symbol: str) -> Dict[str, Any]:
        """Net income / shareholders' equity (ending balance)."""
        income = self._statement(symbol, "income")
        balance = self._statement(symbol, "balance")
        net_income = _row_value(income, _NET_INCOME)
        equity = _row_value(balance, _EQUITY)
        value = _safe_div(net_income, equity)
        note = "Computed on ending equity, not average equity."
        if equity is not None and equity < 0:
            note += " Equity is negative, which inverts the sign and makes the ratio unreliable."
        return self._metric(
            "Return on equity",
            round(value * 100, 2) if value is not None else None,
            "%",
            inputs={"net_income": net_income, "shareholders_equity": equity},
            period=_col_label(income),
            note=note,
        )

    def liquidity_ratio(self, symbol: str) -> Dict[str, Any]:
        """Total assets / total liabilities."""
        balance = self._statement(symbol, "balance")
        assets = _row_value(balance, _TOTAL_ASSETS)
        liabilities = _row_value(balance, _TOTAL_LIABILITIES)
        value = _safe_div(assets, liabilities)
        return self._metric(
            "Liquidity ratio (assets/liabilities)",
            round(value, 2) if value is not None else None,
            "x",
            inputs={"total_assets": assets, "total_liabilities": liabilities},
            period=_col_label(balance),
            note="Coverage of all liabilities by all assets; above 1.0 means positive book equity.",
        )

    def quick_ratio(self, symbol: str) -> Dict[str, Any]:
        """(Current assets - inventory) / current liabilities."""
        balance = self._statement(symbol, "balance")
        current_assets = _row_value(balance, _CURRENT_ASSETS)
        inventory = _row_value(balance, _INVENTORY) or 0.0
        current_liabilities = _row_value(balance, _CURRENT_LIABILITIES)
        numerator = (current_assets - inventory) if current_assets is not None else None
        value = _safe_div(numerator, current_liabilities)
        if value is None:
            value = _to_float(self._info(symbol).get("quickRatio"))
        return self._metric(
            "Quick ratio",
            round(value, 2) if value is not None else None,
            "x",
            inputs={
                "current_assets": current_assets,
                "inventory": inventory,
                "current_liabilities": current_liabilities,
            },
            period=_col_label(balance),
            note="Below 1.0 means liquid assets do not cover liabilities due within a year.",
        )

    def cash_ratio(self, symbol: str) -> Dict[str, Any]:
        """(Cash + short-term investments) / current liabilities."""
        balance = self._statement(symbol, "balance")
        cash_and_sti = _row_value(balance, _CASH_AND_STI)
        if cash_and_sti is None:
            cash = _row_value(balance, _CASH)
            short_term = _row_value(balance, _SHORT_TERM_INV) or 0.0
            cash_and_sti = (cash + short_term) if cash is not None else None
        current_liabilities = _row_value(balance, _CURRENT_LIABILITIES)
        value = _safe_div(cash_and_sti, current_liabilities)
        return self._metric(
            "Cash ratio",
            round(value, 2) if value is not None else None,
            "x",
            inputs={
                "cash_and_short_term_investments": cash_and_sti,
                "current_liabilities": current_liabilities,
            },
            period=_col_label(balance),
        )

    def revenue_growth_yoy(self, symbol: str) -> Dict[str, Any]:
        """Year-over-year revenue growth from the two most recent fiscal years."""
        income = self._statement(symbol, "income")
        latest = _row_value(income, _REVENUE, col=0)
        prior = _row_value(income, _REVENUE, col=1)

        value = None
        period = _col_label(income)
        if latest is not None and prior not in (None, 0):
            value = (latest - prior) / abs(prior)
            period = f"{_col_label(income, 1)} to {_col_label(income, 0)}"
        if value is None:
            value = _to_float(self._info(symbol).get("revenueGrowth"))

        return self._metric(
            "Revenue growth YoY",
            round(value * 100, 2) if value is not None else None,
            "%",
            inputs={"latest_revenue": latest, "prior_revenue": prior},
            period=period,
        )

    def calculate_all_ratios(self, symbol: str) -> Dict[str, Any]:
        """Run every ratio in one pass and return them keyed by name."""
        symbol = symbol.strip().upper()
        calculators = {
            "pe_ratio": self.pe_ratio,
            "ev_ebitda": self.ev_ebitda,
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "roe": self.roe,
            "liquidity_ratio": self.liquidity_ratio,
            "quick_ratio": self.quick_ratio,
            "cash_ratio": self.cash_ratio,
            "revenue_growth_yoy": self.revenue_growth_yoy,
        }
        results: Dict[str, Any] = {"symbol": symbol}
        for key, calculate in calculators.items():
            try:
                results[key] = calculate(symbol)
            except Exception as exc:
                results[key] = {"metric": key, "value": None, "error": str(exc)}
        return results

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    def _get_tools(self) -> List[BaseTool]:
        """Expose the yfinance fetchers and ratio calculators as LLM tools."""

        def _dump(payload: Any) -> str:
            return json.dumps(payload, default=str, indent=2)

        @tool("get_company_info")
        def get_company_info_tool(symbol: str) -> str:
            """Get the company profile, sector, industry, market cap and headline valuation data for a ticker symbol."""
            return _dump(self.get_company_info(symbol))

        @tool("get_stock_price")
        def get_stock_price_tool(symbol: str) -> str:
            """Get the latest stock price, daily move, 52-week range and moving averages for a ticker symbol."""
            return _dump(self.get_stock_price(symbol))

        @tool("get_historical_data")
        def get_historical_data_tool(symbol: str, period: str = "1y") -> str:
            """Get a price-history summary (return, high, low, volatility, periodic closes) for a ticker over a period such as 1mo, 6mo, 1y or 5y."""
            return _dump(self.get_historical_data(symbol, period=period))

        @tool("get_financial_statements")
        def get_financial_statements_tool(symbol: str) -> str:
            """Get the annual income statement, balance sheet and cash-flow line items for a ticker symbol."""
            return _dump(self.get_financial_statements(symbol))

        @tool("calculate_pe_ratio")
        def pe_ratio_tool(symbol: str) -> str:
            """Calculate the trailing price-to-earnings (P/E) ratio for a ticker symbol."""
            return _dump(self.pe_ratio(symbol))

        @tool("calculate_ev_ebitda")
        def ev_ebitda_tool(symbol: str) -> str:
            """Calculate the enterprise-value-to-EBITDA (EV/EBITDA) multiple for a ticker symbol."""
            return _dump(self.ev_ebitda(symbol))

        @tool("calculate_gross_margin")
        def gross_margin_tool(symbol: str) -> str:
            """Calculate the gross margin percentage (gross profit / revenue) for a ticker symbol."""
            return _dump(self.gross_margin(symbol))

        @tool("calculate_operating_margin")
        def operating_margin_tool(symbol: str) -> str:
            """Calculate the operating margin percentage (operating income / revenue) for a ticker symbol."""
            return _dump(self.operating_margin(symbol))

        @tool("calculate_roe")
        def roe_tool(symbol: str) -> str:
            """Calculate return on equity (net income / shareholders' equity) for a ticker symbol."""
            return _dump(self.roe(symbol))

        @tool("calculate_liquidity_ratio")
        def liquidity_ratio_tool(symbol: str) -> str:
            """Calculate the liquidity ratio defined as total assets divided by total liabilities for a ticker symbol."""
            return _dump(self.liquidity_ratio(symbol))

        @tool("calculate_quick_ratio")
        def quick_ratio_tool(symbol: str) -> str:
            """Calculate the quick ratio, (current assets minus inventory) divided by current liabilities, for a ticker symbol."""
            return _dump(self.quick_ratio(symbol))

        @tool("calculate_cash_ratio")
        def cash_ratio_tool(symbol: str) -> str:
            """Calculate the cash ratio, (cash plus short-term investments) divided by current liabilities, for a ticker symbol."""
            return _dump(self.cash_ratio(symbol))

        @tool("calculate_revenue_growth_yoy")
        def revenue_growth_tool(symbol: str) -> str:
            """Calculate year-over-year revenue growth from the last two reported fiscal years for a ticker symbol."""
            return _dump(self.revenue_growth_yoy(symbol))

        @tool("calculate_all_ratios")
        def all_ratios_tool(symbol: str) -> str:
            """Calculate every supported fundamental ratio at once: P/E, EV/EBITDA, gross margin, operating margin, ROE, liquidity ratio, quick ratio, cash ratio and revenue growth."""
            return _dump(self.calculate_all_ratios(symbol))

        return [
            get_company_info_tool,
            get_stock_price_tool,
            get_historical_data_tool,
            get_financial_statements_tool,
            pe_ratio_tool,
            ev_ebitda_tool,
            gross_margin_tool,
            operating_margin_tool,
            roe_tool,
            liquidity_ratio_tool,
            quick_ratio_tool,
            cash_ratio_tool,
            revenue_growth_tool,
            all_ratios_tool,
        ]

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #

    def _get_prompt(self) -> str:
        """Fundamental-analysis system prompt."""
        return """You are a fundamental equity analyst working as part of a multi-agent research team.
Your scope is the financial health and valuation of a single public company. News sentiment and
earnings-surprise analysis belong to other agents, so do not speculate about them.

DATA RULES
- Every number you state must come from a tool result or from the context supplied with the task.
- Never estimate, recall from memory, or invent a missing figure. If a value is not reported, say
  "not reported" and explain what that prevents you from concluding.
- The ratios are computed for you. Read the value and its inputs; do not recompute them.
- Always name the fiscal period a figure belongs to, and the currency where it matters.

HOW TO ANALYZE
1. Business context: what the company does, its sector, and its scale (market cap, revenue).
2. Profitability: gross margin and operating margin, and whether they are strong or thin for that
   sector. Read the trend across periods whenever more than one is reported.
3. Returns: return on equity, noting when leverage or negative equity distorts it.
4. Growth: year-over-year revenue growth, and whether growth and margins move together.
5. Balance sheet: liquidity ratio (assets over liabilities), quick ratio and cash ratio. Treat a
   quick ratio below 1.0 as short-term funding pressure and explain the exposure.
6. Valuation: P/E and EV/EBITDA. Say plainly when negative or missing earnings make P/E
   meaningless, and lean on EV/EBITDA instead. Judge the multiple against the company's own
   growth and margins rather than an index level you cannot verify.

OUTPUT FORMAT
- Summary: two or three sentences on the company's fundamental position.
- Key metrics: a short list of metric, value and period.
- Strengths: bullet points, each tied to a specific figure.
- Risks and weaknesses: bullet points, each tied to a specific figure.
- Valuation view: do the fundamentals justify the multiple?
- Confidence: low, medium or high, plus the data gaps behind that rating.

Be direct about weak fundamentals. Do not soften a poor balance sheet, and do not give buy or sell
advice. Report what the financials show and let the lead agent decide."""

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def _collect_fundamentals(self, symbol: str, period: str) -> Dict[str, Any]:
        """Blocking gather of everything yfinance can give us for one symbol."""
        symbol = symbol.strip().upper()
        bundle: Dict[str, Any] = {"symbol": symbol}
        sections = {
            "company_info": lambda: self.get_company_info(symbol),
            "stock_price": lambda: self.get_stock_price(symbol),
            "price_history": lambda: self.get_historical_data(symbol, period=period),
            "financial_statements": lambda: self.get_financial_statements(symbol),
            "ratios": lambda: self.calculate_all_ratios(symbol),
        }
        for key, fetch in sections.items():
            try:
                bundle[key] = fetch()
            except Exception as exc:
                bundle[key] = {"error": str(exc)}
        return bundle

    async def analyze_financials(
        self,
        symbol: str,
        period: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Fetch fundamentals for `symbol` from yfinance and have the LLM analyze them.

        Args:
            symbol: Ticker symbol, for example "AAPL".
            period: Price-history window; defaults to the agent's history_period.
            context: Extra context merged into the payload sent to the model.

        Returns:
            The AgentResult from the inherited execute(). The fetched data is also
            attached under result.data["fundamentals"].
        """
        symbol = symbol.strip().upper()
        period = period or self.history_period

        # yfinance is blocking, so keep the event loop free while it fetches.
        fundamentals = await asyncio.to_thread(self._collect_fundamentals, symbol, period)
        if context:
            fundamentals["additional_context"] = context

        company = fundamentals.get("company_info") or {}
        display_name = company.get("longName") or company.get("shortName") or symbol

        task = (
            f"Perform a fundamental analysis of {display_name} ({symbol}).\n\n"
            "The company profile, latest price, price history, annual financial statements and all "
            "pre-computed ratios are provided in the context below. Use them as your primary source, "
            "and call a tool only if a figure you need is missing from that context.\n\n"
            "Cover profitability (gross and operating margin), returns (ROE), year-over-year revenue "
            "growth, balance-sheet strength (assets over liabilities, quick ratio, cash ratio) and "
            "valuation (P/E, EV/EBITDA). Finish with the structured output described in your "
            "instructions, including a confidence rating and the data gaps behind it."
        )

        result = await self.execute(task=task, context=fundamentals)
        if result.success:
            result.data["symbol"] = symbol
            result.data["fundamentals"] = fundamentals
        return result