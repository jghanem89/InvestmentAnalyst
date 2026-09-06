"""
Fundamental analysis sub-agent.

Wraps yfinance as a data source and exposes the classic fundamental ratios
(P/E, EV/EBITDA, margins, ROE, liquidity, quick, cash, revenue growth) plus a
simplified discounted cash flow model, both as plain Python methods and as
LangChain tools the LLM can call on its own.
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
_OPERATING_CF = [
    "Operating Cash Flow", "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
]
_CAPEX = ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"]

# Typical gross margin (%) by yfinance industry label, looked up through _norm() so
# the em dashes and ampersands in those labels do not matter. Static reference points
# for "is this margin normal for what the company sells?", not live peer data.
_GROSS_MARGIN_BY_INDUSTRY = {
    "Software - Infrastructure": 72.0,
    "Software - Application": 70.0,
    "Information Technology Services": 30.0,
    "Semiconductors": 50.0,
    "Semiconductor Equipment & Materials": 45.0,
    "Computer Hardware": 35.0,
    "Consumer Electronics": 40.0,
    "Communication Equipment": 50.0,
    "Electronic Components": 30.0,
    "Internet Content & Information": 55.0,
    "Internet Retail": 40.0,
    "Entertainment": 40.0,
    "Telecom Services": 55.0,
    "Biotechnology": 80.0,
    "Drug Manufacturers - General": 70.0,
    "Drug Manufacturers - Specialty & Generic": 55.0,
    "Medical Devices": 60.0,
    "Healthcare Plans": 20.0,
    "Medical Distribution": 10.0,
    "Discount Stores": 25.0,
    "Grocery Stores": 25.0,
    "Restaurants": 30.0,
    "Apparel Retail": 45.0,
    "Auto Manufacturers": 18.0,
    "Aerospace & Defense": 20.0,
    "Airlines": 25.0,
    "Oil & Gas Integrated": 30.0,
    "Oil & Gas E&P": 50.0,
    "Utilities - Regulated Electric": 35.0,
}

# Sector-level fallback when the industry label is missing or unlisted.
_GROSS_MARGIN_BY_SECTOR = {
    "Technology": 55.0,
    "Healthcare": 55.0,
    "Communication Services": 50.0,
    "Financial Services": 60.0,
    "Consumer Cyclical": 35.0,
    "Consumer Defensive": 30.0,
    "Industrials": 30.0,
    "Basic Materials": 25.0,
    "Energy": 30.0,
    "Utilities": 35.0,
    "Real Estate": 45.0,
}

# Last resort when yfinance reports neither an industry nor a sector.
_GROSS_MARGIN_DEFAULT = 35.0

# Typical operating margin (%) for the same industries. Operating margin sits far
# below gross margin because it carries R&D, selling and administrative costs.
_OPERATING_MARGIN_BY_INDUSTRY = {
    "Software - Infrastructure": 30.0,
    "Software - Application": 12.0,
    "Information Technology Services": 10.0,
    "Semiconductors": 25.0,
    "Semiconductor Equipment & Materials": 22.0,
    "Computer Hardware": 10.0,
    "Consumer Electronics": 25.0,
    "Communication Equipment": 15.0,
    "Electronic Components": 10.0,
    "Internet Content & Information": 25.0,
    "Internet Retail": 6.0,
    "Entertainment": 10.0,
    "Telecom Services": 18.0,
    "Biotechnology": 20.0,
    "Drug Manufacturers - General": 25.0,
    "Drug Manufacturers - Specialty & Generic": 15.0,
    "Medical Devices": 18.0,
    "Healthcare Plans": 5.0,
    "Medical Distribution": 2.0,
    "Discount Stores": 5.0,
    "Grocery Stores": 3.0,
    "Restaurants": 12.0,
    "Apparel Retail": 10.0,
    "Auto Manufacturers": 7.0,
    "Aerospace & Defense": 9.0,
    "Airlines": 8.0,
    "Oil & Gas Integrated": 12.0,
    "Oil & Gas E&P": 25.0,
    "Utilities - Regulated Electric": 20.0,
}

_OPERATING_MARGIN_BY_SECTOR = {
    "Technology": 20.0,
    "Healthcare": 12.0,
    "Communication Services": 18.0,
    "Financial Services": 25.0,
    "Consumer Cyclical": 8.0,
    "Consumer Defensive": 7.0,
    "Industrials": 10.0,
    "Basic Materials": 10.0,
    "Energy": 12.0,
    "Utilities": 18.0,
    "Real Estate": 25.0,
}

_OPERATING_MARGIN_DEFAULT = 10.0


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
        "valuation multiples, margins, returns, liquidity, revenue growth and a "
        "simplified discounted cash flow valuation."
    )

    # Simplified DCF assumptions.
    DCF_DISCOUNT_RATE = 0.10
    DCF_TERMINAL_GROWTH = 0.02
    DCF_FORECAST_YEARS = 10

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

        # print(f"Calculated P/E ratio is {value}")

        note = None
        if value is not None:
            if eps <= 0:
                note = "Negative or zero EPS makes P/E meaningless."
            elif value < 15:
                note = "Undervalued"
            elif value > 30:
                note = "Overvalued"
            else:
                note = "Fairly valued"

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
        # print(f"Calculated EV/EBIDTA ratio is {value}")

        note = None
        if value is not None:
            if ebitda <=0:
                note = "Negative EBITDA makes this multiple meaningless."
            elif value < 10:
                note = "Attractive"
            elif value > 15:
                note = "Expensive"
            else:
                note = "Fair"
        
        return self._metric(
            "EV/EBITDA",
            round(value, 2) if value is not None else None,
            "x",
            inputs={"enterprise_value": ev, "ebitda": ebitda},
            period=_col_label(income),
            note=note,
        )

    def _margin_benchmark(
        self,
        symbol: str,
        by_industry: Dict[str, float],
        by_sector: Dict[str, float],
        default: float,
    ) -> Dict[str, Any]:
        """Typical margin for the company's industry, falling back to its sector."""
        info = self._info(symbol)
        industry = info.get("industry")
        sector = info.get("sector")

        index_map = {_norm(label): value for label, value in by_industry.items()}
        benchmark = index_map.get(_norm(industry)) if industry else None
        if benchmark is not None:
            return {"benchmark": benchmark, "peer_group": str(industry), "basis": "industry"}

        index_map = {_norm(label): value for label, value in by_sector.items()}
        benchmark = index_map.get(_norm(sector)) if sector else None
        if benchmark is not None:
            return {"benchmark": benchmark, "peer_group": str(sector), "basis": "sector"}

        return {"benchmark": default, "peer_group": "the broad market", "basis": "default"}

    def gross_margin(self, symbol: str) -> Dict[str, Any]:
        """Gross profit / revenue for the most recent fiscal year, against its industry benchmark."""
        income = self._statement(symbol, "income")
        revenue = _row_value(income, _REVENUE)
        gross = _row_value(income, _GROSS_PROFIT)
        if gross is None:
            cost = _row_value(income, _COST_OF_REVENUE)
            if revenue is not None and cost is not None:
                gross = revenue - cost
        value = _safe_div(gross, revenue)
        margin = round(value * 100, 2) if value is not None else None

        info = self._info(symbol)
        peers = self._margin_benchmark(
            symbol, _GROSS_MARGIN_BY_INDUSTRY, _GROSS_MARGIN_BY_SECTOR, _GROSS_MARGIN_DEFAULT
        )
        benchmark = peers["benchmark"]

        # print(f"Calculated gross margin is {margin}")

        gap = None
        note = None
        if margin is not None:
            gap = round(margin - benchmark, 2)
            if gap > 10:
                note = "Above average"
            elif gap < -10:
                note = "Below Average"
            else:
                note = "Average"

        return self._metric(
            "Gross margin",
            margin,
            "%",
            inputs={"gross_profit": gross, "revenue": revenue},
            period=_col_label(income),
            industry=info.get("industry"),
            sector=info.get("sector"),
            industry_benchmark_pct=benchmark,
            benchmark_basis=peers["basis"],
            vs_benchmark_pp=gap,
            note=note,
        )

    def operating_margin(self, symbol: str) -> Dict[str, Any]:
        """Operating income / revenue for the most recent fiscal year, against its industry benchmark."""
        income = self._statement(symbol, "income")
        revenue = _row_value(income, _REVENUE)
        operating_income = _row_value(income, _OPERATING_INCOME)
        value = _safe_div(operating_income, revenue)
        margin = round(value * 100, 2) if value is not None else None

        info = self._info(symbol)
        peers = self._margin_benchmark(
            symbol, _OPERATING_MARGIN_BY_INDUSTRY, _OPERATING_MARGIN_BY_SECTOR, _OPERATING_MARGIN_DEFAULT
        )
        benchmark = peers["benchmark"]

        # print(f"Calculated oeprating margin is {margin}")
        
        gap = None
        note = None
        if margin is not None:
            # Narrower band than gross margin: operating margins are smaller numbers.
            gap = round(margin - benchmark, 2)
            if gap > 5:
                note = "Above average"
            elif gap < -5:
                note = "Below average"
            else:
                note = "Average"

        return self._metric(
            "Operating margin",
            margin,
            "%",
            inputs={"operating_income": operating_income, "revenue": revenue},
            period=_col_label(income),
            industry=info.get("industry"),
            sector=info.get("sector"),
            industry_benchmark_pct=benchmark,
            benchmark_basis=peers["basis"],
            vs_benchmark_pp=gap,
            note=note,
        )

    def roe(self, symbol: str) -> Dict[str, Any]:
        """Net income / shareholders' equity (ending balance)."""
        income = self._statement(symbol, "income")
        balance = self._statement(symbol, "balance")
        net_income = _row_value(income, _NET_INCOME)
        equity = _row_value(balance, _EQUITY)
        value = _safe_div(net_income, equity)

        # print(f"Calculated ROE is {value}")

        note = None
        if value is not None:
            if value < 0:
                note = "Negative equity - unreliable ratio"
            elif value > 0.2:
                note = "Excellent"
            elif value > 0.15:
                note = "Good"
            elif value > 0.1:
                note = "Average"
            else:
                note = "Below Average"

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

        # print(f"Calculated Liquidity ratio is {value}")

        note = None
        if value is not None:
            if value > 2.0:
                note = "Strong liquidity"
            elif value > 1.0:
                note = "Adequate liquidity"
            else:
                note = "Weak liquidity"
        
        return self._metric(
            "Liquidity ratio (assets/liabilities)",
            round(value, 2) if value is not None else None,
            "x",
            inputs={"total_assets": assets, "total_liabilities": liabilities},
            period=_col_label(balance),
            note=note,
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

        # print(f"Calculated quick ratio is {value}")

        note = None
        if value is not None:
            if value >= 1.0:
                note = "1 year liabilities are covered by liquid assets"
            else:
                note = "1 year liabilities are not covered by liquid assets"

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
            note=note,
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

        # print(f"Calculated cash ratio is {value}")

        note = None
        if value is not None:
            if value > 1.0:
                note = "strong cash ratio"
            elif value > 0.5:
                note = "normal cash ratio"
            else:
                note = "weak cash ratio"

        return self._metric(
            "Cash ratio",
            round(value, 2) if value is not None else None,
            "x",
            inputs={
                "cash_and_short_term_investments": cash_and_sti,
                "current_liabilities": current_liabilities,
            },
            period=_col_label(balance),
            note=note
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

        # print(f"Calculated revenue growth is {value}")

        note = None
        if value is not None:
            if value > 0.5:
                note = "Hyper growth"
            elif value > 0.2:
                note = "High growth"
            elif value > 0.05:
                note = "Moderate growth"
            elif value > 0:
                note = "Stagnant or flat growth"
            else:
                note = "Negative growth"

        return self._metric(
            "Revenue growth YoY",
            round(value * 100, 2) if value is not None else None,
            "%",
            inputs={"latest_revenue": latest, "prior_revenue": prior},
            period=period,
            note=note
        )

    # ------------------------------------------------------------------ #
    # Discounted cash flow
    # ------------------------------------------------------------------ #

    def _free_cash_flows(self, symbol: str) -> List[Dict[str, Any]]:
        """Free cash flow per fiscal year, newest first: operating cash flow + capital expenditure."""
        cashflow = self._statement(symbol, "cashflow")
        if cashflow is None or getattr(cashflow, "empty", True):
            return []

        history: List[Dict[str, Any]] = []
        for col in range(len(cashflow.columns)):
            operating = _row_value(cashflow, _OPERATING_CF, col=col)
            capex = _row_value(cashflow, _CAPEX, col=col)
            if operating is None or capex is None:
                continue
            history.append({
                "period": _col_label(cashflow, col),
                "operating_cash_flow": operating,
                "capital_expenditure": capex,
                # yfinance reports capital expenditure as a negative number, so adding it subtracts.
                "free_cash_flow": operating + capex,
            })
        return history

    def discounted_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Simplified DCF intrinsic value per share.

        Free cash flow is operating cash flow plus capital expenditure, grown at its
        historical CAGR for ten years, discounted at 10%, with a 2% perpetual growth
        terminal value. The total present value is divided by shares outstanding.
        """
        symbol = symbol.strip().upper()
        history = self._free_cash_flows(symbol)
        if not history:
            return self._metric(
                "DCF intrinsic value per share", None, "currency/share",
                symbol=symbol,
                error="yfinance did not report the operating cash flow and capital expenditure "
                      "needed to build free cash flow.",
            )

        base_fcf = history[0]["free_cash_flow"]
        shares = _to_float(self._info(symbol).get("sharesOutstanding"))
        if base_fcf <= 0:
            return self._metric(
                "DCF intrinsic value per share", None, "currency/share",
                symbol=symbol,
                free_cash_flow_history=history,
                error="Latest free cash flow is negative or zero, so a growth-based DCF is not "
                      "meaningful for this company.",
            )
        if shares is None or shares <= 0:
            return self._metric(
                "DCF intrinsic value per share", None, "currency/share",
                symbol=symbol,
                free_cash_flow_history=history,
                error="yfinance did not report shares outstanding, so the total present value "
                      "cannot be converted to a per-share figure.",
            )

        # Growth rate: CAGR across the reported free cash flows, oldest to newest.
        oldest_fcf = history[-1]["free_cash_flow"]
        elapsed_years = len(history) - 1
        if elapsed_years >= 1 and oldest_fcf > 0:
            growth_rate = (base_fcf / oldest_fcf) ** (1.0 / elapsed_years) - 1.0
            growth_basis = (
                f"{elapsed_years}-year free-cash-flow CAGR "
                f"({history[-1]['period']} to {history[0]['period']})"
            )
        else:
            growth_rate = self.DCF_TERMINAL_GROWTH
            growth_basis = "terminal growth rate (free-cash-flow history too short for a CAGR)"

        discount_rate = self.DCF_DISCOUNT_RATE
        terminal_growth = self.DCF_TERMINAL_GROWTH

        # Project the forecast horizon and discount each year back to today.
        projections: List[Dict[str, Any]] = []
        pv_cf = 0.0
        cash_flow = base_fcf
        for year in range(1, self.DCF_FORECAST_YEARS + 1):
            cash_flow *= (1 + growth_rate)
            present_value = cash_flow / (1 + discount_rate) ** year
            pv_cf += present_value
            projections.append({
                "year": year,
                "projected_fcf": round(cash_flow, 2),
                "present_value": round(present_value, 2),
            })

        # Terminal value off the final projected cash flow, then discounted back.
        terminal_value = cash_flow * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_tv = terminal_value / (1 + discount_rate) ** self.DCF_FORECAST_YEARS
        total_pv = pv_cf + pv_tv
        intrinsic_value = total_pv / shares

        price = _to_float(
            self._info(symbol).get("currentPrice")
            or self._info(symbol).get("regularMarketPrice")
            or self._info(symbol).get("previousClose")
        )
        upside = _safe_div((intrinsic_value - price) if price is not None else None, price)

        # print(f"Calculated intrinsic value is {intrinsic_value}")

        note = None
        if upside is not None:
            if upside > 20:
                note = "Severely undervalued: strong buy"
            elif upside > 10:
                note = "Undervalued: buy"
            elif upside > -10:
                note = "fairly valued: hold"
            elif upside > 10:
                note = "overvalued: sell"
            else:
                note = "Severely overvalued: strong sell"

        return self._metric(
            "DCF intrinsic value per share",
            round(intrinsic_value, 2),
            "currency/share",
            symbol=symbol,
            current_price=price,
            upside_vs_price_pct=round(upside * 100, 2) if upside is not None else None,
            assumptions={
                "base_free_cash_flow": base_fcf,
                "base_period": history[0]["period"],
                "growth_rate_pct": round(growth_rate * 100, 2),
                "growth_basis": growth_basis,
                "discount_rate_pct": discount_rate * 100,
                "terminal_growth_pct": terminal_growth * 100,
                "forecast_years": self.DCF_FORECAST_YEARS,
                "shares_outstanding": shares,
            },
            valuation={
                "pv_cf": round(pv_cf, 2),
                "terminal_value": round(terminal_value, 2),
                "pv_tv": round(pv_tv, 2),
                "total_present_value": round(total_pv, 2),
                "intrinsic_value_per_share": round(intrinsic_value, 2),
            },
            projections=projections,
            free_cash_flow_history=history,
            note=note,
        )

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
            """Calculate the trailing price-to-earnings (P/E) ratio for a ticker symbol.
            Assessment found in the note field."""
            return _dump(self.pe_ratio(symbol))

        @tool("calculate_ev_ebitda")
        def ev_ebitda_tool(symbol: str) -> str:
            """Calculate the enterprise-value-to-EBITDA (EV/EBITDA) multiple for a ticker symbol.
            Assessment found in the note field."""
            return _dump(self.ev_ebitda(symbol))

        @tool("calculate_gross_margin")
        def gross_margin_tool(symbol: str) -> str:
            """Calculate the gross margin percentage (gross profit / revenue) for a ticker symbol and compare it to the typical gross margin for that company's industry."""
            return _dump(self.gross_margin(symbol))

        @tool("calculate_operating_margin")
        def operating_margin_tool(symbol: str) -> str:
            """Calculate the operating margin percentage (operating income / revenue) for a ticker symbol and compare it to the typical operating margin for that company's industry.
            Assessment found in the note field."""
            return _dump(self.operating_margin(symbol))

        @tool("calculate_roe")
        def roe_tool(symbol: str) -> str:
            """Calculate return on equity (net income / shareholders' equity) for a ticker symbol.
            Assessment found in the note field."""
            return _dump(self.roe(symbol))

        @tool("calculate_liquidity_ratio")
        def liquidity_ratio_tool(symbol: str) -> str:
            """Calculate the liquidity ratio defined as total assets divided by total liabilities for a ticker symbol.
            Assessment found in the note field."""
            return _dump(self.liquidity_ratio(symbol))

        @tool("calculate_quick_ratio")
        def quick_ratio_tool(symbol: str) -> str:
            """Calculate the quick ratio, (current assets minus inventory) divided by current liabilities, for a ticker symbol.
            Assessment found in the note field."""
            return _dump(self.quick_ratio(symbol))

        @tool("calculate_cash_ratio")
        def cash_ratio_tool(symbol: str) -> str:
            """Calculate the cash ratio, (cash plus short-term investments) divided by current liabilities, for a ticker symbol.
            Assessment found in the note field."""
            return _dump(self.cash_ratio(symbol))

        @tool("calculate_revenue_growth_yoy")
        def revenue_growth_tool(symbol: str) -> str:
            """Calculate year-over-year revenue growth from the last two reported fiscal years for a ticker symbol.
            Assessment found in the note field."""
            return _dump(self.revenue_growth_yoy(symbol))

        @tool("calculate_dcf")
        def dcf_tool(symbol: str) -> str:
            """Run a simplified discounted cash flow (DCF) valuation for a ticker symbol and return the intrinsic value per share. Free cash flow is operating cash flow plus capital expenditure from the cash-flow statement, grown at its historical CAGR over a ten-year forecast, discounted at 10% with a 2% terminal growth rate, then divided by shares outstanding.
            Assessment found in the note field."""
            return _dump(self.discounted_cash_flow(symbol))

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
            dcf_tool
        ]

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #

    def _get_prompt(self) -> str:
        """Fundamental-analysis system prompt."""
        # print("Added fundamental system prompt")
        return """You are a fundamental equity analyst working as part of a multi-agent research team.
        Your scope is the financial health and valuation of a single public company.
        Do not speculate about any metric which doesn't have a corresponding tool.

        ## Responsibilities
        1- Consume financial statements: income statement, balance sheet and cash flow statement
        2- Calculate valuation ratios: P/E and EV/EBITDA
        3- Calculate profitability metrics: gross margin, operating maring and ROE
        4- Calculate liquidity metrics: liquidity ratio, quick ratio and cash ratio
        5- Calculate growth trend: year on year revenue growth
        6- Calculate intrinsic value. If there is a big mismatch between DCF intrinsic value and actual stock price, try to explain why

        ## Output Format
        ---FUNDAMENTAL ANALYSIS---
        1- Metrics Assessments
        2- Key Risks
        3- Investment recommendation of Buy/Hold/Sell with a confidence level of High/Medium/Low
        """

    # ------------------------------------------------------------------ #
    # Main call
    # ------------------------------------------------------------------ #

    async def analyze_financials(
        self,
        symbol: str,
        period: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Analyze fundamentals for `symbol` from yfinance.

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

        company = self.get_company_info(symbol) or {}
        display_name = company.get("longName") or company.get("shortName") or symbol

        # print("Added fundamental task prompt")
        task = (
            f"""Perform a fundamental analysis of {display_name} ({symbol})"""
        )

        result = await self.execute(task=task)
        if result.success:
            result.data["symbol"] = symbol
            result.data["analysis_type"] = "Fundamental"
        return result