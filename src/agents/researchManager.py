"""
Research manager: the orchestrating agent.

Breaks "analyse this company" into specialised analyses, runs the sub-agents
concurrently, aggregates what comes back and asks the LLM for a single
conclusion that has to account for any disagreement between them.

The fan-out is orchestrated in Python rather than by exposing the sub-agents as
LLM tools. A tool-calling model issues one call at a time, so sub-agents behind
tools would run sequentially; `asyncio.gather` actually runs them at once. The
LLM's job here is synthesis, and its tools read the findings that were already
collected.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, tool

try:  # running with `src` on the path
    from src.agents.baseAgent import AgentResult, BaseAgent
    from src.agents.fundamentalAnalyst import FundamentalAnalystAgent
    from src.agents.sentimentAnalyst import SentimentAnalystAgent
    from src.rag.config import RAGConfig
    from src.rag.ingest import FilingIngestionPipeline
except ImportError:  # running from inside `src/agents`
    from baseAgent import AgentResult, BaseAgent
    from fundamentalAnalyst import FundamentalAnalystAgent
    from sentimentAnalyst import SentimentAnalystAgent
    from rag.config import RAGConfig
    from rag.ingest import FilingIngestionPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The sub-agent write-ups end with a stance; these pull it back out so the
# manager can compare directions without asking the LLM to do it.
_STANCE_PATTERN = re.compile(r"\b(strong buy|strong sell|buy|hold|sell)\b", re.IGNORECASE)
_TONE_PATTERN = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)
_BULLISH = {"strong buy", "buy", "positive"}
_BEARISH = {"strong sell", "sell", "negative"}


def _last_match(pattern: re.Pattern, text: str) -> Optional[str]:
    """Last match in the text: both sub-agents put their verdict at the end."""
    matches = pattern.findall(text or "")
    return matches[-1].lower() if matches else None


class ResearchManagerAgent(BaseAgent):
    """
    Top-level agent that plans the research, delegates it and writes the verdict.

    Pipeline for one symbol:
      1. refresh the 10-Q vector store so retrieval sees the latest filing
      2. run every sub-agent concurrently, isolating failures
      3. collect structured metrics and detect contradictions deterministically
      4. have the LLM synthesise a conclusion over the aggregated findings
      5. print the report and save it to reports/<SYMBOL>_<DATE>.md
    """

    DEFAULT_NAME = "research_manager"
    DEFAULT_DESCRIPTION = (
        "Orchestrates company research: delegates fundamental and sentiment "
        "analysis to specialist sub-agents, reconciles their findings and "
        "produces the final recommendation."
    )

    # Sub-agents are slow local models; generous enough not to cut off a working
    # run, short enough that one hung agent cannot stall the whole analysis.
    SUBAGENT_TIMEOUT_SEC = 900.0
    # 10-Q filings pulled per run. Two quarters lets the sentiment agent report
    # the change in tone rather than just a level.
    FILINGS_PER_RUN = 2

    # Thresholds for the deterministic contradiction checks.
    DCF_UPSIDE_THRESHOLD = 15.0     # percent
    SENTIMENT_THRESHOLD = 0.1       # TextBlob polarity, matches the sentiment agent

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False,
        fundamental_agent: Optional[FundamentalAnalystAgent] = None,
        sentiment_agent: Optional[SentimentAnalystAgent] = None,
        reports_dir: Optional[Path] = None,
        ingest_filings: bool = True,
        rag_config: Optional[RAGConfig] = None,
    ):
        # All set before super().__init__: BaseAgent calls _get_tools() and
        # _get_prompt() from its constructor.
        self.reports_dir = Path(reports_dir or PROJECT_ROOT / "reports")
        self.should_ingest = ingest_filings
        self.rag_config = rag_config or RAGConfig()

        self.fundamental_agent = fundamental_agent or FundamentalAnalystAgent(verbose=verbose)
        self.sentiment_agent = sentiment_agent or SentimentAnalystAgent(verbose=verbose)

        # Populated by analyze(); the LLM tools read from here.
        self._findings: Dict[str, Dict[str, Any]] = {}
        self._signals: Dict[str, Any] = {}
        self._contradictions: List[Dict[str, str]] = []
        self._ingestion: Dict[str, Any] = {}

        super().__init__(name=name, description=description, tools=tools, verbose=verbose)

    # ------------------------------------------------------------------ #
    # Delegation plan
    # ------------------------------------------------------------------ #

    def _plan(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Break the request into analyses and name the sub-agent that owns each.

        Adding an analysis type is a matter of appending an entry here; the
        fan-out, aggregation and reporting are all driven off this list.
        """
        return [
            {
                "type": "fundamental",
                "title": "Fundamental analysis",
                "agent": self.fundamental_agent,
                "coroutine": lambda: self.fundamental_agent.analyze_financials(symbol),
                "needs_filings": False,
            },
            {
                "type": "sentiment",
                "title": "News and filing sentiment",
                "agent": self.sentiment_agent,
                "coroutine": lambda: self.sentiment_agent.analyze_sentiment(symbol),
                # Reads the 10-Q vector store, so ingestion has to finish first.
                "needs_filings": True,
            },
        ]

    # ------------------------------------------------------------------ #
    # Step 1: keep the vector store current
    # ------------------------------------------------------------------ #

    def _ingest_sync(self, symbol: str) -> Dict[str, Any]:
        """Blocking ingestion; called off the event loop by _refresh_filings."""
        pipeline = FilingIngestionPipeline(self.rag_config)
        return pipeline.ingest_ticker(symbol, limit=self.FILINGS_PER_RUN, verbose=self.verbose)

    async def _refresh_filings(self, symbol: str) -> Dict[str, Any]:
        """
        Pull any 10-Q filed since the last run into ChromaDB.

        Network and embedding work are both blocking, so this runs in a worker
        thread and the fundamental analysis proceeds alongside it. A failure
        here is not fatal: the sentiment agent falls back to news-only and says
        so in its own output.
        """
        if not self.should_ingest:
            return {"status": "skipped", "reason": "ingest_filings=False"}

        print(f"[{self.name}] refreshing 10-Q filings for {symbol}...")
        try:
            report = await asyncio.to_thread(self._ingest_sync, symbol)
        except Exception as exc:
            print(f"[{self.name}] filing ingestion failed: {exc}")
            return {"status": "failed", "error": str(exc)}

        if report.get("error"):
            print(f"[{self.name}] filing ingestion: {report['error']}")
            return {"status": "failed", "error": report["error"], "report": report}

        indexed = [f for f in report.get("filings", []) if f.get("status") == "indexed"]
        print(
            f"[{self.name}] filings up to date: {len(indexed)} newly indexed, "
            f"{report.get('chunks', 0)} chunks added"
        )
        return {
            "status": "ok",
            "newly_indexed": len(indexed),
            "chunks_added": report.get("chunks", 0),
            "filings": report.get("filings", []),
        }

    # ------------------------------------------------------------------ #
    # Step 2: parallel delegation
    # ------------------------------------------------------------------ #

    async def _run_task(self, task: Dict[str, Any], gate: Optional[asyncio.Task]) -> Dict[str, Any]:
        """
        Run one sub-agent, waiting on `gate` first if it depends on ingestion.

        Every failure mode is captured rather than raised: one sub-agent going
        down should degrade the report, not sink the run.
        """
        started = datetime.now(timezone.utc)
        if gate is not None:
            await gate

        print(f"[{self.name}] delegating {task['type']} analysis...")
        try:
            result: AgentResult = await asyncio.wait_for(
                task["coroutine"](), timeout=self.SUBAGENT_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            return {
                "type": task["type"], "title": task["title"],
                "agent": task["agent"].name, "success": False,
                "error": f"timed out after {self.SUBAGENT_TIMEOUT_SEC:.0f}s",
                "output": "", "data": {}, "elapsed_sec": self.SUBAGENT_TIMEOUT_SEC,
            }
        except Exception as exc:
            return {
                "type": task["type"], "title": task["title"],
                "agent": task["agent"].name, "success": False,
                "error": str(exc), "output": "", "data": {},
                "elapsed_sec": (datetime.now(timezone.utc) - started).total_seconds(),
            }

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        output = result.data.get("output", "") if result.success else ""
        print(f"[{self.name}] {task['type']} analysis returned in {elapsed:.1f}s")

        return {
            "type": task["type"],
            "title": task["title"],
            "agent": task["agent"].name,
            "success": result.success,
            "error": result.error,
            "output": output,
            # The raw LangGraph state is large and of no use downstream.
            "data": {k: v for k, v in result.data.items() if k != "raw_result"},
            "elapsed_sec": round(elapsed, 2),
            "stance": _last_match(_STANCE_PATTERN, output),
            "tone": _last_match(_TONE_PATTERN, output),
        }

    async def _delegate(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Fan the plan out concurrently and collect every result."""
        ingestion_task = asyncio.create_task(self._refresh_filings(symbol))
        plan = self._plan(symbol)

        results = await asyncio.gather(
            *(
                self._run_task(task, ingestion_task if task["needs_filings"] else None)
                for task in plan
            ),
            return_exceptions=False,
        )

        self._ingestion = await ingestion_task
        return {entry["type"]: entry for entry in results}

    # ------------------------------------------------------------------ #
    # Step 3: structured signals and contradictions
    # ------------------------------------------------------------------ #

    def _collect_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Pull the numbers the contradiction checks need.

        These come from the sub-agents' calculation methods rather than from
        their prose, so the checks below run on figures instead of on whatever
        the LLM happened to write.
        """
        signals: Dict[str, Any] = {"symbol": symbol}

        try:
            dcf = self.fundamental_agent.discounted_cash_flow(symbol)
            signals["dcf_upside_pct"] = dcf.get("upside_vs_price_pct")
            signals["dcf_note"] = dcf.get("note")
        except Exception as exc:
            signals["dcf_error"] = str(exc)

        try:
            growth = self.fundamental_agent.revenue_growth_yoy(symbol)
            signals["revenue_growth_pct"] = growth.get("value")
        except Exception as exc:
            signals["revenue_growth_error"] = str(exc)

        try:
            combined = self.sentiment_agent._combined_sentiment(symbol)
            signals["sentiment_polarity"] = combined.get("composite_polarity")
            signals["sentiment_label"] = combined.get("composite_label")
            signals["sentiment_divergence"] = combined.get("divergence")
            signals["sentiment_divergence_flag"] = combined.get("divergence_flag")
            signals["sentiment_confidence"] = combined.get("confidence")
            filing = combined.get("filing") or {}
            signals["filing_tone_trend"] = filing.get("tone_trend")
            signals["filing_date"] = (filing.get("filing") or {}).get("filing_date")
        except Exception as exc:
            signals["sentiment_error"] = str(exc)

        return signals

    def _detect_contradictions(
        self,
        findings: Dict[str, Dict[str, Any]],
        signals: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Flag places where the analyses point in opposite directions.

        Deterministic checks, so the LLM is told where the tension is instead of
        being trusted to notice it. Each entry names the two sides and what it
        means, and the synthesis prompt requires the model to resolve them.
        """
        found: List[Dict[str, str]] = []

        upside = signals.get("dcf_upside_pct")
        polarity = signals.get("sentiment_polarity")
        growth = signals.get("revenue_growth_pct")

        if upside is not None and polarity is not None:
            if upside > self.DCF_UPSIDE_THRESHOLD and polarity < -self.SENTIMENT_THRESHOLD:
                found.append({
                    "kind": "valuation_vs_sentiment",
                    "detail": (
                        f"DCF puts the shares {upside:.1f}% below intrinsic value while "
                        f"sentiment is negative ({polarity:+.3f}). Either the market is "
                        "pricing in something the cash flow model misses, or the pessimism "
                        "is an overreaction."
                    ),
                })
            elif upside < -self.DCF_UPSIDE_THRESHOLD and polarity > self.SENTIMENT_THRESHOLD:
                found.append({
                    "kind": "valuation_vs_sentiment",
                    "detail": (
                        f"DCF puts the shares {abs(upside):.1f}% above intrinsic value while "
                        f"sentiment is positive ({polarity:+.3f}). Optimism is running "
                        "against a stretched valuation."
                    ),
                })

        if growth is not None and polarity is not None:
            if growth < 0 and polarity > self.SENTIMENT_THRESHOLD:
                found.append({
                    "kind": "growth_vs_sentiment",
                    "detail": (
                        f"Revenue is shrinking ({growth:.1f}% YoY) but coverage is positive "
                        f"({polarity:+.3f}). Check whether the optimism rests on something "
                        "other than the top line."
                    ),
                })

        flag = signals.get("sentiment_divergence_flag")
        if flag and flag not in ("aligned", "no_data"):
            found.append({
                "kind": "news_vs_filing",
                "detail": (
                    f"The sentiment agent flagged '{flag}' with a divergence of "
                    f"{signals.get('sentiment_divergence')}: management's own language and "
                    "the press are not telling the same story."
                ),
            })

        if signals.get("filing_tone_trend") == "more negative" and polarity is not None \
                and polarity > self.SENTIMENT_THRESHOLD:
            found.append({
                "kind": "filing_trend_vs_news",
                "detail": (
                    "Management's tone in the latest 10-Q softened versus the prior quarter "
                    "even though news coverage is positive."
                ),
            })

        # Direction stated in the write-ups themselves. Heuristic (it reads the
        # verdict out of the prose), so it is reported as a cross-check only.
        stance = (findings.get("fundamental") or {}).get("stance")
        tone = (findings.get("sentiment") or {}).get("tone")
        if stance and tone:
            if (stance in _BULLISH and tone in _BEARISH) or (stance in _BEARISH and tone in _BULLISH):
                found.append({
                    "kind": "stated_conclusions",
                    "detail": (
                        f"The fundamental analyst concluded '{stance}' while the sentiment "
                        f"analyst concluded '{tone}'. The two write-ups disagree outright."
                    ),
                })

        return found

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    def _get_tools(self) -> List[BaseTool]:
        """
        Tools over the already-collected findings.

        The sub-agent write-ups are long, so they are not all pasted into the
        prompt. The model gets a digest and pulls the full text of whichever
        analysis it needs, plus the contradiction report and raw metrics.
        """

        def _dump(payload: Any) -> str:
            return json.dumps(payload, default=str, indent=2)

        @tool("list_findings")
        def list_findings_tool() -> str:
            """List every analysis the sub-agents returned for this company, with the analysis type, which agent produced it, whether it succeeded, how long it took and the stance it concluded with. Call this first to see what is available."""
            return _dump([
                {
                    "type": entry["type"],
                    "title": entry["title"],
                    "agent": entry["agent"],
                    "success": entry["success"],
                    "error": entry.get("error"),
                    "stance": entry.get("stance"),
                    "tone": entry.get("tone"),
                    "elapsed_sec": entry.get("elapsed_sec"),
                }
                for entry in self._findings.values()
            ])

        @tool("get_analysis")
        def get_analysis_tool(analysis_type: str) -> str:
            """Get the full write-up from one sub-agent. Pass the analysis type exactly as it appears in list_findings, for example "fundamental" or "sentiment"."""
            entry = self._findings.get(analysis_type.strip().lower())
            if not entry:
                return _dump({
                    "error": f"No analysis of type {analysis_type!r}.",
                    "available": sorted(self._findings),
                })
            return _dump({
                "type": entry["type"],
                "agent": entry["agent"],
                "success": entry["success"],
                "error": entry.get("error"),
                "analysis": entry.get("output"),
            })

        @tool("get_contradiction_report")
        def get_contradiction_report_tool() -> str:
            """Get the contradictions detected between the sub-agents' findings, together with the underlying metrics they were computed from: DCF upside, revenue growth, composite sentiment polarity and the news-versus-filing divergence. Use this to reconcile disagreement before writing the conclusion."""
            return _dump({
                "contradictions": self._contradictions,
                "contradiction_count": len(self._contradictions),
                "signals": self._signals,
            })

        @tool("get_filing_status")
        def get_filing_status_tool() -> str:
            """Report whether the SEC 10-Q vector database was refreshed for this company at the start of the run, how many filings were newly indexed, and which filing the sentiment analysis was able to use."""
            return _dump(self._ingestion)

        return [
            list_findings_tool,
            get_analysis_tool,
            get_contradiction_report_tool,
            get_filing_status_tool,
        ]

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #

    def _get_prompt(self) -> str:
        """Synthesis system prompt for the orchestrator."""
        return """You are the research manager of a multi-agent equity research team.
        Specialist sub-agents have already completed their analyses and their findings are
        available through your tools. You do not gather data yourself and you do not repeat
        their work - you reconcile it into one conclusion.

        ## How to work
        1- Call list_findings to see which analyses completed and which failed.
        2- Call get_analysis for each successful analysis to read the full write-up.
        3- Call get_contradiction_report before concluding. It lists detected disagreements
           and the metrics behind them.
        4- Only then write the conclusion.

        ## Handling contradictions
        Contradictions are the most important part of your job. Never average two opposing
        views into a bland middle. For each one, state which side you find more credible and
        why, in terms of the evidence behind it. Useful considerations:
        - A DCF rests on assumptions about growth and discount rate; it is a weak signal when
          the gap to market price is extreme.
        - TextBlob sentiment measures tone, not accuracy. A 10-Q is written by management and
          reviewed by lawyers, so its absolute polarity means little; the change against the
          prior quarter and the gap against press coverage are what carry information.
        - Reported financials are backward looking; sentiment is current. They can legitimately
          disagree when something has changed recently.
        If an analysis failed or was unavailable, say so and lower your confidence rather than
        filling in the gap yourself.

        ## Rules
        - Use only what the sub-agents reported. Never invent a figure.
        - Quote the specific numbers that drive your conclusion.
        - If the findings do not support a clear call, say so.

        ## Output Format
        ---INVESTMENT RESEARCH SUMMARY---
        1- Company Overview: what was analyzed and which analyses completed
        2- Fundamental Findings: the key metrics and what they show
        3- Sentiment Findings: news tone, filing tone, and the change versus last quarter
        4- Contradictions and Resolution: every disagreement found, and which side you favour
        5- Overall Recommendation: Buy/Hold/Sell with a confidence of High/Medium/Low
        6- Key Risks: what would change this conclusion
        """

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def _digest(self, symbol: str) -> str:
        """Compact summary handed to the model so it starts with the shape of the evidence."""
        lines = [f"Company under analysis: {symbol}", "", "Sub-agent results:"]
        for entry in self._findings.values():
            status = "completed" if entry["success"] else f"FAILED ({entry.get('error')})"
            lines.append(f"  - {entry['title']} [{entry['type']}] by {entry['agent']}: {status}")

        lines.append("")
        if self._contradictions:
            lines.append(f"{len(self._contradictions)} contradiction(s) detected:")
            for item in self._contradictions:
                lines.append(f"  - [{item['kind']}] {item['detail']}")
        else:
            lines.append("No contradictions were detected between the analyses.")

        lines.append("")
        lines.append("Use your tools to read the full analyses before concluding.")
        return "\n".join(lines)

    def _report(self, symbol: str, conclusion: str, generated: datetime) -> str:
        """Assemble the markdown report that is printed and saved."""
        lines = [
            f"# Investment Research Report: {symbol}",
            "",
            f"- Generated: {generated.isoformat(timespec='seconds')}",
            f"- Orchestrator: {self.name}",
            f"- Analyses run: {', '.join(sorted(self._findings))}",
        ]

        ingestion = self._ingestion or {}
        if ingestion.get("status") == "ok":
            lines.append(
                f"- SEC filings: refreshed, {ingestion.get('newly_indexed', 0)} newly indexed"
            )
        elif ingestion.get("status") == "skipped":
            lines.append("- SEC filings: ingestion skipped")
        else:
            lines.append(f"- SEC filings: refresh failed ({ingestion.get('error')})")

        lines += ["", "---", "", "## Final Conclusion", "", conclusion.strip(), ""]

        lines += ["---", "", "## Detected Contradictions", ""]
        if self._contradictions:
            for item in self._contradictions:
                lines.append(f"- **{item['kind']}** - {item['detail']}")
        else:
            lines.append("None detected.")
        lines.append("")

        lines += ["---", "", "## Supporting Metrics", "", "```json",
                  json.dumps(self._signals, default=str, indent=2), "```", ""]

        lines += ["---", "", "## Sub-Agent Analyses", ""]
        for entry in self._findings.values():
            lines.append(f"### {entry['title']} ({entry['agent']})")
            lines.append("")
            if entry["success"]:
                lines.append(f"*Completed in {entry.get('elapsed_sec')}s*")
                lines.append("")
                lines.append(entry.get("output", "").strip() or "_No output returned._")
            else:
                lines.append(f"**Failed:** {entry.get('error')}")
            lines.append("")

        return "\n".join(lines)

    def _save_report(self, symbol: str, report: str, generated: datetime) -> Path:
        """
        Write the report to reports/<SYMBOL>_<YYYY-MM-DD>.md.

        One file per symbol per day: re-running on the same day replaces that
        day's report rather than piling up near-identical files.
        """
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{symbol}_{generated.date().isoformat()}.md"
        path.write_text(report, encoding="utf-8")
        return path

    # ------------------------------------------------------------------ #
    # Main call
    # ------------------------------------------------------------------ #

    async def analyze(
        self,
        symbol: str,
        context: Optional[Dict[str, Any]] = None,
        save: bool = True,
        print_report: bool = True,
    ) -> AgentResult:
        """
        Run the full research pipeline for `symbol`.

        Args:
            symbol: Ticker symbol, for example "AAPL".
            context: Extra context merged into the payload sent to the model.
            save: Write the report to the reports directory.
            print_report: Print the report to stdout.

        Returns:
            The AgentResult from the synthesis step, with the findings,
            contradictions, metrics and report path attached to result.data.
        """
        symbol = symbol.strip().upper()
        started = datetime.now(timezone.utc)
        self.reset_state()

        print(f"[{self.name}] starting research on {symbol}")

        # 1 + 2: refresh filings and run the sub-agents concurrently.
        self._findings = await self._delegate(symbol)

        # 3: reconcile.
        self._signals = await asyncio.to_thread(self._collect_signals, symbol)
        self._contradictions = self._detect_contradictions(self._findings, self._signals)
        if self._contradictions:
            print(f"[{self.name}] {len(self._contradictions)} contradiction(s) to resolve")

        succeeded = [e for e in self._findings.values() if e["success"]]
        if not succeeded:
            errors = "; ".join(
                f"{e['type']}: {e.get('error')}" for e in self._findings.values()
            )
            return AgentResult(
                success=False,
                error=f"Every sub-agent failed for {symbol}. {errors}",
                exec_time_sec=(datetime.now(timezone.utc) - started).total_seconds(),
                name=self.name,
            )

        # 4: synthesise.
        print(f"[{self.name}] synthesising final conclusion...")
        task = (
            f"Write the final investment research conclusion for {symbol}. "
            f"{len(succeeded)} of {len(self._findings)} analyses completed. "
            "Read each analysis with your tools, resolve every contradiction in the "
            "contradiction report, and give one recommendation."
        )
        result = await self.execute(task=task, context={**(context or {}), "digest": self._digest(symbol)})

        if not result.success:
            return result

        # 5: report.
        generated = datetime.now(timezone.utc)
        report = self._report(symbol, result.data.get("output", ""), generated)

        if print_report:
            print("\n" + report)
        if save:
            path = self._save_report(symbol, report, generated)
            result.data["report_path"] = str(path)
            print(f"\n[{self.name}] report saved to {path}")

        result.data.update({
            "symbol": symbol,
            "analysis_type": "Research Summary",
            "report": report,
            "findings": {k: {ik: iv for ik, iv in v.items() if ik != "data"}
                         for k, v in self._findings.items()},
            "contradictions": self._contradictions,
            "signals": self._signals,
            "ingestion": self._ingestion,
            "total_time_sec": round((generated - started).total_seconds(), 2),
        })
        return result
