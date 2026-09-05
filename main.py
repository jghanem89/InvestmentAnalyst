import sys
import asyncio
from src.agents.fundamentalAnalyst import FundamentalAnalystAgent

if __name__ == "__main__":
    async def _demo() -> None:
        ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
        agent = FundamentalAnalystAgent(verbose=False)
        result = await agent.analyze_financials(ticker)
        if result.success:
            print(result.data["output"])
        else:
            print(f"Analysis failed: {result.error}")

    asyncio.run(_demo())
