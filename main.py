import sys
import asyncio
from src.agents.researchManager import ResearchManagerAgent

if __name__ == "__main__":
    async def _demo() -> None:
        ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
        manager = ResearchManagerAgent(verbose=False)
        result = await manager.analyze(ticker)
        if not result.success:
            print(f"Analysis failed: {result.error}")

    asyncio.run(_demo())
