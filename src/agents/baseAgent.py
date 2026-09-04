from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from typing import Any, Dict, Optional, List
from pydantic import Field
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone
import settings

class AgentState(BaseModel):
    """Pydantic Model for agent state"""

    name: str
    status: str = "idle"  # idle, running, completed, error
    current_task: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    started: Optional[datetime] = None
    completed: Optional[datetime] = None
    config = ConfigDict(arbitrary_types_allowed=True)


class AgentResult(BaseModel):
    """Pydantic Model for agent results"""

    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    exec_time_sec: float = 0.0
    name: str = ""
    confidence: float = Field(default=0.0, description="Confidence score 0.0-1.0")
    reasoning_steps: int = Field(default=0, description="Number of reasoning steps taken")
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "exec_time_sec": self.execution_time_seconds,
            "name": self.agent_name,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "ts": self.ts.isoformat(),
        }

class BaseAgent():
    """
    Base Agent that other agents will inherit
    It will
    - Initialize ollama as the LLM model (can later imrpove to access any other llm)
    """

    def __init__(
            self,
            name: str,
            description: str,
            tools: Optional[List[BaseTool]] = None,
            max_reasoning_steps: int = 6,
            verbose: bool = False,
            temperature: float = 0.3
    ):
        self.name = name
        self.description = description
        self.tools = tools or self._get_tools()
        self.max_reasoning_steps = max_reasoning_steps
        self.verbose = verbose
        self.temperature = temperature
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            temperature=settings.OLLAMA_TEMP
            )
        self.state = AgentState(name=name)
        self.graph = self._create_graph()

        print(f"Initialized agent: {self.name}")

    @abstractmethod
    def _get_tools(self) -> List[BaseTool]:
        """
        Get the tools for this agent.

        Returns:
            List of tools available to the agent
        """
        raise NotImplementedError("Subclasses must implement _get_tools")

    @abstractmethod
    def _get_prompt(self) -> str:
        """
        Get the prompt for this agent.

        Returns:
            System prompt string
        """
        raise NotImplementedError("Subclasses must implement _get_prompt")

    def _create_graph(self):
        """Create LangGraph"""
        return create_agent(
            model=self.llm,
            tools=self.tools or [],
            system_prompt=self._get_prompt(),
            name=self.name,
            debug=self.verbose,
        )