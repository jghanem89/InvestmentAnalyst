from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from typing import Any, Dict, Optional, List
from pydantic import Field
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone
import json

OLLAMA_MODEL="qwen3:8b"
OLLAMA_TEMP=0.3

class AgentState(BaseModel):
    """Pydantic Model for agent state"""

    name: str
    status: str = "idle"  # idle, executing, completed, error
    current_task: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentResult(BaseModel):
    """Pydantic Model for agent results"""

    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    exec_time_sec: float = 0.0
    name: str = ""
    # reasoning_steps: int = Field(default=0, description="Number of reasoning steps taken")
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "exec_time_sec": self.execution_time_seconds,
            "name": self.agent_name,
            # "reasoning_steps": self.reasoning_steps,
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
            # max_reasoning_steps: int = 6,
            verbose: bool = False
    ):
        self.name = name
        self.description = description
        self.tools = tools or self._get_tools()
        # self.max_reasoning_steps = max_reasoning_steps
        self.verbose = verbose
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=OLLAMA_TEMP
            )
        self.state = AgentState(name=name)
        self.graph = self._create_graph()

        print(f"Initialized agent: {self.name}")

    def __repr__(self) -> str:
            return f"{self.__class__.__name__}(name='{self.name}', status='{self.state.status}')"

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

    def _append_react_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Appends a ReAct prompt to the current task and context for proper reasoning
        """
        react_prompt = """Follow the below step by step approach when reasoning
        1- Think about which data is needed and if we have an appropriate tool
        2- Call the corresponding tools to collect data and observations - never use data that isn't returned by a tool
        3- Analyze the results returned
        4- Return a conclusion supported by observations and specify if confidence is low, medium or high
        """

        new_prompt = react_prompt + task
        if context:
            new_prompt += f"\n\nContext: {json.dumps(context, default=str)}"

        return new_prompt
    
    def reset_state(self) -> None:
        # Reset agent state
        self.state = AgentState(name=self.name)

    def get_state(self) -> Dict[str, Any]:
        # Get agent state
        return self.state.model_dump()

    async def execute(
              self,
              task: str,
              context: Optional[Dict[str, Any]] = None,
              history: Optional[List[BaseMessage]] = None
    ) -> AgentResult:
        """
        Execute task with agent while passing context and history - if any - and return response

        Args:
            task: The planned task
            context: Additional context for the task
            history: Conversation history

        Returns:
            AgentResult with execution results
        """
        start_time = datetime.now(timezone.utc)
        self.state.status = "executing"
        self.state.current_task = task
        self.state.start_time = start_time

        try:
            print(f"Agent {self.name} executing task: {task[:100]}...")

            # Build LLM input
            input_text = self._append_react_prompt(task, context)
            messages = []
            if history:
                messages.extend(history)
            messages.append(HumanMessage(content=input_text))

            # Execute via agent graph
            result = await self.graph.ainvoke({"messages": messages})

            # Get LLM response
            output_messages = result.get("messages", [])
            output = ""
            for msg in reversed(output_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    output = msg.content
                    break

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.state.status = "completed"
            self.state.completion_time = datetime.now(timezone.utc)
            self.state.results[task[:100]] = output

            return AgentResult(
                success=True,
                data={"output": output, "raw_result": result},
                exec_time_sec=execution_time,
                name=self.name
            )
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = str(e)

            self.state.status = "error"
            self.state.errors.append(error_msg)

            print(f"Agent {self.name} error: {error_msg}")

            return AgentResult(
                success=False,
                error=error_msg,
                exec_time_sec=execution_time,
                name=self.name
            )