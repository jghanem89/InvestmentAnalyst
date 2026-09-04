import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Initialize ChatOllama with your local qwen2.5 tag
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.5,
)

# Create a message and invoke the model
messages = [HumanMessage(content="Hello, Qwen! Tell me a fun fact about local AI.")]
response = llm.invoke(messages)

print(response.content)
