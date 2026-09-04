## Objective
The financial analyst agent will provide an analysis summary for a prompted public stock/company.
The reasoning loop will be as follows:
1- Generate a thought/action belonging to the below tasks
    - Compare actual EPS vs estimates from earnings reports
    - Summarize fundamentals (income statement, balance sheet, P/E, gross margins, SEC filings)
    - Perform market/news sentiment analysis
2- Parse thought and action
3- Execute tools and get observations
4- Move to next task in (1)
5- Finish and return analysis summary

It is a multi-agent architecture allowing for specialized agents to perform their roles better. Instead of a single agent that will perform all research and return results, we will have:
1- The main agent that will break down the research into sub-tasks, receives results from the sub-agents and formulate the summary/recommendation
2- A sub-agent responsible for the fundamental analysis sub-task
3- A sub-agent responsible for the earnings analysis sub-task
4- A sub-agent responsible for the sentiment analysis sub-task
5- Potentially a sub-agent for RAG retrieval?

---

## Installation
1- Download repo
2- Install python>=3.14.6
3- Make sure to intall all requirement with the below command in terminal
pip install -r .\requirements.txt
4- Download Ollama
run the below command in Windows PowerShell as administrator
irm https://ollama.com/install.ps1 | iex
