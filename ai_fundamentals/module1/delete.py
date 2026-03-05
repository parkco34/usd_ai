#!/usr/bin/env python
# ==========================================================
# Assignment 1.2 - Question 2a
# AIResearchScout Agent
# Parker Cory
# ==========================================================

# -----------------------
# Install Required Packages
# -----------------------
# %pip install -q -U langchain langchain-community langchain-google-genai arxiv wikipedia google-generativeai

# -----------------------
# Imports
# -----------------------
import os
import arxiv
import wikipedia

from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate

# -----------------------
# Configure Gemini API
# -----------------------
GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# ==========================================================
# ===================== SENSORS ============================
# ==========================================================

# -------- Sensor 1: Wikipedia --------
def wiki_search(query):
    return wikipedia.summary(query, sentences=5)

wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_search,
    description="Search Wikipedia for general background information."
)

# -------- Sensor 2: ArXiv --------
def arxiv_search(query):
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = []
    for result in search.results():
        results.append(
            f"Title: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}\n"
            f"URL: {result.entry_id}\n"
        )
    return "\n\n".join(results)

arxiv_tool = Tool(
    name="ArXiv",
    func=arxiv_search,
    description="Search recent AI research papers from ArXiv."
)

# -------- Sensor 3: LLM Reasoning --------
def reasoning_engine(text):
    return llm.invoke(f"Analyze and summarize this AI content:\n{text}")

reasoning_tool = Tool(
    name="ReasoningEngine",
    func=reasoning_engine,
    description="Summarize and analyze collected AI research."
)

tools = [wiki_tool, arxiv_tool, reasoning_tool]

# ==========================================================
# ===================== ACTUATORS ==========================
# ==========================================================

prompt = PromptTemplate.from_template("""
You are AIResearchScout, an AI agent designed to monitor AI research.

Your objectives:
1. Retrieve relevant information.
2. Summarize key insights.
3. Evaluate relevance to AI safety, alignment, or algorithmic discovery.
4. Produce a structured weekly literature report.

Use available tools when needed.
Question: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==========================================================
# ===================== RUN AGENT ==========================
# ==========================================================

response = agent_executor.invoke({
    "input": "Find 2 recent AI safety papers and summarize their contributions."
})

print(response["output"])

