#!/usr/bin/env python
"""
Creating an agent using Google's API
"""
from dotenv import load_dotenv
import google.genai as genai
import os
import arxiv
import wikipedia
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()

# Getting Client via API key in my .env file
KEY = os.environ['GOOGLE_API_KEY']
client = genai.Client(api_key=KEY)

# Initializing gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    # Rescales logits prior to activation function
    # Reduces entropy for the sake of reproducibiility
    temperature=0,
    google_api_key=KEY
)

# Sensor 1
def wiki(query):
    """
    REturns the first 5 sentences from the Wikipedia article.
    """
    return wikipedia.summary(query, sentences=5)

wiki_tool = Tool(
    name="Wikipedia",
    func=wiki,
    description="Search Wikipedia for general background info"
)

# Sensor 2
def arxiv_search(query):
    """
    Gets FRONTIER knowledge based on query
    """
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = []

    for thing in search.results():
        results.append(
            f"Title: {thing.title}\n"
            f"Authors: {', '.join(a.name for a in thing.authors)}\n"
            f"Summary: {thing.summary}\n"
            f"URL: {thing.entry_id}\n"
        )

    return "\n\n".join(results)

arxiv_tool = Tool(
    name="ArXiv",
    func=arxiv_search,
    description="Search recent AI research papers from ArXiv."
)

# Sensor 3
def reasoning(text):
    """
    Acts as actuator, transforming the collected data into reasoning or insight.
    """
    return llm.invoke(f"Analyze and summarize this AI content: \n{text}")

reasoning_tool = Tool(
    name="ReasoningEngine",
    func=reasoning_engine,
    description="Summarize and analyze collected AI research."
)

tools = [wiki_tool, arxiv_tool, reasoning_tool]

# Prompts
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

# Activate Agent
response = agent_executor.invoke({
    "input": "Find 2 recent AI safety papers and summarize their contributions."
})

print(response["output"])

