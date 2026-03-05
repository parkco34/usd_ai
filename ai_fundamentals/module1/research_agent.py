#!/usr/bin/env python
"""
AI Research Collection Agent - Assignment 1.2 Part 2a)

Automates collection, evaluation, and archival of AI research from multiple sources
using a goal-based agent architecture built on Google's Gemini API + LangChain.

Agent Architecture (Poole & Mackworth Framework):
    Environment:  External AI research ecosystem (web APIs, publications, forums)
    Task Type:    Episodic, partially observable, dynamic, stochastic
    Agent Type:   Goal-based agent with persistent memory

    Sensors (5 external + 1 internal):
        1. arXiv REST API       — structured XML/Atom feed (cs.AI, cs.LG, stat.ML)
        2. DeepMind HTML scraper — semi-structured web parsing
        3. Anthropic HTML scraper — semi-structured web parsing
        4. Alignment Forum GraphQL API — structured JSON via GraphQL
        5. LessWrong GraphQL API      — structured JSON via GraphQL
        6. Archive loader              — reads prior run state from persistent memory

    Actuators (2):
        1. Structured research report  — evaluated, ranked, theme-tagged summary
        2. Persistent JSON archive     — cross-run memory with deduplication and
           trend tracking (research_archive.json)

    Reasoning:
        The LLM (Gemini) acts as the agent's controller. It decides which tools
        to call, compares new results against prior runs, ranks findings by
        relevance, identifies recurring themes, and produces a structured report.
"""
# pip install langchain langchain-google-genai google-generativeai beautifulsoup4 requests python-dotenv
from dotenv import load_dotenv
import google.genai as genai
import os
import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool

# Load API key
load_dotenv()

# Getting Client
KEY = os.environ['GOOGLE_API_KEY']
client = genai.Client(api_key=KEY)

#Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=KEY
)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
ARCHIVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_archive.json")


# ==============================
# Persistent Memory Helpers
# ==============================

def load_archive() -> dict:
    """Load the persistent research archive from disk."""
    if os.path.exists(ARCHIVE_PATH):
        with open(ARCHIVE_PATH, "r") as f:
            return json.load(f)
    return {"runs": [], "seen_titles": [], "topic_counts": {}}


def save_archive(archive: dict) -> None:
    """Write the research archive back to disk (Actuator 2)."""
    with open(ARCHIVE_PATH, "w") as f:
        json.dump(archive, f, indent=2)


# ==============================
# Sensors — Research Collection Tools
# ==============================

@tool
def load_previous_research(query: str = "latest") -> str:
    """Load results from previous agent runs to compare against new findings.
    Returns prior titles, timestamps, and topic trends for deduplication."""
    try:
        archive = load_archive()
        if not archive["runs"]:
            return "No previous runs found. This is the first collection."

        last_run = archive["runs"][-1]
        prior_titles = archive["seen_titles"][-25:]  # last 25 titles
        topic_counts = archive["topic_counts"]

        summary = (
            f"=== Prior Run Memory ===\n"
            f"Last run: {last_run['timestamp']}\n"
            f"Total prior runs: {len(archive['runs'])}\n"
            f"Previously seen titles ({len(prior_titles)}):\n"
        )
        for t in prior_titles:
            summary += f"  - {t}\n"

        if topic_counts:
            summary += "\nTopic frequency (across all runs):\n"
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            for topic, count in sorted_topics[:10]:
                summary += f"  {topic}: {count} occurrences\n"

        return summary

    except Exception as e:
        return f"Error loading archive: {str(e)}"


@tool
def fetch_arxiv_papers(query: str = "recent") -> str:
    """Fetch recent AI/ML papers from arXiv in cs.AI, cs.LG, and stat.ML categories.
    Pass a keyword to search by topic or 'recent' for the latest papers."""
    try:
        categories = "cat:cs.AI+OR+cat:cs.LG+OR+cat:stat.ML"
        if query and query.lower() != "recent":
            search = f"({categories})+AND+all:{query}"
        else:
            search = categories

        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={search}"
            f"&sortBy=submittedDate&sortOrder=descending&max_results=5"
        )

        resp = requests.get(url, headers=HEADERS, timeout=15)
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        results = []
        for entry in entries:
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip()[:300]
            published = entry.find("atom:published", ns).text[:10]
            link = entry.find("atom:id", ns).text
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            results.append(
                f"Title: {title}\n"
                f"Authors: {', '.join(authors[:3])}\n"
                f"Date: {published}\n"
                f"Link: {link}\n"
                f"Summary: {summary}..."
            )

        if not results:
            return "No arXiv papers found for this query."
        return "=== arXiv Papers (cs.AI, cs.LG, stat.ML) ===\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error fetching arXiv: {str(e)}"


@tool
def fetch_deepmind_research(query: str = "recent") -> str:
    """Fetch recent research publications from Google DeepMind."""
    try:
        url = "https://deepmind.google/research/publications/"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        #Look for publication links
        links = soup.find_all("a", href=True)
        results = []
        seen = set()
        for link in links:
            href = link.get("href", "")
            if "/research/publications/" in href and href != "/research/publications/":
                title = link.get_text(strip=True)
                if not title or title in seen or len(title) < 5:
                    continue
                seen.add(title)
                full_link = href if href.startswith("http") else "https://deepmind.google" + href
                results.append(f"Title: {title}\nLink: {full_link}")
                if len(results) >= 5:
                    break

        if not results:
            return "Could not parse DeepMind publications. Site may use JS rendering."
        return "=== DeepMind Research ===\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error fetching DeepMind: {str(e)}"


@tool
def fetch_anthropic_research(query: str = "recent") -> str:
    """Fetch recent research posts from Anthropic's research blog."""
    try:
        url = "https://www.anthropic.com/research"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        #Look for research post links (skip team/category pages)
        links = soup.find_all("a", href=True)
        results = []
        seen = set()
        skip = ["/research/team/", "/research/#"]
        for link in links:
            href = link.get("href", "")
            if "/research/" in href and href != "/research/" and href != "/research":
                if any(s in href for s in skip):
                    continue
                title = link.get_text(strip=True)
                if not title or title in seen or len(title) < 5:
                    continue
                seen.add(title)
                full_link = href if href.startswith("http") else "https://www.anthropic.com" + href
                results.append(f"Title: {title}\nLink: {full_link}")
                if len(results) >= 5:
                    break

        if not results:
            return "Could not parse Anthropic research page. Site may use JS rendering."
        return "=== Anthropic Research ===\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error fetching Anthropic: {str(e)}"


@tool
def fetch_alignment_forum_posts(query: str = "recent") -> str:
    """Fetch recent posts from the AI Alignment Forum."""
    try:
        gql_query = """{
            posts(input: {terms: {limit: 5, meta: false}}) {
                results {
                    title
                    postedAt
                    slug
                    _id
                    baseScore
                    user { displayName }
                }
            }
        }"""
        gql_headers = {**HEADERS, "Content-Type": "application/json"}
        resp = requests.post(
            "https://www.alignmentforum.org/graphql",
            json={"query": gql_query}, headers=gql_headers, timeout=15
        )
        if resp.status_code != 200:
            return f"AI Alignment Forum returned status {resp.status_code}. May be rate-limited."
        data = resp.json()
        posts = data.get("data", {}).get("posts", {}).get("results", [])

        results = []
        for post in posts:
            title = post.get("title", "Untitled")
            author = post.get("user", {}).get("displayName", "Unknown")
            date = post.get("postedAt", "")[:10]
            score = post.get("baseScore", 0)
            pid = post.get("_id", "")
            slug = post.get("slug", "")
            link = f"https://www.alignmentforum.org/posts/{pid}/{slug}"
            results.append(
                f"Title: {title}\n"
                f"Author: {author}\n"
                f"Date: {date}\n"
                f"Score: {score}\n"
                f"Link: {link}"
            )

        if not results:
            return "No AI Alignment Forum posts found."
        return "=== AI Alignment Forum ===\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error fetching Alignment Forum: {str(e)}"


@tool
def fetch_lesswrong_posts(query: str = "recent") -> str:
    """Fetch recent posts from LessWrong."""
    try:
        gql_query = """{
            posts(input: {terms: {limit: 5, meta: false}}) {
                results {
                    title
                    postedAt
                    slug
                    _id
                    baseScore
                    user { displayName }
                }
            }
        }"""
        gql_headers = {**HEADERS, "Content-Type": "application/json"}
        resp = requests.post(
            "https://www.lesswrong.com/graphql",
            json={"query": gql_query}, headers=gql_headers, timeout=15
        )
        if resp.status_code != 200:
            return f"LessWrong returned status {resp.status_code}. May be rate-limited."
        data = resp.json()
        posts = data.get("data", {}).get("posts", {}).get("results", [])

        results = []
        for post in posts:
            title = post.get("title", "Untitled")
            author = post.get("user", {}).get("displayName", "Unknown")
            date = post.get("postedAt", "")[:10]
            score = post.get("baseScore", 0)
            pid = post.get("_id", "")
            slug = post.get("slug", "")
            link = f"https://www.lesswrong.com/posts/{pid}/{slug}"
            results.append(
                f"Title: {title}\n"
                f"Author: {author}\n"
                f"Date: {date}\n"
                f"Score: {score}\n"
                f"Link: {link}"
            )

        if not results:
            return "No LessWrong posts found."
        return "=== LessWrong ===\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error fetching LessWrong: {str(e)}"


# ==============================
# Build the Agent
# ==============================

#Sensors: 5 external sources + 1 internal memory
tools = [
    load_previous_research,
    fetch_arxiv_papers,
    fetch_deepmind_research,
    fetch_anthropic_research,
    fetch_alignment_forum_posts,
    fetch_lesswrong_posts
]

#Agent system prompt — explicit reasoning chain
SYSTEM_PROMPT = (
    "You are an AI research collection and evaluation agent. Follow this reasoning process:\n\n"
    "STEP 1 — MEMORY: First, call load_previous_research to check what was found in prior runs.\n\n"
    "STEP 2 — COLLECT: Call ALL 5 research source tools to gather the latest findings:\n"
    "  - fetch_arxiv_papers\n"
    "  - fetch_deepmind_research\n"
    "  - fetch_anthropic_research\n"
    "  - fetch_alignment_forum_posts\n"
    "  - fetch_lesswrong_posts\n\n"
    "STEP 3 — EVALUATE: After collecting from every source, analyze the results:\n"
    "  a) Flag which items are NEW (not seen in previous runs) vs RECURRING\n"
    "  b) Rank the top 5 most significant findings across all sources by potential impact\n"
    "  c) Identify 2-3 recurring themes or trends that span multiple sources\n\n"
    "STEP 4 — REPORT: Produce a structured research report with these sections:\n"
    "  1. Executive Summary (2-3 sentences on the overall landscape)\n"
    "  2. Source-by-Source Findings (organized list from each source)\n"
    "  3. Top 5 Ranked Items (with brief justification for each ranking)\n"
    "  4. Cross-Source Themes (recurring topics spanning multiple sources)\n"
    "  5. New vs Previously Seen (highlight what changed since last run)\n"
)

#Initialize the agent (LLM controller with tool-calling loop)
agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


# ==============================
# Actuators — Output & Persistence
# ==============================

def extract_titles_from_report(report: str) -> list:
    """Parse titles from the agent's report for deduplication tracking.
    Handles markdown formats like:  *   **Title Text:** description
    and numbered items like:        1.  **Title (Source):** justification"""
    import re
    titles = []
    seen = set()
    #Section headers to skip
    headers = {"arxiv papers", "deepmind research", "anthropic research",
               "ai alignment forum", "lesswrong posts", "ai safety and alignment",
               "interpretability and understanding", "practical challenges"}
    for line in report.split("\n"):
        matches = re.findall(r'\*\*(.+?)\*\*', line)
        for match in matches:
            #Strip colon first, then source annotations like "(Anthropic)"
            clean = match.strip().rstrip(":")
            clean = re.sub(r'\s*\(.*?\)\s*$', '', clean).strip()
            #Skip section headers, short labels, and duplicates
            if len(clean) < 10 or clean.startswith("#"):
                continue
            if any(h in clean.lower() for h in headers):
                continue
            if clean not in seen:
                seen.add(clean)
                titles.append(clean)
    return titles


def extract_topics_from_report(report: str) -> list:
    """Extract topic keywords from the agent's theme analysis."""
    keywords = [
        "alignment", "safety", "interpretability", "reasoning", "agents",
        "reinforcement learning", "language models", "optimization",
        "transformer", "diffusion", "multimodal", "robotics", "scaling",
        "evaluation", "jailbreak", "hallucination", "generalization",
        "fine-tuning", "RLHF", "constitutional AI", "mechanistic",
        "benchmark", "chain-of-thought", "retrieval", "memory"
    ]
    report_lower = report.lower()
    found = [kw for kw in keywords if kw in report_lower]
    return found


def update_archive(report: str) -> dict:
    """Actuator 2: Update persistent memory with new findings (research_archive.json)."""
    archive = load_archive()

    #Extract titles and topics from this run
    new_titles = extract_titles_from_report(report)
    new_topics = extract_topics_from_report(report)

    #Deduplication — identify which titles are genuinely new
    existing = set(archive["seen_titles"])
    novel_titles = [t for t in new_titles if t not in existing]
    recurring_titles = [t for t in new_titles if t in existing]

    #Update seen titles (keep last 200 to prevent unbounded growth)
    archive["seen_titles"] = list(set(archive["seen_titles"] + new_titles))[-200:]

    #Update topic frequency counts
    for topic in new_topics:
        archive["topic_counts"][topic] = archive["topic_counts"].get(topic, 0) + 1

    #Append this run's metadata
    run_record = {
        "timestamp": datetime.now().isoformat(),
        "total_items": len(new_titles),
        "novel_items": len(novel_titles),
        "recurring_items": len(recurring_titles),
        "topics_detected": new_topics
    }
    archive["runs"].append(run_record)

    #Keep only last 50 runs
    if len(archive["runs"]) > 50:
        archive["runs"] = archive["runs"][-50:]

    save_archive(archive)
    return run_record


def generate_report(final_message: str, run_stats: dict) -> None:
    """Actuator 1: Print the structured research report to console."""
    print("\n" + "=" * 60)
    print("  COLLECTED RESEARCH REPORT")
    print("=" * 60)
    print(final_message)
    print("\n" + "-" * 60)
    print("  RUN STATISTICS")
    print("-" * 60)
    print(f"  Timestamp:      {run_stats['timestamp']}")
    print(f"  Items found:    {run_stats['total_items']}")
    print(f"  New items:      {run_stats['novel_items']}")
    print(f"  Recurring:      {run_stats['recurring_items']}")
    print(f"  Topics:         {', '.join(run_stats['topics_detected'])}")
    print("-" * 60)


# ==============================
# Run the Agent
# ==============================

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Research Collection Agent")
    print("  Assignment 1.2 Part 2a)")
    print("=" * 60)

    result = agent.invoke({"messages": [{
        "role": "user",
        "content": (
            "Collect the latest AI research from all 5 sources: "
            "1) arXiv (cs.AI, cs.LG, stat.ML), "
            "2) DeepMind publications, "
            "3) Anthropic research blog, "
            "4) AI Alignment Forum, "
            "5) LessWrong. "
            "First check previous runs, then collect, evaluate, rank, and report."
        )
    }]})

    #Extract the final agent response
    last = result["messages"][-1].content
    if isinstance(last, list):
        final_message = "\n".join(
            block["text"] for block in last if isinstance(block, dict) and "text" in block
        )
    else:
        final_message = str(last)

    #Actuator 2: Update persistent archive with deduplication
    run_stats = update_archive(final_message)

    #Actuator 1: Generate structured report to console
    generate_report(final_message, run_stats)

    #Also save standalone output for this run
    output = {
        "timestamp": run_stats["timestamp"],
        "sources": [
            "arXiv (cs.AI, cs.LG, stat.ML)",
            "DeepMind Research",
            "Anthropic Research Blog",
            "AI Alignment Forum",
            "LessWrong"
        ],
        "summary": final_message,
        "stats": run_stats
    }
    with open("research_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to research_output.json")
    print(f"Archive updated at {ARCHIVE_PATH}")
