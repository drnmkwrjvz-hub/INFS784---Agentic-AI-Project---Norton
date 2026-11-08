import os
import argparse
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from ddgs import DDGS
from langchain.tools import tool

# Explicitly load .env from the script's directory
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

# Fallback: check if key is loaded, and forcefully set os.environ so the underlying openrouter client sees it
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(f"OPENROUTER_API_KEY not found. Please check your .env file at {os.path.join(basedir, '.env')}")

os.environ["OPENROUTER_API_KEY"] = api_key

# Set up argument parsing for lab commands
parser = argparse.ArgumentParser(description="Research AI Agent Lab")
parser.add_argument("--mode", default="basic", choices=["basic", "memory", "deep"], help="Execution mode")
parser.add_argument("--prompt", default="Search for recent news about artificial intelligence milestones.", help="Prompt for the agent")
args = parser.parse_args()

# Configure OpenRouter model
model_id = os.getenv("OPENROUTER_MODEL", "openrouter/free")
model = ChatOpenRouter(
    model=model_id,
    api_key=api_key,
    temperature=0,
    timeout=120,
    max_tokens=1800,
)

@tool
def web_search_tool(query: str, max_results: int = 3) -> str:
    """Search the public web; return titles, URLs, and snippets."""
    max_results = min(max(1, max_results), 5)
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        if not hits:
            return "No results returned; search may be rate-limited."
        rows = []
        for hit in hits:
            rows.append(
                f"Title: {hit.get('title', 'Untitled')}\n"
                f"URL: {hit.get('href', 'No URL')}\n"
                f"Snippet: {hit.get('body', '')[:450]}"
            )
        return "\n---\n".join(rows)
    except Exception as exc:
        return f"Search unavailable: {type(exc).__name__}: {str(exc)[:200]}"

SYSTEM_PROMPT = """You are a Literary & Web Data Research Assistant.

## Workflow
1. Understand the question and choose an evidence strategy.
2. Search the web for leads; verify with permitted source documents.
3. Distinguish confirmed evidence from interpretation.

## Integrity requirements
- Never invent URLs, quotations, line counts, citations or tool outputs.
- Search snippets alone do not prove a source claim.
- Report failed tool calls and uncertainty explicitly.
"""

# Create the agent
agent = create_agent(
    model=model,
    tools=[web_search_tool],
    system_prompt=SYSTEM_PROMPT,
)

print(f"--- Running Agent [Mode: {args.mode}] ---")
result = agent.invoke({
    "messages": [
        {"role": "user", "content": args.prompt}
    ]
})

print("\n### Final Response:")
print(result["messages"][-1].content)