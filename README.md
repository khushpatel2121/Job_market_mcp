# Job Market Intelligence MCP Server

An AI-powered job market research assistant built with FastMCP. Exposes internal tools, databases, and APIs as structured capabilities for LLM agents.

---

## What It Does

An AI agent that helps you research the job market by answering questions like:

- *"What is the salary for a senior AI Engineer in Canada?"*
- *"I know Python and Docker. What skills am I missing for a Data Scientist role?"*
- *"Tell me about Anthropic as a company and what salary I can expect there."*

---

## Architecture

```
LLM Agent (Groq)
      ↓
FastMCP Server  ←  facade between the LLM and all tools
      ↓
┌─────────────────────────────────────────────┐
│  search_web       →  DuckDuckGo API         │
│  get_company_info →  Wikipedia API          │
│  analyze_salary   →  SQLite database        │
│  analyze_skills   →  Groq LLM + Pydantic    │
└─────────────────────────────────────────────┘
```

---

## Tools Exposed

| Tool | Description | Backend |
|---|---|---|
| `search_web` | Search the web for job trends and postings | DuckDuckGo (free) |
| `get_company_info` | Get company background | Wikipedia API (free) |
| `analyze_salary` | Look up salary ranges by role, experience, country | SQLite database |
| `analyze_skills_gap` | Analyze skill gaps for a target role | Groq LLM + Pydantic |

---

## Tech Stack

- **FastMCP** — MCP server framework
- **Groq API** — free LLM inference (llama-3.3-70b-versatile)
- **Pydantic** — LLM output validation and parsing
- **SQLite** — internal salary database
- **DuckDuckGo** — free web search, no API key needed
- **Wikipedia REST API** — free company research, no API key needed

---

## Project Structure

```
job_market_mcp/
├── server.py          # FastMCP server — all 4 tools defined here
├── salary_db.py       # SQLite database setup and queries
├── skills_engine.py   # Groq + Pydantic skills gap analysis
├── agent.py           # LLM inference loop
├── test_client.py     # Tool verification script
├── .env               # API keys (not committed)
└── .gitignore
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/khushpatel2121/Job_market_mcp.git
cd Job_market_mcp
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install fastmcp groq pydantic requests python-dotenv
```

**4. Add your Groq API key**
```bash
touch .env
echo "GROQ_API_KEY=your_key_here" >> .env
```
Get a free key at [console.groq.com](https://console.groq.com)

**5. Run the agent**
```bash
python3 agent.py
```

---

## Sample Queries

```
# No tools needed
How are you?
What can you help me with?

# Single tool
What is the salary for a senior AI Engineer in Canada?
Tell me about Anthropic
I know Python and SQL. What skills am I missing for a Data Scientist role?

# Multiple tools
Tell me about Google and what a senior Backend Engineer earns in the USA
I know Python, Docker and REST APIs. What skills am I missing for an AI 
Engineer role and what salary can I expect in Canada?
```

---

## Testing

```bash
python3 test_client.py
```

Runs all 4 tools with sample inputs and prints results.
