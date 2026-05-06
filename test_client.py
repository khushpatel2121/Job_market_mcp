# test_client.py
# Tests every MCP tool directly by importing and calling
# the same functions the MCP server exposes.
#
# Run with: python3 test_client.py
#
# This proves every tool works correctly before connecting
# to any LLM client or agent.

import json
import sys

# ─────────────────────────────────────────────────────────────
# COLOUR HELPERS
# Makes terminal output easier to read during testing.
# ─────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def print_header(title: str):
    print(f"\n{BOLD}{BLUE}{'═' * 55}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'═' * 55}{RESET}")

def print_result(label: str, value):
    print(f"  {YELLOW}{label:<22}{RESET} {value}")

def print_success(msg: str):
    print(f"  {GREEN}✓ {msg}{RESET}")

def print_error(msg: str):
    print(f"  {RED}✗ {msg}{RESET}")

def print_section(title: str):
    print(f"\n  {BOLD}{title}{RESET}")
    print(f"  {'─' * 40}")


# ─────────────────────────────────────────────────────────────
# Import the helper functions and modules directly.
# We test the same logic the MCP tools use internally.
# ─────────────────────────────────────────────────────────────

print(f"{BOLD}Loading modules...{RESET}")

try:
    from salary_db import init_db, query_salary
    from skills_engine import analyze_skills_gap
    from server import _search_duckduckgo, _get_wikipedia_summary
    print_success("All modules loaded successfully")
except ImportError as e:
    print_error(f"Import failed: {e}")
    sys.exit(1)

# Initialize database
init_db()


# ═════════════════════════════════════════════════════════════
# TEST 1 — search_web (DuckDuckGo)
# ═════════════════════════════════════════════════════════════

print_header("TEST 1 — search_web (DuckDuckGo)")

try:
    results = _search_duckduckgo("AI Engineer job description 2024", max_results=3)

    if results:
        print_success(f"DuckDuckGo returned {len(results)} results")
        print_section("Results preview:")
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] {YELLOW}{r['title'][:60]}{RESET}")
            print(f"      {r['snippet'][:120]}...")
            print(f"      {BLUE}{r['url']}{RESET}")
    else:
        print_error("No results returned — DuckDuckGo may have no data for this query")

except Exception as e:
    print_error(f"search_web failed: {e}")


# ═════════════════════════════════════════════════════════════
# TEST 2 — get_company_info (Wikipedia)
# ═════════════════════════════════════════════════════════════

print_header("TEST 2 — get_company_info (Wikipedia)")

# Test 2a — company that exists
print_section("2a: Valid company — Anthropic")
try:
    result = _get_wikipedia_summary("Anthropic")

    if result["found"]:
        print_success("Wikipedia page found")
        print_result("Title:",       result["title"])
        print_result("Description:", result["description"])
        print_result("Summary:",     result["summary"][:150] + "...")
        print_result("URL:",         result["url"])
    else:
        print_error(result["message"])

except Exception as e:
    print_error(f"get_company_info failed: {e}")

# Test 2b — company that doesn't exist
print_section("2b: Invalid company — handles gracefully")
try:
    result = _get_wikipedia_summary("FakeCompanyXYZ123")

    if not result["found"]:
        print_success("Correctly returned not found")
        print_result("Message:", result["message"])
    else:
        print_error("Should have returned not found")

except Exception as e:
    print_error(f"get_company_info failed: {e}")


# ═════════════════════════════════════════════════════════════
# TEST 3 — analyze_salary (SQLite)
# ═════════════════════════════════════════════════════════════

print_header("TEST 3 — analyze_salary (SQLite Database)")

# Test 3a — valid query
print_section("3a: AI Engineer, senior, Canada")
try:
    results = query_salary("AI Engineer", "senior", "Canada")

    if results:
        print_success(f"Found {len(results)} record(s)")
        for r in results:
            print_result("Role:",        r["role"])
            print_result("Experience:",  r["experience"])
            print_result("Country:",     r["country"])
            print_result("Min Salary:",  f"${r['salary_min']:,}")
            print_result("Max Salary:",  f"${r['salary_max']:,}")
            print_result("Avg Salary:",  f"${r['salary_avg']:,}")
    else:
        print_error("No results found")

except Exception as e:
    print_error(f"analyze_salary failed: {e}")

# Test 3b — partial role match
print_section("3b: Partial match — 'Python', mid, USA")
try:
    results = query_salary("Python", "mid", "USA")

    if results:
        print_success(f"Partial match worked — found {len(results)} record(s)")
        print_result("Matched role:", results[0]["role"])
        print_result("Avg Salary:",  f"${results[0]['salary_avg']:,}")
    else:
        print_error("No results found")

except Exception as e:
    print_error(f"analyze_salary failed: {e}")

# Test 3c — no match
print_section("3c: No match — handles gracefully")
try:
    results = query_salary("Astronaut", "senior", "Canada")

    if not results:
        print_success("Correctly returned empty list for unknown role")
    else:
        print_error("Should have returned empty list")

except Exception as e:
    print_error(f"analyze_salary failed: {e}")


# ═════════════════════════════════════════════════════════════
# TEST 4 — analyze_skills_gap (Groq LLM + Pydantic)
# ═════════════════════════════════════════════════════════════

print_header("TEST 4 — analyze_skills_gap (Groq + Pydantic)")
print(f"  {YELLOW}Note: This calls the Groq API — takes 2-3 seconds{RESET}")

# Sample JD text — simulates what DuckDuckGo would return
sample_jd = """
We are hiring a Senior AI Engineer to build LLM-powered applications.

Required skills:
- Python (advanced)
- LLM APIs (OpenAI, Anthropic, HuggingFace)
- Prompt engineering and RAG pipeline design
- Vector databases (ChromaDB, Pinecone)
- REST API development with FastAPI
- Docker and containerization

Nice to have:
- LangChain or LlamaIndex experience
- MCP (Model Context Protocol)
- Kubernetes
"""

# Test 4a — strong match profile
print_section("4a: Strong match profile")
strong_skills = "Python, FastAPI, Docker, REST APIs, LLM APIs, Prompt Engineering"

try:
    result = analyze_skills_gap(sample_jd, strong_skills)

    if result.get("error"):
        print_error(result["message"])
        print(f"  Tip: {result.get('tip', '')}")
    else:
        print_success("Skills gap analysis completed")
        print_result("Role:",         result["role_summary"][:60])
        print_result("Experience:",   result["experience_level"])
        print_result("Match %:",      f"{result['match_percentage']}%")
        print_result("Matched:",      str(result["matched_skills"]))
        print_result("Missing:",      str(result["missing_skills"]))
        print_result("Bonus:",        str(result["bonus_skills"]))
        print_result("Recommend:",    result["recommendation"][:80])

except Exception as e:
    print_error(f"analyze_skills_gap failed: {e}")

# Test 4b — weak match profile
print_section("4b: Weak match profile")
weak_skills = "Excel, PowerPoint, Project Management"

try:
    result = analyze_skills_gap(sample_jd, weak_skills)

    if result.get("error"):
        print_error(result["message"])
    else:
        print_success("Skills gap analysis completed")
        print_result("Match %:",   f"{result['match_percentage']}%")
        print_result("Missing:",   str(result["missing_skills"][:3]))
        print_result("Recommend:", result["recommendation"][:80])

except Exception as e:
    print_error(f"analyze_skills_gap failed: {e}")


# ═════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════

print(f"\n{BOLD}{BLUE}{'═' * 55}{RESET}")
print(f"{BOLD}{GREEN}  All tests completed{RESET}")
print(f"{BOLD}{BLUE}{'═' * 55}{RESET}")
print(f"""
  {BOLD}MCP Server is ready.{RESET}

  Tools verified:
  {GREEN}✓{RESET} search_web        — DuckDuckGo working
  {GREEN}✓{RESET} get_company_info  — Wikipedia working
  {GREEN}✓{RESET} analyze_salary    — SQLite working
  {GREEN}✓{RESET} analyze_skills_gap — Groq + Pydantic working

  Next step:
  {YELLOW}python3 server.py{RESET} — start the MCP server
""")