# server.py
import os
import re
import json
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv
from salary_db import init_db, query_salary
from skills_engine import analyze_skills_gap as run_gap_analysis

load_dotenv()

mcp = FastMCP(name="Job Market Intelligence Server")
init_db()


# ═════════════════════════════════════════════════════════════
# PRIVATE HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════

def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """
    Internal helper — DuckDuckGo HTML search.
    Fixed regex — no re.DOTALL to avoid recursion on large HTML.
    """
    url = "https://html.duckduckgo.com/html/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            data={"q": query},
            timeout=10
        )
        content = response.text
    except Exception as e:
        return [{"title": "Search unavailable", "snippet": str(e), "url": ""}]

    results = []

    # ── FIX: Use re.findall WITHOUT re.DOTALL ──────────────────
    # re.DOTALL makes . match newlines too, causing the regex
    # engine to recurse deeply on large HTML → RecursionError
    # Without DOTALL, each match stays on one line — safe and fast

    # Split HTML into lines first, then extract per line
    # This avoids multi-line regex entirely
    lines = content.split("\n")

    snippets = []
    titles   = []
    urls     = []

    for line in lines:
        line = line.strip()

        # Extract snippet
        snippet_match = re.search(
            r'class="result__snippet"[^>]*>([^<]+)',
            line
        )
        if snippet_match:
            snippets.append(snippet_match.group(1).strip())

        # Extract title
        title_match = re.search(
            r'class="result__a"[^>]*>([^<]+)',
            line
        )
        if title_match:
            titles.append(title_match.group(1).strip())

        # Extract URL
        url_match = re.search(
            r'class="result__url"[^>]*>([^<]+)',
            line
        )
        if url_match:
            urls.append(url_match.group(1).strip())

    # Combine into result dicts
    for i in range(min(len(snippets), len(titles), max_results)):
        results.append({
            "title":   titles[i]   if i < len(titles)   else "",
            "snippet": snippets[i] if i < len(snippets) else "",
            "url":     urls[i]     if i < len(urls)     else ""
        })

    return results


def _get_wikipedia_summary(company_name: str) -> dict:
    """
    Internal helper — Wikipedia REST API.
    Free, no key needed.
    """
    formatted = company_name.strip().replace(" ", "_")
    url       = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted}"
    headers   = {"User-Agent": "JobMarketMCP/1.0 (educational project)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except Exception as e:
        return {"found": False, "message": str(e)}

    if response.status_code == 404:
        return {
            "found":   False,
            "message": f"No Wikipedia page found for '{company_name}'",
            "tip":     "Try the full official company name e.g. 'Google LLC'"
        }

    data = response.json()

    return {
        "found":       True,
        "title":       data.get("title", ""),
        "summary":     data.get("extract", ""),
        "url":         data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "description": data.get("description", ""),
    }


# ═════════════════════════════════════════════════════════════
# MCP TOOLS
# ═════════════════════════════════════════════════════════════

@mcp.tool()
def search_web(
    query:       str,
    max_results: int = 5
) -> list[dict]:
    """
    Search the web using DuckDuckGo and return relevant results.

    Use this tool when you need to:
    - Find current job postings for a specific role
    - Search for job descriptions to analyze required skills
    - Look up industry trends or job market information
    - Find information that needs to be up to date

    Args:
        query:       The search query. Be specific for better results.
                     Example: "Senior AI Engineer job description 2024"
        max_results: Number of results to return. Between 1 and 10.

    Returns:
        List of dicts with title, snippet, url.
    """
    return _search_duckduckgo(query, max_results)


@mcp.tool()
def get_company_info(
    company_name: str
) -> dict:
    """
    Get background information about a company from Wikipedia.

    Use this tool when you need to:
    - Research what a company does before a job application
    - Understand a company's industry, size, and history
    - Get factual background on a potential employer

    Args:
        company_name: Full company name. Example: "Google", "Anthropic"

    Returns:
        Dict with found, title, summary, description, url.
    """
    return _get_wikipedia_summary(company_name)


@mcp.tool()
def analyze_salary(
    role:       str,
    experience: str,
    country:    str
) -> dict:
    """
    Look up salary ranges for a specific role, experience level,
    and country from our internal salary database.

    Use this tool when you need to:
    - Find out what salary to expect for a given role
    - Compare compensation across experience levels
    - Research compensation benchmarks by country

    Args:
        role:       Job title. Supported: Python Developer, AI Engineer,
                    Data Scientist, Backend Engineer,
                    DevOps Engineer, Product Manager.
                    Partial matches work — "Python" matches "Python Developer"

        experience: One of: "junior", "mid", "senior"

        country:    One of: "Canada", "USA"

    Returns:
        Dict with results list, count, search_params, message.
    """
    results = query_salary(role, experience, country)

    if not results:
        return {
            "results":       [],
            "count":         0,
            "search_params": {
                "role": role, "experience": experience, "country": country
            },
            "message": (
                f"No data found for '{role}' / '{experience}' / '{country}'. "
                f"Try: role='AI Engineer', experience='mid', country='Canada'"
            )
        }

    return {
        "results":       results,
        "count":         len(results),
        "search_params": {
            "role": role, "experience": experience, "country": country
        },
        "message": f"Found {len(results)} record(s)."
    }


@mcp.tool()
def analyze_skills_gap(
    job_title:   str,
    user_skills: str,
) -> dict:
    """
    Analyze the skill gap between a user's current skills and
    what is required for a target job role.

    Uses an LLM to identify required skills for the role,
    then compares against the user's provided skills.

    Use this tool when you need to:
    - Identify skill gaps for a career transition
    - Understand what skills to learn for a target role
    - Get a match percentage against a specific job type
    - Get a prioritized learning recommendation

    Args:
        job_title:   Target role. Example: "AI Engineer", "Data Scientist"

        user_skills: Comma-separated skills.
                     Example: "Python, Docker, SQL, Git, REST APIs"

    Returns:
        Dict with role_summary, experience_level, user_skills,
        matched_skills, missing_skills, bonus_skills,
        match_percentage, recommendation.
    """
    # Build JD text from job title directly — no DuckDuckGo call needed.
    # The LLM already knows standard requirements for common tech roles.
    # This removes one full network call → much faster response.
    jd_text = f"""
    Standard industry job description for a {job_title} role.
    This is a typical mid to senior level {job_title} position
    in the tech industry requiring standard {job_title} skills.
    """

    # run_gap_analysis is the imported skills_engine function
    # NOT this tool — no recursion possible
    result = run_gap_analysis(jd_text, user_skills)
    return result


# ═════════════════════════════════════════════════════════════
# MCP RESOURCES
# ═════════════════════════════════════════════════════════════

@mcp.resource("config://supported-roles")
def get_supported_roles() -> str:
    """List of job roles supported by the salary database."""
    roles = [
        "AI Engineer", "Python Developer", "Data Scientist",
        "Backend Engineer", "DevOps Engineer", "Product Manager"
    ]
    return json.dumps({"supported_roles": roles})


@mcp.resource("config://supported-countries")
def get_supported_countries() -> str:
    """List of countries supported by the salary database."""
    return json.dumps({"supported_countries": ["Canada", "USA"]})


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting Job Market Intelligence MCP Server...")
    print("Tools: search_web, get_company_info, analyze_salary, analyze_skills_gap")
    print("Waiting for MCP client connection...\n")
    mcp.run(transport="stdio")