# skills_engine.py
# Dynamic skills gap analyzer using Groq API + Pydantic output parsing.
#
# Flow:
#   1. Receives raw JD text (from DuckDuckGo search results)
#   2. Sends JD text to Groq API (free, runs LLaMA3 in cloud)
#   3. Pydantic validates the LLM JSON output
#   4. Compares extracted skills against user's provided skills
#   5. Returns structured gap analysis
#
# This demonstrates:
#   - Free cloud LLM integration (Groq + LLaMA3)
#   - Pydantic for LLM output parsing (exactly what you learned)
#   - Sequential tool invocation (search → extract → analyze)

import os
import json
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load the GROQ_API_KEY from .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────
# GROQ CLIENT SETUP
# Groq free tier gives 14,400 requests/day using LLaMA3
# No RAM pressure — inference happens on Groq's servers
# ─────────────────────────────────────────────────────────────

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

GROQ_MODEL = "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# Define exact structure we expect from the LLM.
# If the LLM returns wrong field names or wrong types,
# Pydantic catches it with a clear ValidationError.
# This is exactly the LLM output formatting you learned.
# ─────────────────────────────────────────────────────────────

class SkillsExtractionResult(BaseModel):
    """
    Pydantic model for the LLM's skill extraction output.
    LLM is instructed to return JSON matching this exact shape.
    Pydantic validates every field before we use the data.
    """
    required_skills:     list[str] = Field(
        description="Skills explicitly required in the job description"
    )
    nice_to_have_skills: list[str] = Field(
        description="Skills listed as preferred or bonus"
    )
    experience_level:    str = Field(
        description="Seniority level: junior, mid, or senior"
    )
    role_summary:        str = Field(
        description="One sentence summary of the role"
    )


class GapAnalysisResult(BaseModel):
    """
    Pydantic model for the final gap analysis result.
    This is what our MCP tool returns to the agent.
    """
    role_summary:      str
    experience_level:  str
    user_skills:       list[str]
    matched_skills:    list[str] = Field(
        description="Skills the user has that match the JD"
    )
    missing_skills:    list[str] = Field(
        description="Required skills the user does not have"
    )
    bonus_skills:      list[str] = Field(
        description="Nice-to-have skills the user already has"
    )
    match_percentage:  float = Field(
        description="Percentage of required skills the user has"
    )
    recommendation:    str = Field(
        description="Short actionable recommendation for the user"
    )


# ─────────────────────────────────────────────────────────────
# extract_skills_with_llm()
# Sends raw JD text to Groq API.
# Instructs LLM to respond ONLY in JSON.
# Parses and validates response with Pydantic.
# ─────────────────────────────────────────────────────────────

def extract_skills_with_llm(jd_text: str) -> SkillsExtractionResult:

    prompt = f"""
You are a technical recruiter analyzing a job description.
Extract skills and information from the job description below.

Respond with ONLY valid JSON — no explanation, no markdown, no extra text.
The JSON must match this exact structure:

{{
  "required_skills": ["skill1", "skill2", "skill3"],
  "nice_to_have_skills": ["skill1", "skill2"],
  "experience_level": "junior" or "mid" or "senior",
  "role_summary": "one sentence description of the role"
}}

Job Description:
{jd_text}

JSON response:
"""

    # Call Groq API — same concept as any REST API call
    # Groq runs LLaMA3 on their servers, result comes back fast
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=GROQ_MODEL,
        temperature=0.1,    # low temperature = more consistent JSON output
        max_tokens=1024,
    )

    # Extract the text response
    raw_text = chat_completion.choices[0].message.content.strip()

    # Strip markdown code blocks if LLM wrapped the JSON in them
    # e.g. ```json { ... } ``` → { ... }
    if "```" in raw_text:
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    # Parse raw text into a Python dict
    parsed_json = json.loads(raw_text)

    # Validate with Pydantic — catches wrong fields or wrong types
    result = SkillsExtractionResult.model_validate(parsed_json)

    return result


# ─────────────────────────────────────────────────────────────
# analyze_skills_gap()
# Main function our MCP tool calls.
#
# Parameters:
#   jd_text     — raw job description text from search results
#   user_skills — comma-separated string e.g. "Python, Docker, SQL"
#
# Returns GapAnalysisResult as a plain dict (JSON serializable)
# ─────────────────────────────────────────────────────────────

def analyze_skills_gap(jd_text: str, user_skills: str) -> dict:

    # Step 1 — extract skills from JD using LLM + Pydantic
    try:
        extraction = extract_skills_with_llm(jd_text)
    except json.JSONDecodeError as e:
        return {
            "error": True,
            "message": f"LLM returned invalid JSON: {str(e)}",
            "tip": "Try again — LLM occasionally returns malformed JSON"
        }
    except Exception as e:
        return {
            "error": True,
            "message": f"Failed to extract skills: {str(e)}",
            "tip": "Check your GROQ_API_KEY in the .env file"
        }

    # Step 2 — parse user's skills into a clean lowercase list
    user_skill_list = [
        s.strip().lower()
        for s in user_skills.split(",")
        if s.strip()
    ]

    # Step 3 — compare user skills against extracted JD skills
    # Partial matching so "python" matches "Python (advanced)"
    matched_skills = []
    missing_skills = []
    bonus_skills   = []

    for required in extraction.required_skills:
        required_lower = required.lower()
        user_has_it = any(
            user_skill in required_lower or required_lower in user_skill
            for user_skill in user_skill_list
        )
        if user_has_it:
            matched_skills.append(required)
        else:
            missing_skills.append(required)

    for nice in extraction.nice_to_have_skills:
        nice_lower = nice.lower()
        user_has_it = any(
            user_skill in nice_lower or nice_lower in user_skill
            for user_skill in user_skill_list
        )
        if user_has_it:
            bonus_skills.append(nice)

    # Step 4 — calculate match percentage
    total_required = len(extraction.required_skills)
    matched_count  = len(matched_skills)
    match_pct      = round((matched_count / total_required) * 100, 1) \
                     if total_required > 0 else 0.0

    # Step 5 — generate recommendation
    if match_pct >= 80:
        recommendation = (
            f"Strong match at {match_pct}%. "
            f"Fill these remaining gaps: {', '.join(missing_skills[:2])}."
        )
    elif match_pct >= 50:
        recommendation = (
            f"Moderate match at {match_pct}%. "
            f"Priority skills to learn: {', '.join(missing_skills[:3])}."
        )
    else:
        recommendation = (
            f"Early stage match at {match_pct}%. "
            f"Start with these foundations: {', '.join(missing_skills[:3])}."
        )

    # Step 6 — build final result with Pydantic then return as dict
    result = GapAnalysisResult(
        role_summary     = extraction.role_summary,
        experience_level = extraction.experience_level,
        user_skills      = [s.strip() for s in user_skills.split(",")],
        matched_skills   = matched_skills,
        missing_skills   = missing_skills,
        bonus_skills     = bonus_skills,
        match_percentage = match_pct,
        recommendation   = recommendation
    )

    return result.model_dump()


# ─────────────────────────────────────────────────────────────
# Test directly:  python3 skills_engine.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    sample_jd = """
    We are looking for a Senior AI Engineer to join our team.
    You will design and deploy LLM-powered applications in production.

    Required:
    - Strong Python programming skills
    - Experience with LLM APIs (OpenAI, Anthropic, or HuggingFace)
    - Prompt engineering and RAG pipeline design
    - Vector databases such as Pinecone or ChromaDB
    - REST API development with FastAPI

    Nice to have:
    - Experience with LangChain or LlamaIndex
    - Knowledge of MCP (Model Context Protocol)
    - Docker and containerization
    """

    my_skills = "Python, Docker, SQL, REST APIs, Git"

    print("Sending JD to Groq API for skill extraction...\n")

    result = analyze_skills_gap(sample_jd, my_skills)

    if result.get("error"):
        print(f"Error: {result['message']}")
        print(f"Tip:   {result['tip']}")
    else:
        print(f"Role:             {result['role_summary']}")
        print(f"Experience:       {result['experience_level']}")
        print(f"Match:            {result['match_percentage']}%")
        print(f"Matched skills:   {result['matched_skills']}")
        print(f"Missing skills:   {result['missing_skills']}")
        print(f"Bonus skills:     {result['bonus_skills']}")
        print(f"Recommendation:   {result['recommendation']}")