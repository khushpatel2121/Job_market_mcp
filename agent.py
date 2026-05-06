# agent.py
# The LLM inference agent.
#
# This is the brain of the system. It:
#   1. Connects to our MCP server (server.py)
#   2. Fetches all available tool schemas
#   3. Passes them to Groq so the LLM knows what tools exist
#   4. Runs a conversation loop where Groq decides which tools to call
#   5. Executes tool calls via the MCP client
#   6. Feeds results back to Groq for a final answer

import asyncio
import json
import os
import sys
from groq import Groq
from fastmcp import Client
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# GROQ CLIENT SETUP
# ─────────────────────────────────────────────────────────────

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL  = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


# ─────────────────────────────────────────────────────────────
# convert_tools_to_groq_format()
# Converts FastMCP tool schemas → Groq function calling format
# ─────────────────────────────────────────────────────────────

def convert_tools_to_groq_format(mcp_tools: list) -> list[dict]:
    groq_tools = []
    for tool in mcp_tools:
        groq_tools.append({
            "type": "function",
            "function": {
                "name":        tool.name,
                "description": tool.description,
                "parameters":  tool.inputSchema
            }
        })
    return groq_tools


# ─────────────────────────────────────────────────────────────
# run_agent()
# ─────────────────────────────────────────────────────────────

async def run_agent():

    print(f"\n{BOLD}{BLUE}{'═' * 55}{RESET}")
    print(f"{BOLD}{BLUE}  Job Market Intelligence Agent{RESET}")
    print(f"{BOLD}{BLUE}{'═' * 55}{RESET}")
    print(f"{DIM}  Powered by Groq + FastMCP{RESET}")
    print(f"{DIM}  Type 'quit' or 'exit' to stop{RESET}\n")
    print(f"{DIM}  Connecting to MCP server...{RESET}")

    async with Client("server.py") as mcp_client:

        # ── Tool Discovery ────────────────────────────────────
        # Fetch all tool schemas from MCP server once on startup.
        # These get passed to Groq on every request so the LLM
        # always knows what tools are available.
        mcp_tools  = await mcp_client.list_tools()
        groq_tools = convert_tools_to_groq_format(mcp_tools)

        print(f"{GREEN}  ✓ Connected to MCP server{RESET}")
        print(f"{GREEN}  ✓ Discovered {len(mcp_tools)} tools:{RESET}")
        for t in mcp_tools:
            print(f"{DIM}      - {t.name}{RESET}")
        print()

        # ── Conversation History ──────────────────────────────
        # Full history sent to Groq on every call.
        # Starts with system prompt only.
        # User and assistant messages are appended each turn.
        messages = [
            {
                "role":    "system",
                "content": """You are a Job Market Intelligence Assistant.
You help users research job roles, salaries, company backgrounds,
and skill gaps for career development.

IMPORTANT — only call tools when the user's question genuinely
requires external data. Follow these rules strictly:

USE TOOLS when the user asks about:
- Salary ranges or compensation data  → use analyze_salary
- Skill gaps or career transitions    → use analyze_skills_gap
- Company background or history       → use get_company_info
- Job market trends or job postings   → use search_web

DO NOT USE TOOLS for:
- Greetings or small talk (e.g. "how are you", "hello")
- Questions about your own capabilities (e.g. "what can you do")
- General knowledge you can answer directly
- Follow up questions already answered in the conversation

When you do use tools, only call the ones directly relevant
to the question. Do not call all tools for every question.
Be specific, helpful and actionable in your responses.
When showing salaries, always format with $ and commas."""
            }
        ]

        # ══════════════════════════════════════════════════════
        # OUTER LOOP — conversation loop
        # Runs forever, one iteration = one user question
        # Only exits when user types quit/exit or presses Ctrl+C
        # ══════════════════════════════════════════════════════
        while True:

            # ── Get User Input ────────────────────────────────
            # print() and input() are split so the prompt
            # flushes to terminal immediately before waiting
            try:
                print(f"{BOLD}{CYAN}  You: {RESET}", end="", flush=True)
                user_input = input("").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{DIM}  Goodbye!{RESET}\n")
                break

            # Skip empty input — just show prompt again
            if not user_input:
                continue

            # Exit on quit/exit command
            if user_input.lower() in ("quit", "exit", "bye"):
                print(f"\n{DIM}  Goodbye!{RESET}\n")
                break

            # Add user message to conversation history
            messages.append({
                "role":    "user",
                "content": user_input
            })

            print(f"{DIM}  Thinking...{RESET}")

            # ══════════════════════════════════════════════════
            # INNER LOOP — tool calling loop
            # Runs for ONE user turn until Groq gives a final answer.
            # Each iteration either:
            #   A) Groq returns final answer → print + break inner loop
            #   B) Groq returns tool calls → execute → loop again
            # Breaking the inner loop returns to outer loop
            # which waits for the next user question.
            # ══════════════════════════════════════════════════
            while True:

                # ── Call Groq ─────────────────────────────────
                response = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    tools=groq_tools,
                    tool_choice="auto",
                    max_tokens=2048,
                    temperature=0.3
                )

                assistant_message = response.choices[0].message

                # ── Build Assistant Message Dict ──────────────
                # IMPORTANT: only include tool_calls key if it
                # has actual values. Groq rejects tool_calls: None
                # in subsequent requests — causes 400 error.
                assistant_dict = {
                    "role":    "assistant",
                    "content": assistant_message.content or "",
                }

                if assistant_message.tool_calls:
                    assistant_dict["tool_calls"] = [
                        {
                            "id":       tc.id,
                            "type":     tc.type,
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]

                messages.append(assistant_dict)

                # ── Check for Final Answer ────────────────────
                # If no tool calls → Groq is done for this turn.
                # Print the answer and break the INNER loop only.
                # Outer loop continues → waits for next question.
                if not assistant_message.tool_calls:
                    final_answer = assistant_message.content or ""
                    print(f"\n{BOLD}{GREEN}  Agent:{RESET} {final_answer}\n")
                    sys.stdout.flush()
                    break  # ← breaks INNER loop only, NOT outer loop

                # ── Execute Tool Calls ────────────────────────
                # Groq wants to call one or more tools.
                # Execute each one via the MCP client and feed
                # results back into conversation history.
                print(f"{DIM}  Using tools...{RESET}")

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"{DIM}    → calling {tool_name}({tool_args}){RESET}")

                    try:
                        tool_result = await mcp_client.call_tool(
                            tool_name,
                            tool_args
                        )

                        # FastMCP 3.x returns CallToolResult object
                        # with .content list — not a plain list
                        if (tool_result and
                            tool_result.content and
                            len(tool_result.content) > 0):
                            result_text = tool_result.content[0].text
                        else:
                            result_text = json.dumps(
                                {"result": "No data returned"}
                            )

                    except Exception as e:
                        result_text = json.dumps({
                            "error":   True,
                            "message": str(e)
                        })
                        print(f"{YELLOW}    ⚠ Tool error: {e}{RESET}")

                    # Add tool result to history
                    # Groq reads this on the next inner loop iteration
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      result_text
                    })

                # Inner loop continues → Groq reads tool results
                # and either calls more tools or gives final answer


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_agent())