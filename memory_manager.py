# memory_manager.py
#
# Real, working memory layer for the agent — separate from the salary DB.
#
# Short-term memory is NOT handled here — that's the `messages` list
# already built in agent.py, scoped to one process run.
#
# This module adds the two tiers that were previously just claimed:
#
#   LONG-TERM  — every message from every session gets persisted to
#                disk (memory.db). On startup, the agent loads a short
#                summary of the most recent past session so it isn't
#                starting from zero every time the process restarts.
#
#   EPISODIC   — before answering a new question, we search past
#                logged exchanges for ones that used similar keywords.
#                If we find a strong match, we surface it to the model
#                as extra context ("you answered something similar
#                before"). This is a KEYWORD-OVERLAP match, not an
#                embedding-similarity search — be upfront about that
#                distinction if asked. It's a legitimate, if simple,
#                version of episodic recall.

import sqlite3
import time
import re
from collections import Counter

DB_PATH = "memory.db"

# Common words we don't want dominating keyword overlap scoring
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which",
    "for", "in", "on", "at", "to", "of", "and", "or", "i", "you",
    "me", "my", "can", "do", "does", "how", "tell", "about", "with"
}


def init_memory_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            timestamp  REAL NOT NULL
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session
        ON messages(session_id)
    """)

    conn.commit()
    conn.close()


def log_message(session_id: str, role: str, content: str):
    """Persist one message to long-term storage."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (session_id, role, content, time.time())
    )
    conn.commit()
    conn.close()


def get_last_session_summary(current_session_id: str, max_messages: int = 6) -> str | None:
    """
    LONG-TERM MEMORY.
    Looks up the most recent session that ISN'T the current one,
    and returns a short text summary of what was discussed —
    injected into the system prompt at startup so the agent has
    continuity across process restarts.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT session_id, MAX(timestamp) as last_ts
        FROM messages
        WHERE session_id != ?
        GROUP BY session_id
        ORDER BY last_ts DESC
        LIMIT 1
    """, (current_session_id,))

    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    last_session_id = row["session_id"]

    cursor.execute("""
        SELECT role, content FROM messages
        WHERE session_id = ? AND role = 'user'
        ORDER BY timestamp DESC
        LIMIT ?
    """, (last_session_id, max_messages))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    questions = [r["content"] for r in reversed(rows)]
    return "In the previous session, the user asked about: " + "; ".join(questions)


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def search_episodic(query: str, current_session_id: str, limit: int = 2, min_overlap: int = 2):
    """
    EPISODIC MEMORY (keyword-overlap version).
    Searches past user messages (from any session) for ones that
    share meaningful keywords with the current query. Returns the
    top matches along with how the agent answered them, so the
    model can reference prior interactions instead of starting cold.

    This is intentionally simple — keyword set overlap, not vector
    similarity. Good enough to demonstrate the retrieval pattern,
    not production-grade semantic recall.
    """
    query_keywords = _extract_keywords(query)
    if not query_keywords:
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Pull past user messages (excluding current session) —
    # for a small demo DB this full scan is fine; at scale you'd
    # index keywords in a separate table instead of scanning rows.
    cursor.execute("""
        SELECT id, session_id, content, timestamp
        FROM messages
        WHERE role = 'user' AND session_id != ?
        ORDER BY timestamp DESC
        LIMIT 200
    """, (current_session_id,))

    candidates = cursor.fetchall()

    scored = []
    for row in candidates:
        past_keywords = _extract_keywords(row["content"])
        overlap = len(query_keywords & past_keywords)
        if overlap >= min_overlap:
            scored.append((overlap, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored[:limit]

    results = []
    for overlap, row in top_matches:
        # Find the assistant's reply that followed this user message
        cursor.execute("""
            SELECT content FROM messages
            WHERE session_id = ? AND role = 'assistant' AND timestamp > ?
            ORDER BY timestamp ASC
            LIMIT 1
        """, (row["session_id"], row["timestamp"]))
        reply_row = cursor.fetchone()
        results.append({
            "past_question": row["content"],
            "past_answer":   reply_row["content"] if reply_row else None,
            "overlap_score": overlap
        })

    conn.close()
    return results


if __name__ == "__main__":
    # Quick manual test
    init_memory_db()
    log_message("test_session_1", "user", "What is the salary for AI Engineer in Canada?")
    log_message("test_session_1", "assistant", "Senior AI Engineers in Canada average $175,000.")
    log_message("test_session_2", "user", "What does an AI Engineer make in Canada?")

    print("Last session summary:", get_last_session_summary("test_session_2"))
    print("Episodic matches:", search_episodic("AI Engineer Canada salary", "test_session_2"))