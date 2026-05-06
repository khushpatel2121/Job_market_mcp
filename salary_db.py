# salary_db.py
# Handles everything related to our SQLite salary database.
# This simulates what would be an internal HR / compensation database
# in a real company — exactly what the JD means by "internal databases".

import sqlite3   # built into Python, no install needed

# ─────────────────────────────────────────────────────────────
# SEED DATA
# This is the data we will insert into the database on first run.
# In a real project this would come from an actual HR system or
# a compensation survey dataset. We hardcode it here to keep
# the project self-contained and free.
# Each record = one salary data point.
# ─────────────────────────────────────────────────────────────

SALARY_DATA = [

    # ── Python Developer ──────────────────────────────────────
    ("Python Developer", "junior",  "Canada",  70000,  90000,  80000),
    ("Python Developer", "mid",     "Canada",  90000, 120000, 105000),
    ("Python Developer", "senior",  "Canada", 120000, 160000, 140000),
    ("Python Developer", "junior",  "USA",     80000, 105000,  92000),
    ("Python Developer", "mid",     "USA",    105000, 145000, 125000),
    ("Python Developer", "senior",  "USA",    145000, 200000, 172000),

    # ── AI Engineer ───────────────────────────────────────────
    ("AI Engineer",      "junior",  "Canada",  85000, 110000,  97000),
    ("AI Engineer",      "mid",     "Canada", 110000, 150000, 130000),
    ("AI Engineer",      "senior",  "Canada", 150000, 200000, 175000),
    ("AI Engineer",      "junior",  "USA",    100000, 135000, 117000),
    ("AI Engineer",      "mid",     "USA",    135000, 185000, 160000),
    ("AI Engineer",      "senior",  "USA",    185000, 260000, 220000),

    # ── Data Scientist ────────────────────────────────────────
    ("Data Scientist",   "junior",  "Canada",  75000,  95000,  85000),
    ("Data Scientist",   "mid",     "Canada",  95000, 130000, 112000),
    ("Data Scientist",   "senior",  "Canada", 130000, 175000, 152000),
    ("Data Scientist",   "junior",  "USA",     90000, 115000, 102000),
    ("Data Scientist",   "mid",     "USA",    115000, 155000, 135000),
    ("Data Scientist",   "senior",  "USA",    155000, 220000, 187000),

    # ── Backend Engineer ──────────────────────────────────────
    ("Backend Engineer", "junior",  "Canada",  68000,  88000,  78000),
    ("Backend Engineer", "mid",     "Canada",  88000, 120000, 104000),
    ("Backend Engineer", "senior",  "Canada", 120000, 158000, 139000),
    ("Backend Engineer", "junior",  "USA",     85000, 110000,  97000),
    ("Backend Engineer", "mid",     "USA",    110000, 150000, 130000),
    ("Backend Engineer", "senior",  "USA",    150000, 210000, 180000),

    # ── DevOps Engineer ───────────────────────────────────────
    ("DevOps Engineer",  "junior",  "Canada",  72000,  92000,  82000),
    ("DevOps Engineer",  "mid",     "Canada",  92000, 125000, 108000),
    ("DevOps Engineer",  "senior",  "Canada", 125000, 165000, 145000),
    ("DevOps Engineer",  "junior",  "USA",     88000, 112000, 100000),
    ("DevOps Engineer",  "mid",     "USA",    112000, 152000, 132000),
    ("DevOps Engineer",  "senior",  "USA",    152000, 215000, 183000),

    # ── Product Manager ───────────────────────────────────────
    ("Product Manager",  "junior",  "Canada",  75000,  95000,  85000),
    ("Product Manager",  "mid",     "Canada",  95000, 130000, 112000),
    ("Product Manager",  "senior",  "Canada", 130000, 170000, 150000),
    ("Product Manager",  "junior",  "USA",     90000, 120000, 105000),
    ("Product Manager",  "mid",     "USA",    120000, 165000, 142000),
    ("Product Manager",  "senior",  "USA",    165000, 230000, 197000),

]

# ─────────────────────────────────────────────────────────────
# DATABASE PATH
# This is where SQLite will create the database file on disk.
# When you run the project, a file called "salaries.db" will
# appear in your project folder.
# ─────────────────────────────────────────────────────────────

DB_PATH = "salaries.db"


# ─────────────────────────────────────────────────────────────
# init_db()
# Creates the database, creates the table, inserts seed data.
# Safe to call multiple times — it checks before inserting
# so data is never duplicated.
# ─────────────────────────────────────────────────────────────

def init_db():
    # sqlite3.connect() creates the .db file if it doesn't exist yet
    conn = sqlite3.connect(DB_PATH)

    # A cursor is what actually executes SQL statements
    cursor = conn.cursor()

    # CREATE TABLE IF NOT EXISTS means this won't fail if the
    # table already exists — safe to run multiple times
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS salaries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            role            TEXT NOT NULL,
            experience      TEXT NOT NULL,
            country         TEXT NOT NULL,
            salary_min      INTEGER NOT NULL,
            salary_max      INTEGER NOT NULL,
            salary_avg      INTEGER NOT NULL
        )
    """)

    # Only insert seed data if the table is empty
    # This prevents duplicates if init_db() is called again
    cursor.execute("SELECT COUNT(*) FROM salaries")
    count = cursor.fetchone()[0]

    if count == 0:
        cursor.executemany("""
            INSERT INTO salaries
                (role, experience, country, salary_min, salary_max, salary_avg)
            VALUES (?, ?, ?, ?, ?, ?)
        """, SALARY_DATA)
        print(f"Seeded {len(SALARY_DATA)} salary records into database.")
    else:
        print(f"Database already has {count} records. Skipping seed.")

    # commit() saves the changes permanently to the file
    conn.commit()

    # Always close the connection when done
    conn.close()


# ─────────────────────────────────────────────────────────────
# query_salary()
# This is the function our MCP tool will call directly.
# It takes a role, experience level, and country as filters
# and returns matching salary records from the database.
#
# Parameters:
#   role       — job title to search for e.g. "AI Engineer"
#   experience — "junior", "mid", or "senior"
#   country    — "Canada" or "USA"
#
# Returns a list of dicts — one dict per matching row.
# ─────────────────────────────────────────────────────────────

def query_salary(
    role: str,
    experience: str,
    country: str
) -> list[dict]:

    conn = sqlite3.connect(DB_PATH)

    # row_factory makes each row behave like a dict
    # so we can access columns by name: row["salary_avg"]
    # instead of by index: row[5]
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    # We use LIKE with % wildcards so partial matches work.
    # e.g. searching "Python" will match "Python Developer"
    # UPPER() makes the search case-insensitive
    cursor.execute("""
        SELECT role, experience, country,
               salary_min, salary_max, salary_avg
        FROM salaries
        WHERE UPPER(role)       LIKE UPPER(?)
          AND UPPER(experience) = UPPER(?)
          AND UPPER(country)    = UPPER(?)
    """, (f"%{role}%", experience, country))

    rows = cursor.fetchall()
    conn.close()

    # Convert each sqlite3.Row object into a plain Python dict
    # This is important — MCP tools need to return JSON-serializable data
    return [dict(row) for row in rows]


# ─────────────────────────────────────────────────────────────
# This block only runs when you execute this file directly:
#   python3 salary_db.py
# It will NOT run when this file is imported by server.py
# Use it to test that the database is working correctly.
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialize the database
    init_db()

    # Test a query and print the results
    print("\nTest query — AI Engineer, senior, Canada:")
    results = query_salary("AI Engineer", "senior", "Canada")
    for row in results:
        print(row)

    print("\nTest query — Python Developer, mid, USA:")
    results = query_salary("Python Developer", "mid", "USA")
    for row in results:
        print(row)