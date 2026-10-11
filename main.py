import json
import os
import re
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(title="Acme Appliances Ledger Agent")

DATA_FILE = Path(__file__).with_name("ledger_export.ndjson")
MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
IST = ZoneInfo("Asia/Kolkata")

USD_PER_UNIT = {
    "USD": 1.0,
    "EUR": 1.03,
    "INR": 0.01145,
}

# Connect in-memory SQLite database
conn = sqlite3.connect(":memory:", check_same_thread=False)

def init_db():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Ledger file not found: {DATA_FILE}")

    latest = {}
    with DATA_FILE.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            order = json.loads(line)
            oid = order["id"]
            if oid not in latest or order["updated_at"] > latest[oid]["updated_at"]:
                latest[oid] = order

    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id TEXT PRIMARY KEY,
        customer TEXT,
        region TEXT,
        product TEXT,
        qty INTEGER,
        unit_price REAL,
        amount REAL,
        currency TEXT,
        amount_usd REAL,
        status TEXT,
        created_at TEXT,
        updated_at TEXT,
        year INTEGER,
        month INTEGER,
        month_name TEXT,
        quarter INTEGER,
        date TEXT
    )
    """)

    rows = []
    for r in latest.values():
        cr = datetime.fromisoformat(r["created_at"])
        if cr.tzinfo is None:
            cr = cr.replace(tzinfo=IST)
        cr_ist = cr.astimezone(IST)

        up = datetime.fromisoformat(r["updated_at"])
        if up.tzinfo is None:
            up = up.replace(tzinfo=IST)
        up_ist = up.astimezone(IST)

        rate = USD_PER_UNIT.get(r["currency"], 1.0)
        amount_usd = float(r["amount"]) * rate

        rows.append((
            r["id"],
            r["customer"],
            r["region"],
            r["product"],
            int(r["qty"]),
            float(r["unit_price"]),
            float(r["amount"]),
            r["currency"],
            amount_usd,
            r["status"],
            cr_ist.isoformat(),
            up_ist.isoformat(),
            cr_ist.year,
            cr_ist.month,
            cr_ist.strftime("%B"),
            (cr_ist.month - 1) // 3 + 1,
            cr_ist.strftime("%Y-%m-%d")
        ))

    cursor.executemany("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    print(f"Database initialized with {len(rows)} unique orders.")

init_db()

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from the environment")
    return genai.Client(api_key=api_key)

SQL_PROMPT_TEMPLATE = """You are an expert SQL engineer. Given a plain English question about an order ledger, write a single SQLite query for the `orders` table to answer it.
Return ONLY the raw SQL query. Do not wrap in markdown (no ```sql). No explanations. No comments.

SQLite Schema:
TABLE orders (
    id TEXT,            -- e.g. 'ORD-11483'
    customer TEXT,      -- customer ID, e.g. 'C1132'
    region TEXT,        -- 'Central', 'East', 'North', 'South', 'West'
    product TEXT,       -- 'Air Fryer', 'Blender', 'Grinder', 'Juicer', 'Kettle', 'Mixer', 'Rice Cooker', 'Toaster'
    qty INTEGER,        -- quantity of units
    unit_price REAL,    -- price per unit in local currency
    amount REAL,        -- total amount in local currency
    currency TEXT,      -- 'USD', 'EUR', 'INR'
    amount_usd REAL,    -- amount in USD. ALWAYS use this for money, revenue, sales, and refunds!
    status TEXT,        -- 'paid', 'refunded', 'void'
    year INTEGER,       -- 2026
    month INTEGER,      -- 1 to 12 (1=January, 2=February, 3=March, 4=April, 5=May, 6=June)
    month_name TEXT,    -- 'January', 'February', 'March', 'April', 'May', 'June'
    quarter INTEGER,    -- 1 (Jan-Mar) or 2 (Apr-Jun)
    date TEXT           -- 'YYYY-MM-DD'
);

Rules:
1. "Revenue", "sales", or customer spending ALWAYS means status = 'paid'. Sum amount_usd: ROUND(SUM(amount_usd), 2).
2. "Refunds" or "refunded" ALWAYS means status = 'refunded'. Sum amount_usd: ROUND(SUM(amount_usd), 2).
3. If asked for a product name, return the exact product name: SELECT product ...
4. If asked for a customer ID, return the customer ID: SELECT customer ...
5. If asked for unique/distinct customers, use COUNT(DISTINCT customer).
6. If asked for total orders or number of orders, use COUNT(*).
7. If asked for total quantity / units sold, use SUM(qty).
8. If asked for top / most / highest, use ORDER BY ... DESC LIMIT 1.
9. If asked for lowest / least, use ORDER BY ... ASC LIMIT 1.
10. All monetary values must be rounded to 2 decimal places: ROUND(..., 2).
11. Return a query that produces exactly 1 row and 1 column.

Question: {question}
SQL:"""

def generate_sql(question: str) -> str:
    client = get_client()
    prompt = SQL_PROMPT_TEMPLATE.format(question=question)

    last_error = None
    for model_name in MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            if response.text:
                sql = response.text.strip()
                sql = re.sub(r"^```sql", "", sql, flags=re.IGNORECASE)
                sql = re.sub(r"^```", "", sql)
                sql = re.sub(r"```$", "", sql).strip()
                return sql.rstrip(";")
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            last_error = e

    raise last_error or RuntimeError("All Gemini models failed")

@app.post("/")
def solve(request: Question):
    try:
        print(f"Received question: {request.question}")
        sql = generate_sql(request.question)
        print(f"Generated SQL: {sql}")

        cursor = conn.cursor()
        row = cursor.execute(sql).fetchone()
        if not row:
            raise ValueError("Query returned no results")

        ans = row[0]
        if isinstance(ans, float):
            ans = round(ans, 2)
        elif isinstance(ans, str):
            ans = ans.strip()

        print(f"Calculated answer: {ans}")
        return {"answer": ans}

    except Exception as exc:
        print(f"Ledger Agent error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"Error answering question: {exc}",
        ) from exc

@app.get("/")
def health():
    cursor = conn.cursor()
    count = cursor.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    return {
        "service": "Acme Appliances Ledger Agent",
        "orders_loaded": count,
        "llm_enabled": bool(os.getenv("GEMINI_API_KEY")),
    }
