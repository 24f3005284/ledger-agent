
import json
import os
import re
import traceback
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(title="Acme Appliances Ledger Agent")

DATA_FILE = Path(__file__).with_name("ledger_export.ndjson")
MODEL = "gemini-3.8-flash"
IST = ZoneInfo("Asia/Kolkata")

USD_PER_UNIT = {
    "USD": 1.0,
    "EUR": 1.03,
    "INR": 0.01145,
}

REGIONS = {"North", "South", "East", "West", "Central"}
CURRENCIES = set(USD_PER_UNIT)
STATUSES = {"paid", "refunded", "void", "all"}
METRICS = {"revenue", "refunds", "count", "top_product"}


def load_orders():
    """Load the latest version of each order from the NDJSON export."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Ledger file not found: {DATA_FILE}")

    latest = {}

    with DATA_FILE.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                order = json.loads(line)
                order_id = order["id"]
                previous = latest.get(order_id)

                if (
                    previous is None
                    or order["updated_at"] > previous["updated_at"]
                ):
                    latest[order_id] = order

            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid ledger record on line {line_number}"
                ) from exc

    return list(latest.values())


ORDERS = load_orders()
PRODUCTS = sorted({order["product"] for order in ORDERS})


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def get_client():
    """Create the Gemini client using the configured environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from the server environment"
        )

    return genai.Client(api_key=api_key)


def interpret_question(question: str) -> dict:
    """Use Gemini to translate natural language into a structured query."""
    client = get_client()

    schema_description = {
        "metric": "revenue, refunds, count, or top_product",
        "region": "exact region name or null",
        "product": "exact product name or null",
        "customer_id": "customer ID such as C1132 or null",
        "month": "integer 1-12 or null",
        "year": "integer year or null",
        "start_date": "YYYY-MM-DD or null",
        "end_date": "YYYY-MM-DD or null",
        "currency": "USD, EUR, or INR",
        "status": "paid, refunded, void, or all",
    }

    prompt = f"""
You convert questions about an order ledger into structured JSON queries.

Return exactly one JSON object with these fields:
{json.dumps(schema_description)}

Rules:
- Never return SQL, Python, code, or explanations.
- revenue means the sum of paid order amounts.
- refunds means the sum of refunded order amounts.
- count means the number of matching orders.
- top_product means the product with the largest total quantity.
- A revenue question defaults to status paid.
- A refund question uses metric refunds and status refunded.
- For an order-count question, use the status explicitly requested.
  If no status is specified, use all.
- For a monetary question, default currency to USD unless another
  currency is explicitly requested.
- Use null for filters the user did not specify.
- If a month and year are specified, populate both fields.
- For date ranges, use inclusive start_date and end_date.
- If the user asks about a particular customer, extract the customer ID.
- If the question is ambiguous, choose the closest supported metric.
- Use only the exact region and product names listed below.

Available regions: {json.dumps(sorted(REGIONS))}
Available products: {json.dumps(PRODUCTS)}
Available currencies: {json.dumps(sorted(CURRENCIES))}

User question:
{question}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    query = json.loads(response.text)

    if not isinstance(query, dict):
        raise ValueError("Gemini did not return a JSON object")

    if query.get("metric") not in METRICS:
        raise ValueError("Gemini returned an unsupported metric")

    region = query.get("region")
    if region is not None and region not in REGIONS:
        raise ValueError("Gemini returned an invalid region")

    product = query.get("product")
    if product is not None and product not in PRODUCTS:
        raise ValueError("Gemini returned an invalid product")

    currency = query.get("currency")
    if currency not in CURRENCIES:
        query["currency"] = "USD"

    status = query.get("status")
    if status not in STATUSES:
        query["status"] = (
            "refunded"
            if query["metric"] == "refunds"
            else "all" if query["metric"] == "count"
            else "paid"
        )

    if query["metric"] == "refunds":
        query["status"] = "refunded"

    # Validate customer IDs before using them as filters.
    customer_id = query.get("customer_id")
    if customer_id is not None:
        customer_id = str(customer_id).strip().upper()
        if not re.fullmatch(r"C\d+", customer_id):
            raise ValueError("Gemini returned an invalid customer ID")
        query["customer_id"] = customer_id

    # Validate dates and normalize numeric fields.
    for field in ("month", "year"):
        value = query.get(field)
        if value is not None:
            query[field] = int(value)

    if query.get("month") is not None:
        if query["month"] not in range(1, 13):
            raise ValueError("Invalid month")
        if query.get("year") is None:
            raise ValueError("Month queries require a year")

    if query.get("year") is not None:
        if query["year"] not in range(2000, 2101):
            raise ValueError("Invalid year")

    for field in ("start_date", "end_date"):
        value = query.get(field)
        if value is not None:
            date.fromisoformat(value)

    return query


def order_date(order):
    return (
        datetime.fromisoformat(order["created_at"])
        .astimezone(IST)
        .date()
    )


def amount_in_currency(order, target_currency):
    """Convert the order's source currency using the supplied exchange rates."""
    source_currency = order["currency"]

    if source_currency not in USD_PER_UNIT:
        raise ValueError(f"Unsupported source currency: {source_currency}")

    amount_usd = float(order["amount"]) * USD_PER_UNIT[source_currency]
    return amount_usd / USD_PER_UNIT[target_currency]


def answer_query(query):
    rows = ORDERS[:]

    if query.get("region"):
        rows = [
            order for order in rows
            if order["region"] == query["region"]
        ]

    if query.get("product"):
        rows = [
            order for order in rows
            if order["product"] == query["product"]
        ]

    if query.get("customer_id"):
        customer_id = query["customer_id"]
        rows = [
            order for order in rows
            if order["customer"].upper() == customer_id
        ]

    month = query.get("month")
    year = query.get("year")

    if month is not None:
        rows = [
            order for order in rows
            if order_date(order).month == month
            and order_date(order).year == year
        ]

    start_date = query.get("start_date")
    end_date = query.get("end_date")

    if start_date:
        start = date.fromisoformat(start_date)
        rows = [order for order in rows if order_date(order) >= start]

    if end_date:
        end = date.fromisoformat(end_date)
        rows = [order for order in rows if order_date(order) <= end]

    status = query.get("status", "paid")

    if status != "all":
        rows = [
            order for order in rows
            if order["status"] == status
        ]

    metric = query["metric"]

    if metric == "count":
        return len(rows)

    if metric == "top_product":
        quantities = defaultdict(int)

        for order in rows:
            quantities[order["product"]] += int(order["qty"])

        if not quantities:
            return "No matching records"

        return max(quantities, key=quantities.get)

    currency = query.get("currency", "USD")

    total = sum(
        amount_in_currency(order, currency)
        for order in rows
    )

    return round(total, 2)


@app.post("/")
def solve(request: Question):
    try:
        print(f"Received question: {request.question}")

        query = interpret_question(request.question)
        print(f"Structured query: {query}")

        answer = answer_query(query)
        print(f"Calculated answer: {answer}")

        return {"answer": answer}

    except Exception as exc:
        print(f"Ledger Agent error: {type(exc).__name__}: {exc}")
        traceback.print_exc()

        # Keep internal errors and credentials out of the public response.
        raise HTTPException(
            status_code=502,
            detail="Unable to interpret or answer the question right now",
        ) from exc


@app.get("/")
def health():
    return {
        "service": "Acme Appliances Ledger Agent",
        "orders_loaded": len(ORDERS),
        "llm_enabled": bool(os.getenv("GEMINI_API_KEY")),
    }
