"""One-off loader for the Supabase tables backing the Fragrance API.

Run once (or after resetting the tables): python scripts/seed_supabase.py
Pass a table name to load only one: python scripts/seed_supabase.py perfumes

Requires SUPABASE_URL and SUPABASE_KEY (a secret key, not the publishable
key) in the environment.

Uses raw httpx rather than the supabase-py client: the client's .insert()
was observed silently double-inserting every batch (likely a client-side
retry resending an already-successful write). Plain httpx.Client with no
retry logic does not have this problem.
"""
import csv
import os
import sys

import httpx

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FILES = {
    "perfumes": os.path.join(DATA_DIR, "perfume_database_cleaned.csv"),
    "collection": os.path.join(DATA_DIR, "collection.csv"),
    "wishlist": os.path.join(DATA_DIR, "wishlist.csv"),
}

BATCH_SIZE = 500


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            notes = (row.get("notes") or "").strip()
            if not notes:
                continue
            rows.append({"brand": row["brand"], "perfume": row["perfume"], "notes": notes})
        return rows


def seed(client, table, path):
    rows = load_rows(path)
    print(f"{table}: {len(rows)} rows to insert")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        resp = client.post(url, json=batch)
        if resp.status_code >= 300:
            raise RuntimeError(f"insert failed at row {i}: {resp.status_code} {resp.text}")
        print(f"  inserted {i + len(batch)}/{len(rows)}")


if __name__ == "__main__":
    only_table = sys.argv[1] if len(sys.argv) > 1 else None
    with httpx.Client(headers=HEADERS, timeout=60) as client:
        for name, path in FILES.items():
            if only_table and name != only_table:
                continue
            seed(client, name, path)
    print("done")
