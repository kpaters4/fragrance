"""Supabase client for the Fragrance API.

Requires SUPABASE_URL and SUPABASE_KEY in the environment. The key must be a
secret/service key (not the publishable key) since build.py and serve.py run
server-side only.
"""
import os

from supabase import Client, create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client
