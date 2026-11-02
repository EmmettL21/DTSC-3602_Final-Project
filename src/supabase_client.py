"""Thin Supabase client wrapper, shared by the upload script and the dashboard.

Both read `SUPABASE_URL`/`SUPABASE_KEY` the same way: a local `.env` (via
`python-dotenv`) when running on a laptop, or real environment variables when
running as a Modal secret -- one credential code path for both. Neither
constant is ever hardcoded; this module raises a clear, catchable error when
they're absent instead of letting a bare `KeyError`/`None`-client crash
whatever imports it.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


class SupabaseNotConfigured(RuntimeError):
    """Raised when SUPABASE_URL / SUPABASE_KEY aren't set.

    Callers (the dashboard, the upload script) catch this and show a setup
    message instead of letting the app crash with no explanation.
    """


@lru_cache(maxsize=1)
def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise SupabaseNotConfigured(
            "SUPABASE_URL and SUPABASE_KEY are not set. Copy .env.example to "
            ".env and fill in your project's URL and API key (Project "
            "Settings -> API in the Supabase dashboard), or set them as real "
            "environment variables / a Modal secret in production."
        )
    return create_client(url, key)
