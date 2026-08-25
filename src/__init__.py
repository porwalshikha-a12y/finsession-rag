"""FinSession-RAG package.

Loads a .env file (if present) when the package is first imported, so
OPENAI_API_KEY and PGVECTOR_DSN can live in a gitignored file instead of
being exported in every shell. Real environment variables always win over
.env values.
"""
from __future__ import annotations

try:  # python-dotenv is optional; the project works without it
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover
    pass
