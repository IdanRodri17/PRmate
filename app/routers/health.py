"""
Liveness and health probe endpoints.

  GET /        — app metadata for "is this thing on?" checks
  GET /health  — minimal health check for Render-style polling

Both endpoints are CHEAP by design: no DB queries, no API calls, no
LLM. They must return within milliseconds because they're polled
frequently by external monitoring (Render, uptime checkers, the
GitHub App's own webhook delivery infrastructure).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings

# APIRouter is FastAPI's mechanism for grouping related endpoints in a
# separate module. Routes registered on this router get mounted under
# the FastAPI app via app.include_router() in main.py.
#
# The `tags` argument controls how endpoints group in /docs. Health
# endpoints under one tag, GitHub endpoints under another — keeps the
# auto-generated Swagger UI readable.
router = APIRouter(tags=["health"])


@router.get("/")
async def home() -> dict:
    """Liveness probe — returns app metadata."""
    settings = get_settings()
    return {
        "app": "PRmate",
        "version": "v0.1-foundation",
        "status": "alive",
        "environment": settings.environment,
    }


@router.get("/health")
async def health() -> dict:
    """
    Health check endpoint — Render and most platforms poll this every
    few seconds. Keep it cheap.
    """
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — verify both health endpoints respond via the integrated app
# ─────────────────────────────────────────────────────────────────────────
def _smoke_test() -> None:
    """
    Run with: python -m app.routers.health

    Imports the full app (with all routers and lifespan wired) and hits
    the two health endpoints. The TestClient context manager runs the
    lifespan, so the checkpointer opens and the graph compiles — even
    though these endpoints don't need the graph, this proves they
    coexist correctly with the full startup.
    """
    from fastapi.testclient import TestClient

    # Imported here, not at module top, to avoid the circular-import
    # cost when main.py imports this router file. The cycle is fine
    # because Python's import machinery memoizes loaded modules; this
    # just defers the import until smoke-test time.
    from main import app

    print("=" * 60)
    print("Health router smoke test")
    print("=" * 60)

    with TestClient(app) as client:
        r = client.get("/")
        print(f"  GET /              → {r.status_code} {r.json()}")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

        r = client.get("/health")
        print(f"  GET /health        → {r.status_code} {r.json()}")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    print("=" * 60)
    print("✓ Health endpoints working")
    print("=" * 60)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )
    _smoke_test()
