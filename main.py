"""
PRmate — FastAPI application entry point.

This module is intentionally THIN. Its only responsibilities are:
  1. Define the lifespan context manager (open checkpointer, build graph)
  2. Construct the FastAPI app instance
  3. Mount the routers

All endpoint logic lives in app/routers/. All business logic lives in
app/github/, app/graph/, and app/schemas/. main.py is the composition
root — it wires the pieces together but contains no business logic itself.

Adding a new endpoint group (e.g., V6's pull_request_review handler for
HITL resume): create app/routers/<name>.py, then add one
app.include_router(<name>.router) call below. Nothing else changes.

Run locally:
  uvicorn main:app --reload --port 8000

Expose to GitHub (Lesson 13):
  ngrok http 8000
  Update the GitHub App's webhook URL to the ngrok HTTPS URL.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.graph.builder import build_graph
from app.graph.checkpointer import get_checkpointer
from app.routers import health, webhooks

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Lifespan — app-scoped resource management
# ─────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Open the checkpointer, compile the graph, store on app.state.

    The yield is inside the `async with` block, so the SQLite connection
    stays open for the entire app lifetime. On shutdown (uvicorn SIGTERM),
    control returns to the yield, the with-block exits, and the
    connection closes cleanly.
    """
    settings = get_settings()
    logger.info("PRmate starting up (environment=%s)", settings.environment)

    async with get_checkpointer() as saver:
        app.state.graph = build_graph(saver)
        logger.info("Graph compiled — ready to receive webhooks")

        yield

        logger.info("PRmate shutting down — checkpointer will close cleanly")


# ─────────────────────────────────────────────────────────────────────────
# App instance + router mounts
# ─────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PRmate",
    description="AI-powered Pull Request review agent (V1)",
    lifespan=lifespan,
)

# include_router merges routes from each router into the main app.
# Order is purely conventional here (no path conflicts); listing
# health first keeps "what's the entry point?" easy to read.
app.include_router(health.router)
app.include_router(webhooks.router)


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — integration check
# ─────────────────────────────────────────────────────────────────────────
def _smoke_test() -> None:
    """
    Run with: python main.py

    Quick integration check: lifespan runs, all expected routes are
    mounted, home + health respond.

    For per-router behavioral tests:
      python -m app.routers.health      (health endpoints)
      python -m app.routers.webhooks    (full webhook pipeline — posts to GitHub)
    """
    from fastapi.testclient import TestClient

    print("=" * 60)
    print("Integrated app smoke test")
    print("=" * 60)

    with TestClient(app) as client:
        # Verify all expected routes are registered. app.routes contains
        # entries for every registered route; we extract the path set
        # and check our three V1 paths are present.
        paths = {route.path for route in app.routes}
        expected = {"/", "/health", "/webhook"}
        missing = expected - paths
        assert not missing, f"Missing routes: {missing}"
        print(f"  Routes mounted: {sorted(expected)}")

        # Quick liveness check
        r = client.get("/")
        print(f"  GET /          → {r.status_code} status={r.json()['status']}")
        assert r.status_code == 200

        r = client.get("/health")
        print(f"  GET /health    → {r.status_code} {r.json()}")
        assert r.status_code == 200

    print("=" * 60)
    print("✓ App starts cleanly with all routes mounted")
    print("=" * 60)
    print("\nFor full webhook pipeline tests:")
    print("  python -m app.routers.webhooks")
    print("For health-only tests:")
    print("  python -m app.routers.health")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )
    _smoke_test()
