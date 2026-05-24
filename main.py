"""
PRmate — FastAPI application entry point.

This module wires every PRmate component together:
  - Settings (Pydantic Settings, app/config.py)
  - GitHub webhook signature verification (app/github/webhooks.py)
  - LangGraph builder + compiled graph (app/graph/builder.py)
  - SQLite checkpointer (app/graph/checkpointer.py)

The lifespan context manager handles app-scoped resources:
  - Open the SQLite checkpointer at startup
  - Compile the graph against it
  - Stash the compiled graph on app.state
  - Close the checkpointer on shutdown

V1 surface:
  GET  /           Liveness probe; returns app metadata
  GET  /health     Render-style health check
  POST /webhook    The GitHub webhook receiver

Run locally:
  uvicorn main:app --reload --port 8000

Expose to GitHub (Lesson 13):
  ngrok http 8000
  Update the GitHub App's webhook URL to the ngrok HTTPS URL.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from app.config import get_settings
from app.github.webhooks import dispatch_webhook, verify_signature
from app.graph.builder import build_graph
from app.graph.checkpointer import get_checkpointer
from app.graph.state import initial_state, thread_id_for
from app.schemas.github import PullRequestWebhook

# Module-level logger — every log line is prefixed with "__main__"
# when run directly, or "main" when imported by uvicorn.
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Lifespan — manage app-scoped resources
# ─────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Open the checkpointer, compile the graph, store on app.state.

    The async-with for get_checkpointer() is INSIDE lifespan so the saver
    stays alive for the whole app lifetime. The `yield` is the boundary
    between startup (lines above) and shutdown (lines below). When uvicorn
    receives SIGTERM and shuts the app down, control returns to the `yield`,
    the `async with` exits, and the SQLite connection closes cleanly.
    """
    settings = get_settings()
    logger.info("PRmate starting up (environment=%s)", settings.environment)

    # The checkpointer's async context manager will hold the SQLite
    # connection open for the entire app run.
    async with get_checkpointer() as saver:
        # Compile the graph once, against this checkpointer.
        # app.state.graph is the same compiled object for every request —
        # thread-safe, reusable across different PRs via different
        # thread_ids in the per-request config.
        app.state.graph = build_graph(saver)
        logger.info("Graph compiled — ready to receive webhooks")

        yield  # ← App runs here; lifespan suspends until shutdown

        logger.info("PRmate shutting down — checkpointer will close cleanly")


# ─────────────────────────────────────────────────────────────────────────
# App instance
# ─────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PRmate",
    description="AI-powered Pull Request review agent (V1)",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────
@app.get("/")
async def home() -> dict:
    """
    Liveness probe — returns app metadata.

    Used as a quick "is this thing on?" check, both manually (curl
    http://localhost:8000/) and by ngrok's automatic inspection.
    """
    settings = get_settings()
    return {
        "app": "PRmate",
        "version": "v0.1-foundation",
        "status": "alive",
        "environment": settings.environment,
    }


@app.get("/health")
async def health() -> dict:
    """
    Health check endpoint — Render and most platforms poll this every
    few seconds. Keep it cheap: no DB queries, no API calls, no LLM.
    A 200 response is the signal the app is alive and accepting traffic.
    """
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request) -> dict:
    """
    The GitHub webhook receiver — the entry point for every PR event.

    Flow:
      1. Read raw body (bytes) — required for HMAC verification
      2. Verify X-Hub-Signature-256 against the webhook secret
      3. Dispatch via app/github/webhooks.py:
         - ping events → {"pong": True}
         - unhandled events → {"ignored": True, ...}
         - draft PRs → {"ignored": True, "reason": "draft_pr"}
         - real PR actions (opened/synchronize/reopened) → {"accepted": True, ...}
      4. If accepted, parse the body again into PullRequestWebhook and
         invoke the compiled graph
      5. Return the final result (dispatch metadata + graph outputs)
    """
    # ─── Step 1: raw body ───
    # We MUST grab raw bytes before any JSON parsing, because the HMAC
    # is computed over the exact bytes GitHub sent. Re-serializing
    # parsed JSON would produce different bytes (key order, whitespace)
    # and the signature check would fail.
    raw_body = await request.body()

    # ─── Step 2: HMAC verification ───
    # This is the security boundary. Anything past this line is trusted
    # to have come from GitHub (or someone holding our webhook secret).
    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(raw_body, signature):
        logger.warning("Webhook rejected: invalid HMAC signature")
        # 401 is the HTTP convention for "you didn't prove your identity."
        # GitHub will see this and stop retrying after 3 attempts.
        raise HTTPException(status_code=401, detail="invalid signature")

    # ─── Step 3: dispatch (validation + filtering) ───
    # dispatch_webhook handles event-type routing, action filtering,
    # draft PR filtering, and returns a status dict describing what
    # should happen next.
    event_type = request.headers.get("X-GitHub-Event")
    result = await dispatch_webhook(event_type, raw_body)

    # ─── Step 4: graph invocation (only if dispatch accepted) ───
    if result.get("accepted"):
        # Re-parse the body to get the typed Pydantic object. There's a
        # mild inefficiency here — dispatch_webhook already parsed once
        # internally — but the duplicate JSON parse is microseconds and
        # keeps the dispatch return shape simple (a status dict, not a
        # tuple of (status, object)). If V6 starts caring about per-
        # webhook latency this is a candidate refactor.
        try:
            payload = PullRequestWebhook(**json.loads(raw_body))
        except (json.JSONDecodeError, ValidationError) as exc:
            # If we got here, dispatch already parsed successfully but
            # WE just failed — that means a bug in our schema, not
            # in GitHub's payload. Log loudly and return 500.
            logger.exception("Re-parse failed after dispatch accepted")
            raise HTTPException(status_code=500, detail="internal parse error") from exc

        # Build initial state from webhook identity fields.
        state = initial_state(
            pr_number=payload.number,
            repo_full_name=payload.repository.full_name,
            installation_id=payload.installation.id,
        )

        # thread_id is the checkpointer's primary key. Same PR pushed
        # twice = same thread_id = same persisted graph thread.
        thread_id = thread_id_for(state)
        config = {"configurable": {"thread_id": thread_id}}

        logger.info(
            "Invoking graph for %s#%d (thread_id=%s)",
            payload.repository.full_name,
            payload.number,
            thread_id,
        )

        # ⚠️  V6 NOTE: in V1 we await the graph synchronously, which
        # works because the V1 flow takes only a few seconds. V6 adds
        # LLM calls, RAG retrieval, and a judge loop — easily 30+ seconds.
        # That exceeds GitHub's default 10-second webhook timeout, which
        # will mark the delivery as failed and trigger retries (and
        # potentially duplicate processing). The V6 fix is to schedule
        # graph invocation as a BackgroundTask (FastAPI built-in) or a
        # job queue (RQ/Celery), and return 200 to GitHub immediately.
        # We're not doing it now because (a) V1 is fast enough, and
        # (b) backgrounding without idempotency creates a worse class
        # of bug than the timeout it prevents.
        final_state = await request.app.state.graph.ainvoke(state, config=config)

        # Merge graph outputs into the response so the GitHub Recent
        # Deliveries UI shows useful info when you click through.
        result["thread_id"] = thread_id
        result["posted_comment_url"] = final_state.get("posted_comment_url")

    return result


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — exercise the full pipeline in-process
# ─────────────────────────────────────────────────────────────────────────
def _smoke_test() -> None:
    """
    Test the full FastAPI app end-to-end using TestClient.

    TestClient:
      - Runs lifespan startup before the first request (with-block enter)
      - Sends requests synchronously, internally driving the async handlers
      - Runs lifespan shutdown when the with-block exits

    This test exercises:
      1. GET /            → 200 with app metadata
      2. GET /health      → 200 with status ok
      3. POST /webhook with bad signature → 401
      4. POST /webhook with ping event    → 200 with pong
      5. POST /webhook with pull_request  → 200, full graph runs, GitHub gets a comment

    HEADS UP: test #5 posts another LGTM comment to your DocTor PR.
    Same V1 caveat as before — V6 fixes idempotency.
    """
    import hashlib
    import hmac

    from fastapi.testclient import TestClient

    settings = get_settings()

    if not settings.smoke_test_ready:
        print(
            "⚠  Skipping smoke test — set TEST_INSTALLATION_ID, "
            "TEST_REPO_FULL_NAME, TEST_PR_NUMBER in .env"
        )
        return

    assert settings.test_installation_id is not None
    assert settings.test_repo_full_name is not None
    assert settings.test_pr_number is not None

    secret = settings.github_webhook_secret.get_secret_value().encode("utf-8")

    print("=" * 60)
    print("FastAPI app smoke test")
    print("=" * 60)

    # TestClient's context manager triggers lifespan — this is where
    # the checkpointer opens and the graph compiles.
    with TestClient(app) as client:
        # ─── Test 1: home ───
        r = client.get("/")
        print(f"  GET /              → {r.status_code} {r.json()}")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

        # ─── Test 2: health ───
        r = client.get("/health")
        print(f"  GET /health        → {r.status_code} {r.json()}")
        assert r.status_code == 200

        # ─── Test 3: invalid signature ───
        r = client.post(
            "/webhook",
            content=b"{}",
            headers={
                "X-Hub-Signature-256": "sha256=deadbeef",
                "X-GitHub-Event": "ping",
            },
        )
        print(f"  POST /webhook bad sig  → {r.status_code} (expected 401)")
        assert r.status_code == 401

        # ─── Test 4: valid ping ───
        ping_body = b"{}"
        ping_sig = "sha256=" + hmac.new(secret, ping_body, hashlib.sha256).hexdigest()
        r = client.post(
            "/webhook",
            content=ping_body,
            headers={"X-Hub-Signature-256": ping_sig, "X-GitHub-Event": "ping"},
        )
        print(f"  POST /webhook ping     → {r.status_code} {r.json()}")
        assert r.status_code == 200
        assert r.json().get("pong") is True

        # ─── Test 5: full pull_request flow ───
        # Build a realistic payload using our smoke test config.
        pr_body_dict = {
            "action": "opened",
            "number": settings.test_pr_number,
            "pull_request": {
                "number": settings.test_pr_number,
                "title": "Lesson 12 smoke test",
                "head": {"ref": "test", "sha": "a" * 40},
                "base": {"ref": "main", "sha": "b" * 40},
                "draft": False,
            },
            "repository": {
                "full_name": settings.test_repo_full_name,
                "private": False,
            },
            "installation": {"id": settings.test_installation_id},
        }
        pr_body = json.dumps(pr_body_dict).encode("utf-8")
        pr_sig = "sha256=" + hmac.new(secret, pr_body, hashlib.sha256).hexdigest()

        print("\n  POST /webhook pull_request — full pipeline, will POST to GitHub...")
        r = client.post(
            "/webhook",
            content=pr_body,
            headers={
                "X-Hub-Signature-256": pr_sig,
                "X-GitHub-Event": "pull_request",
            },
        )
        print(f"  Status:   {r.status_code}")
        print(f"  Response: {r.json()}")

        assert r.status_code == 200
        body = r.json()
        assert body.get("accepted") is True
        assert body.get("posted_comment_url") is not None

    print("\n" + "=" * 60)
    print("✓ Full pipeline working: webhook → verify → dispatch → graph → GitHub")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )
    _smoke_test()
