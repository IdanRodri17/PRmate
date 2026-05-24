"""
GitHub webhook receiver.

  POST /webhook  — HMAC-verify, dispatch, invoke graph for accepted events

The compiled graph is set on app.state.graph during the app's lifespan
(see main.py); this handler reads it via request.app.state.graph.
That indirection is what lets us compile the graph once at startup
and reuse it across thousands of webhooks without per-request setup.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.github.webhooks import dispatch_webhook, verify_signature
from app.graph.state import initial_state, thread_id_for
from app.schemas.github import PullRequestWebhook

logger = logging.getLogger(__name__)

router = APIRouter(tags=["github"])


@router.post("/webhook")
async def webhook(request: Request) -> dict:
    """
    GitHub webhook receiver.

    Flow:
      1. Read raw body (bytes) — required for HMAC verification before
         any JSON parsing destroys the original byte sequence
      2. Verify X-Hub-Signature-256 against the App's webhook secret
      3. Dispatch via app/github/webhooks.py (event filter, action
         filter, draft PR filter)
      4. For accepted PR events, parse → initial_state → invoke graph
      5. Return result dict (becomes the response body GitHub records
         in its Recent Deliveries UI for debugging)
    """
    # ─── 1. Raw bytes BEFORE any JSON parsing ───
    raw_body = await request.body()

    # ─── 2. HMAC verification — the security boundary ───
    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(raw_body, signature):
        logger.warning("Webhook rejected: invalid HMAC signature")
        raise HTTPException(status_code=401, detail="invalid signature")

    # ─── 3. Dispatch (validation + filtering) ───
    event_type = request.headers.get("X-GitHub-Event")
    result = await dispatch_webhook(event_type, raw_body)

    # ─── 4. Graph invocation (only if dispatch accepted) ───
    if result.get("accepted"):
        try:
            payload = PullRequestWebhook(**json.loads(raw_body))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.exception("Re-parse failed after dispatch accepted")
            raise HTTPException(status_code=500, detail="internal parse error") from exc

        state = initial_state(
            pr_number=payload.number,
            repo_full_name=payload.repository.full_name,
            installation_id=payload.installation.id,
        )
        thread_id = thread_id_for(state)
        config = {"configurable": {"thread_id": thread_id}}

        logger.info(
            "Invoking graph for %s#%d (thread_id=%s)",
            payload.repository.full_name,
            payload.number,
            thread_id,
        )

        # ⚠️ V6 NOTE: synchronous graph invocation works for V1's fast
        # two-API-call flow but will exceed GitHub's 10-second webhook
        # timeout when V6 adds LLM + RAG + judge. V6 fix: schedule
        # graph.ainvoke as a BackgroundTask and return 200 immediately.
        final_state = await request.app.state.graph.ainvoke(state, config=config)

        result["thread_id"] = thread_id
        result["posted_comment_url"] = final_state.get("posted_comment_url")

    return result


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — exercise the full webhook pipeline
# ─────────────────────────────────────────────────────────────────────────
def _smoke_test() -> None:
    """
    Run with: python -m app.routers.webhooks

    Four tests:
      1. Bad signature → 401
      2. Valid ping → 200 + pong
      3. Full pull_request flow → 200 + posted_comment_url
         (this WILL post another LGTM comment to your test PR;
          V6 will add idempotency)
    """
    import hashlib
    import hmac

    from fastapi.testclient import TestClient

    from app.config import get_settings
    from main import app

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
    print("Webhook router smoke test")
    print("=" * 60)

    with TestClient(app) as client:
        # ─── Test 1: bad signature ───
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

        # ─── Test 2: valid ping ───
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

        # ─── Test 3: full pull_request flow ───
        pr_body_dict = {
            "action": "opened",
            "number": settings.test_pr_number,
            "pull_request": {
                "number": settings.test_pr_number,
                "title": "Lesson 12.5 router refactor smoke test",
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
        assert r.json().get("accepted") is True
        assert r.json().get("posted_comment_url") is not None

    print("=" * 60)
    print("✓ Webhook routes working end-to-end")
    print("=" * 60)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )
    _smoke_test()
