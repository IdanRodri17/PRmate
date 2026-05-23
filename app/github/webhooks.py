"""
GitHub webhook receiver: signature verification + event dispatch.

This module is the security boundary between the public internet and
PRmate's internal logic. Every incoming webhook MUST pass HMAC signature
verification before any further processing.

V1 surface:
  - verify_signature: HMAC-SHA256 check against the webhook secret
  - dispatch_webhook: routes by X-GitHub-Event header

The actual graph invocation is wired in Lesson 11 (graph/builder.py + main.py).
For now, valid pull_request webhooks just log a "would have started graph"
message — proves the webhook plumbing works without graph complexity.

Reference:
  https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging

from pydantic import ValidationError

from app.config import get_settings
from app.schemas.github import PullRequestWebhook

# Module-level logger — every log line from this file is prefixed with
# "app.github.webhooks", making it easy to filter logs in production.
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
# The HTTP header GitHub uses for the signature. Always sha256-prefixed.
_SIGNATURE_HEADER = "X-Hub-Signature-256"

# The HTTP header GitHub uses to identify the event type.
_EVENT_HEADER = "X-GitHub-Event"

# Actions on the pull_request event that PRmate reacts to.
# GitHub sends pull_request webhooks for many other actions (labeled,
# assigned, edited, closed, etc.) — we ignore those.
_HANDLED_PR_ACTIONS = frozenset({"opened", "synchronize", "reopened"})


# ─────────────────────────────────────────────────────────────────────────
# Signature verification — the security boundary
# ─────────────────────────────────────────────────────────────────────────
def verify_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """
    Verify the X-Hub-Signature-256 header against the raw request body.

    GitHub computes HMAC-SHA256 over the raw body using the shared
    webhook secret, hex-encodes it, prefixes with "sha256=", and sends
    that as the X-Hub-Signature-256 header. We recompute and compare.

    Args:
        raw_body:          The exact bytes GitHub sent. NOT a re-serialized
                           JSON — that would produce a different hash.
        signature_header:  The full header value, e.g. "sha256=abc123...".
                           None if GitHub didn't send the header (always
                           a security failure).

    Returns:
        True if the signature is valid AND the secret was checked.
        False otherwise — caller MUST reject the request.

    Security:
        Uses hmac.compare_digest() for timing-safe comparison. Never use
        == for comparing hashes; it's vulnerable to timing attacks where
        an attacker can deduce the correct hash byte-by-byte by measuring
        response times.
    """
    # Missing header = definitely not from GitHub.
    if signature_header is None:
        logger.warning("Webhook rejected: missing X-Hub-Signature-256 header")
        return False

    # The header should look like "sha256=<hex>". If the prefix is wrong,
    # someone is probing the endpoint or sending malformed requests.
    if not signature_header.startswith("sha256="):
        logger.warning("Webhook rejected: signature header missing sha256= prefix")
        return False

    # Load the secret on demand. We do this here (not at module import)
    # so tests can override the secret per-test if needed.
    settings = get_settings()
    secret = settings.github_webhook_secret.get_secret_value().encode("utf-8")

    # Recompute the HMAC over the raw body using SHA-256.
    # The result is a hashlib digest object; we hex-encode and prefix.
    computed_hmac = hmac.new(
        key=secret,
        msg=raw_body,
        digestmod=hashlib.sha256,
    )
    expected = "sha256=" + computed_hmac.hexdigest()

    # Constant-time comparison — see docstring for why == is unsafe here.
    # compare_digest returns True only if both strings are equal AND were
    # checked in constant time regardless of where they differ.
    return hmac.compare_digest(expected, signature_header)


# ─────────────────────────────────────────────────────────────────────────
# Event dispatch — routing by X-GitHub-Event header
# ─────────────────────────────────────────────────────────────────────────
async def dispatch_webhook(
    event_type: str | None,
    raw_body: bytes,
) -> dict:
    """
    Route a verified webhook to the right handler.

    Caller is responsible for signature verification BEFORE invoking this.
    Once we're here, we trust the payload came from GitHub.

    Args:
        event_type: Value of the X-GitHub-Event header. None if absent
                    (which would be very unusual for a real GitHub webhook).
        raw_body:   The raw request bytes — we'll parse JSON ourselves.

    Returns:
        A small dict describing what we did. Becomes the HTTP response
        body, useful for debugging via GitHub's "Recent Deliveries" UI
        in the App settings page.
    """
    # Defensive: if there's no event type, log and bail. Real GitHub
    # always sends this header; missing it suggests a probing request.
    if event_type is None:
        logger.warning("Webhook missing X-GitHub-Event header")
        return {"ignored": True, "reason": "no_event_type"}

    # ─── 'ping' event: GitHub's "did this work?" health check ───
    # Sent once when the App is installed, and whenever you click
    # "Redeliver" in the App's Recent Deliveries UI.
    if event_type == "ping":
        logger.info("Received ping webhook")
        return {"pong": True}

    # ─── 'pull_request' event: PRmate's primary trigger ───
    if event_type == "pull_request":
        return await _handle_pull_request(raw_body)

    # ─── Anything else: acknowledge with 200 but do nothing ───
    # GitHub retries non-2xx responses. Returning 200 for unhandled
    # events keeps GitHub from spamming us with retries.
    logger.info("Received unhandled event: %s", event_type)
    return {"ignored": True, "reason": "unhandled_event_type", "event": event_type}


# ─────────────────────────────────────────────────────────────────────────
# Individual event handlers
# ─────────────────────────────────────────────────────────────────────────
async def _handle_pull_request(raw_body: bytes) -> dict:
    """
    Process a pull_request webhook.

    V1 behavior: parse the payload, filter for actions we care about,
    log what we would have done. The actual graph invocation lands in
    Lesson 11 — this function will be the integration point.
    """
    # Parse the raw bytes as JSON. We do this manually (not via
    # request.json()) because the verifier needed the raw bytes,
    # and now we're past verification.
    try:
        payload_dict = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.exception("Webhook body was not valid JSON")
        return {"error": "invalid_json"}

    # Validate against the schema. Pydantic raises ValidationError on
    # missing required fields — this would only happen if GitHub
    # changed their payload format (rare but possible).
    try:
        webhook = PullRequestWebhook(**payload_dict)
    except ValidationError as exc:
        logger.error("Webhook payload failed validation: %s", exc)
        return {"error": "invalid_payload"}

    # Filter for actions we react to.
    if webhook.action not in _HANDLED_PR_ACTIONS:
        logger.info(
            "Ignoring pull_request action=%s on %s#%d",
            webhook.action,
            webhook.repository.full_name,
            webhook.number,
        )
        return {"ignored": True, "reason": "unhandled_action", "action": webhook.action}

    # Skip draft PRs — they're not ready for review.
    if webhook.pull_request.draft:
        logger.info(
            "Ignoring draft PR %s#%d",
            webhook.repository.full_name,
            webhook.number,
        )
        return {"ignored": True, "reason": "draft_pr"}

    # ─── V1: log what we would have done ───
    # Lesson 11 replaces this log with the actual graph invocation.
    logger.info(
        "Would start graph for %s#%d (action=%s, head_sha=%s, installation=%d)",
        webhook.repository.full_name,
        webhook.number,
        webhook.action,
        webhook.pull_request.head.sha[:7],
        webhook.installation.id,
    )

    return {
        "accepted": True,
        "repo": webhook.repository.full_name,
        "pr_number": webhook.number,
        "action": webhook.action,
        # The head SHA appears in our response for debugging via GitHub's
        # Recent Deliveries UI — handy when verifying idempotency in V6.
        "head_sha": webhook.pull_request.head.sha,
    }


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — exercises both verification AND dispatch
# ─────────────────────────────────────────────────────────────────────────
def _smoke_test() -> None:
    """
    Verifies the full webhook handling pipeline in-process:

      1. Construct a known body
      2. Compute its signature with the real secret from .env
      3. Confirm verify_signature() accepts it
      4. Confirm verify_signature() REJECTS a tampered version
      5. Confirm verify_signature() REJECTS a missing header
      6. Run dispatch_webhook through three event types
    """
    import asyncio

    settings = get_settings()
    secret = settings.github_webhook_secret.get_secret_value().encode("utf-8")

    print("=" * 60)
    print("Webhook handler smoke test")
    print("=" * 60)

    # ─── Build a realistic body matching our schema ───
    body_dict = {
        "action": "opened",
        "number": 1,
        "pull_request": {
            "number": 1,
            "title": "Test PR for webhook smoke test",
            "head": {"ref": "feature", "sha": "a" * 40},
            "base": {"ref": "main", "sha": "b" * 40},
            "draft": False,
        },
        "repository": {"full_name": "IdanRodri17/DocTor", "private": False},
        "installation": {"id": 87778117},
    }
    raw_body = json.dumps(body_dict).encode("utf-8")

    # ─── Compute the correct signature ───
    correct_sig = "sha256=" + hmac.new(secret, raw_body, hashlib.sha256).hexdigest()

    # ─── Test 1: correct signature accepted ───
    print(f"  Test 1 (valid signature):   ", end="")
    assert verify_signature(raw_body, correct_sig) is True
    print("✓ accepted")

    # ─── Test 2: tampered body rejected ───
    tampered = raw_body + b" "
    print(f"  Test 2 (tampered body):     ", end="")
    assert verify_signature(tampered, correct_sig) is False
    print("✓ rejected")

    # ─── Test 3: missing header rejected ───
    print(f"  Test 3 (no signature hdr):  ", end="")
    assert verify_signature(raw_body, None) is False
    print("✓ rejected")

    # ─── Test 4: malformed header rejected ───
    print(f"  Test 4 (bad prefix):        ", end="")
    assert verify_signature(raw_body, "md5=whatever") is False
    print("✓ rejected")

    # ─── Test 5: dispatch a ping ───
    print(f"  Test 5 (ping dispatch):     ", end="")
    result = asyncio.run(dispatch_webhook("ping", b"{}"))
    assert result == {"pong": True}, f"got {result}"
    print("✓ pong")

    # ─── Test 6: dispatch a pull_request ───
    print(f"  Test 6 (pull_request):      ", end="")
    result = asyncio.run(dispatch_webhook("pull_request", raw_body))
    assert result.get("accepted") is True, f"got {result}"
    print("✓ accepted")

    # ─── Test 7: dispatch an unhandled event ───
    print(f"  Test 7 (unhandled event):   ", end="")
    result = asyncio.run(dispatch_webhook("issues", b"{}"))
    assert result.get("ignored") is True, f"got {result}"
    print("✓ ignored")

    # ─── Test 8: dispatch a draft PR ───
    print(f"  Test 8 (draft PR):          ", end="")
    draft_body_dict = dict(body_dict)
    draft_body_dict["pull_request"] = dict(body_dict["pull_request"])
    draft_body_dict["pull_request"]["draft"] = True
    draft_body = json.dumps(draft_body_dict).encode("utf-8")
    result = asyncio.run(dispatch_webhook("pull_request", draft_body))
    assert result.get("ignored") is True and result.get("reason") == "draft_pr"
    print("✓ skipped")

    print("=" * 60)
    print("✓ All 8 webhook tests passed")
    print("=" * 60)


if __name__ == "__main__":
    # Configure basic logging so the "Would start graph for..." line
    # actually appears during the smoke test.
    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )
    _smoke_test()
