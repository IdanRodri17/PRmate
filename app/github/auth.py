"""
GitHub App authentication: JWT generation and installation token exchange.

GitHub Apps use a two-layer auth model:
  1. Sign a short-lived JWT with the App's RSA private key
     → proves "I am PRmate the application"
  2. Exchange that JWT for an installation access token
     → proves "I am PRmate, acting on behalf of installation X"

API calls (post comments, fetch diffs) use the installation token, NOT the JWT.

Reference:
  https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import httpx
import jwt as pyjwt

from app.config import get_settings

if TYPE_CHECKING:
    # Only imported for type hints — avoids circular import risk
    from app.config import Settings


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
# GitHub's maximum JWT lifetime is 10 minutes. We use 9 to give ourselves
# a safety margin against clock skew between our server and GitHub's.
_JWT_LIFETIME_SECONDS = 9 * 60

# Setting iat (issued-at) to slightly in the past protects us from
# clock skew the OTHER direction — if our clock is ahead of GitHub's,
# a "now" iat could be rejected as being in the future.
_JWT_IAT_SKEW_SECONDS = 60

# Installation tokens are valid for 1 hour. We refresh when they have
# less than this many seconds left — to ensure the token survives any
# API call we're about to make with it.
_INSTALLATION_TOKEN_REFRESH_BUFFER_SECONDS = 60

# GitHub API base — kept as a module constant for easy override in tests
_GITHUB_API_BASE = "https://api.github.com"


# ─────────────────────────────────────────────────────────────────────────
# In-memory cache for installation tokens
# ─────────────────────────────────────────────────────────────────────────
# Key:   installation_id (int)
# Value: tuple of (token string, UTC datetime when it expires)
#
# Why a module-level dict and not a class?
#   - Simplicity: V1 runs as a single process with one worker
#   - We'll revisit this in V6 if Render multi-worker becomes relevant
#
# Why store the expiration as a datetime instead of a duration?
#   - We compare against datetime.now() to decide if refresh is needed
#   - GitHub returns expires_at as an ISO timestamp, so this matches
_installation_token_cache: dict[int, tuple[str, datetime]] = {}


# ─────────────────────────────────────────────────────────────────────────
# Step 1 — Load and decode the App's private key
# ─────────────────────────────────────────────────────────────────────────
def _load_private_key() -> bytes:
    """
    Decode the base64-encoded private key from settings.

    Returns the raw PEM bytes that PyJWT needs for RS256 signing.

    The key lives base64-encoded in env vars because:
      - .env files can't reliably handle multi-line PEM content
      - Render env vars definitely can't handle multi-line strings
    """
    settings = get_settings()
    # .get_secret_value() unwraps the SecretStr safely (only at the moment
    # of use; the value is never stored as a plain string anywhere else)
    b64_key = settings.github_private_key_b64.get_secret_value()
    # base64.b64decode returns bytes — which is exactly what pyjwt wants
    # for the `key` parameter when signing RS256 JWTs
    return base64.b64decode(b64_key)


# ─────────────────────────────────────────────────────────────────────────
# Step 2 — Generate a JWT proving "I am PRmate"
# ─────────────────────────────────────────────────────────────────────────
def generate_app_jwt() -> str:
    """
    Create a short-lived JWT that authenticates PRmate as a GitHub App.

    This JWT identifies PRmate to GitHub. It does NOT identify any specific
    user or installation — for that, we exchange it for an installation
    access token (see get_installation_token below).

    GitHub's requirements:
      - alg = RS256 (RSA with SHA-256)
      - iss = App ID (our numeric identifier)
      - iat = unix timestamp, no more than 60s in the future
      - exp = unix timestamp, no more than 10 minutes after iat

    Returns:
        A JWT string, valid for ~9 minutes.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)

    payload = {
        # "issued at" — slightly in the past to handle clock skew
        "iat": int((now - timedelta(seconds=_JWT_IAT_SKEW_SECONDS)).timestamp()),
        # "expires at" — 9 minutes from now (under GitHub's 10-min cap)
        "exp": int((now + timedelta(seconds=_JWT_LIFETIME_SECONDS)).timestamp()),
        # "issuer" — our GitHub App ID, identifies WHO created this token
        "iss": settings.github_app_id,
    }

    # pyjwt.encode() handles the heavy crypto:
    #   1. Base64-encodes the header and payload
    #   2. Signs them with our private key using RS256
    #   3. Concatenates header.payload.signature with dots
    token = pyjwt.encode(
        payload=payload,
        key=_load_private_key(),
        algorithm="RS256",
    )

    # pyjwt 2.x returns str directly; earlier versions returned bytes.
    # We pinned 2.10.1 so this is guaranteed to be str, but the
    # explicit isinstance check documents the assumption for readers.
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    return token


# ─────────────────────────────────────────────────────────────────────────
# Step 3 — Exchange the JWT for an installation access token
# ─────────────────────────────────────────────────────────────────────────
async def get_installation_token(installation_id: int) -> str:
    """
    Return a valid installation access token for the given installation.

    Caches the token for its lifetime to avoid hitting GitHub on every API
    call. Refreshes proactively when the cached token has less than
    _INSTALLATION_TOKEN_REFRESH_BUFFER_SECONDS remaining.

    Args:
        installation_id: The numeric ID of a specific PRmate installation.
                         GitHub sends this in every webhook payload.

    Returns:
        A token string (prefixed with "ghs_") usable as a Bearer token
        for repository-level API calls on behalf of this installation.

    Raises:
        httpx.HTTPStatusError: If GitHub rejects the JWT or the
                               installation_id is invalid.
    """
    # ─── Cache check ───
    cached = _installation_token_cache.get(installation_id)
    if cached is not None:
        token, expires_at = cached
        # If the token has more than our buffer left, use it as-is
        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
        if remaining > _INSTALLATION_TOKEN_REFRESH_BUFFER_SECONDS:
            return token

    # ─── Mint a new token ───
    # Step A: sign a JWT to authenticate ourselves as the App
    app_jwt = generate_app_jwt()

    # Step B: call GitHub's installation token endpoint
    url = f"{_GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {app_jwt}",
        # The Accept header pins us to a specific API version — protects
        # us from breaking changes if GitHub updates the response format
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # We use httpx.AsyncClient because FastAPI's webhook handler is async.
    # Using a sync `requests` call inside an async handler would block
    # the event loop — bad practice that hurts throughput.
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers)
        # Raise an exception for 4xx/5xx — the caller (or webhook handler)
        # will catch and log. We don't swallow errors here.
        response.raise_for_status()
        data = response.json()

    # GitHub returns expires_at as an ISO 8601 timestamp:
    #   "2026-05-21T18:30:00Z"
    # We parse it to a datetime for cache comparison.
    # The replace() handles the "Z" suffix that fromisoformat doesn't
    # accept in Python < 3.11; on 3.11+ this is a no-op.
    token = data["token"]
    expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))

    # ─── Cache the result ───
    _installation_token_cache[installation_id] = (token, expires_at)

    return token


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — verify JWT generation works end-to-end with GitHub
# ─────────────────────────────────────────────────────────────────────────
async def _smoke_test() -> None:
    """
    Verifies that:
      1. The private key loads from .env and decodes from base64
      2. A signed JWT is accepted by GitHub's /app endpoint
      3. (Optional) installation token exchange works if an
         installation_id is provided

    GitHub's /app endpoint returns metadata about the App itself and
    requires only the JWT (no installation token). It's the canonical
    "did my JWT work?" check.
    """
    print("=" * 60)
    print("Testing GitHub App JWT authentication...")
    print("=" * 60)

    # Generate the JWT
    token = generate_app_jwt()
    print(f"  JWT generated:  {token[:40]}... ({len(token)} chars)")

    # Call /app to verify GitHub accepts it
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{_GITHUB_API_BASE}/app", headers=headers)

    if response.status_code == 200:
        app_data = response.json()
        print(f"  GitHub accepts JWT ✓")
        print(f"  App name:       {app_data['name']}")
        print(f"  App ID:         {app_data['id']}")
        print(f"  Owner:          {app_data['owner']['login']}")
    else:
        print(f"  ✗ GitHub rejected JWT: {response.status_code}")
        print(f"  Response: {response.text}")
        return

    print("=" * 60)
    print("✓ Authentication module working correctly")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    # asyncio.run is the modern way to run async code from sync entrypoints.
    # Equivalent to creating an event loop, running the coroutine, and closing.
    asyncio.run(_smoke_test())
