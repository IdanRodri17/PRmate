"""
Minimal GitHub REST API client for PRmate.

This module is the ONLY place in PRmate that knows about GitHub's HTTP API.
Every other module calls these functions and works with Python dicts.

V1 surface area:
  - get_pull_request: fetch PR metadata (title, head SHA, files, etc.)
  - get_pull_request_diff: fetch the unified diff text
  - post_issue_comment: post a plain comment in the PR conversation
                       (V1 uses this for the "LGTM" stub; V6 graduates to
                        /reviews for inline comments)

Reference:
  https://docs.github.com/en/rest
"""

from __future__ import annotations

import httpx

from app.github.auth import get_installation_token

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
# Base URL for the GitHub REST API. Module-level constant so tests can
# monkey-patch it to a mock server if needed.
_GITHUB_API_BASE = "https://api.github.com"

# Default timeout for every HTTP call. GitHub's API is generally fast
# (P99 well under 2 seconds), so 10 seconds is generous but bounded.
# Without this, network issues cause webhook handlers to hang forever.
_DEFAULT_TIMEOUT_SECONDS = 10.0

# API version pin — protects us from breaking changes if GitHub updates
# response formats. The Accept header asks for the stable JSON shape.
_API_VERSION = "2022-11-28"


# ─────────────────────────────────────────────────────────────────────────
# Internal helper — build standard headers for an installation
# ─────────────────────────────────────────────────────────────────────────
async def _auth_headers(
    installation_id: int, accept: str = "application/vnd.github+json"
) -> dict[str, str]:
    """
    Construct the headers for an authenticated GitHub API call.

    Centralized so every API method gets identical auth treatment —
    one place to update if GitHub changes header requirements.

    Args:
        installation_id: Which installation we're acting on behalf of.
        accept: The Accept header value. Defaults to JSON, but
                get_pull_request_diff() overrides this with the
                diff media type to receive raw unified-diff text.
    """
    token = await get_installation_token(installation_id)
    return {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": _API_VERSION,
    }


# ─────────────────────────────────────────────────────────────────────────
# Operation 1 — Fetch PR metadata
# ─────────────────────────────────────────────────────────────────────────
async def get_pull_request(
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
) -> dict:
    """
    Fetch metadata for a single Pull Request.

    Returns the parsed JSON response from GET /repos/{owner}/{repo}/pulls/{number}.
    Useful fields in the response:
      - title:        the PR title
      - body:         the PR description (Markdown)
      - state:        "open" or "closed"
      - head.sha:     commit SHA the PR points to — our idempotency key
      - base.ref:     target branch (usually "main")
      - changed_files: number of files modified
      - user.login:   author username

    Args:
        installation_id: From the webhook payload.
        repo_full_name:  "owner/repo" string, e.g. "IdanRodri17/DocTor".
        pr_number:       The PR number, e.g. 42.

    Raises:
        httpx.HTTPStatusError: If GitHub returns 4xx/5xx (e.g., PR not found
                               or installation lacks permission).
    """
    url = f"{_GITHUB_API_BASE}/repos/{repo_full_name}/pulls/{pr_number}"
    headers = await _auth_headers(installation_id)

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


# ─────────────────────────────────────────────────────────────────────────
# Operation 2 — Fetch PR diff (raw unified diff text)
# ─────────────────────────────────────────────────────────────────────────
async def get_pull_request_diff(
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
) -> str:
    """
    Fetch the unified diff for a PR as raw text.

    Same URL as get_pull_request, but uses a different Accept header to ask
    GitHub for the diff format instead of JSON metadata. This is the same
    text you'd see by appending ".diff" to a PR's web URL.

    Returns:
        The full unified diff as a single string. Example:

            diff --git a/foo.py b/foo.py
            index 1234..5678 100644
            --- a/foo.py
            +++ b/foo.py
            @@ -10,3 +10,4 @@ def hello():
                 return "world"
            +    raise NotImplementedError()

    Note:
        For huge PRs (thousands of changed lines), this string can be many
        MB. V2+ may need to chunk or filter before sending to the LLM,
        but V1 just hands it to a stub node that doesn't read it.
    """
    url = f"{_GITHUB_API_BASE}/repos/{repo_full_name}/pulls/{pr_number}"
    # Overriding the Accept header is what makes GitHub return diff text
    # instead of JSON. Same URL, different content-type negotiation.
    headers = await _auth_headers(
        installation_id,
        accept="application/vnd.github.v3.diff",
    )

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        # response.text gives us the diff as a str (decoded from UTF-8).
        # Not response.json() — the response isn't JSON in this case.
        return response.text


# ─────────────────────────────────────────────────────────────────────────
# Operation 3 — Post a comment to the PR conversation
# ─────────────────────────────────────────────────────────────────────────
async def post_issue_comment(
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
    body: str,
) -> dict:
    """
    Post a comment on a PR's conversation tab.

    Uses the *issues* endpoint, not the *pulls* endpoint, because GitHub
    models the conversation tab as an issue thread (every PR is also an
    issue under the hood, sharing the same numeric ID).

    For V1, this is how we post the "LGTM" stub. V6 replaces this with
    the proper /reviews flow for inline comments + summary body.

    Args:
        installation_id: From the webhook payload.
        repo_full_name:  "owner/repo".
        pr_number:       The PR number (which is also its issue number).
        body:            Markdown-formatted comment text.

    Returns:
        The parsed JSON of the created comment, including its id and url.
        We return this so callers can later edit or delete the comment
        (useful for V6's idempotency: don't post twice for the same SHA).
    """
    url = f"{_GITHUB_API_BASE}/repos/{repo_full_name}/issues/{pr_number}/comments"
    headers = await _auth_headers(installation_id)
    # The request body is a JSON object with a single "body" key
    # containing the Markdown text.
    payload = {"body": body}

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — exercises all three operations against a real repo
# ─────────────────────────────────────────────────────────────────────────
async def _smoke_test() -> None:
    """
    To run this test, you must edit the constants below with values
    from a real PR on a repo where PRmate is installed.

    Steps to set up:
      1. Make sure PRmate is installed on at least one of your repos
         (your DocTor fork is a good choice).
      2. Open any PR on that repo (even a trivial one, e.g. add a comment
         to the README).
      3. Find the installation_id by visiting:
            https://github.com/settings/installations
         Click "Configure" next to PRmate-Idan; the URL ends with
         /installations/<this_number>
      4. Fill in INSTALLATION_ID, REPO, PR_NUMBER below.
    """
    # ───── FILL THESE IN ─────
    INSTALLATION_ID = 0
    REPO = "IdanRodri17/DocTor"
    PR_NUMBER = 1
    # ─────────────────────────

    if INSTALLATION_ID == 0:
        print(
            "⚠  Skipping smoke test — fill in INSTALLATION_ID, REPO, PR_NUMBER above."
        )
        print("    See the docstring of _smoke_test() for instructions.")
        return

    print("=" * 60)
    print(f"Testing GitHub client against {REPO} PR #{PR_NUMBER}")
    print("=" * 60)

    # Operation 1: fetch metadata
    pr = await get_pull_request(INSTALLATION_ID, REPO, PR_NUMBER)
    print(f"  PR title:       {pr['title']}")
    print(f"  Author:         {pr['user']['login']}")
    print(f"  Head SHA:       {pr['head']['sha'][:7]}")
    print(f"  Changed files:  {pr['changed_files']}")

    # Operation 2: fetch diff
    diff = await get_pull_request_diff(INSTALLATION_ID, REPO, PR_NUMBER)
    diff_preview = diff.split("\n")[0] if diff else "(empty)"
    print(f"  Diff size:      {len(diff)} chars")
    print(f"  Diff starts:    {diff_preview}")

    # Operation 3: post a comment
    comment = await post_issue_comment(
        INSTALLATION_ID,
        REPO,
        PR_NUMBER,
        body="🤖 Hello from PRmate! This is a smoke test from Lesson 5.",
    )
    print(f"  Comment posted: {comment['html_url']}")

    print("=" * 60)
    print("✓ All three operations working")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_smoke_test())
