"""
Pydantic models for GitHub webhook payloads.

PRmate only models the fields it actually uses. GitHub adds fields to
webhooks regularly; extra fields are silently ignored (extra="ignore")
so new GitHub features never break PRmate.

V1 surface:
  - PullRequestWebhook: the "pull_request" event envelope

V6 will add:
  - PullRequestReviewWebhook: the "pull_request_review" event envelope
  - HumanReviewDecision: parsed review action (approved/changes_requested/comment)

Reference:
  https://docs.github.com/en/webhooks/webhook-events-and-payloads
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────────────────
# Reusable inner objects
# ─────────────────────────────────────────────────────────────────────────
# These models represent GitHub objects that appear in multiple webhook
# events. Defined once, reused across event types.


class _IgnoreExtraModel(BaseModel):
    """
    Base class for all GitHub webhook models.

    Sets extra="ignore" so PRmate doesn't crash when GitHub adds new
    fields to webhooks. This is mandatory for third-party API payloads.

    Underscore prefix marks this as a private helper — callers should
    inherit from it, not instantiate it directly.
    """

    model_config = ConfigDict(extra="ignore")


class RepositoryPayload(_IgnoreExtraModel):
    """
    A GitHub repository as it appears in webhook payloads.

    PRmate uses full_name to build API URLs (e.g. /repos/{full_name}/pulls).
    """

    # full_name is the "owner/repo" string, e.g. "IdanRodri17/DocTor".
    # This is THE most-used field across all PRmate code — every API call
    # to GitHub starts with this string.
    full_name: str

    # We also capture private status so future versions can decide
    # whether to be more conservative on private repos (e.g. log less).
    private: bool = False


class GitRefPayload(_IgnoreExtraModel):
    """
    A reference to a specific commit on a branch — used for both
    pull_request.head and pull_request.base.

    PRmate uses 'sha' as the idempotency key in V6: don't re-post a draft
    review for the same commit twice. We capture 'ref' too because some
    log messages benefit from knowing the branch name.
    """

    # sha is the 40-character commit hash, e.g. "785d435a8c..."
    # This is the unique identifier for "the code at this exact moment".
    sha: str

    # ref is the branch name, e.g. "prmate-smoke-test" or "main".
    # Less critical than sha but useful for human-readable logs.
    ref: str


class PullRequestPayload(_IgnoreExtraModel):
    """
    The 'pull_request' object as it appears in webhook payloads.

    GitHub's PR object has ~80 fields; we model only what PRmate uses.
    """

    # The PR number, e.g. 42. This appears in URLs and is GitHub's
    # primary identifier for a PR within a repository.
    number: int

    # The PR title — useful for log messages and (in V2+) for the
    # classify_change node to better understand the change category.
    title: str

    # head = the branch being merged FROM (the "source" of the changes)
    # base = the branch being merged INTO (the "target")
    # We need both, but head.sha is the critical one for idempotency.
    head: GitRefPayload
    base: GitRefPayload

    # 'draft' tells us if this is a draft PR. PRmate skips draft PRs
    # by default in V1 — they're work-in-progress, not ready for review.
    # Default False because the field is sometimes missing on older repos.
    draft: bool = False


class InstallationRef(_IgnoreExtraModel):
    """
    A reference to a specific App installation.

    GitHub sends this in every webhook so PRmate knows WHICH installation
    to mint a token for. Critical: without this, PRmate can't authenticate
    back to GitHub to do anything.
    """

    # The installation ID — what we pass to get_installation_token().
    # GitHub uses 'id' (a Python builtin), so we don't shadow it; we
    # just accept that this attribute name is fine in context.
    id: int


# ─────────────────────────────────────────────────────────────────────────
# Top-level webhook event envelopes
# ─────────────────────────────────────────────────────────────────────────
# These models map directly to the JSON GitHub POSTs to /webhook.


class PullRequestWebhook(_IgnoreExtraModel):
    """
    The payload for the 'pull_request' webhook event.

    GitHub sends this for many actions on a PR — opened, closed, edited,
    labeled, synchronize, ready_for_review, etc. PRmate filters by
    `action` and only reacts to a subset (see webhooks.py for the filter).

    Example webhook flow:
      1. Developer opens PR #5 on IdanRodri17/DocTor
      2. GitHub POSTs to /webhook with X-GitHub-Event: pull_request
      3. Body parses to PullRequestWebhook(action="opened", number=5, ...)
      4. Our handler checks action in {"opened", "synchronize"}
      5. If yes, kicks off the LangGraph
    """

    # The specific action that triggered this event.
    # We care about: "opened" (new PR), "synchronize" (new commits pushed),
    # "reopened" (closed PR was reopened).
    # We ignore: "labeled", "edited", "closed", "assigned", etc.
    action: str

    # The PR number — also available inside pull_request, but GitHub
    # promotes it to the top level for convenience.
    number: int

    # The full PR object (nested).
    # The Field alias is just for clarity — we want the Python attribute
    # to be `pull_request`, matching GitHub's exact JSON key.
    pull_request: PullRequestPayload

    # The repository the PR belongs to.
    repository: RepositoryPayload

    # The installation that triggered this event.
    # Without this, PRmate can't authenticate to call back to GitHub.
    installation: InstallationRef


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — parse a realistic example payload to verify the schema
# ─────────────────────────────────────────────────────────────────────────
def _smoke_test() -> None:
    """
    Validates the schema against a representative pull_request webhook.

    The example payload below is a minimal but realistic sample.
    Real GitHub payloads have ~10x more fields — but thanks to
    extra="ignore", our model handles them just fine.
    """
    example_payload = {
        "action": "opened",
        "number": 1,
        "pull_request": {
            "id": 2156789012,
            "number": 1,
            "title": "Add 'test' to the license section of README",
            "body": "Trivial test PR",
            "user": {"login": "IdanRodri17", "id": 12345},
            "head": {
                "ref": "prmate-smoke-test",
                "sha": "785d435a8c43e8b2156789012345abcdef123456",
                # Extra fields GitHub sends that we don't model:
                "repo": {"name": "DocTor"},
                "user": {"login": "IdanRodri17"},
            },
            "base": {
                "ref": "main",
                "sha": "abc1234def5678901234567890abcdef12345678",
                "repo": {"name": "DocTor"},
            },
            "draft": False,
            # Extras that should be silently ignored:
            "merged": False,
            "mergeable": True,
            "labels": [],
            "comments": 0,
        },
        "repository": {
            "id": 987654321,
            "full_name": "IdanRodri17/DocTor",
            "name": "DocTor",
            "private": False,
            # Extras:
            "owner": {"login": "IdanRodri17"},
            "default_branch": "main",
        },
        "installation": {"id": 87778117},
        # Top-level extras that should be ignored:
        "sender": {"login": "IdanRodri17"},
        "organization": None,
    }

    # The actual parse — this is what webhook handlers do at runtime
    webhook = PullRequestWebhook(**example_payload)

    print("=" * 60)
    print("Schema parsed example payload successfully ✓")
    print("=" * 60)
    print(f"  Action:           {webhook.action}")
    print(f"  PR number:        {webhook.number}")
    print(f"  PR title:         {webhook.pull_request.title}")
    print(f"  Head ref:         {webhook.pull_request.head.ref}")
    print(f"  Head SHA:         {webhook.pull_request.head.sha[:7]}")
    print(f"  Base ref:         {webhook.pull_request.base.ref}")
    print(f"  Repository:       {webhook.repository.full_name}")
    print(f"  Repo is private:  {webhook.repository.private}")
    print(f"  Is draft PR:      {webhook.pull_request.draft}")
    print(f"  Installation ID:  {webhook.installation.id}")
    print("=" * 60)
    print("✓ Webhook schema working — ready for Lesson 7")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
