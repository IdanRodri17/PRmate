"""
LangGraph node functions for PRmate V1.

A node is a function that:
  1. Takes the current ReviewState as input
  2. Does some unit of work (API call, LLM call, computation)
  3. Returns a dict containing ONLY the state fields it wants to update

LangGraph merges the returned dict into State automatically. Do NOT return
the full state, do NOT mutate state in place — return only the delta.

V1 surface:
  - fetch_pr_metadata: pure read; populates pr_title and pr_head_sha
  - draft_comments:    side-effecting; posts a stub "LGTM" comment to the
                       PR and populates draft_comment_body + posted_comment_url

V2+ will grow this module with classify_change, retrieve_similar_past_reviews,
rerank_context, agent_loop, llm_as_judge, etc. Each version adds nodes;
existing nodes are extended, not rewritten.

Reference:
  https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes
"""

from __future__ import annotations

import logging

from app.github import client as gh
from app.graph.state import ReviewState

# Module-level logger — every log line from this file is prefixed with
# "app.graph.nodes", easy to filter in production output.
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# V1 stub content
# ─────────────────────────────────────────────────────────────────────────
# The hardcoded comment body for V1. Lives as a module constant so the
# smoke test and the actual node read from the same source — if you want
# to tweak the text, you tweak it in one place.
#
# The 🤖 emoji at the start is intentional: when V2 starts posting real
# reviews, the visual marker lets you immediately distinguish "V1 stub
# from a forgotten dev install" from "real V2 output".
_V1_STUB_COMMENT_BODY = (
    "🤖 **PRmate V1 stub**\n\n"
    "LGTM! This is a placeholder while V1 wires up the foundation. "
    "V2 introduces actual review logic with classification and structured comments."
)


# ─────────────────────────────────────────────────────────────────────────
# Node 1: fetch_pr_metadata
# ─────────────────────────────────────────────────────────────────────────
async def fetch_pr_metadata(state: ReviewState) -> dict:
    """
    Fetch the PR's metadata from GitHub and write it into State.

    This is a PURE node — it reads from an external API but produces no
    side effects on GitHub. Running it twice for the same PR yields the
    same result (modulo new commits being pushed, which would change the
    head SHA — that's the point: capture the SHA at fetch time).

    State reads:
      - installation_id  (for auth)
      - repo_full_name   (for the API URL)
      - pr_number        (for the API URL)

    State writes:
      - pr_title         (from GitHub's "title" field)
      - pr_head_sha      (from GitHub's "head.sha" field — our idempotency
                          anchor; in V6 we'll check this before posting)

    Returns:
        A partial state dict containing only the fields above. LangGraph
        merges this into the running State automatically.

    Raises:
        httpx.HTTPStatusError: If the PR doesn't exist, was deleted, or
                               this installation lacks permission. The
                               exception propagates to graph.invoke()'s
                               caller — V1 lets FastAPI return 500 and
                               GitHub retries the webhook delivery.
    """
    # Pull what we need from state. Reading via state["..."] (not .get())
    # is deliberate — these are identity fields, they MUST exist. If one
    # is missing, KeyError immediately is better than a silent None
    # turning into a confusing API error two function calls later.
    installation_id = state["installation_id"]
    repo = state["repo_full_name"]
    pr_number = state["pr_number"]

    # One API call. We don't fetch the diff here even though our client
    # has get_pull_request_diff — V1's State has no diff field, and the
    # V1 draft_comments stub doesn't read it. Adding the call would
    # waste rate budget and complicate this node. V2's classify_change
    # is where the diff becomes necessary; we'll extend the node then.
    pr = await gh.get_pull_request(
        installation_id=installation_id,
        repo_full_name=repo,
        pr_number=pr_number,
    )

    # Structured log line — easy to grep, contains the three identifiers
    # plus the SHA prefix that lets you correlate this run with GitHub's
    # Recent Deliveries UI when debugging.
    logger.info(
        "fetched PR metadata for %s#%d (title=%r, head=%s)",
        repo,
        pr_number,
        pr["title"],
        pr["head"]["sha"][:7],
    )

    # Return ONLY the fields we touched. LangGraph merges this into State.
    # Notice we don't return pr_number, repo_full_name, or installation_id —
    # those are identity, set at graph invocation, and we never overwrite them.
    return {
        "pr_title": pr["title"],
        "pr_head_sha": pr["head"]["sha"],
    }


# ─────────────────────────────────────────────────────────────────────────
# Node 2: draft_comments  (V1 stub)
# ─────────────────────────────────────────────────────────────────────────
async def draft_comments(state: ReviewState) -> dict:
    """
    Generate the review comment and post it to the PR.

    This is a SIDE-EFFECTING node — it POSTs to GitHub, which mutates
    state visible to the outside world (a comment appears on the PR).

    V1 behavior is deliberately trivial: hardcoded "LGTM" body, single
    POST, write the resulting URL back to state.

    ⚠️  V6 NOTE — when interrupt() lands in this part of the graph:
        Re-running this node will post the comment twice. The fix is to
        check `state["posted_comment_url"]` at the top of the function
        and skip the POST if it's already set (meaning a previous run of
        this same node, for this same SHA, already completed the post).
        We're not implementing that in V1 because there's no resume
        pathway yet to trigger the double-post. Adding idempotency
        before there's a way to test it just adds untested code paths.

    State reads:
      - installation_id, repo_full_name, pr_number (for the API call)

    State writes:
      - draft_comment_body  (the text we generated — useful for logging
                             and, in V2+, for the judge node to evaluate)
      - posted_comment_url  (the URL GitHub returned after the POST)

    Returns:
        Partial state dict. As with fetch_pr_metadata, we return only
        the two fields we touched.
    """
    # Compose the comment body. In V1 this is a constant — but we still
    # write it to state because (a) the State schema documents that this
    # field is populated by this node, and (b) V2's classifier-aware
    # version will produce real content here and we want the field
    # already wired through to the downstream judge node.
    body = _V1_STUB_COMMENT_BODY

    # Post to GitHub. This is THE side effect — once this line returns
    # successfully, the comment exists on GitHub and there's no undo.
    # If anything below this line raises, we've posted but not recorded
    # the URL in state. V6's idempotency check (see the note above)
    # handles that case by recognizing "we already posted for this SHA"
    # via state inspection on retry.
    comment = await gh.post_issue_comment(
        installation_id=state["installation_id"],
        repo_full_name=state["repo_full_name"],
        pr_number=state["pr_number"],
        body=body,
    )

    logger.info(
        "posted V1 stub comment to %s#%d → %s",
        state["repo_full_name"],
        state["pr_number"],
        comment["html_url"],
    )

    return {
        "draft_comment_body": body,
        "posted_comment_url": comment["html_url"],
    }


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — exercise both nodes end-to-end against a real PR
# ─────────────────────────────────────────────────────────────────────────
async def _smoke_test() -> None:
    """
    Run both V1 nodes in sequence, exactly as the graph will in Lesson 11.

    Reads TEST_INSTALLATION_ID, TEST_REPO_FULL_NAME, TEST_PR_NUMBER from
    .env via the Settings module. Skips gracefully if any are unset.

    What this verifies:
      - fetch_pr_metadata calls GitHub and populates pr_title + pr_head_sha
      - draft_comments posts the stub comment and populates the URL field
      - Both nodes return ONLY their delta, not the full state
      - The state merge pattern works (we manually merge here to simulate
        what LangGraph does in Lesson 11)
    """
    from app.config import get_settings
    from app.graph.state import initial_state

    settings = get_settings()

    # Bail early with a clear message if smoke test config is missing.
    # The smoke_test_ready property keeps the "are we configured?" check
    # in one place — if we ever add a fourth test variable, only Settings
    # needs to know about it.
    if not settings.smoke_test_ready:
        print(
            "⚠  Skipping smoke test — set TEST_INSTALLATION_ID, "
            "TEST_REPO_FULL_NAME, TEST_PR_NUMBER in .env"
        )
        return

    # These are guaranteed non-None (smoke_test_ready verified it), but
    # mypy/IDE doesn't know that across a property boundary. The explicit
    # asserts narrow the types AND document the invariant for readers.
    assert settings.test_installation_id is not None
    assert settings.test_repo_full_name is not None
    assert settings.test_pr_number is not None

    print("=" * 60)
    print(
        f"V1 nodes smoke test against "
        f"{settings.test_repo_full_name} PR #{settings.test_pr_number}"
    )
    print("=" * 60)

    # Build the initial state — exactly what the webhook handler will
    # construct in Lesson 12.
    state = initial_state(
        pr_number=settings.test_pr_number,
        repo_full_name=settings.test_repo_full_name,
        installation_id=settings.test_installation_id,
    )

    print("\n  --- Initial state ---")
    print(f"  pr_title:           {state['pr_title']}   ← absent")
    print(f"  pr_head_sha:        {state['pr_head_sha']}   ← absent")

    # ─── Run node 1 ───
    print("\n  Running fetch_pr_metadata...")
    delta_1 = await fetch_pr_metadata(state)

    # Verify the delta has exactly the two fields we expect.
    assert set(delta_1.keys()) == {"pr_title", "pr_head_sha"}, (
        f"fetch_pr_metadata should return only pr_title and pr_head_sha, "
        f"got {set(delta_1.keys())}"
    )
    # Merge into state — this is what LangGraph does for us automatically.
    state.update(delta_1)

    print(f"  ✓ pr_title:         {state['pr_title']!r}")
    print(f"  ✓ pr_head_sha:      {state['pr_head_sha'][:7]}...")

    # ─── Run node 2 ───
    print("\n  Running draft_comments...")
    delta_2 = await draft_comments(state)

    assert set(delta_2.keys()) == {"draft_comment_body", "posted_comment_url"}, (
        f"draft_comments should return only draft_comment_body and "
        f"posted_comment_url, got {set(delta_2.keys())}"
    )
    state.update(delta_2)

    print(f"  ✓ draft_comment_body: {state['draft_comment_body'][:50]}...")
    print(f"  ✓ posted_comment_url: {state['posted_comment_url']}")

    # ─── Final state assertions ───
    print("\n  --- Final state ---")
    assert state["pr_number"] == settings.test_pr_number
    assert state["repo_full_name"] == settings.test_repo_full_name
    assert state["installation_id"] == settings.test_installation_id
    assert state["pr_title"] is not None
    assert state["pr_head_sha"] is not None
    assert state["draft_comment_body"] is not None
    assert state["posted_comment_url"] is not None

    print("  ✓ All identity fields preserved")
    print("  ✓ All computed fields populated")

    print("\n" + "=" * 60)
    print("✓ Both V1 nodes working — go check the PR on GitHub")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )
    asyncio.run(_smoke_test())
