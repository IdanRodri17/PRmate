"""
LangGraph state schema for PRmate.

ReviewState is the single workspace that every graph node reads from and
writes to. It is the only communication channel between nodes — no node
calls another node directly, no node returns values to its caller. They
all just mutate State.

V1 schema is intentionally minimal. Two nodes, six fields:
  - 3 identity fields (set at graph invocation)
  - 2 computed fields (populated by fetch_pr_metadata)
  - 1 output field (populated by draft_comments)

Each version grows the schema:
  - V2 adds change_category, messages (with add_messages reducer)
  - V3 adds tool-related fields for the agent loop
  - V4 adds RAG fields (retrieved_past_reviews, reranked_context)
  - V5 adds quality_score, judge_reasoning, needs_retry
  - V6 adds posted_review_id (idempotency), human_decision

Reference:
  https://langchain-ai.github.io/langgraph/concepts/low_level/#state
"""

from __future__ import annotations

from typing import Optional, TypedDict


# ─────────────────────────────────────────────────────────────────────────
# ReviewState — the V1 workspace
# ─────────────────────────────────────────────────────────────────────────
class ReviewState(TypedDict):
    """
    The complete state of a single PR review run.

    Lifecycle of a V1 graph invocation:

      1. Webhook handler builds an initial state with only identity fields
         populated (pr_number, repo_full_name, installation_id). All other
         fields are absent — meaning they'll read as None.

      2. fetch_pr_metadata runs first. It uses the identity fields to call
         GitHub's API, then writes pr_title and pr_head_sha into state.

      3. draft_comments runs next. It produces a comment body (V1 stub:
         hardcoded "LGTM"). Writes draft_comment_body. Then posts the
         comment to GitHub via the client. Writes posted_comment_url.

      4. Graph reaches END. The final state is returned to the caller
         (the webhook handler), which logs the outcome.

    Why TypedDict and not Pydantic BaseModel:
      - TypedDict is LangGraph's idiomatic state type
      - Zero runtime validation overhead (validation happens at node
        boundaries via mypy/IDE during development, not at runtime)
      - Plays naturally with LangGraph's reducer mechanism

    Why every non-identity field is Optional:
      - Fields don't exist when the graph starts; they get populated
        as their producer nodes run
      - Optional[X] makes that "filled in later" semantics explicit
      - Forces consumer nodes to defensively handle missing values
    """

    # ─────────────────────────────────────────────────────────────────────
    # Identity fields — set at graph invocation, never modified
    # ─────────────────────────────────────────────────────────────────────
    # These three together uniquely identify a graph run. They match the
    # checkpointer's thread_id components — by convention we'll use
    # thread_id = f"{repo_full_name}#{pr_number}" so the checkpointer
    # can find paused runs from the resume webhook in V6.

    # The PR number, e.g. 1. Used in API URLs.
    pr_number: int

    # The full repository name, e.g. "IdanRodri17/DocTor".
    # Used in every GitHub API URL we construct.
    repo_full_name: str

    # The GitHub App installation ID. Used to mint installation tokens
    # for authenticated API calls back to GitHub.
    installation_id: int

    # ─────────────────────────────────────────────────────────────────────
    # Computed fields — populated during execution
    # ─────────────────────────────────────────────────────────────────────
    # These start as None (absent) and are filled in by their producer
    # nodes. The "owner" of each field is documented as a comment so
    # readers know exactly where a field gets populated.

    # Set by: fetch_pr_metadata
    # The PR title, e.g. "Add 'test' to the license section of README".
    # Used in V1 for log messages; in V2+ it'll feed the classifier.
    pr_title: Optional[str]

    # Set by: fetch_pr_metadata
    # The commit SHA the PR currently points to. Critical in V6 as the
    # idempotency key for "don't post twice for the same code state."
    # We capture it in V1 to keep the field shape stable across versions
    # — easier than retrofitting later.
    pr_head_sha: Optional[str]

    # Set by: draft_comments
    # The body of the comment we'll post on the PR. V1 is a hardcoded
    # stub ("LGTM"). V2+ replaces this with actual LLM-generated content.
    draft_comment_body: Optional[str]

    # ─────────────────────────────────────────────────────────────────────
    # Output fields — terminal artifacts of a completed run
    # ─────────────────────────────────────────────────────────────────────
    # Set by: draft_comments (after the GitHub POST succeeds)
    # The URL of the posted comment, e.g.:
    #   "https://github.com/owner/repo/pull/1#issuecomment-12345"
    # Useful for logging, debugging, and (in V6) for idempotency checks:
    # if this is already set when the node runs, don't post again.
    posted_comment_url: Optional[str]


# ─────────────────────────────────────────────────────────────────────────
# Helper: build the initial state from webhook payload data
# ─────────────────────────────────────────────────────────────────────────
def initial_state(
    *,
    pr_number: int,
    repo_full_name: str,
    installation_id: int,
) -> ReviewState:
    """
    Construct the State that seeds a new graph run.

    Only identity fields are set; everything else is absent (will read
    as None). This is the entry point from the webhook handler — in
    Lesson 12, main.py's webhook endpoint will call this with values
    extracted from the PullRequestWebhook payload.

    The keyword-only signature (the `*,`) forces callers to use named
    arguments. Three int/str fields in a row are easy to swap by accident
    if positional — keyword-only makes call sites self-documenting.

    Args:
        pr_number:        The PR number (e.g., 1).
        repo_full_name:   The "owner/repo" string (e.g., "IdanRodri17/DocTor").
        installation_id:  The GitHub App installation ID.

    Returns:
        A ReviewState dict with only identity fields populated.
        Computed and output fields are not set — they'll read as None.
    """
    # We construct as a regular dict and cast via the TypedDict.
    # Python's type system doesn't enforce TypedDict at runtime;
    # this is documentation for IDE/mypy, not validation.
    return ReviewState(
        pr_number=pr_number,
        repo_full_name=repo_full_name,
        installation_id=installation_id,
        # Explicit None for documentation — these would default to absent
        # if omitted, but listing them keeps the State's shape visible.
        pr_title=None,
        pr_head_sha=None,
        draft_comment_body=None,
        posted_comment_url=None,
    )


# ─────────────────────────────────────────────────────────────────────────
# Helper: build the thread_id string from State identity fields
# ─────────────────────────────────────────────────────────────────────────
def thread_id_for(state: ReviewState) -> str:
    """
    Compute the LangGraph thread_id for this state.

    The thread_id is the checkpointer's primary key. Critical property:
    two graph invocations with the same thread_id are treated as the
    SAME logical conversation — the second invocation resumes the first.

    For PRmate, the natural identity is (repo, pr_number). Different PRs
    on the same repo must get different thread_ids; the same PR pushed
    multiple times stays on one thread (so V6's HITL can resume it).

    Format: "{repo_full_name}#{pr_number}"
    Example: "IdanRodri17/DocTor#1"

    This is a free function (not a method) because TypedDicts don't have
    methods. Putting it next to the schema keeps the convention visible.
    """
    return f"{state['repo_full_name']}#{state['pr_number']}"


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — verify shape and helpers
# ─────────────────────────────────────────────────────────────────────────
def _smoke_test() -> None:
    """
    Validates:
      1. initial_state() produces a state with only identity fields set
      2. thread_id_for() formats correctly
      3. The schema's fields can be assigned and read with the expected
         types (verified via IDE/mypy; this just confirms no obvious bugs)
    """
    # ─── Build an initial state ───
    state = initial_state(
        pr_number=42,
        repo_full_name="IdanRodri17/DocTor",
        installation_id=87778117,
    )

    print("=" * 60)
    print("ReviewState schema smoke test")
    print("=" * 60)
    print("\n  --- Initial state (identity fields populated) ---")
    print(f"  pr_number:           {state['pr_number']}")
    print(f"  repo_full_name:      {state['repo_full_name']}")
    print(f"  installation_id:     {state['installation_id']}")
    print(f"  pr_title:            {state['pr_title']}        ← not yet computed")
    print(f"  pr_head_sha:         {state['pr_head_sha']}        ← not yet computed")
    print(
        f"  draft_comment_body:  {state['draft_comment_body']}        ← not yet computed"
    )
    print(
        f"  posted_comment_url:  {state['posted_comment_url']}        ← not yet computed"
    )

    # ─── Verify thread_id format ───
    tid = thread_id_for(state)
    print(f"\n  thread_id:           {tid}")
    assert tid == "IdanRodri17/DocTor#42", f"unexpected thread_id: {tid}"

    # ─── Simulate fetch_pr_metadata writing into state ───
    # This is what the Lesson 9 node will do — mutate state with the
    # GitHub API response. TypedDict mutation is just dict mutation.
    state["pr_title"] = "Add 'test' to the license section of README"
    state["pr_head_sha"] = "785d435a8c43e8b2156789012345abcdef123456"

    print("\n  --- After simulated fetch_pr_metadata ---")
    print(f"  pr_title:            {state['pr_title']}")
    print(f"  pr_head_sha:         {state['pr_head_sha'][:7]}...")

    # ─── Simulate draft_comments writing into state ───
    state["draft_comment_body"] = "🤖 LGTM (V1 stub)"
    state["posted_comment_url"] = (
        "https://github.com/IdanRodri17/DocTor/pull/42#issuecomment-99999"
    )

    print("\n  --- After simulated draft_comments ---")
    print(f"  draft_comment_body:  {state['draft_comment_body']}")
    print(f"  posted_comment_url:  {state['posted_comment_url']}")

    # ─── Schema sanity: identity fields unchanged throughout ───
    assert state["pr_number"] == 42
    assert state["repo_full_name"] == "IdanRodri17/DocTor"
    assert state["installation_id"] == 87778117

    print("\n" + "=" * 60)
    print("✓ ReviewState behaves as expected")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
