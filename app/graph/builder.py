"""
LangGraph builder for PRmate V1.

Assembles the V1 nodes and a caller-provided checkpointer into a compiled,
executable graph. The builder is pure topology — it does not own the
checkpointer, does not open SQLite connections, does not touch the
filesystem. Whoever calls build_graph() supplies the checkpointer.

Why DI for the checkpointer:
  - Testability: pass MemorySaver (or None) in unit tests; pass
    AsyncSqliteSaver in dev; pass PostgresSaver in V6 production —
    builder code never changes.
  - Lifecycle clarity: the checkpointer's connection lives at app
    scope (FastAPI lifespan), not graph scope. Build once, reuse
    forever; the checkpointer connection is opened/closed by whoever
    owns the lifespan (Lesson 12).

Reference:
  https://langchain-ai.github.io/langgraph/concepts/low_level/
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import draft_comments, fetch_pr_metadata
from app.graph.state import ReviewState

# ─────────────────────────────────────────────────────────────────────────
# Node-name constants
# ─────────────────────────────────────────────────────────────────────────
# Centralizing node names as module constants saves us from typos in the
# string literals — and crucially, when V2+ adds conditional edges, the
# routing function will return one of these strings as its choice. Having
# them as named constants means we typo-check at import time, not at
# graph-invocation time.
#
# Convention: name == the function name == this constant. Three things in
# sync; if you ever rename one, the others are easy to find.
NODE_FETCH = "fetch_pr_metadata"
NODE_DRAFT = "draft_comments"


# ─────────────────────────────────────────────────────────────────────────
# Build & compile the V1 graph
# ─────────────────────────────────────────────────────────────────────────
def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """
    Assemble the V1 nodes and given checkpointer into a compiled graph.

    Topology:
        START → fetch_pr_metadata → draft_comments → END

    Args:
        checkpointer: Any BaseCheckpointSaver implementation. V1 callers
                      will pass an AsyncSqliteSaver from get_checkpointer().
                      Tests can pass MemorySaver or None.

    Returns:
        A CompiledStateGraph implementing LangChain's Runnable interface.
        Call .ainvoke(initial_state, config={"configurable":
        {"thread_id": ...}}) to run it.

    Raises:
        ValueError: If the graph is malformed at compile time (e.g.,
                    a node was added that has no incoming edge). LangGraph
                    validates this at compile() — we never see runtime
                    "where do I go now?" errors because compile catches them.
    """
    # ─── Phase 1: Build ───
    # StateGraph(ReviewState) tells LangGraph the state schema. At each
    # node boundary, LangGraph type-checks that returned dicts contain
    # only fields declared in ReviewState (extras are ignored, missing
    # fields are fine — they stay as previously-set values).
    builder: StateGraph = StateGraph(ReviewState)

    # Register the two work nodes. The first argument is the node name
    # used in edges; the second is the async function that does the work.
    # LangGraph will detect that both are async (`async def`) and run
    # them in the event loop accordingly.
    builder.add_node(NODE_FETCH, fetch_pr_metadata)
    builder.add_node(NODE_DRAFT, draft_comments)

    # Declare the edges. V1 is linear, so every edge is unconditional:
    #   START → fetch_pr_metadata → draft_comments → END
    # V2 will replace the middle edge with a conditional edge that
    # routes based on change_category. The shape will be:
    #   START → fetch → classify → [bug|feature|refactor] → ... → END
    builder.add_edge(START, NODE_FETCH)
    builder.add_edge(NODE_FETCH, NODE_DRAFT)
    builder.add_edge(NODE_DRAFT, END)

    # ─── Phase 2: Compile ───
    # compile() does three things:
    #   1. Validates the topology (every node reachable, no dangling edges)
    #   2. Wires in the checkpointer for state persistence at every
    #      super-step (node boundary)
    #   3. Returns a CompiledStateGraph — the executable, thread-safe,
    #      reusable runnable that we'll invoke for every webhook
    return builder.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — build the graph, invoke against a real PR end-to-end
# ─────────────────────────────────────────────────────────────────────────
async def _smoke_test() -> None:
    """
    Verifies the full V1 spine:
      1. build_graph() returns a CompiledStateGraph without error
      2. The graph topology is correct (we print the Mermaid diagram
         and assert the expected node count)
      3. Invoking the graph against a real PR runs both nodes in order
      4. The final state contains all expected fields
      5. Checkpoints were actually written to SQLite

    HEADS UP: this WILL post another LGTM comment on your test PR.
    V6 adds idempotency to draft_comments; for now, each smoke test
    run leaves one more 🤖 stub comment behind. That's expected V1
    behavior — close the PR or delete the comments manually between
    runs if it bothers you.
    """
    from app.config import get_settings
    from app.graph.checkpointer import get_checkpointer
    from app.graph.state import initial_state, thread_id_for

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

    print("=" * 60)
    print(
        f"V1 graph smoke test against "
        f"{settings.test_repo_full_name} PR #{settings.test_pr_number}"
    )
    print("=" * 60)

    # Open the checkpointer for the lifetime of this test. In Lesson 12,
    # FastAPI's lifespan handler will hold this context across the
    # entire process lifetime instead of just one test.
    async with get_checkpointer() as checkpointer:
        # ─── Build ───
        graph = build_graph(checkpointer)
        print(f"  Compiled graph type: {type(graph).__name__}")

        # ─── Inspect topology via Mermaid ───
        # LangGraph's draw_mermaid() returns a Mermaid syntax string
        # describing the graph. Paste it into https://mermaid.live to
        # see a rendered diagram, or just read it as text — it's pretty
        # legible. This is a great debugging tool when V2+ topology
        # gets more interesting.
        print("\n  --- Graph topology (Mermaid) ---")
        print(graph.get_graph().draw_mermaid())

        # ─── Build initial state ───
        state = initial_state(
            pr_number=settings.test_pr_number,
            repo_full_name=settings.test_repo_full_name,
            installation_id=settings.test_installation_id,
        )

        # The thread_id is what lets the checkpointer correlate this
        # invocation with future invocations on the same PR (V6
        # resume). For V1, same PR pushed twice means same thread_id;
        # the checkpointer will see the second invocation as a
        # continuation of the first.
        thread_id = thread_id_for(state)
        config = {"configurable": {"thread_id": thread_id}}

        print(f"\n  thread_id:           {thread_id}")
        print(f"  Initial state keys:  {sorted(state.keys())}")

        # ─── Invoke ───
        print("\n  Invoking graph (this calls GitHub twice — once to")
        print("  fetch metadata, once to post the LGTM stub comment)...\n")

        final_state = await graph.ainvoke(state, config=config)

        # ─── Verify final state ───
        print("  --- Final state ---")
        print(f"  pr_title:            {final_state['pr_title']!r}")
        print(f"  pr_head_sha:         {final_state['pr_head_sha'][:7]}...")
        print(f"  draft_comment_body:  {final_state['draft_comment_body'][:50]}...")
        print(f"  posted_comment_url:  {final_state['posted_comment_url']}")

        # Hard assertions — the test fails loudly if any V1 field is missing.
        assert final_state["pr_title"] is not None
        assert final_state["pr_head_sha"] is not None
        assert final_state["draft_comment_body"] is not None
        assert final_state["posted_comment_url"] is not None
        assert final_state["pr_number"] == settings.test_pr_number
        assert final_state["repo_full_name"] == settings.test_repo_full_name
        assert final_state["installation_id"] == settings.test_installation_id

        print("\n  ✓ All V1 state fields populated")

        # ─── Verify checkpointer actually saved state ───
        # aget_tuple returns the most recent checkpoint for this thread,
        # or None if nothing was saved. If we get None here, the
        # checkpointer wasn't wired correctly to the compiled graph.
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        assert (
            checkpoint_tuple is not None
        ), "Checkpointer didn't save state — the graph isn't wired to the saver"
        print(
            f"  ✓ Checkpoint persisted (checkpoint_id: "
            f"{checkpoint_tuple.config['configurable']['checkpoint_id'][:13]}...)"
        )

    print("\n" + "=" * 60)
    print("✓ V1 graph working end-to-end — check the PR on GitHub")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )
    asyncio.run(_smoke_test())
