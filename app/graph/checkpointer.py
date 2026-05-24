"""
SQLite checkpointer for LangGraph state persistence.

A checkpointer saves State at every node boundary, keyed by thread_id.
This is what makes LangGraph's HITL pause-and-resume work in V6 — and
even without HITL in V1, it's how we survive crashes and webhook retries.

V1 uses SQLite for local dev simplicity. V6 swaps to PostgresSaver for
Render production (Render's free tier has no persistent disk, so a
SQLite file would be wiped on every redeploy).

Lifecycle:
    AsyncSqliteSaver is an async context manager — it opens an aiosqlite
    connection, yields a saver bound to it, and closes the connection
    on exit.

    Two usage patterns:

    1) Per-request (what the smoke test does):
           async with get_checkpointer() as saver:
               graph = builder.compile(checkpointer=saver)
               await graph.ainvoke(state, config)

    2) App-wide via FastAPI lifespan (what Lesson 12 will do):
           Enter the context once at startup, keep the saver alive for
           the lifetime of the process, exit on shutdown. The compiled
           graph is reused across every webhook.

Reference:
    https://langchain-ai.github.io/langgraph/concepts/persistence/
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Async context manager: open a SQLite checkpointer
# ─────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """
    Yield a ready-to-use AsyncSqliteSaver bound to settings.sqlite_db_path.

    Wraps AsyncSqliteSaver.from_conn_string with three additions:
      1. Ensures the parent directory of the DB file exists. SQLite
         won't create missing directories on its own — it'd just fail
         to connect with an unhelpful error.
      2. Calls saver.setup() to create the checkpoint and writes tables
         if they don't already exist. Idempotent — calling on an
         already-initialized DB is a no-op.
      3. Logs the path on open and close so when you're debugging
         "where is my state actually stored?" the answer is two
         scrolls up in the log, not twenty minutes of grepping.

    Yields:
        An AsyncSqliteSaver instance, ready to be passed as the
        `checkpointer` argument to a compiled graph.

    Usage:
        async with get_checkpointer() as saver:
            graph = builder.compile(checkpointer=saver)
            await graph.ainvoke(state, config)
    """
    settings = get_settings()
    db_path = settings.sqlite_db_path

    # SQLite won't auto-create the data/ directory if it's missing. We
    # do it ourselves with parents=True (create any missing intermediate
    # dirs) and exist_ok=True (don't crash if data/ is already there).
    # This makes the very first run on a fresh clone succeed without a
    # manual `mkdir data/` step.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("opening SQLite checkpointer at %s", db_path)

    # from_conn_string is the AsyncSqliteSaver's classmethod that
    # returns an async context manager. It opens the underlying
    # aiosqlite connection, hands us back a saver, and closes the
    # connection automatically when the async-with block exits.
    #
    # We pass str(db_path) because the underlying aiosqlite.connect()
    # signature expects a string path, not a Path object.
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        # setup() creates the checkpoints/writes/blobs tables if they
        # don't already exist. It's idempotent — safe to call on every
        # open. Calling it explicitly here means we don't depend on
        # whether from_conn_string runs setup internally (the answer
        # differs across langgraph versions and we don't want to bet).
        await saver.setup()

        yield saver

    logger.info("closed SQLite checkpointer at %s", db_path)


# ─────────────────────────────────────────────────────────────────────────
# Smoke test — verify the checkpointer opens, creates the file, closes
# ─────────────────────────────────────────────────────────────────────────
async def _smoke_test() -> None:
    """
    Verifies:
      1. data/ directory is created on first run if absent
      2. The SQLite file is created at settings.sqlite_db_path
      3. setup() runs without raising
      4. The file is on disk and non-empty after the saver closes
         (a non-empty file means setup() actually wrote the schema)

    Does NOT verify that checkpoints can be saved/loaded — that test
    happens naturally in Lesson 12 when the full graph runs end-to-end.
    Here we just prove the connection layer works.
    """
    settings = get_settings()
    db_path = settings.sqlite_db_path

    print("=" * 60)
    print("Checkpointer smoke test")
    print("=" * 60)
    print(f"  DB path (relative):  {db_path}")
    print(f"  DB path (absolute):  {db_path.resolve()}")

    # Track whether the file existed BEFORE the test — affects what we
    # expect to be true after. On a fresh clone, this is False (we
    # expect the file to be created). On rerun, this is True (we expect
    # the file to still exist and not be corrupted).
    existed_before = db_path.exists()
    print(f"  Existed before test: {existed_before}")

    # Open and close the checkpointer; this should create the file
    # if absent, and run setup() to create tables if absent.
    async with get_checkpointer() as saver:
        print(f"  Saver type:          {type(saver).__name__}")
        print("  ✓ Checkpointer opened successfully")

    # ─── Post-conditions ───
    assert db_path.exists(), f"DB file was not created at {db_path}"
    size = db_path.stat().st_size
    assert (
        size > 0
    ), f"DB file is empty at {db_path} (setup() should have written schema)"

    print(f"  ✓ DB file exists at {db_path}")
    print(f"  ✓ DB file size: {size} bytes")
    print("=" * 60)
    print("✓ Checkpointer module working — ready for Lesson 11")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s"
    )
    asyncio.run(_smoke_test())
