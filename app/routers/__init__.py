"""HTTP router modules for PRmate.

Each module here exposes a FastAPI APIRouter that gets mounted in
main.py via app.include_router(). Adding a new endpoint group
(e.g., V6's HITL resume handler) is a new file here + one
include_router() line in main.py — nothing else changes.
"""
