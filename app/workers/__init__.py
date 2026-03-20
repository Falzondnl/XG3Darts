"""
Darts autonomous background workers.

Workers run as asyncio tasks within the application lifespan and handle
all autonomous live operations: feed monitoring, auto-seeding live matches,
match-completion detection, and settlement triggering.
"""
from app.workers.live_ops_worker import DartsLiveOpsWorker, get_live_ops_worker

__all__ = ["DartsLiveOpsWorker", "get_live_ops_worker"]
