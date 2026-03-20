"""
XG3 Darts — Settlement / Grading Package.

Provides market grading for all 15 darts market types after match completion.
All grades are computed from real match result data — no hardcoded outcomes.
"""
from settlement.darts_settlement_service import (
    DartsSettlementService,
    MatchResult,
    GradeResult,
    GradeOutcome,
    MarketGrade,
    SettlementReport,
)

__all__ = [
    "DartsSettlementService",
    "MatchResult",
    "GradeResult",
    "GradeOutcome",
    "MarketGrade",
    "SettlementReport",
]
