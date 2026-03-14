"""
Canonical error hierarchy for the XG3 Darts pricing engine.

All engine-layer and data-layer errors derive from these base classes so
that callers can catch at the appropriate granularity.
"""
from __future__ import annotations


class DartsEngineError(Exception):
    """
    Raised when the pricing engine encounters an unrecoverable computation error.

    Examples: invalid state, probability domain violation, unsupported format.
    """


class DartsDataError(Exception):
    """
    Raised when required input data is missing, insufficient, or malformed.

    Examples: player not found, regime gate failed, stats below threshold.
    """


class DartsMarketClosedError(DartsEngineError):
    """
    Raised when a market cannot be opened due to data sufficiency gates.

    Subclass of DartsEngineError so callers that catch the parent still handle it.
    """


class DartsFormatError(DartsEngineError):
    """Raised when a competition format code is unknown or misconfigured."""
