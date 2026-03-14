"""PDC website scraper — fetches live fixture and result updates."""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

BASE_URL = "https://www.pdc.tv"
REQUEST_DELAY = 2.0


async def fetch_live_results(
    session: aiohttp.ClientSession,
    tournament_id: int,
) -> Optional[dict[str, Any]]:
    """
    Fetch live/recent results for a PDC tournament.

    Parameters
    ----------
    session:
        Active aiohttp session.
    tournament_id:
        PDC tournament ID.

    Returns
    -------
    dict | None
        Parsed result data or None on failure.

    Raises
    ------
    NotImplementedError
        Live PDC scraping requires site-specific HTML parsing.
    """
    raise NotImplementedError(
        "PDC live scraping requires reverse-engineering the current PDC.tv "
        "HTML structure. Implement after inspecting the live site."
    )
