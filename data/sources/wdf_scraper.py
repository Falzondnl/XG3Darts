"""WDF website scraper — fetches WDF rankings and tournament data."""
from __future__ import annotations

from typing import Any, Optional

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

BASE_URL = "https://www.wdf.one"


async def fetch_rankings(
    session: aiohttp.ClientSession,
) -> Optional[list[dict[str, Any]]]:
    """
    Fetch current WDF rankings.

    Returns
    -------
    list[dict] | None

    Raises
    ------
    NotImplementedError
        WDF site scraping requires site-specific HTML parsing.
    """
    raise NotImplementedError(
        "WDF scraping requires inspecting the wdf.one site structure. "
        "Implement after auditing the current HTML layout."
    )
