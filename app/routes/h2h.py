"""
Darts head-to-head route.

GET /api/v1/darts/h2h?player1=X&player2=Y&limit=10

Returns historical head-to-head statistics between two darts players.
Darts matches are scored in legs (e.g. 7-5 in a best-of-13 match).

The current data/raw/odds/pdc/ directory contains HTML-format odds files
but no parseable match-result CSVs.  When no structured result data is
found the endpoint returns an empty record with
data_source="no_historical_data".

Response schema
---------------
{
  "player1": str,
  "player2": str,
  "total_matches": int,
  "p1_wins": int,
  "p2_wins": int,
  "avg_legs_p1": float | null,
  "avg_legs_p2": float | null,
  "recent_matches": [
    {"date": str, "player1": str, "player2": str,
     "score": str, "winner": str}
  ],
  "data_source": str
}
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["h2h"])

# ---------------------------------------------------------------------------
# Data directory — icehockey/api/routes/h2h.py →
# XG3Darts/ is the package root.
# Layout: XG3Darts/app/routes/h2h.py → XG3Darts/
# ---------------------------------------------------------------------------
_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # XG3Darts/
_DATA_DIR = _PKG_ROOT / "data"


# ---------------------------------------------------------------------------
# Data loader — scan for any CSV with player match-result columns.
# Accepted column sets (case-insensitive):
#   {date, player1, player2, legs1, legs2}
#   {date, p1, p2, legs_p1, legs_p2}
#   {date, winner, loser, winner_legs, loser_legs}
# ---------------------------------------------------------------------------


def _to_int(val: Any) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _to_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _load_match_history() -> list[dict[str, Any]]:
    """Scan for darts match result CSVs.  Returns [] when none found."""
    import csv

    candidate_sets = [
        {"date", "player1", "player2", "legs1", "legs2"},
        {"date", "p1", "p2", "legs_p1", "legs_p2"},
        {"date", "winner", "loser", "winner_legs", "loser_legs"},
    ]
    col_map = {
        frozenset({"date", "player1", "player2", "legs1", "legs2"}): (
            "player1", "player2", "legs1", "legs2"
        ),
        frozenset({"date", "p1", "p2", "legs_p1", "legs_p2"}): (
            "p1", "p2", "legs_p1", "legs_p2"
        ),
        frozenset({"date", "winner", "loser", "winner_legs", "loser_legs"}): (
            "winner", "loser", "winner_legs", "loser_legs"
        ),
    }

    matches: list[dict[str, Any]] = []

    if not _DATA_DIR.exists():
        return matches

    for csv_path in sorted(_DATA_DIR.rglob("*.csv")):
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    continue
                cols = {c.strip().lower() for c in reader.fieldnames}
                mapping: Optional[tuple] = None
                for candidate in candidate_sets:
                    if candidate <= cols:
                        mapping = col_map[frozenset(candidate)]
                        break
                if mapping is None:
                    continue
                p1_col, p2_col, l1_col, l2_col = mapping
                for row in reader:
                    matches.append(
                        {
                            "date": row.get("date", ""),
                            "player1": row.get(p1_col, "").strip(),
                            "player2": row.get(p2_col, "").strip(),
                            "legs1": _to_int(row.get(l1_col)),
                            "legs2": _to_int(row.get(l2_col)),
                        }
                    )
        except Exception as exc:
            log.debug("h2h: skipping %s — %s", csv_path, exc)

    return matches


_MATCH_HISTORY: list[dict[str, Any]] = _load_match_history()
_DATA_SOURCE: str = (
    f"darts_match_history_csv ({len(_MATCH_HISTORY)} records)"
    if _MATCH_HISTORY
    else "no_historical_data"
)

log.info(
    "darts.h2h: data loaded",
    records=len(_MATCH_HISTORY),
    data_source=_DATA_SOURCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(name: str) -> str:
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/h2h",
    summary="Darts head-to-head history",
    response_class=JSONResponse,
)
async def darts_h2h(
    player1: str = Query(..., description="First player name"),
    player2: str = Query(..., description="Second player name"),
    limit: int = Query(10, ge=1, le=100, description="Maximum recent matches to return"),
) -> JSONResponse:
    """
    Return historical head-to-head leg record between two darts players.

    Returns data_source="no_historical_data" when no match CSV is available.
    """
    p1_norm = _normalise(player1)
    p2_norm = _normalise(player2)

    if not _MATCH_HISTORY:
        return JSONResponse(
            content={
                "player1": player1,
                "player2": player2,
                "total_matches": 0,
                "p1_wins": 0,
                "p2_wins": 0,
                "avg_legs_p1": None,
                "avg_legs_p2": None,
                "recent_matches": [],
                "data_source": "no_historical_data",
            }
        )

    head_to_head: list[dict[str, Any]] = []
    for m in _MATCH_HISTORY:
        a_norm = _normalise(m["player1"])
        b_norm = _normalise(m["player2"])
        if (a_norm == p1_norm and b_norm == p2_norm) or (
            a_norm == p2_norm and b_norm == p1_norm
        ):
            head_to_head.append(m)

    total = len(head_to_head)

    if total == 0:
        return JSONResponse(
            content={
                "player1": player1,
                "player2": player2,
                "total_matches": 0,
                "p1_wins": 0,
                "p2_wins": 0,
                "avg_legs_p1": None,
                "avg_legs_p2": None,
                "recent_matches": [],
                "data_source": _DATA_SOURCE,
            }
        )

    h2h_sorted = sorted(head_to_head, key=lambda x: x.get("date", ""), reverse=True)

    p1_wins = 0
    p2_wins = 0
    p1_legs_total = 0
    p2_legs_total = 0
    valid_count = 0
    recent: list[dict[str, Any]] = []

    for m in h2h_sorted:
        a_norm = _normalise(m["player1"])
        if a_norm == p1_norm:
            l1 = m["legs1"]
            l2 = m["legs2"]
        else:
            l1 = m["legs2"]
            l2 = m["legs1"]

        if l1 is not None and l2 is not None:
            valid_count += 1
            p1_legs_total += l1
            p2_legs_total += l2
            score = f"{l1}-{l2}"
            if l1 > l2:
                winner = player1
                p1_wins += 1
            elif l2 > l1:
                winner = player2
                p2_wins += 1
            else:
                winner = "draw"
        else:
            score = None
            winner = "unknown"

        recent.append(
            {
                "date": m.get("date", ""),
                "player1": player1,
                "player2": player2,
                "score": score,
                "winner": winner,
            }
        )

    recent = recent[:limit]

    return JSONResponse(
        content={
            "player1": player1,
            "player2": player2,
            "total_matches": total,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "avg_legs_p1": round(p1_legs_total / valid_count, 2) if valid_count > 0 else None,
            "avg_legs_p2": round(p2_legs_total / valid_count, 2) if valid_count > 0 else None,
            "recent_matches": recent,
            "data_source": _DATA_SOURCE,
        }
    )
