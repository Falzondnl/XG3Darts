"""
Darts player form / momentum route.

GET /api/v1/darts/form/{player_id}?limit=10

Returns recent form statistics for a darts player.
Darts matches are scored in legs won.

The current data/ directory contains no parseable match-result CSVs.
When no structured result data is found the endpoint returns an empty
form record with data_source="no_historical_data".

Response schema
---------------
{
  "player_id": str,
  "last_n_results": [
    {"opponent": str, "result": "W"|"L", "score": str, "date": str}
  ],
  "win_rate_last10": float | null,
  "win_rate_last5": float | null,
  "avg_checkout_pct": float | null,
  "form_string": str,   # last 5 results e.g. "WWLWW"
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

router = APIRouter(tags=["form"])

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # XG3Darts/
_DATA_DIR = _PKG_ROOT / "data"


# ---------------------------------------------------------------------------
# Data loader
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
                    checkout_pct = _to_float(row.get("checkout_pct") or row.get("double_pct"))
                    matches.append(
                        {
                            "date": row.get("date", ""),
                            "player1": row.get(p1_col, "").strip(),
                            "player2": row.get(p2_col, "").strip(),
                            "legs1": _to_int(row.get(l1_col)),
                            "legs2": _to_int(row.get(l2_col)),
                            "checkout_pct_p1": checkout_pct,
                        }
                    )
        except Exception as exc:
            log.debug("form: skipping %s — %s", csv_path, exc)

    return matches


_MATCH_HISTORY: list[dict[str, Any]] = _load_match_history()
_DATA_SOURCE: str = (
    f"darts_match_history_csv ({len(_MATCH_HISTORY)} records)"
    if _MATCH_HISTORY
    else "no_historical_data"
)

log.info(
    "darts.form: data loaded",
    records=len(_MATCH_HISTORY),
    data_source=_DATA_SOURCE,
)


def _normalise(name: str) -> str:
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/form/{player_id}",
    summary="Darts player recent form",
    response_class=JSONResponse,
)
async def darts_form(
    player_id: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
) -> JSONResponse:
    """
    Return recent form statistics for a darts player.

    Returns data_source="no_historical_data" when no match CSV is available.
    """
    player_norm = _normalise(player_id)

    if not _MATCH_HISTORY:
        return JSONResponse(
            content={
                "player_id": player_id,
                "last_n_results": [],
                "win_rate_last10": None,
                "win_rate_last5": None,
                "avg_checkout_pct": None,
                "form_string": "",
                "data_source": "no_historical_data",
            }
        )

    player_matches: list[dict[str, Any]] = []
    for m in _MATCH_HISTORY:
        p1_norm = _normalise(m["player1"])
        p2_norm = _normalise(m["player2"])
        if p1_norm == player_norm or p2_norm == player_norm:
            player_matches.append(m)

    if not player_matches:
        return JSONResponse(
            content={
                "player_id": player_id,
                "last_n_results": [],
                "win_rate_last10": None,
                "win_rate_last5": None,
                "avg_checkout_pct": None,
                "form_string": "",
                "data_source": _DATA_SOURCE,
            }
        )

    player_matches_sorted = sorted(
        player_matches, key=lambda x: x.get("date", ""), reverse=True
    )
    last_n = player_matches_sorted[:limit]

    results: list[dict[str, Any]] = []
    wins = 0
    checkout_sum = 0.0
    checkout_count = 0
    form_chars: list[str] = []

    for m in last_n:
        p1_norm_m = _normalise(m["player1"])
        is_p1 = p1_norm_m == player_norm
        opponent = m["player2"] if is_p1 else m["player1"]

        l_me = m["legs1"] if is_p1 else m["legs2"]
        l_opp = m["legs2"] if is_p1 else m["legs1"]

        if l_me is not None and l_opp is not None:
            score = f"{l_me}-{l_opp}"
            if l_me > l_opp:
                result_char = "W"
                wins += 1
            else:
                result_char = "L"
        else:
            score = None
            result_char = "U"

        if is_p1 and m.get("checkout_pct_p1") is not None:
            checkout_sum += m["checkout_pct_p1"]
            checkout_count += 1

        form_chars.append(result_char)
        results.append(
            {
                "opponent": opponent,
                "result": result_char,
                "score": score,
                "date": m.get("date", ""),
            }
        )

    n = len(results)
    known = [r for r in results if r["result"] in ("W", "L")]
    n_known = len(known)

    win_rate_last10: Optional[float] = (
        round(wins / n_known, 4) if n_known > 0 else None
    )
    last5_known = [r for r in known[:5]]
    win_rate_last5: Optional[float] = None
    if last5_known:
        w5 = sum(1 for r in last5_known if r["result"] == "W")
        win_rate_last5 = round(w5 / len(last5_known), 4)

    avg_checkout: Optional[float] = (
        round(checkout_sum / checkout_count, 4) if checkout_count > 0 else None
    )

    form_string = "".join(c if c != "U" else "?" for c in form_chars[:5])

    return JSONResponse(
        content={
            "player_id": player_id,
            "last_n_results": results,
            "win_rate_last10": win_rate_last10,
            "win_rate_last5": win_rate_last5,
            "avg_checkout_pct": avg_checkout,
            "form_string": form_string,
            "data_source": _DATA_SOURCE,
        }
    )
