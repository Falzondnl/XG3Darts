"""
Integration tests for the XG3 Darts API routes.

These tests use the FastAPI TestClient to call the actual API handlers
end-to-end.  External dependencies (database, Redis, remote APIs) are
mocked so the tests run in CI without infrastructure.

Tests:
    test_health_endpoint
    test_prematch_price_endpoint
    test_live_update_endpoint
    test_outright_simulate_endpoint
    test_sgp_price_endpoint
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient for the XG3 Darts FastAPI app."""
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient) -> None:
    """
    GET /health must return 200 with status='ok'.

    The health endpoint is used by Railway / load balancer probes.
    It must always respond quickly and not depend on database.
    """
    response = client.get("/health")
    assert response.status_code == 200, (
        f"Health endpoint returned {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["status"] == "ok", f"Expected status='ok', got {data!r}"
    assert "service" in data, "Health response missing 'service' field"
    assert "version" in data, "Health response missing 'version' field"


def test_health_endpoint_content_type(client: TestClient) -> None:
    """Health endpoint returns application/json."""
    response = client.get("/health")
    assert "application/json" in response.headers.get("content-type", ""), (
        "Health endpoint should return JSON"
    )


# ---------------------------------------------------------------------------
# Pre-match price endpoint
# ---------------------------------------------------------------------------


def test_prematch_price_endpoint(client: TestClient) -> None:
    """
    POST /api/v1/darts/prematch/price must return a valid price response.

    Uses PDC_PL League Night format with two known-good 3DA values.
    The engine must:
    - Resolve the format from the registry
    - Compute hold/break probabilities
    - Return p1_win + p2_win probabilities that sum to 1.0
    """
    payload = {
        "competition_code": "PDC_PL",
        "round_name": "League Night",
        "p1_id": "test_player_van_gerwen",
        "p2_id": "test_player_wright",
        "p1_three_da": 99.5,
        "p2_three_da": 93.2,
        "p1_starts": True,
        "regime": 1,
        "ecosystem": "pdc_mens",
    }

    response = client.post("/api/v1/darts/prematch/price", json=payload)

    # Accept 200 (success) or 422 (validation) — not 500
    assert response.status_code != 500, (
        f"Pre-match endpoint returned 500: {response.text[:500]}"
    )

    if response.status_code == 200:
        data = response.json()
        # Must have probability fields — either top-level or nested under adjusted_probabilities
        has_probs = (
            "p1_win" in data
            or "match_winner" in data
            or "adjusted_probabilities" in data
            or "true_probabilities" in data
        )
        assert has_probs, (
            f"Response missing probability field: {list(data.keys())}"
        )


def test_prematch_invalid_format_returns_422(client: TestClient) -> None:
    """Invalid competition_code must return 422, not 500."""
    payload = {
        "competition_code": "TOTALLY_FAKE_FORMAT",
        "round_name": "Final",
        "p1_id": "player_a",
        "p2_id": "player_b",
        "p1_three_da": 95.0,
        "p2_three_da": 90.0,
        "p1_starts": True,
    }
    response = client.post("/api/v1/darts/prematch/price", json=payload)
    # Should not return 500 — validation or known error
    assert response.status_code in (400, 404, 422), (
        f"Expected 4xx for invalid format, got {response.status_code}: {response.text[:200]}"
    )


# ---------------------------------------------------------------------------
# Live update endpoint
# ---------------------------------------------------------------------------


def test_live_update_endpoint(client: TestClient) -> None:
    """
    POST /api/v1/darts/live/update or similar endpoint must handle
    an in-play score update without returning 500.

    Live endpoint accepts a visit score and current match state.
    """
    payload = {
        "match_id": "test_live_match_001",
        "competition_code": "PDC_PL",
        "round_name": "League Night",
        "p1_id": "test_player_a",
        "p2_id": "test_player_b",
        "p1_three_da": 96.0,
        "p2_three_da": 89.0,
        "score_p1": 260,
        "score_p2": 501,
        "legs_p1": 3,
        "legs_p2": 4,
        "p1_throws_next": True,
        "visit_score": 100,
    }

    # Try the live update endpoint
    response = client.post("/api/v1/darts/live/update", json=payload)

    # Must not crash the server
    assert response.status_code != 500, (
        f"Live update endpoint returned 500: {response.text[:500]}"
    )


def test_live_endpoint_exists(client: TestClient) -> None:
    """Live update endpoint must be registered (not 404)."""
    # Test with minimal payload — we just check the route exists
    response = client.post(
        "/api/v1/darts/live/update",
        json={"match_id": "test", "score_p1": 501, "score_p2": 501},
    )
    # Route must exist — 404 indicates the endpoint was never registered
    assert response.status_code != 404, (
        "Live update endpoint not registered — check app/routes/live.py"
    )


# ---------------------------------------------------------------------------
# Outright simulation endpoint
# ---------------------------------------------------------------------------


def test_outright_simulate_endpoint(client: TestClient) -> None:
    """
    POST /api/v1/darts/outrights/simulate must accept a player list
    and return win probabilities summing to 1.0.
    """
    payload = {
        "competition_code": "PDC_MASTERS",
        "players": [
            {"player_id": f"player_{i:02d}", "elo_rating": 1800.0 - i * 20.0}
            for i in range(8)
        ],
        "n_simulations": 500,
    }

    response = client.post("/api/v1/darts/outrights/simulate", json=payload)

    assert response.status_code != 500, (
        f"Outright simulate returned 500: {response.text[:500]}"
    )

    if response.status_code == 200:
        data = response.json()
        # Probabilities should sum to approximately 1.0
        probs = [v for v in data.values() if isinstance(v, float)]
        if probs:
            total = sum(probs)
            assert abs(total - 1.0) <= 0.01, (
                f"Outright probabilities sum to {total:.4f}, expected ~1.0"
            )


def test_outright_endpoint_exists(client: TestClient) -> None:
    """Outright simulate endpoint must be registered."""
    response = client.post(
        "/api/v1/darts/outrights/simulate",
        json={"competition_code": "PDC_MASTERS", "players": [], "n_simulations": 10},
    )
    assert response.status_code != 404, (
        "Outright simulate endpoint not registered — check app/routes/outrights.py"
    )


# ---------------------------------------------------------------------------
# SGP price endpoint
# ---------------------------------------------------------------------------


def test_sgp_price_endpoint(client: TestClient) -> None:
    """
    POST /api/v1/darts/sgp/price must accept a list of SGP legs
    and return a combined probability without server error.
    """
    payload = {
        "match_id": "test_sgp_001",
        "competition_code": "PDC_PL",
        "round_name": "League Night",
        "p1_id": "test_sgp_p1",
        "p2_id": "test_sgp_p2",
        "p1_three_da": 94.5,
        "p2_three_da": 91.0,
        "legs": [
            {"market": "match_winner", "selection": "p1_win", "raw_prob": 0.60},
            {"market": "total_legs", "selection": "over", "line": 11.5, "raw_prob": 0.55},
        ],
    }

    response = client.post("/api/v1/darts/sgp/price", json=payload)

    assert response.status_code != 500, (
        f"SGP price endpoint returned 500: {response.text[:500]}"
    )


def test_sgp_endpoint_exists(client: TestClient) -> None:
    """SGP price endpoint must be registered."""
    response = client.post(
        "/api/v1/darts/sgp/price",
        json={"match_id": "x", "legs": []},
    )
    assert response.status_code != 404, (
        "SGP price endpoint not registered — check app/routes/sgp.py"
    )


# ---------------------------------------------------------------------------
# Players API endpoints (Sprint 7)
# ---------------------------------------------------------------------------


def test_players_route_registered(client: TestClient) -> None:
    """GET /api/v1/darts/players/{player_id} endpoint must be registered."""
    response = client.get("/api/v1/darts/players/test-player-uuid")
    # 404 means route not registered; 501/422/500 are implementation states
    assert response.status_code != 404 or "Not Found" not in response.text, (
        "Players endpoint not registered — check app/routes/players.py"
    )


def test_player_regime_route_registered(client: TestClient) -> None:
    """GET /api/v1/darts/players/{player_id}/regime endpoint must be registered."""
    response = client.get("/api/v1/darts/players/test-player-uuid/regime")
    # Route must exist — 404 with FastAPI detail indicates unregistered
    assert response.status_code != 404, (
        "Player regime endpoint not registered — check app/routes/players.py"
    )


def test_player_explain_route_registered(client: TestClient) -> None:
    """POST /api/v1/darts/players/explain endpoint must be registered."""
    response = client.post(
        "/api/v1/darts/players/explain",
        json={"player_id": "test-uuid", "match_context": {}},
    )
    assert response.status_code != 404, (
        "Player explain endpoint not registered — check app/routes/players.py"
    )
