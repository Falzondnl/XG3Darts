"""
Unit tests for API key authentication middleware (app/auth.py).

Tests:
  - Dev mode (no keys configured): all requests allowed
  - Production mode: missing key → 401
  - Production mode: invalid key → 401
  - Production mode: valid key → allowed
  - Health/docs paths bypass auth even in production mode
  - Timing-safe comparison (HMAC)
"""
from __future__ import annotations

import hashlib
import hmac
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app_with_keys(api_keys: str, salt: str = "testsalt") -> tuple[FastAPI, TestClient]:
    """
    Build a minimal FastAPI app with auth middleware and the given key registry.

    Sets API_KEYS and API_KEY_SALT env vars, reloads auth module so
    _VALID_KEY_HASHES is built with the test salt.

    Caller is responsible for restoring env vars in teardown.
    """
    import sys

    # Set env vars BEFORE reloading auth so both the registry and _hash_key use them
    os.environ["API_KEYS"] = api_keys
    os.environ["API_KEY_SALT"] = salt

    # Force reload of auth module so _VALID_KEY_HASHES is rebuilt
    if "app.auth" in sys.modules:
        del sys.modules["app.auth"]

    from app.auth import api_key_middleware

    app = FastAPI()
    app.middleware("http")(api_key_middleware)

    @app.get("/api/v1/darts/test")
    async def test_route():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    client = TestClient(app, raise_server_exceptions=True)
    return app, client


def _hmac_hash(key: str, salt: str = "testsalt") -> str:
    return hmac.new(salt.encode(), key.encode(), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Dev mode (no API_KEYS configured)
# ---------------------------------------------------------------------------

class TestDevMode:
    """When API_KEYS is empty, all requests pass through (dev mode)."""

    def setup_method(self):
        import sys
        if "app.auth" in sys.modules:
            del sys.modules["app.auth"]
        os.environ.pop("API_KEYS", None)
        os.environ.pop("API_KEY_SALT", None)
        from app.auth import api_key_middleware
        app = FastAPI()
        app.middleware("http")(api_key_middleware)

        @app.get("/api/v1/darts/test")
        async def test_route():
            return {"ok": True}

        self.client = TestClient(app)

    def test_request_without_key_allowed_in_dev(self):
        resp = self.client.get("/api/v1/darts/test")
        assert resp.status_code == 200

    def test_request_with_any_key_allowed_in_dev(self):
        resp = self.client.get("/api/v1/darts/test", headers={"X-Api-Key": "anything"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Production mode (API_KEYS set)
# ---------------------------------------------------------------------------

class TestProductionMode:
    """When API_KEYS is set, only valid keys are accepted."""

    SALT = "xg3_test_salt"
    VALID_KEY = "XG3-test-key-abc123"

    def setup_method(self):
        _, self.client = _make_app_with_keys(self.VALID_KEY, self.SALT)

    def teardown_method(self):
        os.environ.pop("API_KEYS", None)
        os.environ.pop("API_KEY_SALT", None)

    def test_missing_key_returns_401(self):
        resp = self.client.get("/api/v1/darts/test")
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"] == "unauthorized"
        assert "X-Api-Key" in body["message"]

    def test_wrong_key_returns_401(self):
        resp = self.client.get(
            "/api/v1/darts/test",
            headers={"X-Api-Key": "XG3-wrong-key"},
        )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"] == "unauthorized"

    def test_valid_key_allowed(self):
        resp = self.client.get(
            "/api/v1/darts/test",
            headers={"X-Api-Key": self.VALID_KEY},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_case_sensitive_key(self):
        """Keys must match exactly — case sensitive."""
        resp = self.client.get(
            "/api/v1/darts/test",
            headers={"X-Api-Key": self.VALID_KEY.upper()},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Public path bypass
# ---------------------------------------------------------------------------

class TestPublicPathBypass:
    """Health/docs/etc. paths must bypass auth in all modes."""

    VALID_KEY = "XG3-prod-key-xyz"

    def setup_method(self):
        import sys
        if "app.auth" in sys.modules:
            del sys.modules["app.auth"]
        os.environ["API_KEYS"] = self.VALID_KEY
        os.environ["API_KEY_SALT"] = "bypass_salt"
        from app.auth import api_key_middleware

        app = FastAPI()
        app.middleware("http")(api_key_middleware)

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/ready")
        async def ready():
            return {"status": "ready"}

        @app.get("/api/v1/darts/protected")
        async def protected():
            return {"secret": True}

        self.client = TestClient(app)

    def teardown_method(self):
        os.environ.pop("API_KEYS", None)
        os.environ.pop("API_KEY_SALT", None)

    def test_health_bypasses_auth(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_ready_bypasses_auth(self):
        resp = self.client.get("/ready")
        assert resp.status_code == 200

    def test_protected_route_requires_key(self):
        resp = self.client.get("/api/v1/darts/protected")
        assert resp.status_code == 401

    def test_protected_route_with_key(self):
        resp = self.client.get(
            "/api/v1/darts/protected",
            headers={"X-Api-Key": self.VALID_KEY},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Multiple keys
# ---------------------------------------------------------------------------

class TestMultipleKeys:
    """Multiple keys can be active simultaneously."""

    KEY_A = "XG3-client-alpha"
    KEY_B = "XG3-client-beta"
    SALT = "multi_salt"

    def setup_method(self):
        _, self.client = _make_app_with_keys(f"{self.KEY_A},{self.KEY_B}", self.SALT)

    def teardown_method(self):
        os.environ.pop("API_KEYS", None)
        os.environ.pop("API_KEY_SALT", None)

    def test_key_a_accepted(self):
        resp = self.client.get(
            "/api/v1/darts/test",
            headers={"X-Api-Key": self.KEY_A},
        )
        assert resp.status_code == 200

    def test_key_b_accepted(self):
        resp = self.client.get(
            "/api/v1/darts/test",
            headers={"X-Api-Key": self.KEY_B},
        )
        assert resp.status_code == 200

    def test_unknown_key_rejected(self):
        resp = self.client.get(
            "/api/v1/darts/test",
            headers={"X-Api-Key": "XG3-client-unknown"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# HMAC hash helper
# ---------------------------------------------------------------------------

class TestHmacHash:
    """The HMAC hash function produces consistent output."""

    def test_same_input_same_output(self):
        h1 = _hmac_hash("testkey", "testsalt")
        h2 = _hmac_hash("testkey", "testsalt")
        assert h1 == h2

    def test_different_key_different_hash(self):
        h1 = _hmac_hash("key1", "salt")
        h2 = _hmac_hash("key2", "salt")
        assert h1 != h2

    def test_different_salt_different_hash(self):
        h1 = _hmac_hash("key", "salt1")
        h2 = _hmac_hash("key", "salt2")
        assert h1 != h2

    def test_hash_is_hex_64_chars(self):
        h = _hmac_hash("anykey", "anysalt")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
