"""NF-XG3-REDIS-DSN-LOG-DISCLOSURE-001 (SEC-3A-R5B, 2026-09-30).

RED before the fix: `app/workers/live_ops_worker.py` logged the full REDIS_URL, so the Redis
password was written to the container log on every client initialisation.

These tests lock the structural property: no credential-bearing connection
string is ever handed to the logger. They use a synthetic DSN only -- no real
credential appears in this file or in the test output.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER_PATH = _REPO_ROOT / "app/redis_dsn.py"

# A synthetic DSN. The password is deliberately distinctive so any leak is
# unambiguous, and deliberately fake so this file holds no real secret.
_FAKE_PASSWORD = "sYnThEtIc-NoT-a-ReAl-PaSsWoRd-3a5b7c9d"
_FAKE_DSN = f"redis://:{_FAKE_PASSWORD}@redis-host.internal:6379/0"


def _load_helper():
    spec = importlib.util.spec_from_file_location("_r5b_redis_dsn", _HELPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


redis_log_target = _load_helper().redis_log_target


# ---------------------------------------------------------------- helper ----

def test_helper_strips_the_credential_entirely():
    out = redis_log_target(_FAKE_DSN)
    assert _FAKE_PASSWORD not in out
    assert "@" not in out
    assert ":" in out  # still describes host:port
    assert out == "redis://redis-host.internal:6379/0"


@pytest.mark.parametrize(
    "dsn",
    [
        f"redis://:{_FAKE_PASSWORD}@h:6379/0",
        f"redis://user:{_FAKE_PASSWORD}@h:6379",
        f"rediss://user:{_FAKE_PASSWORD}@h:6380/2",
        f"redis://:{_FAKE_PASSWORD}@[2001:db8::1]:6379/1",
        f"redis://:{_FAKE_PASSWORD}@h",
        f"unix:///var/run/redis.sock?password={_FAKE_PASSWORD}",
        f"redis://:{_FAKE_PASSWORD}@h:6379/0?ssl_cert_reqs=none",
    ],
)
def test_helper_never_emits_the_password_for_any_url_shape(dsn):
    out = redis_log_target(dsn)
    assert _FAKE_PASSWORD not in out, dsn
    assert "@" not in out, dsn


def test_helper_handles_absent_and_malformed_without_falling_back_to_raw():
    assert redis_log_target(None) == "unset"
    assert redis_log_target("") == "unset"
    # A malformed value must not be echoed back verbatim.
    weird = f"::::{_FAKE_PASSWORD}"
    assert _FAKE_PASSWORD not in redis_log_target(weird)


def test_helper_is_structural_not_a_masking_pass():
    """Two different passwords of different lengths must give identical output.

    That can only hold if the credential is never part of the result, which is
    what distinguishes this from redacting a known secret or fixed-width mask.
    """
    a = redis_log_target("redis://:short@h:6379/0")
    b = redis_log_target(f"redis://:{'x' * 512}@h:6379/0")
    assert a == b == "redis://h:6379/0"


# ------------------------------------------------------- static source lock --

_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
_URLY = ("redis_url", "REDIS_URL", "redis_dsn", "REDIS_DSN")


def _is_logger_call(node: ast.Call) -> bool:
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr in _LOG_METHODS


def _mentions_redis_url(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in _URLY:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in _URLY:
            return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if "redis://" in sub.value or "rediss://" in sub.value:
                return True
    return False


def _source_files():
    for rel in ["app", "engines", "data", "db"]:
        base = _REPO_ROOT / rel
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "/tests/" in path.as_posix() or path.name.startswith("test_"):
                continue
            yield path


def test_no_logger_call_anywhere_receives_a_raw_redis_dsn():
    """The lock. A logger call may never receive a redis URL as an argument.

    Passing it through the helper is fine: the helper's return value is a
    credential-free description, and the call node then references the helper,
    not the URL, as the outermost expression.
    """
    offenders = []
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_logger_call(node):
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                # Allowed: the value is a call to the redaction helper.
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                    if arg.func.id == "redis_log_target":
                        continue
                if _mentions_redis_url(arg):
                    offenders.append(
                        f"{path.relative_to(_REPO_ROOT)}:{node.lineno}"
                    )
    assert not offenders, (
        "logger call receives a raw Redis DSN (credential disclosure): "
        + ", ".join(sorted(set(offenders)))
    )


def test_the_patched_site_still_logs_something_useful():
    """Guard against 'fixing' the leak by deleting the observability."""
    site = (_REPO_ROOT / "app/workers/live_ops_worker.py").read_text(encoding="utf-8", errors="replace")
    assert "redis_connected" in site
    assert "redis_log_target" in site
