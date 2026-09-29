"""Credential-free description of a Redis DSN, safe to hand to a logger.

NF-XG3-REDIS-DSN-LOG-DISCLOSURE-001 (2026-09-30, checkpoint SEC-3A-R5B).

The Redis connection string embeds the Redis password in its userinfo section,
so it must never be passed to logging. This helper does not mask or redact the
credential: it never handles it. The description is rebuilt from the parsed
scheme, hostname, port and path only, so no userinfo can survive regardless of
the password's length, contents or URL format.
"""

from __future__ import annotations

from urllib.parse import urlsplit

__all__ = ["redis_log_target"]


def redis_log_target(url: str | None) -> str:
    """Return a credential-free "scheme://host:port/db" description of *url*.

    Returns ``"unset"`` for an empty value and ``"unparseable"`` when the URL
    has no host, so callers always get a loggable string and never fall back to
    printing the raw DSN.
    """
    if not url:
        return "unset"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "unparseable"

    host = parts.hostname
    if not host:
        # Covers unix:///path sockets and malformed values alike. The path is
        # not reproduced, because a unix DSN can carry ?password= in its query.
        return "unparseable"
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"

    target = f"{parts.scheme or 'redis'}://{host}"
    if parts.port:
        target = f"{target}:{parts.port}"
    if parts.path and parts.path != "/":
        target = f"{target}{parts.path}"
    return target
