"""
Regression lock manager.

Tracks all completed sprint implementations.
Prevents regressions by locking critical module checksums.

Two modes of operation:
1. Checksum-based locking of source files (RegressionLockManager class).
2. Price-output regression locking (save_lock / assert_matches_lock — legacy,
   retained for backward compatibility with sprint 1-6 tests).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Critical modules — Sprint 7 locks ALL of these
# ---------------------------------------------------------------------------

CRITICAL_MODULES: list[str] = [
    "engines/state_layer/score_state.py",
    "engines/leg_layer/markov_chain.py",
    "engines/leg_layer/hold_break_model.py",
    "engines/match_layer/match_combinatorics.py",
    "engines/match_layer/world_matchplay_engine.py",
    "engines/match_layer/premier_league_engine.py",
    "competition/format_registry.py",
    "competition/draw_result.py",
    "elo/elo_pipeline.py",
    "calibration/beta_calibrator.py",
    "calibration/market_calibrators.py",
    "margin/blending_engine.py",
    "margin/shin_margin.py",
    "sgp/correlation_estimator.py",
    "sgp/copula_builder.py",
    "props/prop_180.py",
    "props/prop_nine_darter.py",
    "outrights/tournament_simulator.py",
    "app/routes/live.py",
    # Lock depth expansion (2026-03-25)
    "app/main.py",
    "app/routes/prematch.py",
    "app/routes/events.py",
    "app/routes/sgp.py",
    "app/routes/settlement.py",
    "app/routes/outrights.py",
    "app/routes/props.py",
    "app/routes/worldcup.py",
    "app/routes/players.py",
    "app/routes/trader.py",
    "app/routes/liability.py",
    "app/routes/feeds.py",
    "app/routes/monitoring.py",
    # Alembic isolation lock (DARTS-TABLES-MIGRATION-NEEDED fix 2026-04-30)
    "alembic/env.py",
]

FIXED_BUGS: list[str] = [
    "DARTS-LIVE-AUTH-001: pricing_authority='live_model_only' added to "
    "LivePriceResponse and LiveMarketsResponse in app/routes/live.py — "
    "sprint 44 live-blend assessment: darts live pricing uses visit-state "
    "Markov chain without Pinnacle logit-space blend.",
    "DARTS-TABLES-MIGRATION-NEEDED: alembic/env.py now sets "
    "version_table='alembic_version_darts' in both run_migrations_offline() and "
    "do_run_migrations() — prevents collision with gateway's alembic_version table "
    "on the shared Supabase DB. All 13 darts tables (darts_competitions, "
    "darts_matches, darts_legs, darts_visits, darts_player_stats, "
    "darts_coverage_regimes, darts_elo_ratings, darts_gdpr_consents, "
    "darts_ml_model_artifacts, darts_liability_limits, darts_bet_exposure, "
    "darts_trader_overrides + existing darts_players) created in production and "
    "alembic_version_darts stamped at revision 004. Fix applied 2026-04-30.",
]


# ---------------------------------------------------------------------------
# RegressionLockManager — checksum-based locking
# ---------------------------------------------------------------------------


class RegressionLockManager:
    """
    Manages SHA-256 checksum locks for critical source modules.

    The lock state is stored in ``lock_state.json`` at the project root.
    Running ``verify()`` detects any unexpected changes to locked modules.

    Parameters
    ----------
    project_root:
        Absolute path to the project root (directory containing
        ``engines/``, ``competition/``, etc.).
    """

    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root)
        self.lock_file = self.project_root / "lock_state.json"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _module_path(self, relative: str) -> Path:
        return self.project_root / relative

    def _sha256(self, path: Path) -> Optional[str]:
        """Compute SHA-256 hex digest of a file, or None if it doesn't exist."""
        if not path.exists():
            return None
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_checksums(self) -> dict[str, str]:
        """
        Compute SHA-256 checksums for all CRITICAL_MODULES.

        Returns
        -------
        dict[str, str]
            Mapping of relative module path → SHA-256 hex digest.
            Missing files are recorded as ``"MISSING"``.
        """
        checksums: dict[str, str] = {}
        for rel in CRITICAL_MODULES:
            path = self._module_path(rel)
            digest = self._sha256(path)
            checksums[rel] = digest if digest is not None else "MISSING"
        return checksums

    def lock(self) -> None:
        """
        Compute checksums for all critical modules and save to lock_state.json.

        Overwrites any existing lock file.  Run this after every intentional
        change to a critical module.
        """
        checksums = self.compute_checksums()
        state = {
            "locked_at": datetime.now(tz=timezone.utc).isoformat(),
            "project_root": str(self.project_root),
            "fixed_bugs": FIXED_BUGS,
            "checksums": checksums,
        }
        with self.lock_file.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

        missing = [m for m, v in checksums.items() if v == "MISSING"]
        logger.info(
            "regression_lock_saved",
            path=str(self.lock_file),
            modules_locked=len(checksums),
            modules_missing=len(missing),
        )
        if missing:
            logger.warning("regression_lock_missing_modules", missing=missing)

    def verify(self) -> tuple[bool, list[str]]:
        """
        Verify that all critical modules still match their locked checksums.

        Returns
        -------
        (ok, changed_modules):
            ``ok`` is True iff all checksums match.
            ``changed_modules`` is the list of relative paths that have changed
            or are now missing when they weren't before.

        Notes
        -----
        If no lock file exists, ``verify()`` returns ``(True, [])`` — the first
        run always passes (caller should then run ``lock()``).
        """
        if not self.lock_file.exists():
            logger.warning(
                "regression_lock_no_lock_file",
                path=str(self.lock_file),
            )
            return True, []

        with self.lock_file.open("r", encoding="utf-8") as fh:
            state = json.load(fh)

        locked_checksums: dict[str, str] = state.get("checksums", {})
        current_checksums = self.compute_checksums()

        changed: list[str] = []
        for rel, locked_digest in locked_checksums.items():
            current_digest = current_checksums.get(rel, "MISSING")
            if current_digest != locked_digest:
                changed.append(rel)
                logger.error(
                    "regression_lock_violation",
                    module=rel,
                    locked=locked_digest[:16] + "...",
                    current=(current_digest[:16] + "...") if current_digest != "MISSING" else "MISSING",
                )

        # Also flag any modules in CRITICAL_MODULES that weren't in the lock
        for rel in CRITICAL_MODULES:
            if rel not in locked_checksums:
                changed.append(rel)
                logger.warning("regression_lock_new_module_not_locked", module=rel)

        ok = len(changed) == 0
        if ok:
            logger.info(
                "regression_lock_verify_ok",
                modules_checked=len(locked_checksums),
            )
        else:
            logger.error(
                "regression_lock_verify_failed",
                changed_count=len(changed),
                changed=changed,
            )
        return ok, changed

    def generate_report(self) -> str:
        """
        Generate a human-readable regression lock status report.

        Returns
        -------
        str
            Multi-line report string.
        """
        lines: list[str] = [
            "=" * 70,
            "XG3 Darts — Regression Lock Status Report",
            f"Generated: {datetime.now(tz=timezone.utc).isoformat()}",
            f"Project root: {self.project_root}",
            "=" * 70,
        ]

        if not self.lock_file.exists():
            lines.append("STATUS: NO LOCK FILE FOUND")
            lines.append(f"  Run RegressionLockManager.lock() to create {self.lock_file}")
            return "\n".join(lines)

        with self.lock_file.open("r", encoding="utf-8") as fh:
            state = json.load(fh)

        locked_at = state.get("locked_at", "unknown")
        locked_checksums: dict[str, str] = state.get("checksums", {})
        current_checksums = self.compute_checksums()

        lines.append(f"Lock file: {self.lock_file}")
        lines.append(f"Locked at: {locked_at}")
        lines.append(f"Modules tracked: {len(CRITICAL_MODULES)}")
        lines.append(f"Fixed bugs: {len(FIXED_BUGS)}")
        lines.append("")
        lines.append("Module Status:")

        all_ok = True
        for rel in CRITICAL_MODULES:
            locked = locked_checksums.get(rel)
            current = current_checksums.get(rel, "MISSING")
            if locked is None:
                status = "  [NEW   ]"
                all_ok = False
            elif current == "MISSING":
                status = "  [MISS  ]"
                all_ok = False
            elif current != locked:
                status = "  [CHANGE]"
                all_ok = False
            else:
                status = "  [OK    ]"
            lines.append(f"{status} {rel}")

        lines.append("")
        if all_ok:
            lines.append("OVERALL STATUS: PASS — all checksums match")
        else:
            lines.append("OVERALL STATUS: FAIL — one or more modules have changed")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy price-output regression locking (sprint 1-6 compatibility)
# ---------------------------------------------------------------------------

class RegressionLockError(Exception):
    """Raised when current output violates a stored regression lock."""


_LOCK_DIR = Path(__file__).parent / "locks"
_DEFAULT_TOLERANCE = 0.001


def save_lock(scenario_id: str, output: dict[str, Any]) -> Path:
    """
    Save a regression lock file for a scenario.

    Parameters
    ----------
    scenario_id:
        Unique scenario identifier.
    output:
        Model output dict to lock.

    Returns
    -------
    Path
        Path of the saved lock file.
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _LOCK_DIR / f"{scenario_id}.json"
    with lock_path.open("w", encoding="utf-8") as fh:
        json.dump({"scenario_id": scenario_id, "output": output}, fh, indent=2)
    return lock_path


def load_lock(scenario_id: str) -> Optional[dict[str, Any]]:
    """
    Load a regression lock file.

    Parameters
    ----------
    scenario_id:
        Unique scenario identifier.

    Returns
    -------
    dict | None
        Locked output, or None if no lock file exists.
    """
    lock_path = _LOCK_DIR / f"{scenario_id}.json"
    if not lock_path.exists():
        return None
    with lock_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("output")


def assert_matches_lock(
    scenario_id: str,
    current_output: dict[str, Any],
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
    update_lock: bool = False,
) -> None:
    """
    Assert that current model output matches the stored lock.

    Parameters
    ----------
    scenario_id:
        Unique scenario identifier.
    current_output:
        Current model output to compare.
    tolerance:
        Maximum allowed fractional deviation per numeric field.
    update_lock:
        If True, update the lock file instead of failing.

    Raises
    ------
    RegressionLockError
        If any numeric value deviates beyond tolerance.
    """
    if update_lock:
        save_lock(scenario_id, current_output)
        return

    locked = load_lock(scenario_id)
    if locked is None:
        save_lock(scenario_id, current_output)
        return

    violations: list[str] = []
    _compare_dicts(locked, current_output, path="", violations=violations, tol=tolerance)

    if violations:
        raise RegressionLockError(
            f"Regression lock violated for '{scenario_id}':\n"
            + "\n".join(violations)
        )


def _compare_dicts(
    locked: Any,
    current: Any,
    path: str,
    violations: list[str],
    tol: float,
) -> None:
    """Recursively compare locked and current values."""
    if isinstance(locked, dict) and isinstance(current, dict):
        for key in locked:
            if key not in current:
                violations.append(f"  {path}.{key}: missing in current output")
                continue
            _compare_dicts(locked[key], current[key], f"{path}.{key}", violations, tol)
    elif isinstance(locked, (int, float)) and isinstance(current, (int, float)):
        if locked != 0:
            rel_diff = abs(current - locked) / abs(locked)
            if rel_diff > tol:
                violations.append(
                    f"  {path}: locked={locked:.6f}, current={current:.6f}, "
                    f"rel_diff={rel_diff:.4%} > tolerance={tol:.4%}"
                )
    elif locked != current:
        violations.append(f"  {path}: locked={locked!r}, current={current!r}")


# ---------------------------------------------------------------------------
# CLI entry point for CI verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.stderr.write("Usage: regression_lock_manager.py [lock|verify|report]\n")
        sys.exit(1)

    command = sys.argv[1]
    project_root = Path(__file__).resolve().parent.parent.parent

    manager = RegressionLockManager(str(project_root))

    if command == "lock":
        manager.lock()
        sys.stdout.write("Lock file created.\n")
        sys.exit(0)

    elif command == "verify":
        ok, changed = manager.verify()
        if ok:
            sys.stdout.write("Regression lock: PASS\n")
            sys.exit(0)
        else:
            sys.stderr.write(f"Regression lock: FAIL — changed: {changed}\n")
            sys.exit(1)

    elif command == "report":
        report = manager.generate_report()
        sys.stdout.write(report + "\n")
        ok, _ = manager.verify()
        sys.exit(0 if ok else 1)

    else:
        sys.stderr.write(f"Unknown command: {command!r}\n")
        sys.exit(1)
