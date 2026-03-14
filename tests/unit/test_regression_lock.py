"""
Unit tests for the regression lock manager (Sprint 7).

Tests:
- test_regression_lock_creates_checksums
- test_regression_lock_detects_change
- test_regression_lock_verify_clean
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from tests.regression.regression_lock_manager import (
    CRITICAL_MODULES,
    RegressionLockManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_temp_project(tmp_path: Path, modules: list[str]) -> Path:
    """
    Create a minimal temporary project tree with dummy Python files.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    modules:
        List of relative module paths to create.

    Returns
    -------
    Path
        The project root (tmp_path itself).
    """
    for rel in modules:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# dummy module: {rel}\nVERSION = 1\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegressionLockCreatesChecksums:
    """G: RegressionLockManager.compute_checksums() returns correct data."""

    def test_regression_lock_creates_checksums(self, tmp_path: Path) -> None:
        """
        compute_checksums() returns a dict with one entry per CRITICAL_MODULE.
        Files that exist have a 64-char hex digest; missing files are 'MISSING'.
        """
        modules = [
            "engines/state_layer/score_state.py",
            "engines/leg_layer/markov_chain.py",
        ]
        _make_temp_project(tmp_path, modules)
        manager = RegressionLockManager(str(tmp_path))

        checksums = manager.compute_checksums()

        assert isinstance(checksums, dict), "checksums must be a dict"
        assert len(checksums) == len(CRITICAL_MODULES), (
            f"Expected {len(CRITICAL_MODULES)} entries, got {len(checksums)}"
        )

        # Created modules have valid hex digests
        for rel in modules:
            assert rel in checksums, f"Missing key: {rel}"
            digest = checksums[rel]
            assert digest != "MISSING", f"{rel} exists but returned MISSING"
            assert len(digest) == 64, f"SHA-256 should be 64 hex chars, got {len(digest)}"

        # Modules NOT created in tmp_path are MISSING
        for rel in CRITICAL_MODULES:
            if rel not in modules:
                assert checksums[rel] == "MISSING", (
                    f"{rel} should be MISSING but got {checksums[rel]!r}"
                )

    def test_regression_lock_creates_checksums_all_present(self, tmp_path: Path) -> None:
        """compute_checksums() returns 64-char digests when all modules exist."""
        _make_temp_project(tmp_path, CRITICAL_MODULES)
        manager = RegressionLockManager(str(tmp_path))

        checksums = manager.compute_checksums()

        for rel in CRITICAL_MODULES:
            digest = checksums.get(rel)
            assert digest is not None
            assert digest != "MISSING", f"{rel} should not be MISSING"
            assert len(digest) == 64

    def test_lock_writes_json_file(self, tmp_path: Path) -> None:
        """RegressionLockManager.lock() writes a valid JSON lock_state.json."""
        _make_temp_project(tmp_path, CRITICAL_MODULES)
        manager = RegressionLockManager(str(tmp_path))
        manager.lock()

        assert manager.lock_file.exists(), "lock_state.json should be created"

        with manager.lock_file.open("r") as fh:
            state = json.load(fh)

        assert "checksums" in state
        assert "locked_at" in state
        assert len(state["checksums"]) == len(CRITICAL_MODULES)


class TestRegressionLockDetectsChange:
    """G: RegressionLockManager.verify() detects modified modules."""

    def test_regression_lock_detects_change(self, tmp_path: Path) -> None:
        """
        After locking, modifying a critical module causes verify() to return
        (False, [changed_module]).
        """
        changed_module = "engines/state_layer/score_state.py"
        _make_temp_project(tmp_path, CRITICAL_MODULES)
        manager = RegressionLockManager(str(tmp_path))

        # Create the initial lock
        manager.lock()

        # Modify one critical module
        module_path = tmp_path / changed_module
        module_path.write_text(
            "# MODIFIED — Sprint 7 regression test\nVERSION = 99\n",
            encoding="utf-8",
        )

        ok, changed = manager.verify()

        assert not ok, "verify() should return False after a module is changed"
        assert changed_module in changed, (
            f"Expected {changed_module!r} in changed list, got {changed}"
        )

    def test_regression_lock_detects_multiple_changes(self, tmp_path: Path) -> None:
        """Multiple changed modules are all reported in the changed list."""
        changed_modules = [
            "engines/leg_layer/markov_chain.py",
            "competition/format_registry.py",
        ]
        _make_temp_project(tmp_path, CRITICAL_MODULES)
        manager = RegressionLockManager(str(tmp_path))
        manager.lock()

        for rel in changed_modules:
            (tmp_path / rel).write_text("# CHANGED\n", encoding="utf-8")

        ok, changed = manager.verify()

        assert not ok
        for rel in changed_modules:
            assert rel in changed, f"{rel} should be in changed list"

    def test_regression_lock_detects_deleted_file(self, tmp_path: Path) -> None:
        """A module that existed at lock time but is deleted is reported as changed."""
        _make_temp_project(tmp_path, CRITICAL_MODULES)
        manager = RegressionLockManager(str(tmp_path))
        manager.lock()

        target = "margin/shin_margin.py"
        (tmp_path / target).unlink()

        ok, changed = manager.verify()

        assert not ok
        assert target in changed


class TestRegressionLockVerifyClean:
    """G: verify() returns (True, []) when no changes have occurred."""

    def test_regression_lock_verify_clean(self, tmp_path: Path) -> None:
        """
        Immediately after locking, verify() should return (True, []).
        """
        _make_temp_project(tmp_path, CRITICAL_MODULES)
        manager = RegressionLockManager(str(tmp_path))
        manager.lock()

        ok, changed = manager.verify()

        assert ok, f"verify() should pass immediately after lock(); changed={changed}"
        assert changed == [], f"No modules should be changed; got {changed}"

    def test_regression_lock_verify_no_lock_file_passes(self, tmp_path: Path) -> None:
        """verify() passes (True, []) when no lock file exists yet."""
        manager = RegressionLockManager(str(tmp_path))

        ok, changed = manager.verify()

        assert ok, "No lock file should not be treated as a failure"
        assert changed == []

    def test_regression_lock_verify_idempotent(self, tmp_path: Path) -> None:
        """Calling verify() multiple times without changes always passes."""
        _make_temp_project(tmp_path, CRITICAL_MODULES)
        manager = RegressionLockManager(str(tmp_path))
        manager.lock()

        for _ in range(3):
            ok, changed = manager.verify()
            assert ok
            assert changed == []

    def test_generate_report_includes_all_modules(self, tmp_path: Path) -> None:
        """generate_report() mentions every CRITICAL_MODULE."""
        _make_temp_project(tmp_path, CRITICAL_MODULES)
        manager = RegressionLockManager(str(tmp_path))
        manager.lock()

        report = manager.generate_report()

        assert isinstance(report, str)
        assert "PASS" in report, "Clean lock should show PASS in report"
        for rel in CRITICAL_MODULES:
            # The report should contain at least the filename part
            filename = Path(rel).name
            assert filename in report, f"{filename} should appear in report"

    def test_generate_report_shows_fail_on_change(self, tmp_path: Path) -> None:
        """generate_report() shows FAIL when a module has changed."""
        _make_temp_project(tmp_path, CRITICAL_MODULES)
        manager = RegressionLockManager(str(tmp_path))
        manager.lock()

        (tmp_path / "engines/leg_layer/markov_chain.py").write_text(
            "# MODIFIED\n", encoding="utf-8"
        )

        report = manager.generate_report()
        assert "FAIL" in report
