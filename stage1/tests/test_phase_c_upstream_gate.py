"""Regression tests for Phase C upstream-validity gate (v4 PRIORITY 1).

Verifies that ``stage1.run_phase_c._assert_phase_b_passed`` hard-fails by
default when the Phase B summary reports anything other than
``run_status == "passed"``, and that the explicit ``--allow-failed-upstream``
override behaves exactly as documented.

These tests are torch-free — the gate function reads only
``phase_b_summary.json`` and never touches torch / numpy.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stage1.run_phase_c import (  # noqa: E402
    FailedUpstreamError,
    _assert_phase_b_passed,
)


def _write_phase_b_summary(
    run_dir, *, run_status=None, failure_reason=None, omit_status=False,
):
    """Write a minimal phase_b_summary.json into ``run_dir``."""
    body = {"phase": "B"}
    if not omit_status:
        body["run_status"] = run_status
        body["failure_reason"] = failure_reason
    path = os.path.join(str(run_dir), "phase_b_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f)
    return path


# ─── Default-strict path ─────────────────────────────────────────────────────


def test_passed_upstream_returns_summary(tmp_path):
    """Happy path: run_status='passed' returns the summary dict."""
    _write_phase_b_summary(tmp_path, run_status="passed")
    out = _assert_phase_b_passed(str(tmp_path))
    assert out["run_status"] == "passed"


def test_failed_upstream_hard_fails_by_default(tmp_path):
    """run_status='failed' must raise FailedUpstreamError under default."""
    _write_phase_b_summary(
        tmp_path, run_status="failed",
        failure_reason="sanity_check_failed: missing anchors",
    )
    with pytest.raises(FailedUpstreamError) as exc:
        _assert_phase_b_passed(str(tmp_path))
    msg = str(exc.value)
    assert "run_status='failed'" in msg
    assert "missing anchors" in msg
    # Must point at the override knob in the diagnostic.
    assert "--allow-failed-upstream" in msg


def test_pending_upstream_hard_fails_by_default(tmp_path):
    """A 'pending' run is half-written and must not propagate."""
    _write_phase_b_summary(tmp_path, run_status="pending")
    with pytest.raises(FailedUpstreamError):
        _assert_phase_b_passed(str(tmp_path))


def test_unknown_upstream_status_hard_fails(tmp_path):
    """Any string other than 'passed' is rejected — no silent acceptance."""
    _write_phase_b_summary(tmp_path, run_status="unknown_status_xyz")
    with pytest.raises(FailedUpstreamError) as exc:
        _assert_phase_b_passed(str(tmp_path))
    assert "unknown_status_xyz" in str(exc.value)


def test_missing_run_status_field_hard_fails(tmp_path):
    """Legacy Phase B run without run_status field cannot be proven valid."""
    _write_phase_b_summary(tmp_path, omit_status=True)
    with pytest.raises(FailedUpstreamError) as exc:
        _assert_phase_b_passed(str(tmp_path))
    assert "no `run_status` field" in str(exc.value)


def test_missing_summary_file_hard_fails(tmp_path):
    """No phase_b_summary.json at all → refuse."""
    with pytest.raises(FailedUpstreamError) as exc:
        _assert_phase_b_passed(str(tmp_path))
    assert "not found" in str(exc.value)


def test_unreadable_summary_file_hard_fails(tmp_path):
    """Corrupted JSON → refuse with a clear diagnostic."""
    path = os.path.join(str(tmp_path), "phase_b_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("not valid json {")
    with pytest.raises(FailedUpstreamError) as exc:
        _assert_phase_b_passed(str(tmp_path))
    assert "unreadable" in str(exc.value)


# ─── Override path ───────────────────────────────────────────────────────────


def test_override_allows_failed_upstream(tmp_path, caplog):
    """allow_failed_upstream=True returns the summary AND emits a loud warning."""
    _write_phase_b_summary(
        tmp_path, run_status="failed",
        failure_reason="wording_violations: x",
    )
    with caplog.at_level(logging.WARNING, logger="stage1.run_phase_c"):
        out = _assert_phase_b_passed(
            str(tmp_path), allow_failed_upstream=True,
        )
    assert out["run_status"] == "failed"
    # The warning must mention the override so it shows up in operator logs.
    assert any("UPSTREAM-GATE OVERRIDDEN" in r.getMessage() for r in caplog.records)


def test_override_allows_missing_status(tmp_path, caplog):
    """Override also bypasses the missing-run_status hard fail."""
    _write_phase_b_summary(tmp_path, omit_status=True)
    with caplog.at_level(logging.WARNING, logger="stage1.run_phase_c"):
        out = _assert_phase_b_passed(
            str(tmp_path), allow_failed_upstream=True,
        )
    assert "run_status" not in out
    assert any("UPSTREAM-GATE OVERRIDDEN" in r.getMessage() for r in caplog.records)


def test_override_does_not_silently_swallow_missing_file(tmp_path):
    """Override only relaxes the run_status check, not the file-existence
    check — a missing summary still hard-fails."""
    with pytest.raises(FailedUpstreamError):
        _assert_phase_b_passed(
            str(tmp_path), allow_failed_upstream=True,
        )


# ─── Source-level guardrails ─────────────────────────────────────────────────


def test_run_phase_c_calls_gate_before_creating_run_dir():
    """Refactor-safety: the gate MUST run before _create_run_dir to avoid
    leaving an empty Phase C dir on hard fail."""
    path = os.path.join(_REPO_ROOT, "stage1", "run_phase_c.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    gate_idx = src.find("_assert_phase_b_passed(")
    create_idx = src.find("_create_run_dir(run_name)")
    assert gate_idx != -1 and create_idx != -1
    assert gate_idx < create_idx, (
        "_assert_phase_b_passed must run BEFORE _create_run_dir so a failed "
        "upstream does not create empty Phase C run directories."
    )


def test_run_phase_c_records_upstream_gate_block_in_summary():
    """Auditability: the override decision must persist into the Phase C summary."""
    path = os.path.join(_REPO_ROOT, "stage1", "run_phase_c.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert '"upstream_gate"' in src
    assert '"allow_failed_upstream"' in src
    assert '"phase_b_run_status"' in src


def test_cli_exposes_allow_failed_upstream_flag():
    """The override must be reachable via CLI."""
    path = os.path.join(_REPO_ROOT, "stage1", "run_phase_c.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "--allow-failed-upstream" in src
    assert "allow_failed_upstream=args.allow_failed_upstream" in src
