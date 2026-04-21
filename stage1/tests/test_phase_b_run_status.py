"""Regression tests for Phase B run-status artifact marking (YELLOW-LIGHT v3 P3).

Verifies that:

1. ``write_phase_b_status_artifacts`` writes phase_b_summary.json/txt and
   RUN_STATUS.txt with consistent status across all three sentinels.
2. A failed-path persist embeds an explicit ``run_status: "failed"`` in the
   JSON, a ``RUN STATUS: FAILED`` banner at the top of the TXT, a
   ``FAILURE REASON:`` line, and a single-token failed sentinel — so a
   downstream user cannot mistake a failed Phase B run for a valid completed
   one.
3. ``run_phase_b.py`` source still routes both the wording-failure and the
   sanity-check-failure paths through ``_persist_summary("failed", ...)``
   BEFORE raising. This is the source-level guardrail against a future
   refactor that silently restores the old "raise without marking" pattern.

These tests run without torch — the helper module is intentionally
torch-free.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stage1.utils.run_status import (
    RUN_STATUS_FAILED,
    RUN_STATUS_PASSED,
    RUN_STATUS_PENDING,
    build_status_banner,
    write_phase_b_status_artifacts,
)


# ─── Banner construction ─────────────────────────────────────────────────────


def test_passed_banner_contains_passed_marker():
    banner = build_status_banner(RUN_STATUS_PASSED)
    assert any("RUN STATUS: PASSED" in ln for ln in banner)
    assert not any("FAILURE REASON" in ln for ln in banner)


def test_failed_banner_contains_failed_marker_and_reason():
    banner = build_status_banner(RUN_STATUS_FAILED, "wording_violations: x")
    assert any("RUN STATUS: FAILED" in ln for ln in banner)
    assert any("FAILURE REASON: wording_violations: x" in ln for ln in banner)


def test_pending_banner_does_not_falsely_signal_passed():
    banner = build_status_banner(RUN_STATUS_PENDING)
    text = "\n".join(banner)
    assert "PENDING" in text
    assert "PASSED" not in text
    assert "FAILED" not in text


def test_invalid_status_rejected():
    with pytest.raises(ValueError):
        build_status_banner("ok")  # not a real status


# ─── Artifact write ──────────────────────────────────────────────────────────


def _body() -> list:
    return [
        "=" * 60,
        "PHASE B — RESTORATION INTERVENTION RESULTS",
        "=" * 60,
        "",
        "no_patch_accuracy: 0.32",
        "clean_baseline_accuracy: 0.80",
        "",
    ]


def _read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_failed_artifacts_cannot_look_valid(tmp_path):
    summary = {"phase": "B", "no_patch_accuracy": 0.32, "clean_baseline_accuracy": 0.80}
    write_phase_b_status_artifacts(
        str(tmp_path), summary, _body(), RUN_STATUS_FAILED,
        failure_reason="cross_check_failed: anchor parity mismatch",
    )

    # JSON: top-level run_status + failure_reason
    with open(tmp_path / "phase_b_summary.json", encoding="utf-8") as f:
        out = json.load(f)
    assert out["run_status"] == RUN_STATUS_FAILED
    assert "anchor parity mismatch" in out["failure_reason"]

    # TXT: leading FAILED banner with reason — visible to any human glance.
    txt = _read_text(tmp_path / "phase_b_summary.txt")
    head = txt.splitlines()[:6]
    assert any("RUN STATUS: FAILED" in ln for ln in head)
    assert any("anchor parity mismatch" in ln for ln in head)
    # The body still appears AFTER the banner.
    assert "PHASE B — RESTORATION INTERVENTION RESULTS" in txt

    # Single-token sentinel
    sentinel = _read_text(tmp_path / "RUN_STATUS.txt")
    assert sentinel.startswith("FAILED")
    assert "failure_reason" in sentinel


def test_passed_artifacts_are_unambiguous(tmp_path):
    summary = {"phase": "B"}
    write_phase_b_status_artifacts(
        str(tmp_path), summary, _body(), RUN_STATUS_PASSED,
    )
    with open(tmp_path / "phase_b_summary.json", encoding="utf-8") as f:
        out = json.load(f)
    assert out["run_status"] == RUN_STATUS_PASSED
    assert out["failure_reason"] is None
    txt = _read_text(tmp_path / "phase_b_summary.txt")
    assert "RUN STATUS: PASSED" in txt
    assert "FAILURE REASON" not in txt
    assert _read_text(tmp_path / "RUN_STATUS.txt").startswith("PASSED")


def test_persist_overwrites_pending_with_final_status(tmp_path):
    """Simulate the production sequence: pending → failed. Final state must
    not contain stale 'pending' wording anywhere."""
    summary = {"phase": "B"}
    write_phase_b_status_artifacts(
        str(tmp_path), summary, _body(), RUN_STATUS_PENDING,
    )
    write_phase_b_status_artifacts(
        str(tmp_path), summary, _body(), RUN_STATUS_FAILED,
        failure_reason="sanity_check_failed: missing anchors",
    )
    txt = _read_text(tmp_path / "phase_b_summary.txt")
    assert "RUN STATUS: FAILED" in txt
    assert "RUN STATUS: PENDING" not in txt
    sentinel = _read_text(tmp_path / "RUN_STATUS.txt")
    assert sentinel.startswith("FAILED")
    assert "PENDING" not in sentinel


# ─── Source-level guardrails for the production wiring ──────────────────────


def _read_run_phase_b_src():
    with open(
        os.path.join(_REPO_ROOT, "stage1", "run_phase_b.py"), encoding="utf-8",
    ) as f:
        return f.read()


def test_run_phase_b_marks_wording_failure_before_raising():
    """Wording-gate failure path MUST persist a failed status before raising."""
    src = _read_run_phase_b_src()
    # The wording branch contains the violation print and the persist call.
    assert "Conservative-wording gate FAILED" in src
    # Look for the failed persist immediately associated with wording.
    wording_idx = src.find("wording_violations:")
    assert wording_idx != -1, "wording_reason assembly missing in run_phase_b.py"
    persist_idx = src.find("_persist_summary(RUN_STATUS_FAILED", wording_idx)
    raise_idx = src.find("raise RuntimeError", wording_idx)
    assert persist_idx != -1, (
        "wording-failure path missing _persist_summary(RUN_STATUS_FAILED, ...) call"
    )
    assert persist_idx < raise_idx, (
        "wording-failure persist must happen BEFORE raise — otherwise the "
        "summary stays at 'pending' on failure (defeats P3)."
    )


def test_run_phase_b_marks_sanity_failure_before_raising():
    """Sanity-check failure path MUST persist a failed status before raising."""
    src = _read_run_phase_b_src()
    sanity_idx = src.find("sanity_check_failed:")
    assert sanity_idx != -1, (
        "sanity-failure reason assembly missing in run_phase_b.py — the "
        "fail-closed marking may have been removed."
    )
    persist_idx = src.find("_persist_summary(RUN_STATUS_FAILED", sanity_idx)
    raise_idx = src.find("raise RuntimeError", sanity_idx)
    assert persist_idx != -1
    assert persist_idx < raise_idx, (
        "sanity-failure persist must happen BEFORE raise."
    )


def test_run_phase_b_finalises_passed_status_on_success():
    src = _read_run_phase_b_src()
    assert "_persist_summary(RUN_STATUS_PASSED)" in src, (
        "run_phase_b.py: missing the success-path persist of RUN_STATUS_PASSED."
    )
