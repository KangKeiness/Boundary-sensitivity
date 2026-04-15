"""
Tests for the grid-intersection self-consistency check in stage1/run_phase_a.py.

Tests directly exercise the production _check_grid_intersection helper extracted
from stage1.run_phase_a.
"""
import json
import os
import tempfile

import pytest

from stage1.run_phase_a import _check_grid_intersection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary_json(run_dir: str) -> str:
    """Write a minimal phase_a_summary.json and return its path."""
    path = os.path.join(run_dir, "phase_a_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"phase": "A", "description": "test"}, f)
    return path


def _make_rows(r_pos2: dict, r_b8w4: dict) -> list:
    """Build a rows list with the required 'condition' keys for both intersection points."""
    return [
        {"condition": "fixed_w4_pos2", **r_pos2},
        {"condition": "fixed_b8_w4", **r_b8w4},
    ]


# ---------------------------------------------------------------------------
# Scenario (a): matching accuracy/degradation, fld within 1e-3
# Expected: no raise, no grid_intersection_fld_warning key in JSON
# ---------------------------------------------------------------------------

def test_scenario_a_no_raise_no_warning():
    """Both fld deltas are 5e-4 (< 1e-3) and accuracy/degradation match exactly."""
    r_pos2 = {"accuracy": 0.72, "degradation": 0.04, "fld_cos": 0.1200, "fld_l2": 3.5000}
    r_b8w4 = {"accuracy": 0.72, "degradation": 0.04, "fld_cos": 0.1205, "fld_l2": 3.5004}
    # fld_cos delta = 0.0005, fld_l2 delta = 0.0004 — both < 1e-3

    with tempfile.TemporaryDirectory() as run_dir:
        summary_path = _make_summary_json(run_dir)
        # Must not raise
        _check_grid_intersection(_make_rows(r_pos2, r_b8w4), run_dir)
        # No warning key written
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "grid_intersection_fld_warning" not in data, (
            "Expected no fld warning key when fld delta < 1e-3"
        )


# ---------------------------------------------------------------------------
# Scenario (b): matching accuracy/degradation, fld delta 1e-2 (> 1e-3)
# Expected: no raise, warning key written into phase_a_summary.json
# ---------------------------------------------------------------------------

def test_scenario_b_no_raise_but_writes_warning():
    """fld_cos delta = 1e-2 (> 1e-3); accuracy/degradation match exactly."""
    r_pos2 = {"accuracy": 0.68, "degradation": 0.08, "fld_cos": 0.1500, "fld_l2": 4.0000}
    r_b8w4 = {"accuracy": 0.68, "degradation": 0.08, "fld_cos": 0.1600, "fld_l2": 4.0000}
    # fld_cos delta = 0.01 > 1e-3; fld_l2 delta = 0.0

    with tempfile.TemporaryDirectory() as run_dir:
        summary_path = _make_summary_json(run_dir)
        # Must not raise
        _check_grid_intersection(_make_rows(r_pos2, r_b8w4), run_dir)
        # Warning key must be present
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "grid_intersection_fld_warning" in data, (
            "Expected grid_intersection_fld_warning key when fld_cos delta > 1e-3"
        )
        warning = data["grid_intersection_fld_warning"]
        assert warning["tolerance"] == 1e-3
        assert "fld_cos" in warning["details"]
        assert abs(warning["details"]["fld_cos"]["delta"] - 0.01) < 1e-9
        # fld_l2 should NOT appear (delta == 0 < 1e-3)
        assert "fld_l2" not in warning["details"]


# ---------------------------------------------------------------------------
# Scenario (c): mismatched accuracy
# Expected: raises RuntimeError
# ---------------------------------------------------------------------------

def test_scenario_c_mismatched_accuracy_raises():
    """accuracy differs — must raise RuntimeError."""
    r_pos2 = {"accuracy": 0.72, "degradation": 0.04, "fld_cos": 0.12, "fld_l2": 3.5}
    r_b8w4 = {"accuracy": 0.70, "degradation": 0.06, "fld_cos": 0.12, "fld_l2": 3.5}
    # accuracy delta = 0.02, degradation delta = 0.02

    with tempfile.TemporaryDirectory() as run_dir:
        _make_summary_json(run_dir)
        with pytest.raises(RuntimeError, match="logic bug"):
            _check_grid_intersection(_make_rows(r_pos2, r_b8w4), run_dir)


# ---------------------------------------------------------------------------
# Scenario (d): only one intersection condition present
# Expected: no raise, no warning written (silent skip)
# ---------------------------------------------------------------------------

def test_scenario_d_only_one_condition_silent_skip():
    """Only fixed_w4_pos2 is present — check skips silently without raise or warning."""
    rows = [
        {"condition": "fixed_w4_pos2", "accuracy": 0.72, "degradation": 0.04,
         "fld_cos": 0.12, "fld_l2": 3.5},
        {"condition": "some_other_condition", "accuracy": 0.65, "degradation": 0.10,
         "fld_cos": 0.15, "fld_l2": 4.0},
    ]

    with tempfile.TemporaryDirectory() as run_dir:
        summary_path = _make_summary_json(run_dir)
        # Must not raise
        _check_grid_intersection(rows, run_dir)
        # No warning key written
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "grid_intersection_fld_warning" not in data, (
            "Expected no fld warning key when only one intersection condition is present"
        )
