"""Tests for condition-name enumeration in ``post_analysis.compute_bpd_sweep``.

Verifies that the Phase C-introduced ``_enumerate_conditions`` helper:
1. Preserves byte-identical order for legacy ``hard_swap_b`` / ``random_donor_b``
   prefixes.
2. Recognizes all five new naming families (``fixed_w4_``, ``fixed_b8_``,
   ``random_fixed_``, ``patch_``, ``corrupt_``).
3. Returns a superset of the pre-change key set so no Phase A / Stage 2
   downstream caller regresses.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ``post_analysis`` imports ``torch`` at module top; skip cleanly if absent.
pytest.importorskip("torch")
pytest.importorskip("scipy")

from stage1.analysis.post_analysis import (  # noqa: E402
    CONDITION_NAME_PREFIXES,
    _enumerate_conditions,
    _infer_b_for_condition,
)


def test_enumerate_conditions_hard_swap_and_random_donor_byte_identical_to_legacy():
    """Legacy-only input: order must exactly reproduce the pre-Phase-C loop.

    Pre-change loop:
        for b in boundary_grid:
            for prefix in ("hard_swap_b", "random_donor_b"):
                cond = f"{prefix}{b}"
    """
    boundary_grid = [4, 8]
    hs = {
        "no_swap": object(),
        "hard_swap_b4": object(),
        "random_donor_b4": object(),
        "hard_swap_b8": object(),
        "random_donor_b8": object(),
    }
    expected = [
        "hard_swap_b4", "random_donor_b4",
        "hard_swap_b8", "random_donor_b8",
    ]
    assert _enumerate_conditions(hs, boundary_grid) == expected


def test_enumerate_conditions_recognizes_fixed_w4():
    hs = {f"fixed_w4_pos{i}": object() for i in (1, 2, 3, 4)}
    out = _enumerate_conditions(hs, boundary_grid=[])
    # All four recognized, in sorted order.
    assert out == sorted(hs.keys())


def test_enumerate_conditions_recognizes_fixed_b8():
    hs = {f"fixed_b8_w{i}": object() for i in (2, 4, 8)}
    out = _enumerate_conditions(hs, boundary_grid=[])
    assert set(out) == set(hs.keys())
    assert out == sorted(hs.keys())


def test_enumerate_conditions_recognizes_random_fixed():
    hs = {"random_fixed_w4_pos1": object(), "random_fixed_b8_w4": object()}
    out = _enumerate_conditions(hs, boundary_grid=[])
    assert set(out) == set(hs.keys())


def test_enumerate_conditions_recognizes_patch_prefix():
    hs = {"patch_boundary_local": object(), "patch_recovery_full": object()}
    out = _enumerate_conditions(hs, boundary_grid=[])
    assert set(out) == set(hs.keys())


def test_enumerate_conditions_recognizes_corrupt_prefix():
    hs = {"corrupt_recovery_full": object()}
    out = _enumerate_conditions(hs, boundary_grid=[])
    assert out == ["corrupt_recovery_full"]


def test_enumerate_conditions_mixed_legacy_first_then_new_sorted():
    """Legacy prefixes are emitted FIRST (in grid order), new prefixes AFTER."""
    boundary_grid = [8]
    hs = {
        "no_swap": object(),
        "hard_swap_b8": object(),
        "random_donor_b8": object(),
        "patch_recovery_full": object(),
        "corrupt_recovery_full": object(),
        "fixed_b8_w4": object(),
    }
    out = _enumerate_conditions(hs, boundary_grid)
    # First two entries are legacy in pre-change order.
    assert out[:2] == ["hard_swap_b8", "random_donor_b8"]
    # Remaining entries are sorted alphabetically.
    assert out[2:] == sorted([
        "corrupt_recovery_full", "fixed_b8_w4", "patch_recovery_full",
    ])


def test_infer_b_for_condition_patch_and_corrupt():
    assert _infer_b_for_condition("patch_recovery_full", {"boundary_grid": [8]}) == 8
    assert _infer_b_for_condition("corrupt_recovery_full", {"boundary_grid": [8]}) == 8


def test_infer_b_for_condition_fixed_b_family():
    assert _infer_b_for_condition("fixed_b8_w4", {}) == 8
    assert _infer_b_for_condition("fixed_b12_w2", {}) == 12


def test_condition_name_prefixes_constant_contains_all_families():
    for p in (
        "hard_swap_b", "random_donor_b", "fixed_w4_",
        "fixed_b8_", "random_fixed_", "patch_", "corrupt_",
    ):
        assert p in CONDITION_NAME_PREFIXES
