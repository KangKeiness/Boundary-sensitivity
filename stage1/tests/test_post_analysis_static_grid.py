"""Regression tests for v4 PRIORITY 3 — fixed_w4_* / random_fixed_w4_*
boundary inference must NOT depend on optional heavy imports.

Background:
    ``post_analysis._infer_b_for_condition`` previously imported
    ``stage1.models.composer.PHASE_A_GRID`` at call time. ``composer.py``
    imports transformers at module load, so the inference path silently broke
    in any analysis environment lacking the heavy import (and its torch /
    transformers dependency chain).

v4 P3 fixes this by inlining a static grid mirror, ``_STATIC_PHASE_A_GRID``,
that is the *primary* source. Composer is consulted only as an upgrade-path
drift check.

These tests cover:
    1. The static grid produces the correct boundary for every fixed_w4 /
       random_fixed_w4 condition WITHOUT triggering the composer import path.
    2. The static grid stays in sync with composer.PHASE_A_GRID (skipped
       gracefully if composer can't be imported in the current environment).
    3. Drift is loud, not silent.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ─── Static-grid lookups (no heavy import required) ─────────────────────────


@pytest.mark.parametrize("cond, expected_b", [
    ("fixed_w4_pos1", 4),
    ("fixed_w4_pos2", 8),
    ("fixed_w4_pos3", 12),
    ("fixed_w4_pos4", 16),
    ("random_fixed_w4_pos1", 4),
    ("random_fixed_w4_pos2", 8),
    ("random_fixed_w4_pos3", 12),
    ("random_fixed_w4_pos4", 16),
    ("fixed_b8_w2", 8),
    ("fixed_b8_w4", 8),
    ("fixed_b8_w6", 8),
    ("fixed_b8_w8", 8),
    ("random_fixed_b8_w2", 8),
    ("random_fixed_b8_w4", 8),
    ("random_fixed_b8_w6", 8),
    ("random_fixed_b8_w8", 8),
])
def test_infer_b_uses_static_grid(cond, expected_b):
    """Static grid must yield the correct b without raising."""
    from stage1.analysis.post_analysis import _infer_b_for_condition
    assert _infer_b_for_condition(cond, run_data={}) == expected_b


def test_infer_b_works_when_composer_unimportable(monkeypatch):
    """Simulate a torch/transformers-less environment: composer import raises.

    Pre-v4 behavior: fixed_w4_* raised ValueError. v4 behavior: static grid
    answers correctly without ever needing composer.
    """
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "stage1.models.composer" or name.endswith(".composer"):
            raise ImportError("simulated heavy-import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from stage1.analysis.post_analysis import _infer_b_for_condition
    # All four fixed_w4 positions must succeed even with composer unimportable.
    assert _infer_b_for_condition("fixed_w4_pos1", run_data={}) == 4
    assert _infer_b_for_condition("fixed_w4_pos2", run_data={}) == 8
    assert _infer_b_for_condition("fixed_w4_pos3", run_data={}) == 12
    assert _infer_b_for_condition("fixed_w4_pos4", run_data={}) == 16
    assert _infer_b_for_condition("random_fixed_w4_pos1", run_data={}) == 4


def test_unknown_fixed_w_condition_still_raises():
    """Explicit failure when a new fixed_w family is added without updating
    the static table — better than silently returning a wrong value."""
    from stage1.analysis.post_analysis import _infer_b_for_condition
    with pytest.raises(ValueError) as exc:
        _infer_b_for_condition("fixed_w99_unknown", run_data={})
    assert "_STATIC_PHASE_A_GRID" in str(exc.value)


# ─── Drift detection ─────────────────────────────────────────────────────────


def test_static_grid_matches_composer_grid_when_importable():
    """If composer imports successfully, both tables must agree byte-for-byte.

    Skipped in environments without transformers (per project memory: torch is
    installed but transformers may not be). The drift-detection inside
    ``_infer_b_for_condition`` provides additional protection at call time.
    """
    try:
        from stage1.models.composer import PHASE_A_GRID
    except Exception as exc:  # pragma: no cover — env-dependent
        pytest.skip(f"composer not importable in this env: {exc!r}")
    from stage1.analysis.post_analysis import _STATIC_PHASE_A_GRID
    assert _STATIC_PHASE_A_GRID == PHASE_A_GRID, (
        "static and composer Phase A grids have drifted; re-sync the two."
    )


def test_inferred_drift_is_loud_not_silent(monkeypatch):
    """If the static grid says (4, 8) but composer says (5, 9), the second
    call to ``_infer_b_for_condition`` must hard-fail with a clear message,
    not silently disagree."""
    from stage1.analysis import post_analysis
    # Patch the static grid to a known-different value.
    fake_static = dict(post_analysis._STATIC_PHASE_A_GRID)
    fake_static["fixed_w4_pos1"] = (5, 9)
    monkeypatch.setattr(post_analysis, "_STATIC_PHASE_A_GRID", fake_static)
    # Inject a fake composer module with the *original* (4, 8).
    import sys
    import types
    fake_mod = types.ModuleType("stage1.models.composer")
    fake_mod.PHASE_A_GRID = {"fixed_w4_pos1": (4, 8)}
    monkeypatch.setitem(sys.modules, "stage1.models.composer", fake_mod)
    with pytest.raises(ValueError) as exc:
        post_analysis._infer_b_for_condition("fixed_w4_pos1", run_data={})
    assert "drift" in str(exc.value).lower()
