"""Integration-level regression tests for the Phase B anchor gate.

YELLOW-LIGHT v3 PRIORITY 1.

Unlike the existing source-shape and predicate-only tests, these tests exercise
the production ``evaluate_phase_b_anchor_gate`` end-to-end against real on-disk
fake Phase A / Stage 1 run directories. Every test path goes through:

    - ``glob`` over a real temp directory tree
    - ``json.load`` of real ``phase_a_summary.json`` / ``evaluation.json``
    - ``json.load`` of real ``manifest.json`` files containing real parity blocks
    - ``check_manifest_parity`` filtering against a real current_parity dict
    - ``phase_a_condition_accuracy`` extraction with the same code path the
      production run uses
    - the actual missing/failed/passing decision logic in the production module

If a future refactor silently breaks any of these layers, these tests will fail
at the integration boundary, not at a copied-in-test predicate that may have
drifted from the real implementation.

The tests do NOT require torch — ``stage1.utils.anchor_gate`` is intentionally
torch-free so it is fully exercisable in any environment.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stage1.utils.anchor_gate import (
    PHASE_A_CROSS_CHECK_TOL,
    AnchorGateResult,
    evaluate_phase_b_anchor_gate,
    load_latest_phase_a_summary,
    load_latest_stage1_hard_swap_b8,
)


# ─── Fixtures: real on-disk fake run trees ───────────────────────────────────


def _canonical_parity() -> Dict[str, Any]:
    """The parity block both sides should agree on for a passing run."""
    return {
        "models": {
            "recipient": "Qwen/Qwen2.5-1.5B-Instruct",
            "donor": "Qwen/Qwen2.5-1.5B",
            "recipient_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "donor_revision": "8faed761d45a263340a0528343f099c05c9a4323",
        },
        "dataset": {"name": "mgsm", "lang": "zh", "split": "test"},
        "generation": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 512},
        "hidden_state": {"pooling": "last_token"},
    }


def _write_phase_a_run(
    base: str,
    run_id: str,
    *,
    parity: Optional[Dict[str, Any]],
    no_swap_acc: Optional[float] = 0.80,
    hard_swap_acc: Optional[float] = None,
    omit_manifest: bool = False,
    corrupt_summary: bool = False,
) -> str:
    """Create one fake Phase A run dir under ``base / "phase_a" / run_id``.

    ``parity`` may be set to None to emit a manifest with no parity block (which
    triggers the parity check to fail since required fields are missing).
    """
    run_dir = os.path.join(base, "phase_a", run_id)
    os.makedirs(run_dir, exist_ok=True)
    summary: Dict[str, Any] = {"phase": "A"}
    if not corrupt_summary:
        summary["baseline_accuracy"] = no_swap_acc
        all_conditions: List[Dict[str, Any]] = []
        if hard_swap_acc is not None:
            all_conditions.append(
                {"condition": "hard_swap_b8", "accuracy": hard_swap_acc}
            )
        if no_swap_acc is not None:
            all_conditions.append({"condition": "no_swap", "accuracy": no_swap_acc})
        summary["all_conditions"] = all_conditions
    summary_path = os.path.join(run_dir, "phase_a_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        if corrupt_summary:
            f.write("not valid json {")
        else:
            json.dump(summary, f)
    if not omit_manifest:
        manifest = {"phase": "A"}
        if parity is not None:
            manifest["parity"] = parity
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
    return run_dir


def _write_stage1_run(
    base: str,
    run_id: str,
    *,
    parity: Optional[Dict[str, Any]],
    hard_swap_acc: Optional[float] = 0.32,
    no_swap_acc: Optional[float] = 0.80,
    omit_manifest: bool = False,
) -> str:
    """Create one fake Stage 1 sweep run dir under ``base / run_id``."""
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=True)
    eval_dict: Dict[str, Any] = {"baseline_accuracy": no_swap_acc, "accuracies": {}}
    if hard_swap_acc is not None:
        eval_dict["accuracies"]["hard_swap_b8"] = {"accuracy": hard_swap_acc}
    if no_swap_acc is not None:
        eval_dict["accuracies"]["no_swap"] = {"accuracy": no_swap_acc}
    with open(os.path.join(run_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(eval_dict, f)
    if not omit_manifest:
        manifest = {"phase": "1"}
        if parity is not None:
            manifest["parity"] = parity
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
    return run_dir


@pytest.fixture
def empty_anchor_dirs(tmp_path):
    """Return (phase_a_dir, stage1_dir) for a pristine outputs tree."""
    base = tmp_path / "outputs"
    (base / "phase_a").mkdir(parents=True)
    return str(base / "phase_a"), str(base)


@pytest.fixture
def both_anchors_compatible(tmp_path):
    """Phase A has no_swap; Stage 1 has hard_swap_b8 + no_swap. All parity-OK."""
    base = tmp_path / "outputs"
    (base / "phase_a").mkdir(parents=True)
    parity = _canonical_parity()
    _write_phase_a_run(str(base), "run_20260420_000000",
                       parity=parity, no_swap_acc=0.80, hard_swap_acc=None)
    _write_stage1_run(str(base), "run_20260419_000000",
                      parity=parity, hard_swap_acc=0.32, no_swap_acc=0.80)
    return str(base / "phase_a"), str(base), parity


# ─── PRIORITY 1.A — missing hard_swap_b8 anchor → hard fail ──────────────────


def test_full_run_missing_hard_swap_anchor_hard_fails(empty_anchor_dirs):
    """No Stage 1 run + Phase A only has no_swap → hard_swap_b8 missing.

    Exercises the actual loader paths: phase_a_summary.json glob hit returns a
    summary without hard_swap_b8 row → ``phase_a_condition_accuracy`` returns
    None → no fallback Stage 1 run exists → anchor_hs is None → full-run gate
    appends "hard_swap_b8" to missing_anchors.
    """
    pa_dir, s1_dir = empty_anchor_dirs
    parity = _canonical_parity()
    _write_phase_a_run(os.path.dirname(pa_dir), "run_20260420_000000",
                       parity=parity, no_swap_acc=0.80, hard_swap_acc=None)

    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.80,
        sanity=False, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert isinstance(result, AnchorGateResult)
    assert result.passed is None
    assert "hard_swap_b8" in result.missing_anchors
    assert "no_swap" not in result.missing_anchors
    # Phase A no_swap was found and recorded.
    assert result.phase_a_no_swap_accuracy == 0.80
    assert result.anchor_no_swap_source == "phase_a"


# ─── PRIORITY 1.B — missing no_swap anchor → hard fail ───────────────────────


def test_full_run_missing_no_swap_anchor_hard_fails(empty_anchor_dirs):
    """Stage 1 has hard_swap_b8 but no no_swap; Phase A absent.

    Note: the Stage 1 evaluation.json fallback uses ``baseline_accuracy`` when
    accuracies.no_swap is absent. To produce a true "no no_swap" condition we
    drop both the row and the baseline_accuracy field.
    """
    pa_dir, s1_dir = empty_anchor_dirs
    parity = _canonical_parity()
    # Manually write a Stage 1 run with hard_swap_b8 but NO no_swap and NO baseline.
    run_dir = os.path.join(s1_dir, "run_20260419_000000")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"accuracies": {"hard_swap_b8": {"accuracy": 0.32}}}, f,
        )
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"parity": parity}, f)

    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.80,
        sanity=False, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert result.passed is None
    assert "no_swap" in result.missing_anchors
    assert "hard_swap_b8" not in result.missing_anchors
    assert result.anchor_hard_swap_source == "stage1"
    assert result.anchor_hard_swap_b8_accuracy == 0.32


# ─── PRIORITY 1.C — both anchors present and parity-compatible → pass ────────


def test_full_run_both_anchors_compatible_passes(both_anchors_compatible):
    """End-to-end: real loaders + real parity check + real decision → passed=True."""
    pa_dir, s1_dir, parity = both_anchors_compatible
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.80,
        sanity=False, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert result.passed is True
    assert not result.missing_anchors
    assert not result.failed_anchors
    assert result.anchor_hard_swap_source == "stage1"
    assert result.anchor_no_swap_source == "phase_a"
    assert result.anchor_hard_swap_b8_accuracy == 0.32
    assert result.anchor_no_swap_accuracy == 0.80


# ─── PRIORITY 1.D — anchor exists but parity-incompatible → hard fail ────────


def test_full_run_phase_a_parity_incompatible_anchor_rejected(empty_anchor_dirs):
    """Phase A run has wrong max_new_tokens — must be rejected by parity filter,
    so its no_swap accuracy MUST NOT propagate to anchor_no_swap.
    """
    pa_dir, s1_dir = empty_anchor_dirs
    correct = _canonical_parity()
    bad = _canonical_parity()
    bad["generation"]["max_new_tokens"] = 256  # mismatched!
    _write_phase_a_run(os.path.dirname(pa_dir), "run_20260420_000000",
                       parity=bad, no_swap_acc=0.80, hard_swap_acc=None)
    # No Stage 1 run.
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.80,
        sanity=False, current_parity=correct,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    # Phase A parity-rejected → no_swap unavailable → both anchors missing.
    assert result.passed is None
    assert "no_swap" in result.missing_anchors
    assert "hard_swap_b8" in result.missing_anchors
    assert result.phase_a_summary_path is None
    assert result.anchor_no_swap_source is None


def test_full_run_stage1_parity_incompatible_anchor_rejected(empty_anchor_dirs):
    """Stage 1 run with mismatched generation.max_new_tokens must be rejected.

    This is the exact footgun the workflow doc warns about: stage1_main.yaml
    defaults to max_new_tokens=256 while stage2_confound.yaml uses 512.
    Without the parity filter, the Stage 1 hard_swap_b8 would be silently used
    even though it was decoded under a different budget.
    """
    pa_dir, s1_dir = empty_anchor_dirs
    correct = _canonical_parity()
    bad = _canonical_parity()
    bad["generation"]["max_new_tokens"] = 256
    _write_stage1_run(s1_dir, "run_20260419_000000",
                      parity=bad, hard_swap_acc=0.32, no_swap_acc=0.80)
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.80,
        sanity=False, current_parity=correct,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert result.passed is None
    assert "hard_swap_b8" in result.missing_anchors
    assert "no_swap" in result.missing_anchors
    assert result.stage1_evaluation_path is None


def test_full_run_manifest_missing_anchor_rejected(empty_anchor_dirs):
    """A Phase A run without manifest.json must be rejected (cannot prove parity)."""
    pa_dir, s1_dir = empty_anchor_dirs
    parity = _canonical_parity()
    _write_phase_a_run(os.path.dirname(pa_dir), "run_20260420_000000",
                       parity=parity, no_swap_acc=0.80, omit_manifest=True)
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.80,
        sanity=False, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert result.passed is None
    # Both rejected: no_swap (Phase A had it but manifest missing) AND
    # hard_swap_b8 (no Stage 1 run).
    assert "no_swap" in result.missing_anchors
    assert "hard_swap_b8" in result.missing_anchors


# ─── PRIORITY 1.E — both present + within tolerance → pass; outside → fail ───


def test_full_run_both_present_within_tolerance_passes(both_anchors_compatible):
    pa_dir, s1_dir, parity = both_anchors_compatible
    # Anchors are 0.32 / 0.80; offer measurements within 0.008 of each.
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.324, clean_baseline_acc=0.802,
        sanity=False, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert result.passed is True


def test_full_run_hard_swap_outside_tolerance_fails(both_anchors_compatible):
    pa_dir, s1_dir, parity = both_anchors_compatible
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.50, clean_baseline_acc=0.80,
        sanity=False, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert result.passed is False
    assert any("hard_swap_b8" in f for f in result.failed_anchors)


def test_full_run_no_swap_outside_tolerance_fails(both_anchors_compatible):
    pa_dir, s1_dir, parity = both_anchors_compatible
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.50,
        sanity=False, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert result.passed is False
    assert any("no_swap" in f for f in result.failed_anchors)


# ─── PRIORITY 1.F — newest-run preference + skipping corrupt summaries ───────


def test_loader_prefers_newer_run_dir(both_anchors_compatible):
    """Two Phase A runs: newer is parity-compatible, older is also compatible.

    The newest must win because ``sorted(..., reverse=True)`` puts the
    timestamped dir name at the top. This regression-protects the precedence.
    """
    pa_dir, s1_dir, parity = both_anchors_compatible
    base = os.path.dirname(pa_dir)
    # Add an older Phase A run with a different no_swap accuracy.
    _write_phase_a_run(base, "run_20260101_000000",
                       parity=parity, no_swap_acc=0.50, hard_swap_acc=None)
    summary, path = load_latest_phase_a_summary(pa_dir, parity)
    assert summary is not None
    # The newer (run_20260420_000000) must be picked.
    assert "20260420" in os.path.basename(os.path.dirname(path))


def test_loader_skips_corrupt_summary_falls_back_to_older(empty_anchor_dirs):
    """If newest summary is unparseable, loader skips and tries older."""
    pa_dir, s1_dir = empty_anchor_dirs
    parity = _canonical_parity()
    base = os.path.dirname(pa_dir)
    _write_phase_a_run(base, "run_20260101_000000",
                       parity=parity, no_swap_acc=0.77, hard_swap_acc=None)
    _write_phase_a_run(base, "run_20260420_000000",
                       parity=parity, corrupt_summary=True)
    summary, path = load_latest_phase_a_summary(pa_dir, parity)
    assert summary is not None
    assert "20260101" in os.path.basename(os.path.dirname(path))


# ─── PRIORITY 1.G — sanity mode is genuinely relaxed ─────────────────────────


def test_sanity_mode_passes_with_no_anchors(empty_anchor_dirs):
    """Sanity mode + no anchors → passed=None (treated by callers as 'skipped')."""
    pa_dir, s1_dir = empty_anchor_dirs
    parity = _canonical_parity()
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.80,
        sanity=True, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert result.passed is None
    assert not result.missing_anchors  # sanity mode does not record missing


def test_sanity_mode_passes_with_only_one_anchor(empty_anchor_dirs):
    """Sanity mode + only no_swap available → passed=True."""
    pa_dir, s1_dir = empty_anchor_dirs
    parity = _canonical_parity()
    base = os.path.dirname(pa_dir)
    _write_phase_a_run(base, "run_20260420_000000",
                       parity=parity, no_swap_acc=0.80, hard_swap_acc=None)
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.80,
        sanity=True, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    assert result.passed is True


# ─── PRIORITY 1.H — to_summary_dict serialises faithfully ────────────────────


def test_summary_dict_contains_required_fields(both_anchors_compatible):
    pa_dir, s1_dir, parity = both_anchors_compatible
    result = evaluate_phase_b_anchor_gate(
        no_patch_acc=0.32, clean_baseline_acc=0.80,
        sanity=False, current_parity=parity,
        phase_a_dir=pa_dir, stage1_dir=s1_dir,
    )
    d = result.to_summary_dict(phase_a_outputs_dir=pa_dir)
    for key in (
        "phase_a_summary_path", "phase_a_outputs_dir",
        "phase_a_hard_swap_b8_accuracy", "phase_a_no_swap_accuracy",
        "stage1_evaluation_path", "stage1_hard_swap_b8_accuracy",
        "stage1_no_swap_accuracy",
        "anchor_hard_swap_b8_accuracy", "anchor_hard_swap_source",
        "anchor_no_swap_accuracy", "anchor_no_swap_source",
        "tolerance", "passed", "missing_anchors", "failed_anchors", "note",
    ):
        assert key in d, f"missing key in summary dict: {key}"


# ─── PRIORITY 1.I — production loaders exposed and usable ────────────────────


def test_load_latest_phase_a_summary_returns_none_when_empty(empty_anchor_dirs):
    pa_dir, _ = empty_anchor_dirs
    summary, path = load_latest_phase_a_summary(pa_dir, _canonical_parity())
    assert summary is None and path is None


def test_load_latest_stage1_returns_none_when_empty(empty_anchor_dirs):
    _, s1_dir = empty_anchor_dirs
    hs, ns, p = load_latest_stage1_hard_swap_b8(s1_dir, _canonical_parity())
    assert hs is None and ns is None and p is None
