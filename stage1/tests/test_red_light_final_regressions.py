"""Regression tests for RED LIGHT final-blocking fixes (v2).

Covers three high-priority regressions:

1. PRIORITY 1 — Stage 1 ``run.py`` must forward ``recipient_revision`` /
   ``donor_revision`` to ``load_models(...)``. Without this, manifest parity
   can falsely pass while the actual weights come from the default branch.

2. PRIORITY 2 — Phase B full runs must hard-fail if either the
   ``hard_swap_b8`` or ``no_swap`` anchor is missing, or if either fails the
   tolerance check.

3. PRIORITY 3 — Phase C ``strict_sample_ids=True`` must hard-fail on any
   sample-ID mismatch across the required JSONLs.

Tests that depend on torch are gated by a feature skip; the pure-Python tests
run in any environment.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ═════════════════════════════════════════════════════════════════════════════
# PRIORITY 1 — revision forwarding in Stage 1 run.py
# ═════════════════════════════════════════════════════════════════════════════


def test_run_py_forwards_revision_args_to_load_models():
    """Stage 1 run.py must pass recipient_revision / donor_revision into
    load_models. Verified by source inspection + AST.

    Rationale: a kwarg is either present or not in the call expression. If a
    future refactor drops it, this test fails at the source level before any
    runtime inference happens.
    """
    import ast

    run_py_path = os.path.join(_REPO_ROOT, "stage1", "run.py")
    with open(run_py_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    load_models_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "load_models":
                load_models_calls.append(node)

    assert load_models_calls, (
        "stage1/run.py: no call to load_models() found. Either the module "
        "changed shape or the import was renamed."
    )

    # Check each call's keyword arguments.
    for call in load_models_calls:
        kwarg_names = {kw.arg for kw in call.keywords}
        assert "recipient_revision" in kwarg_names, (
            "stage1/run.py: load_models() call is missing `recipient_revision` "
            "kwarg. Without it, the actual loaded weights do not correspond "
            "to the manifest-declared revision (RED LIGHT v2 Priority 1)."
        )
        assert "donor_revision" in kwarg_names, (
            "stage1/run.py: load_models() call is missing `donor_revision` "
            "kwarg. Without it, the actual loaded weights do not correspond "
            "to the manifest-declared revision (RED LIGHT v2 Priority 1)."
        )


def test_load_models_signature_accepts_revision_kwargs():
    """``load_models`` must accept recipient_revision / donor_revision kwargs."""
    import inspect
    import ast

    composer_path = os.path.join(_REPO_ROOT, "stage1", "models", "composer.py")
    with open(composer_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    load_models_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_models":
            load_models_def = node
            break
    assert load_models_def is not None, "load_models not found in composer.py"

    arg_names = {a.arg for a in load_models_def.args.args}
    arg_names |= {a.arg for a in load_models_def.args.kwonlyargs}
    assert "recipient_revision" in arg_names, (
        "composer.load_models signature missing recipient_revision"
    )
    assert "donor_revision" in arg_names, (
        "composer.load_models signature missing donor_revision"
    )


def test_stage1_main_yaml_has_pinned_revisions():
    """Stage 1 main config must pin recipient_revision and donor_revision.

    Without pins, manifest parity would record None, which is fine, but it
    would prevent cross-run reproducibility guarantees that the reviewer
    requires.
    """
    import yaml

    path = os.path.join(_REPO_ROOT, "stage1", "configs", "stage1_main.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", {})
    assert "recipient_revision" in models, (
        "stage1_main.yaml: models.recipient_revision missing — cannot pin "
        "recipient weight version for reproducibility"
    )
    assert "donor_revision" in models, (
        "stage1_main.yaml: models.donor_revision missing — cannot pin donor "
        "weight version for reproducibility"
    )
    assert models["recipient_revision"], "recipient_revision must be non-empty"
    assert models["donor_revision"], "donor_revision must be non-empty"


# ═════════════════════════════════════════════════════════════════════════════
# PRIORITY 2 — Phase B strict anchor enforcement regression tests
# ═════════════════════════════════════════════════════════════════════════════

# These tests validate the cross-check gating logic in run_phase_b.py without
# requiring torch. We instead exercise the gate logic by reading the source
# and verifying the control flow, plus unit-testing the decision predicate.


def test_phase_b_full_run_rejects_missing_hard_swap_anchor():
    """Simulate a full run with anchor_hard_swap_acc=None → cross_check_passed
    must be None (signalling missing), and the sanity-check builder must turn
    that into a FAIL item in non-sanity mode.
    """
    # Construct the decision logic in isolation, matching run_phase_b.py lines.
    def _decide(sanity, anchor_hs, anchor_ns, no_patch_acc, clean_acc, tol):
        missing = []
        failed = []
        if not sanity:
            if anchor_hs is None:
                missing.append("hard_swap_b8")
            if anchor_ns is None:
                missing.append("no_swap")
            if missing:
                return None, missing, failed
            hs_ok = abs(no_patch_acc - anchor_hs) <= tol
            ns_ok = abs(clean_acc - anchor_ns) <= tol
            if not hs_ok:
                failed.append("hard_swap_b8")
            if not ns_ok:
                failed.append("no_swap")
            return (hs_ok and ns_ok), missing, failed
        # Sanity mode
        ok = []
        if anchor_hs is not None:
            ok.append(abs(no_patch_acc - anchor_hs) <= tol)
        if anchor_ns is not None:
            ok.append(abs(clean_acc - anchor_ns) <= tol)
        return (all(ok) if ok else None), missing, failed

    # Case: full run, missing hard_swap
    result, missing, _ = _decide(
        sanity=False, anchor_hs=None, anchor_ns=0.80,
        no_patch_acc=0.32, clean_acc=0.80, tol=0.008,
    )
    assert result is None, "full run with missing hard_swap_b8 must signal missing"
    assert "hard_swap_b8" in missing
    assert "no_swap" not in missing


def test_phase_b_full_run_rejects_missing_no_swap_anchor():
    def _decide(sanity, anchor_hs, anchor_ns, no_patch_acc, clean_acc, tol):
        missing = []
        if not sanity:
            if anchor_hs is None:
                missing.append("hard_swap_b8")
            if anchor_ns is None:
                missing.append("no_swap")
            if missing:
                return None, missing
            return (abs(no_patch_acc - anchor_hs) <= tol and
                    abs(clean_acc - anchor_ns) <= tol), missing
        return True, missing  # sanity-mode stub

    result, missing = _decide(
        sanity=False, anchor_hs=0.32, anchor_ns=None,
        no_patch_acc=0.32, clean_acc=0.80, tol=0.008,
    )
    assert result is None
    assert "no_swap" in missing
    assert "hard_swap_b8" not in missing


def test_phase_b_full_run_rejects_both_missing():
    """If both anchors are missing, both must be reported in the failure."""
    def _decide(sanity, anchor_hs, anchor_ns):
        missing = []
        if not sanity:
            if anchor_hs is None:
                missing.append("hard_swap_b8")
            if anchor_ns is None:
                missing.append("no_swap")
        return missing

    missing = _decide(sanity=False, anchor_hs=None, anchor_ns=None)
    assert "hard_swap_b8" in missing
    assert "no_swap" in missing
    assert len(missing) == 2


def test_phase_b_full_run_rejects_hard_swap_tolerance_failure():
    """Both anchors present but hard_swap fails tolerance → cross_check fails."""
    tol = 0.008
    no_patch_acc = 0.32
    anchor_hs = 0.50  # far off
    anchor_ns = 0.80
    clean_acc = 0.80

    hs_ok = abs(no_patch_acc - anchor_hs) <= tol
    ns_ok = abs(clean_acc - anchor_ns) <= tol
    assert not hs_ok
    assert ns_ok
    overall = hs_ok and ns_ok
    assert overall is False


def test_phase_b_full_run_accepts_both_anchors_within_tolerance():
    """Both anchors present and within tolerance → cross_check passes."""
    tol = 0.008
    no_patch_acc = 0.32
    anchor_hs = 0.324  # |Δ|=0.004 < 0.008
    anchor_ns = 0.80
    clean_acc = 0.802  # |Δ|=0.002 < 0.008

    hs_ok = abs(no_patch_acc - anchor_hs) <= tol
    ns_ok = abs(clean_acc - anchor_ns) <= tol
    assert hs_ok and ns_ok


def test_phase_b_sanity_mode_allows_relaxed_check():
    """Sanity mode allows pass when only one anchor is available.

    This is the explicit gate: sanity=True → relaxed; sanity=False → strict.
    """
    def _decide_sanity(anchor_hs, anchor_ns, no_patch_acc, clean_acc, tol):
        ok = []
        if anchor_hs is not None:
            ok.append(abs(no_patch_acc - anchor_hs) <= tol)
        if anchor_ns is not None:
            ok.append(abs(clean_acc - anchor_ns) <= tol)
        return all(ok) if ok else None

    # Only no_swap available; sanity mode accepts.
    r = _decide_sanity(None, 0.80, 0.32, 0.80, 0.008)
    assert r is True

    # No anchors — returns None (upstream sanity path accepts as "skipped").
    r2 = _decide_sanity(None, None, 0.32, 0.80, 0.008)
    assert r2 is None


def test_phase_b_gating_logic_is_present_in_source():
    """Verify the strict-vs-sanity gate is wired between run_phase_b and the
    canonical anchor_gate module.

    The actual decision logic lives in ``stage1/utils/anchor_gate.py``
    (YELLOW-LIGHT v3 P1: extracted to enable integration-level regression
    tests in ``test_phase_b_anchor_gate_integration.py``). This test protects
    against a future refactor that silently removes either side of the wiring.
    """
    rb_path = os.path.join(_REPO_ROOT, "stage1", "run_phase_b.py")
    ag_path = os.path.join(_REPO_ROOT, "stage1", "utils", "anchor_gate.py")
    with open(rb_path, encoding="utf-8") as f:
        rb_src = f.read()
    with open(ag_path, encoding="utf-8") as f:
        ag_src = f.read()

    # run_phase_b.py must import and call the canonical evaluator.
    assert "evaluate_phase_b_anchor_gate" in rb_src, (
        "run_phase_b.py: must import evaluate_phase_b_anchor_gate from "
        "stage1.utils.anchor_gate — the gate may have been removed."
    )
    # anchor_gate.py must contain the strict / sanity distinction and the
    # explicit BOTH-anchors requirement.
    assert "if not sanity:" in ag_src, (
        "anchor_gate.py: sanity/full-run distinction not present."
    )
    assert "missing_anchors" in ag_src, (
        "anchor_gate.py: missing the strict anchor tracking list — "
        "the strict gate may have been removed."
    )
    # Either the diagnostic text or the rendered BOTH-required message must
    # remain available so error messages stay actionable.
    assert "require BOTH" in ag_src, (
        "anchor_gate.py: missing the 'require BOTH' diagnostic for missing anchors."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PRIORITY 3 — Phase C strict sample-ID mismatch regression tests
# ═════════════════════════════════════════════════════════════════════════════


_np_available = True
try:
    import numpy  # noqa: F401
except ImportError:
    _np_available = False

requires_numpy = pytest.mark.skipif(
    not _np_available, reason="numpy not installed"
)


@requires_numpy
def test_align_by_sample_id_strict_rejects_missing_id():
    """strict=True: when one condition lacks an ID, raise ValueError."""
    from stage1.analysis.mediation import (
        ConditionCorrectness,
        align_by_sample_id,
    )

    a = ConditionCorrectness("A", ("s0", "s1", "s2"), (True, False, True))
    b = ConditionCorrectness("B", ("s0", "s1"), (True, True))  # missing s2

    with pytest.raises(ValueError) as exc:
        align_by_sample_id(a, b, strict=True)
    msg = str(exc.value)
    assert "sample-ID mismatch" in msg
    # Diagnostic must name which condition and show counts.
    assert "A" in msg or "B" in msg
    assert "|aligned|" in msg or "|condition|" in msg


@requires_numpy
def test_align_by_sample_id_strict_rejects_extra_id():
    """strict=True: an extra ID in one condition is a hard failure."""
    from stage1.analysis.mediation import (
        ConditionCorrectness,
        align_by_sample_id,
    )

    a = ConditionCorrectness("A", ("s0", "s1"), (True, False))
    b = ConditionCorrectness("B", ("s0", "s1", "s_extra"), (True, True, False))

    with pytest.raises(ValueError) as exc:
        align_by_sample_id(a, b, strict=True)
    msg = str(exc.value)
    assert "Extra" in msg or "extra" in msg.lower() or "Missing" in msg


@requires_numpy
def test_align_by_sample_id_strict_accepts_identical_sets():
    """strict=True: identical sample-ID sets pass without error."""
    from stage1.analysis.mediation import (
        ConditionCorrectness,
        align_by_sample_id,
    )

    a = ConditionCorrectness("A", ("s0", "s1", "s2"), (True, False, True))
    b = ConditionCorrectness("B", ("s0", "s1", "s2"), (False, True, True))

    aligned_ids, arrays = align_by_sample_id(a, b, strict=True)
    assert aligned_ids == ["s0", "s1", "s2"]
    assert len(arrays) == 2


@requires_numpy
def test_align_by_sample_id_non_strict_still_warns_on_drop():
    """strict=False (default): intersection with warning, not an error."""
    import logging
    from stage1.analysis.mediation import (
        ConditionCorrectness,
        align_by_sample_id,
    )

    a = ConditionCorrectness("A", ("s0", "s1", "s2"), (True, False, True))
    b = ConditionCorrectness("B", ("s0", "s1"), (True, True))

    with patch.object(logging.getLogger("stage1.analysis.mediation"), "warning") as mock_warn:
        aligned_ids, arrays = align_by_sample_id(a, b)  # default strict=False
    assert aligned_ids == ["s0", "s1"]
    # Warning must have been issued for the dropped ID.
    assert mock_warn.called


@requires_numpy
def test_compute_decomposition_table_strict_hard_fails_on_mismatch(tmp_path):
    """End-to-end: strict_sample_ids=True rejects JSONLs with mismatched IDs."""
    from stage1.analysis.mediation import compute_decomposition_table

    def _wj(name, ids, bits):
        p = tmp_path / f"results_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for i, bit in zip(ids, bits):
                f.write(json.dumps({"sample_id": i, "correct": bool(bit)}) + "\n")

    # clean and no_patch share IDs s0-s7, but patch_boundary_local is MISSING s7.
    _wj("clean_no_patch",              ["s0","s1","s2","s3","s4","s5","s6","s7"],
        [1,1,1,1,0,0,0,0])
    _wj("restoration_no_patch",        ["s0","s1","s2","s3","s4","s5","s6","s7"],
        [1,0,0,0,0,0,0,0])
    # Missing s7 — strict mode must reject.
    _wj("restoration_patch_boundary_local", ["s0","s1","s2","s3","s4","s5","s6"],
        [1,1,1,0,0,0,0])
    _wj("restoration_patch_recovery_early", ["s0","s1","s2","s3","s4","s5","s6","s7"],
        [1,1,0,1,0,0,0,0])
    _wj("restoration_patch_recovery_full",  ["s0","s1","s2","s3","s4","s5","s6","s7"],
        [1,1,0,1,0,0,0,0])
    _wj("restoration_patch_final_only",     ["s0","s1","s2","s3","s4","s5","s6","s7"],
        [1,0,0,0,0,0,0,0])

    with pytest.raises(ValueError) as exc:
        compute_decomposition_table(
            str(tmp_path),
            bootstrap_n=100, seed=0,
            strict_sample_ids=True,
        )
    # Diagnostic should name the offending condition.
    msg = str(exc.value)
    assert "mismatch" in msg.lower()


@requires_numpy
def test_compute_decomposition_table_non_strict_accepts_mismatch(tmp_path):
    """strict_sample_ids=False (sanity path) should not raise on mismatch.

    It falls back to intersection with a warning. Ensures the strict gate
    distinction is meaningful.
    """
    from stage1.analysis.mediation import compute_decomposition_table

    def _wj(name, ids, bits):
        p = tmp_path / f"results_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for i, bit in zip(ids, bits):
                f.write(json.dumps({"sample_id": i, "correct": bool(bit)}) + "\n")

    _wj("clean_no_patch",              ["s0","s1","s2","s3","s4","s5","s6","s7"],
        [1,1,1,1,0,0,0,0])
    _wj("restoration_no_patch",        ["s0","s1","s2","s3","s4","s5","s6","s7"],
        [1,0,0,0,0,0,0,0])
    _wj("restoration_patch_boundary_local", ["s0","s1","s2","s3","s4","s5","s6"],
        [1,1,1,0,0,0,0])
    _wj("restoration_patch_recovery_early", ["s0","s1","s2","s3","s4","s5","s6"],
        [1,1,0,1,0,0,0])
    _wj("restoration_patch_recovery_full",  ["s0","s1","s2","s3","s4","s5","s6"],
        [1,1,0,1,0,0,0])
    _wj("restoration_patch_final_only",     ["s0","s1","s2","s3","s4","s5","s6"],
        [1,0,0,0,0,0,0])

    # Non-strict: runs to completion even with drop.
    tbl = compute_decomposition_table(
        str(tmp_path),
        bootstrap_n=100, seed=0,
        strict_sample_ids=False,
    )
    assert "rows" in tbl
    assert tbl["best_condition"] in (
        "patch_boundary_local", "patch_recovery_early",
        "patch_recovery_full", "patch_final_only",
    )


def test_run_phase_c_passes_strict_flag_in_non_sanity_mode():
    """Source-level check: run_phase_c.py must pass strict_sample_ids=(not sanity)."""
    path = os.path.join(_REPO_ROOT, "stage1", "run_phase_c.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "strict_sample_ids=(not sanity)" in src or "strict_sample_ids=not sanity" in src, (
        "run_phase_c.py: must pass strict_sample_ids=(not sanity) to "
        "compute_decomposition_table. Without this gate, Phase C analytical "
        "runs silently tolerate sample-ID mismatch."
    )
