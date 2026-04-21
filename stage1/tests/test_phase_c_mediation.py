"""Unit tests for Phase C mediation-style decomposition.

Covers:
- JSONL loader (happy path, duplicate-id error, missing-field error)
- sample_id alignment with drops + warning
- restoration_effect / residual_effect deterministic bootstrap CI
- restoration_proportion null / unstable branches
- compute_decomposition_table best-condition tie-break (alphabetical)
- Forbidden-phrases gate over Phase C vocabulary
- End-to-end CLI against the synthetic fixture directory
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from stage1.analysis.mediation import (
    CLAIM_ELIGIBLE_CONDITIONS,
    EPSILON_DENOM,
    ConditionCorrectness,
    align_by_sample_id,
    compute_decomposition_table,
    load_condition_correctness,
    residual_effect,
    restoration_effect,
    restoration_proportion,
)
from stage1.utils.wording import (
    FORBIDDEN_PHRASES_PHASE_C,
    check_artifacts_for_forbidden,
)


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "phase_b_run_fixture",
)


# ─── Spec-literal header constant (standalone; no fixture needed) ───────────


def test_phase_c_txt_header_byte_equals_spec_literal():
    """Spec §11.10 mandates the exact em-dash header string."""
    from stage1.run_phase_c import PHASE_C_TXT_HEADER
    expected = (
        "Phase C \u2014 mediation-style decomposition of prompt-side "
        "restoration intervention (not a formal NIE/NDE decomposition)"
    )
    assert PHASE_C_TXT_HEADER == expected, (
        f"PHASE_C_TXT_HEADER drift vs spec §11.10:\n"
        f"  got     : {PHASE_C_TXT_HEADER!r}\n"
        f"  expected: {expected!r}"
    )
    # Em-dash (U+2014), not ASCII hyphen-minus.
    assert "\u2014" in PHASE_C_TXT_HEADER
    assert "Phase C - " not in PHASE_C_TXT_HEADER


# ─── Loader ─────────────────────────────────────────────────────────────────


def test_load_condition_correctness_ok(tmp_path):
    p = tmp_path / "results_clean_no_patch.jsonl"
    p.write_text(
        '{"sample_id": "s0", "correct": true}\n'
        '{"sample_id": "s1", "correct": false}\n'
        '{"sample_id": "s2", "correct": true}\n',
        encoding="utf-8",
    )
    c = load_condition_correctness(str(p))
    assert c.condition == "clean_no_patch"
    assert c.sample_ids == ("s0", "s1", "s2")
    assert c.correct == (True, False, True)


def test_load_condition_correctness_duplicate_sample_id_raises(tmp_path):
    p = tmp_path / "results_clean_no_patch.jsonl"
    p.write_text(
        '{"sample_id": "s0", "correct": true}\n'
        '{"sample_id": "s0", "correct": false}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_condition_correctness(str(p))


def test_load_condition_correctness_missing_field_raises(tmp_path):
    p = tmp_path / "results_clean_no_patch.jsonl"
    p.write_text('{"sample_id": "s0"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="'correct'"):
        load_condition_correctness(str(p))


def test_load_condition_correctness_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_condition_correctness(str(tmp_path / "results_nope.jsonl"))


# ─── Alignment ──────────────────────────────────────────────────────────────


def test_align_by_sample_id_drops_and_logs(caplog):
    a = ConditionCorrectness("A", ("a", "b", "c"), (True, False, True))
    b = ConditionCorrectness("B", ("b", "c", "d"), (True, True, False))
    with caplog.at_level(logging.WARNING):
        aligned_ids, arrays = align_by_sample_id(a, b)
    assert aligned_ids == ["b", "c"]
    assert len(arrays) == 2
    assert arrays[0].tolist() == [0, 1]
    assert arrays[1].tolist() == [1, 1]
    # Both conditions dropped exactly one sample_id each.
    assert "dropped" in caplog.text.lower()


# ─── Deterministic point + CI ───────────────────────────────────────────────


def test_restoration_effect_point_and_ci_deterministic():
    # Pinned 8-sample patterns.
    # patched = 0b11100000, no_patch = 0b10000000
    patched = ConditionCorrectness(
        "patch_recovery_full",
        tuple(f"s{i}" for i in range(8)),
        (True, True, True, False, False, False, False, False),
    )
    no_patch = ConditionCorrectness(
        "no_patch",
        tuple(f"s{i}" for i in range(8)),
        (True, False, False, False, False, False, False, False),
    )
    r1 = restoration_effect(patched, no_patch, bootstrap_n=1000, seed=0, ci=0.95)
    r2 = restoration_effect(patched, no_patch, bootstrap_n=1000, seed=0, ci=0.95)
    assert r1 == r2
    assert abs(r1["point"] - 0.25) < 1e-9
    assert r1["n_aligned"] == 8
    # CI bounds are finite.
    assert np.isfinite(r1["ci_lo"])
    assert np.isfinite(r1["ci_hi"])
    # Sanity: CI contains the point.
    assert r1["ci_lo"] <= r1["point"] <= r1["ci_hi"] + 1e-9


def test_residual_effect_point_and_ci_deterministic():
    clean = ConditionCorrectness(
        "clean_no_patch",
        tuple(f"s{i}" for i in range(8)),
        (True, True, True, True, False, False, False, False),
    )
    best = ConditionCorrectness(
        "patch_recovery_full",
        tuple(f"s{i}" for i in range(8)),
        (True, True, True, False, False, False, False, False),
    )
    r1 = residual_effect(clean, best, bootstrap_n=1000, seed=0, ci=0.95)
    r2 = residual_effect(clean, best, bootstrap_n=1000, seed=0, ci=0.95)
    assert r1 == r2
    assert abs(r1["point"] - 0.125) < 1e-9


# ─── Proportion branches ────────────────────────────────────────────────────


def test_restoration_proportion_null_when_denom_below_epsilon():
    # acc_clean - acc_no_patch = 0.002 < 0.005. Use 500 samples to get 1/500=0.002.
    n = 500
    clean_bits = [True] + [False] * (n - 1)      # acc = 0.002
    no_patch_bits = [False] * n                   # acc = 0.0
    best_bits = [True] * n                        # acc = 1.0 (irrelevant)
    clean = ConditionCorrectness(
        "clean_no_patch",
        tuple(f"s{i}" for i in range(n)),
        tuple(clean_bits),
    )
    no_patch = ConditionCorrectness(
        "no_patch",
        tuple(f"s{i}" for i in range(n)),
        tuple(no_patch_bits),
    )
    best = ConditionCorrectness(
        "patch_recovery_full",
        tuple(f"s{i}" for i in range(n)),
        tuple(best_bits),
    )
    res = restoration_proportion(clean, no_patch, best, bootstrap_n=200, seed=0)
    assert res["point"] is None
    assert res["ci_reason"] == "denominator_below_epsilon"
    assert abs(res["denom_point"]) < EPSILON_DENOM


def test_restoration_proportion_unstable_denominator_branch():
    # Small n with near-zero denominator → many resamples cross zero.
    n = 20
    # acc_clean = 3/20 = 0.15, acc_no_patch = 2/20 = 0.10 → denom = 0.05.
    # Denominator is above epsilon on the full set but fragile under resampling:
    # many bootstrap draws will have zero-difference.
    clean_bits = [True, True, True] + [False] * 17
    no_patch_bits = [True, True] + [False] * 18
    best_bits = [True, True, True, True, True] + [False] * 15
    clean = ConditionCorrectness(
        "clean_no_patch",
        tuple(f"s{i}" for i in range(n)),
        tuple(clean_bits),
    )
    no_patch = ConditionCorrectness(
        "no_patch",
        tuple(f"s{i}" for i in range(n)),
        tuple(no_patch_bits),
    )
    best = ConditionCorrectness(
        "patch_recovery_full",
        tuple(f"s{i}" for i in range(n)),
        tuple(best_bits),
    )
    res = restoration_proportion(
        clean, no_patch, best, bootstrap_n=1000, seed=0, epsilon_denom=0.02,
    )
    # Point remains a float (denom_point ≥ epsilon_denom holds on full set).
    assert isinstance(res["point"], float)
    assert res["ci_reason"] == "unstable_denominator"
    assert res["ci_lo"] is None
    assert res["ci_hi"] is None


# ─── Orchestration ──────────────────────────────────────────────────────────


def test_compute_decomposition_table_best_condition_tie_break_alphabetical(tmp_path):
    # Build a tiny phase_b_run_dir with two tied claim-eligible conditions.
    def _wj(name: str, bits):
        p = tmp_path / f"results_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for i, b in enumerate(bits):
                f.write(json.dumps({"sample_id": f"s{i}",
                                    "correct": bool(b)}) + "\n")

    # no_patch and clean set so denom > epsilon.
    _wj("clean_no_patch",              [1, 1, 1, 1, 0, 0, 0, 0])  # 0.5
    _wj("restoration_no_patch",        [1, 0, 0, 0, 0, 0, 0, 0])  # 0.125
    # Two claim-eligible conditions with identical restoration_effect = 0.25.
    _wj("restoration_patch_boundary_local",  [1, 1, 1, 0, 0, 0, 0, 0])  # 0.375
    _wj("restoration_patch_recovery_early",  [1, 1, 0, 1, 0, 0, 0, 0])  # 0.375
    _wj("restoration_patch_recovery_full",   [1, 1, 0, 1, 0, 0, 0, 0])  # 0.375
    _wj("restoration_patch_final_only",      [1, 0, 0, 0, 0, 0, 0, 0])  # 0.125

    # phase_b_summary.json required for provenance (compute_decomposition_table
    # itself doesn't read it, only run_phase_c does — but we still need the file
    # for the CLI test. For this unit test we stop at compute_decomposition_table.
    tbl = compute_decomposition_table(str(tmp_path), bootstrap_n=100, seed=0)
    assert tbl["best_condition"] == "patch_boundary_local"


def test_phase_c_forbidden_phrases_gate_extended_phase_c_terms_flagged(tmp_path):
    # Create one file per phrase and assert every phrase is flagged.
    files = []
    for i, phrase in enumerate(FORBIDDEN_PHRASES_PHASE_C):
        p = tmp_path / f"artifact_{i}.txt"
        p.write_text(f"prefix {phrase} suffix\n", encoding="utf-8")
        files.append(str(p))
    violations = check_artifacts_for_forbidden(
        files, phrases=FORBIDDEN_PHRASES_PHASE_C,
    )
    assert len(violations) >= len(FORBIDDEN_PHRASES_PHASE_C)
    for phrase in FORBIDDEN_PHRASES_PHASE_C:
        assert any(phrase in v for v in violations), \
            f"phrase {phrase!r} not flagged"


def test_phase_c_forbidden_phrases_gate_does_not_flag_phase_c_vocabulary(tmp_path):
    # Phase C core vocabulary is allowed in Phase C gate.
    p = tmp_path / "good.txt"
    p.write_text(
        "restoration effect 0.25; residual effect 0.125; restoration proportion 0.6667.",
        encoding="utf-8",
    )
    violations = check_artifacts_for_forbidden(
        [str(p)], phrases=FORBIDDEN_PHRASES_PHASE_C,
    )
    assert violations == []


# ─── End-to-end CLI against the fixture ─────────────────────────────────────


def test_phase_c_cli_sanity_end_to_end_against_fixture():
    """Invoke the Phase C module in-process, pointing at the shipped fixture."""
    from stage1.run_phase_c import run_phase_c

    run_dir = run_phase_c(
        phase_b_run=FIXTURE_DIR, sanity=False,
        bootstrap_n=200, seed=0, ci=0.95, run_name="unit_test",
    )
    try:
        csv_path = os.path.join(run_dir, "phase_c_decomposition_table.csv")
        json_path = os.path.join(run_dir, "phase_c_summary.json")
        txt_path = os.path.join(run_dir, "phase_c_summary.txt")
        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)
        assert os.path.exists(txt_path)

        with open(json_path, encoding="utf-8") as f:
            summary = json.load(f)
        from stage1.run_phase_c import MANDATED_CAVEAT, PHASE_C_TXT_HEADER
        assert summary["caveat"] == MANDATED_CAVEAT
        assert summary["phase"] == "C"
        assert summary["best_condition"] in CLAIM_ELIGIBLE_CONDITIONS
        assert summary["forbidden_phrases_gate"] == []
        assert "prompt-side restoration intervention" in summary["methodology"]

        with open(txt_path, encoding="utf-8") as f:
            txt_body = f.read()
        assert MANDATED_CAVEAT in txt_body
        assert PHASE_C_TXT_HEADER in txt_body

        # Spec §11.10: PHASE_C_TXT_HEADER must byte-equal the mandated literal
        # (em-dash, not ASCII hyphen). Asserted against the spec literal.
        SPEC_HEADER_LITERAL = (
            "Phase C \u2014 mediation-style decomposition of prompt-side "
            "restoration intervention (not a formal NIE/NDE decomposition)"
        )
        assert PHASE_C_TXT_HEADER == SPEC_HEADER_LITERAL, (
            f"PHASE_C_TXT_HEADER drift vs spec §11.10:\n"
            f"  got     : {PHASE_C_TXT_HEADER!r}\n"
            f"  expected: {SPEC_HEADER_LITERAL!r}"
        )
        assert SPEC_HEADER_LITERAL in txt_body
        # acc_cross_check_tolerance recorded in environment (RED LIGHT Fix D: tightened).
        assert summary["environment"]["acc_cross_check_tolerance"] == 5e-5

        # Determinism: second invocation should produce bytewise-equal CSV.
        run_dir2 = run_phase_c(
            phase_b_run=FIXTURE_DIR, sanity=False,
            bootstrap_n=200, seed=0, ci=0.95, run_name="unit_test_2",
        )
        try:
            csv_path2 = os.path.join(run_dir2, "phase_c_decomposition_table.csv")
            h1 = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
            h2 = hashlib.sha256(open(csv_path2, "rb").read()).hexdigest()
            assert h1 == h2, "CSV is not bytewise-deterministic across runs"
        finally:
            import shutil
            shutil.rmtree(run_dir2, ignore_errors=True)
    finally:
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)
