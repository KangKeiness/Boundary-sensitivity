"""Source-level guardrails for stage1.run hardening paths.

These tests are intentionally static so they run even when heavy ML
dependencies are unavailable in the test interpreter.
"""

from pathlib import Path


_RUN_PY = Path(__file__).resolve().parents[1] / "run.py"


def _source() -> str:
    return _RUN_PY.read_text(encoding="utf-8")


def test_random_donor_seed_formula_used_in_stage1_run():
    src = _source()
    assert "compute_random_donor_seed(" in src
    assert "seed_base*1000 + b*100 + t" in src


def test_self_verification_checks_boundary_table_length_and_order():
    src = _source()
    assert "boundary_table length mismatch" in src
    assert "boundary_table ordering mismatch" in src


def test_self_verification_checks_expected_result_and_bds_conditions():
    src = _source()
    assert "missing results_*.jsonl conditions" in src
    assert "missing bds_*.json conditions" in src
    assert "random_donor_b" in src
    assert "hard_swap_b" in src
