"""Stage 1 hardening — hidden-state artifact self-verification tests.

Covers:
  * Happy path: well-formed .pt files pass verification.
  * Sample-ID set mismatch raises.
  * Wrong layer count raises.
  * Wrong hidden size raises.
  * Wrong dtype across samples raises.
  * Empty run dir raises FileNotFoundError.
  * raise_on_error=False path returns reports without raising.
  * summarise_reports yields a manifest-friendly schema.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stage1.utils.hidden_state_verify import (
    HiddenStateVerificationError,
    summarise_reports,
    verify_hidden_state_artifacts,
)


def _write_hs(
    run_dir: Path,
    condition: str,
    sample_ids,
    *,
    layers: int = 28,
    hidden: int = 1536,
    dtype=torch.float32,
):
    path = run_dir / f"hidden_states_{condition}.pt"
    hs = {sid: torch.zeros((layers, hidden), dtype=dtype) for sid in sample_ids}
    torch.save(hs, str(path))
    return path


@pytest.fixture
def good_run_dir(tmp_path):
    sample_ids = ["s0", "s1", "s2"]
    _write_hs(tmp_path, "no_swap", sample_ids)
    _write_hs(tmp_path, "fixed_w4_pos1", sample_ids)
    return tmp_path, sample_ids


def test_happy_path(good_run_dir):
    run_dir, sample_ids = good_run_dir
    reports = verify_hidden_state_artifacts(
        str(run_dir), sample_ids,
        expected_layer_count=28, expected_hidden_size=1536,
    )
    assert len(reports) == 2
    assert all(r["ok"] for r in reports)
    summary = summarise_reports(reports)
    assert summary["all_ok"] is True
    assert summary["n_artifacts"] == 2


def test_missing_sample_id_raises(tmp_path):
    _write_hs(tmp_path, "cond", ["s0", "s1"])
    with pytest.raises(HiddenStateVerificationError, match="missing sample_ids"):
        verify_hidden_state_artifacts(str(tmp_path), ["s0", "s1", "s2"])


def test_unexpected_sample_id_raises(tmp_path):
    _write_hs(tmp_path, "cond", ["s0", "s1", "extra"])
    with pytest.raises(HiddenStateVerificationError, match="unexpected sample_ids"):
        verify_hidden_state_artifacts(str(tmp_path), ["s0", "s1"])


def test_wrong_layer_count_raises(tmp_path):
    _write_hs(tmp_path, "cond", ["s0"], layers=12)
    with pytest.raises(HiddenStateVerificationError, match="layer_count=12"):
        verify_hidden_state_artifacts(
            str(tmp_path), ["s0"], expected_layer_count=28,
        )


def test_wrong_hidden_size_raises(tmp_path):
    _write_hs(tmp_path, "cond", ["s0"], hidden=2048)
    with pytest.raises(HiddenStateVerificationError, match="hidden_size=2048"):
        verify_hidden_state_artifacts(
            str(tmp_path), ["s0"], expected_hidden_size=1536,
        )


def test_inconsistent_dtype_raises(tmp_path):
    path = tmp_path / "hidden_states_cond.pt"
    hs = {
        "s0": torch.zeros((4, 8), dtype=torch.float32),
        "s1": torch.zeros((4, 8), dtype=torch.float16),
    }
    torch.save(hs, str(path))
    with pytest.raises(HiddenStateVerificationError, match="inconsistent dtype"):
        verify_hidden_state_artifacts(str(tmp_path), ["s0", "s1"])


def test_empty_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No hidden_states"):
        verify_hidden_state_artifacts(str(tmp_path), ["s0"])


def test_nonexistent_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_hidden_state_artifacts(
            str(tmp_path / "does_not_exist"), ["s0"],
        )


def test_raise_on_error_false_returns_reports(tmp_path):
    _write_hs(tmp_path, "cond", ["s0"], layers=12)
    reports = verify_hidden_state_artifacts(
        str(tmp_path), ["s0"],
        expected_layer_count=28, raise_on_error=False,
    )
    assert len(reports) == 1
    assert reports[0]["ok"] is False
    assert any("layer_count=12" in e for e in reports[0]["errors"])


def test_summary_schema(good_run_dir):
    run_dir, sample_ids = good_run_dir
    reports = verify_hidden_state_artifacts(str(run_dir), sample_ids)
    summary = summarise_reports(reports)
    keys = {"n_artifacts", "all_ok", "artifacts"}
    assert keys.issubset(summary.keys())
    for art in summary["artifacts"]:
        assert {"condition", "n_samples", "layer_count", "hidden_size",
                "dtype", "ok", "errors"}.issubset(art.keys())
