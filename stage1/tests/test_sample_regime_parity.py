"""Regression tests for sample-regime parity (v4 PRIORITY 2).

Verifies that ``stage1.utils.manifest_parity`` rejects anchor candidates that
agree on every other field but differ in:

  - debug-vs-full mode (``dataset.debug_n``)
  - ``sample_regime.mode``
  - ``sample_regime.sample_count``
  - ``sample_regime.sample_ordering_sha256``

These are scientific blockers — a debug subset that happens to share a config
must NOT be accepted as a valid anchor for a full-mode run.

Tests are torch-free.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stage1.utils.manifest_parity import (  # noqa: E402
    check_manifest_parity,
    compute_sample_ordering_sha256,
    compute_sample_regime,
    extract_parity_block,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _fake_config(debug_n=None):
    """Minimal config-like object accepted by extract_parity_block."""
    return SimpleNamespace(
        models=SimpleNamespace(
            recipient="r", donor="d",
            recipient_revision="rev_r", donor_revision="rev_d",
        ),
        dataset=SimpleNamespace(name="mgsm", lang="zh", split="test", debug_n=debug_n),
        generation=SimpleNamespace(do_sample=False, temperature=0.0, max_new_tokens=512),
        hidden_state=SimpleNamespace(pooling="last_token"),
    )


def _full_sample_ids():
    return [f"s{i}" for i in range(250)]


# ─── Sample-ordering hash determinism ───────────────────────────────────────


def test_ordering_hash_is_deterministic():
    ids = ["s0", "s1", "s2"]
    assert compute_sample_ordering_sha256(ids) == compute_sample_ordering_sha256(ids)


def test_ordering_hash_changes_with_order():
    """Ordering, not just set membership, must affect the digest."""
    a = compute_sample_ordering_sha256(["s0", "s1", "s2"])
    b = compute_sample_ordering_sha256(["s2", "s1", "s0"])
    assert a != b


def test_ordering_hash_changes_with_membership():
    a = compute_sample_ordering_sha256(["s0", "s1", "s2"])
    b = compute_sample_ordering_sha256(["s0", "s1", "s3"])
    assert a != b


def test_ordering_hash_avoids_concat_collisions():
    """Null-separator must prevent ['s', '0s1'] colliding with ['s0', 's1']."""
    a = compute_sample_ordering_sha256(["s", "0s1"])
    b = compute_sample_ordering_sha256(["s0", "s1"])
    assert a != b


# ─── Sample-regime block shape ──────────────────────────────────────────────


def test_full_mode_sample_regime_shape():
    cfg = _fake_config(debug_n=None)
    regime = compute_sample_regime(cfg, _full_sample_ids())
    assert regime["mode"] == "full"
    assert regime["debug_n"] is None
    assert regime["sample_count"] == 250
    assert isinstance(regime["sample_ordering_sha256"], str)
    assert len(regime["sample_ordering_sha256"]) == 64


def test_debug_mode_sample_regime_shape():
    cfg = _fake_config(debug_n=5)
    regime = compute_sample_regime(cfg, ["s0", "s1", "s2", "s3", "s4"])
    assert regime["mode"] == "debug"
    assert regime["debug_n"] == 5
    assert regime["sample_count"] == 5


# ─── Parity rejection cases ─────────────────────────────────────────────────


def _parity_with_samples(cfg, sample_ids):
    return extract_parity_block(cfg, sample_ids=sample_ids)


def test_parity_rejects_debug_vs_full_mismatch():
    """Same model+gen+dataset, but one is debug and the other is full → reject."""
    full_cfg = _fake_config(debug_n=None)
    debug_cfg = _fake_config(debug_n=5)
    src = {"parity": _parity_with_samples(full_cfg, _full_sample_ids())}
    tgt = _parity_with_samples(debug_cfg, ["s0", "s1", "s2", "s3", "s4"])
    mismatches = check_manifest_parity(src, tgt)
    paths = " ".join(mismatches)
    assert "sample_regime.mode" in paths or "dataset.debug_n" in paths
    assert mismatches  # at least one


def test_parity_rejects_sample_count_mismatch():
    """Both full but with different sample counts → reject."""
    cfg = _fake_config(debug_n=None)
    src = {"parity": _parity_with_samples(cfg, [f"s{i}" for i in range(100)])}
    tgt = _parity_with_samples(cfg, [f"s{i}" for i in range(101)])
    mismatches = check_manifest_parity(src, tgt)
    paths = " ".join(mismatches)
    assert "sample_count" in paths


def test_parity_rejects_ordering_signature_mismatch():
    """Same count, same set, different order → reject."""
    cfg = _fake_config(debug_n=None)
    ids = [f"s{i}" for i in range(10)]
    src = {"parity": _parity_with_samples(cfg, ids)}
    # Reverse order → identical set, identical count, different signature.
    tgt = _parity_with_samples(cfg, list(reversed(ids)))
    mismatches = check_manifest_parity(src, tgt)
    paths = " ".join(mismatches)
    assert "sample_ordering_sha256" in paths


def test_parity_passes_with_identical_sample_regime():
    cfg = _fake_config(debug_n=None)
    ids = _full_sample_ids()
    src = {"parity": _parity_with_samples(cfg, ids)}
    tgt = _parity_with_samples(cfg, ids)
    assert check_manifest_parity(src, tgt) == []


def test_parity_rejects_anchor_without_sample_regime():
    """A pre-v4 anchor manifest (no sample_regime block) MUST be rejected
    against a v4 config that includes one. This forces anchor regeneration
    rather than silent acceptance of unverifiable upstream runs."""
    cfg = _fake_config(debug_n=None)
    # Source has no sample_regime block at all.
    src = {"parity": {
        "models": {"recipient": "r", "donor": "d",
                   "recipient_revision": "rev_r", "donor_revision": "rev_d"},
        "dataset": {"name": "mgsm", "lang": "zh", "split": "test", "debug_n": None},
        "generation": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 512},
        "hidden_state": {"pooling": "last_token"},
        # sample_regime intentionally omitted (legacy)
    }}
    tgt = _parity_with_samples(cfg, _full_sample_ids())
    mismatches = check_manifest_parity(src, tgt)
    # Three sample_regime fields (mode, sample_count, sample_ordering_sha256)
    # must be flagged as missing-in-source.
    flagged = [m for m in mismatches if "sample_regime" in m]
    assert len(flagged) >= 3
    assert all("missing in source" in m for m in flagged)


def test_debug_n_null_vs_missing_treated_as_full_equivalent():
    """A pre-v4 manifest without dataset.debug_n must NOT cause a spurious
    mismatch against a current full-mode config (debug_n=None). This is the
    same null-equivalence treatment as for revision fields."""
    cfg = _fake_config(debug_n=None)
    ids = _full_sample_ids()
    src = {"parity": {
        "models": {"recipient": "r", "donor": "d",
                   "recipient_revision": "rev_r", "donor_revision": "rev_d"},
        "dataset": {"name": "mgsm", "lang": "zh", "split": "test"},
        # debug_n omitted entirely.
        "generation": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 512},
        "hidden_state": {"pooling": "last_token"},
        "sample_regime": compute_sample_regime(cfg, ids),
    }}
    tgt = _parity_with_samples(cfg, ids)
    # debug_n missing-vs-None must NOT itself produce a mismatch.
    mismatches = check_manifest_parity(src, tgt)
    debug_n_paths = [m for m in mismatches if "dataset.debug_n" in m]
    assert debug_n_paths == [], (
        "debug_n=None vs missing should be equivalent (both = full run): %s"
        % debug_n_paths
    )


# ─── Diagnostic clarity ─────────────────────────────────────────────────────


def test_mismatch_message_names_offending_field(tmp_path):
    """Logs/diagnostics should name the specific sample_regime field that
    failed so operators don't have to guess."""
    cfg = _fake_config(debug_n=None)
    src = {"parity": _parity_with_samples(cfg, ["s0", "s1"])}
    tgt = _parity_with_samples(cfg, ["s0", "s1", "s2"])
    mismatches = check_manifest_parity(src, tgt)
    assert any("sample_count" in m for m in mismatches)
    # The current vs source values must appear in the message.
    sample_count_msg = next(m for m in mismatches if "sample_count" in m)
    assert "2" in sample_count_msg and "3" in sample_count_msg
