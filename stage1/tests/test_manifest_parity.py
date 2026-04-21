"""Tests for manifest parity checker (RED LIGHT Priority 2).

Covers:
- Parity passes when fields match
- Parity fails on model mismatch
- Parity fails on generation config mismatch
- Parity fails on missing manifest
- Backward compat: manifests with config sub-key
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stage1.utils.manifest_parity import (
    ManifestParityError,
    check_manifest_parity,
    check_manifest_parity_or_raise,
    extract_parity_block,
    load_manifest_from_run_dir,
)


def _make_parity_block(**overrides):
    """Build a default parity block, overriding nested keys as needed."""
    block = {
        "models": {
            "recipient": "Qwen/Qwen2.5-1.5B-Instruct",
            "donor": "Qwen/Qwen2.5-1.5B",
            "recipient_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "donor_revision": "8faed761d45a263340a0528343f099c05c9a4323",
        },
        "dataset": {
            "name": "mgsm",
            "lang": "zh",
            "split": "test",
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
        "hidden_state": {
            "pooling": "last_token",
        },
    }
    for key, val in overrides.items():
        parts = key.split(".")
        d = block
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = val
    return block


def test_parity_passes_when_identical():
    source = {"parity": _make_parity_block()}
    target = _make_parity_block()
    mismatches = check_manifest_parity(source, target)
    assert mismatches == []


def test_parity_fails_on_model_mismatch():
    source = {"parity": _make_parity_block()}
    target = _make_parity_block(**{"models.recipient": "Qwen/Qwen2.5-3B-Instruct"})
    mismatches = check_manifest_parity(source, target)
    assert len(mismatches) == 1
    assert "recipient model identifier" in mismatches[0]


def test_parity_fails_on_generation_mismatch():
    source = {"parity": _make_parity_block()}
    target = _make_parity_block(**{"generation.max_new_tokens": 256})
    mismatches = check_manifest_parity(source, target)
    assert len(mismatches) == 1
    assert "max_new_tokens" in mismatches[0]


def test_parity_fails_on_dataset_mismatch():
    source = {"parity": _make_parity_block()}
    target = _make_parity_block(**{"dataset.lang": "en"})
    mismatches = check_manifest_parity(source, target)
    assert len(mismatches) == 1
    assert "dataset language" in mismatches[0]


def test_parity_or_raise_on_mismatch():
    source = {"parity": _make_parity_block()}
    target = _make_parity_block(**{"models.donor": "other/model"})
    with pytest.raises(ManifestParityError) as exc_info:
        check_manifest_parity_or_raise(
            source, target,
            source_path="/fake/manifest.json",
            target_desc="test",
        )
    assert "donor model identifier" in str(exc_info.value)


def test_parity_backward_compat_config_sub_key():
    """Older Stage 1 manifests embed config under 'config' key."""
    source = {
        "config": _make_parity_block(),
    }
    target = _make_parity_block()
    mismatches = check_manifest_parity(source, target)
    assert mismatches == []


def test_load_manifest_from_run_dir(tmp_path):
    manifest = {"phase": "A", "parity": _make_parity_block()}
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    loaded = load_manifest_from_run_dir(str(tmp_path))
    assert loaded["phase"] == "A"


def test_load_manifest_from_run_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest_from_run_dir(str(tmp_path))


def test_parity_missing_field_in_source():
    """Source manifest missing a required field flags it."""
    source = {"parity": _make_parity_block()}
    del source["parity"]["generation"]["max_new_tokens"]
    target = _make_parity_block()
    mismatches = check_manifest_parity(source, target)
    assert len(mismatches) == 1
    assert "missing in source" in mismatches[0]
