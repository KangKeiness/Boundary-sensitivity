"""Stage 1 hardening — runtime provenance manifest tests.

Covers:
  * ``build_runtime_provenance`` returns the expected schema with all
    required fields populated (or stamped 'unavailable').
  * Dataset block falls back to configured pin when no loader stamp exists.
  * Dataset block prefers the realised loader stamp when present.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stage1.utils.provenance import build_runtime_provenance


REQUIRED_KEYS = {
    "git_sha",
    "python_version",
    "torch_version",
    "transformers_version",
    "numpy_version",
    "scipy_version",
    "platform",
    "hostname",
    "command",
    "cwd",
    "timestamp",
    "config_path",
}


def test_schema_contains_required_keys():
    block = build_runtime_provenance()
    assert REQUIRED_KEYS.issubset(block.keys())


def test_command_is_sys_argv():
    block = build_runtime_provenance()
    assert block["command"] == list(sys.argv)


def test_dataset_falls_back_to_config_pin():
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            name="mgsm", lang="zh", split="test",
            revision="rev-pin", expected_sha256="deadbeef",
        ),
    )
    block = build_runtime_provenance(config=cfg)
    assert block["dataset"]["revision"] == "rev-pin"
    assert block["dataset"]["expected_sha256_pin"] == "deadbeef"


def test_dataset_prefers_loader_stamp():
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            name="mgsm", lang="zh", split="test",
            revision="rev-pin", expected_sha256="deadbeef",
            _provenance={
                "name": "mgsm", "lang": "zh", "split": "test",
                "revision": "rev-stamp", "sha256": "cafebabe",
                "expected_sha256_pin": "deadbeef",
                "row_count_raw": 250,
                "source": "https://example/zh.tsv",
                "cache_path": "/tmp/x",
            },
        ),
    )
    block = build_runtime_provenance(config=cfg)
    assert block["dataset"]["revision"] == "rev-stamp"
    assert block["dataset"]["sha256"] == "cafebabe"


def test_extra_fields_merge_in():
    block = build_runtime_provenance(extra={"phase": "A", "seed": 42})
    assert block["phase"] == "A"
    assert block["seed"] == 42


def test_versions_are_strings():
    block = build_runtime_provenance()
    for k in ("torch_version", "transformers_version", "numpy_version", "scipy_version"):
        assert isinstance(block[k], str)
        assert block[k] != ""


def test_config_path_passed_through(tmp_path):
    cfg_path = str(tmp_path / "stage1.yaml")
    block = build_runtime_provenance(config_path=cfg_path)
    assert block["config_path"] == cfg_path
