"""Stage 1 hardening — dataset provenance tests.

Covers:
  * Configured ``revision`` flows into the download URL.
  * SHA-256 mismatch raises ``DatasetProvenanceError``.
  * Successful load stamps ``config.dataset._provenance`` with realised
    revision + sha256.
  * Provenance sidecar (``*.sha256.json``) is written next to the cache.
  * ``build_dataset_provenance`` schema is stable.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stage1.data import loader as loader_mod
from stage1.data.loader import (
    DatasetProvenanceError,
    KNOWN_DATASET_SHA256,
    build_dataset_provenance,
    load_mgsm,
)


def _make_tsv(path: Path, n_rows: int = 3) -> str:
    """Write a tiny TSV in MGSM format and return its sha256."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for i in range(n_rows):
            w.writerow([f"What is {i}+{i}?", str(2 * i)])
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fake_config(
    *,
    lang: str = "en",  # not in KNOWN_DATASET_SHA256 → no implicit pin
    revision: str = "abc123def456",
    expected_sha256: Any = None,
    debug_n: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(
            name="mgsm",
            lang=lang,
            split="test",
            debug_n=debug_n,
            revision=revision,
            expected_sha256=expected_sha256,
        ),
    )


@pytest.fixture
def patch_download(tmp_path, monkeypatch):
    """Replace ``_download_tsv`` with a function that writes ``tmp_path/<lang>__<rev>.tsv``."""

    written: dict = {"path": None, "calls": []}

    def fake_download(lang: str, revision: str) -> Path:
        rev_safe = revision.replace("/", "_")
        target = tmp_path / f"mgsm_{lang}__{rev_safe}.tsv"
        if not target.exists():
            _make_tsv(target, n_rows=3)
        written["path"] = target
        written["calls"].append((lang, revision))
        return target

    monkeypatch.setattr(loader_mod, "_download_tsv", fake_download)
    # Also redirect the cache dir for the sidecar write step.
    monkeypatch.setattr(loader_mod, "CACHE_DIR", tmp_path)
    return written


def test_revision_flows_into_download(patch_download):
    cfg = _fake_config(revision="rev-from-config")
    # No expected SHA → loader logs a warning and stamps the realised SHA.
    samples = load_mgsm(cfg)
    assert len(samples) == 3
    assert ("en", "rev-from-config") in patch_download["calls"]


def test_sha_mismatch_raises(patch_download):
    cfg = _fake_config(
        revision="revstrict",
        expected_sha256="0" * 64,  # impossible SHA
    )
    with pytest.raises(DatasetProvenanceError) as excinfo:
        load_mgsm(cfg)
    msg = str(excinfo.value)
    assert "SHA-256 mismatch" in msg
    assert "0" * 64 in msg
    assert "Refusing to run" in msg


def test_provenance_stamped_on_config(patch_download):
    cfg = _fake_config(revision="rev-stamp", expected_sha256=None)
    samples = load_mgsm(cfg)
    prov = cfg.dataset._provenance
    assert prov["revision"] == "rev-stamp"
    assert prov["lang"] == "en"
    assert len(prov["sha256"]) == 64
    assert prov["row_count_raw"] == 3
    assert "/resolve/rev-stamp/" in prov["source"]


def test_provenance_sidecar_written(patch_download, tmp_path):
    cfg = _fake_config(revision="rev-side", expected_sha256=None)
    load_mgsm(cfg)
    sidecar = tmp_path / "mgsm_en__rev-side.tsv.sha256.json"
    assert sidecar.exists()
    with open(sidecar, encoding="utf-8") as f:
        body = json.load(f)
    assert body["revision"] == "rev-side"
    assert body["lang"] == "en"


def test_known_pin_for_zh_present():
    """The mgsm_zh known SHA-256 pin must remain in KNOWN_DATASET_SHA256."""
    assert "zh" in KNOWN_DATASET_SHA256
    assert len(KNOWN_DATASET_SHA256["zh"]) == 64


def test_build_dataset_provenance_schema(tmp_path):
    cfg = _fake_config(revision="rev-x", expected_sha256="deadbeef")
    block = build_dataset_provenance(
        cfg,
        realised_sha256="cafebabe",
        revision="rev-x",
        cache_path=tmp_path / "f.tsv",
        raw_row_count=10,
    )
    assert block["revision"] == "rev-x"
    assert block["sha256"] == "cafebabe"
    assert block["expected_sha256_pin"] == "deadbeef"
    assert block["row_count_raw"] == 10
    assert block["lang"] == "en"
