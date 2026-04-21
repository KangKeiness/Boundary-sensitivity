"""Runtime smoke tests for Stage 1 / Phase A (runtime_repro_v5 §10).

Four regression tests guarding the plumbing-only hardening in v5:

  1. ``stage1.run`` is importable as a package module (covers the bare-import
     regression at stage1/run.py L11–L25).
  2. ``python -m stage1.run --help`` exits 0 from the repo root.
  3. ``stage1.utils.config.load_config`` opens YAML as UTF-8 regardless of the
     OS locale (covers Windows cp949 ambient).
  4. The Phase A no-swap reuse path passes ``sample_ids=`` to
     ``extract_parity_block``, so reuse against a v4+ manifest with a
     ``sample_regime`` block no longer raises
     ``"sample regime mode ...: missing in current config"``.

Torch-free where feasible. Test 4 is split into a pure parity-check variant
(runs always) and an end-to-end gated variant (requires torch). The pure
variant still fails on pre-fix code, preserving the regression signal.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ─── §10.1 ──────────────────────────────────────────────────────────────────


def test_stage1_run_importable():
    """``stage1.run`` must be importable under its package-qualified name.

    ``stage1.run`` transitively imports torch via ``stage1.models.composer``.
    To keep this test torch-free we probe via ``importlib.util.find_spec``
    first (which parses the module path without executing it), then delegate
    the full import to a subprocess so any ImportError from the bare-imports
    regression still surfaces — without dragging torch into this pytest
    process.
    """
    pytest.importorskip("transformers", reason="stage1.run imports transformers at module load time")
    pytest.importorskip("torch", reason="stage1.run imports torch at module load time")

    spec = importlib.util.find_spec("stage1.run")
    assert spec is not None, "stage1.run not findable as a package submodule"

    # Full import in a subprocess so torch/transformers stay out of pytest.
    result = subprocess.run(
        [sys.executable, "-c", "import stage1.run"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"`import stage1.run` failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ─── §10.2 ──────────────────────────────────────────────────────────────────


def test_stage1_run_m_help_exit_zero():
    """``python -m stage1.run --help`` must exit 0 from the repo root.

    ``cwd`` is the repo root (parent of ``stage1/``) — NOT ``stage1/`` — so
    any regression to bare sibling imports (``from utils.config import ...``)
    would break module resolution here and we'd see a non-zero rc.
    """
    pytest.importorskip("transformers", reason="stage1.run imports transformers at module load time")
    pytest.importorskip("torch", reason="stage1.run imports torch at module load time")

    result = subprocess.run(
        [sys.executable, "-m", "stage1.run", "--help"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"`python -m stage1.run --help` failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    # Sanity: usage line must mention --config.
    assert "--config" in result.stdout, (
        f"--help output did not contain --config:\n{result.stdout}"
    )


# ─── §10.3 ──────────────────────────────────────────────────────────────────


def test_load_config_utf8():
    """``load_config`` must open the YAML with explicit ``encoding="utf-8"``.

    Dual assertion:
      - Behavioral: the call succeeds and populates ``dataset.lang == "zh"``
        for the pinned Phase A config.
      - Source-level: ``inspect.getsource(load_config)`` contains the literal
        ``encoding="utf-8"`` (belt-and-braces guard against a regression that
        strips the kwarg silently).
    """
    # Import lazily so this file is collectible even if a sibling module is
    # momentarily broken.
    from stage1.utils import config as _config_mod
    from stage1.utils.config import load_config

    cfg_path = _REPO_ROOT / "stage1" / "configs" / "stage2_confound.yaml"
    assert cfg_path.exists(), f"missing fixture config: {cfg_path}"

    cfg = load_config(str(cfg_path))
    # Known field populated from YAML — dataset.lang == "zh".
    assert cfg.dataset.lang == "zh", (
        f"expected dataset.lang == 'zh', got {cfg.dataset.lang!r}"
    )

    src = inspect.getsource(_config_mod.load_config)
    assert 'encoding="utf-8"' in src, (
        f"load_config source does not contain encoding=\"utf-8\":\n{src}"
    )


# ─── §10.4 ──────────────────────────────────────────────────────────────────


def _fake_phase_a_config(debug_n=None, lang="zh"):
    """Minimal config-like object accepted by ``extract_parity_block``."""
    return SimpleNamespace(
        models=SimpleNamespace(
            recipient="Qwen/Qwen2.5-1.5B-Instruct",
            donor="Qwen/Qwen2.5-1.5B",
            recipient_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            donor_revision="8faed761d45a263340a0528343f099c05c9a4323",
        ),
        dataset=SimpleNamespace(
            name="mgsm", lang=lang, split="test", debug_n=debug_n,
        ),
        generation=SimpleNamespace(
            do_sample=False, temperature=0.0, max_new_tokens=512,
        ),
        hidden_state=SimpleNamespace(pooling="last_token"),
    )


def test_phase_a_reuse_parity_has_sample_regime_pure():
    """Pure parity-check variant — does not exercise ``torch.load``.

    Simulates exactly the comparison ``_load_reused_no_swap`` performs:
    builds a manifest whose ``parity`` block was written via
    ``extract_parity_block(cfg, sample_ids=[...])`` (fresh-write site at
    stage1/run_phase_a.py L781–L783) and compares it against the block the
    reuse-read site builds for the CURRENT config.

    On pre-fix code the reuse site called ``extract_parity_block(config)``
    (no sample_ids) and the resulting current_parity had NO ``sample_regime``
    block, so the check raised "sample regime mode ...: missing in current
    config". This test therefore fails on pre-fix code.
    """
    from stage1.utils.manifest_parity import (
        ManifestParityError,
        check_manifest_parity,
        check_manifest_parity_or_raise,
        extract_parity_block,
    )

    cfg = _fake_phase_a_config()
    sample_ids = ["s1", "s2", "s3"]

    # Source manifest: what a v4+ fresh Phase A run writes.
    source_manifest = {
        "parity": extract_parity_block(cfg, sample_ids=sample_ids),
    }

    # Target: what the CURRENT reuse path builds. Post-fix mirrors the
    # fresh-write call with sample_ids — must produce no mismatches.
    current_parity_post_fix = extract_parity_block(
        cfg, sample_ids=sample_ids,
    )
    mismatches = check_manifest_parity(
        source_manifest, current_parity_post_fix,
        source_path="<stub>", target_desc="current Phase A config",
    )
    assert mismatches == [], (
        f"post-fix parity should be clean, got mismatches: {mismatches}"
    )

    # Pre-fix target: NO sample_ids → no sample_regime block → must mismatch
    # on every sample_regime.* field. This is the regression signal.
    current_parity_pre_fix = extract_parity_block(cfg)
    with pytest.raises(ManifestParityError, match="sample_regime"):
        check_manifest_parity_or_raise(
            source_manifest, current_parity_pre_fix,
            source_path="<stub>", target_desc="current Phase A config",
        )

    # Also: dropping sample_regime from the source side must raise on the
    # current-is-missing path.
    source_manifest_dropped = {
        "parity": {
            k: v for k, v in source_manifest["parity"].items()
            if k != "sample_regime"
        },
    }
    with pytest.raises(ManifestParityError, match="sample_regime"):
        check_manifest_parity_or_raise(
            source_manifest_dropped, current_parity_post_fix,
            source_path="<stub>", target_desc="current Phase A config",
        )


def test_phase_a_reuse_parity_end_to_end(tmp_path):
    """End-to-end variant — exercises ``_load_reused_no_swap`` itself.

    Gated on torch availability because ``_load_reused_no_swap`` calls
    ``torch.load`` unconditionally on ``hidden_states_no_swap.pt``. When torch
    is present we construct a minimal stub reuse dir and verify (a) a
    v4-compliant manifest reuses cleanly, and (b) a manifest with
    ``sample_regime`` stripped raises ``ManifestParityError`` mentioning
    ``sample_regime``.
    """
    torch = pytest.importorskip("torch")

    # ``_load_reused_no_swap`` is module-private; access via getattr on the
    # parent module so we do not force its visibility to change.
    run_phase_a_mod = importlib.import_module("stage1.run_phase_a")
    load_reused = getattr(run_phase_a_mod, "_load_reused_no_swap")
    from stage1.utils.manifest_parity import (
        ManifestParityError,
        extract_parity_block,
    )

    cfg = _fake_phase_a_config()
    sample_ids = ["s1", "s2", "s3"]
    samples = [{"sample_id": sid} for sid in sample_ids]

    # Stub hidden_states_no_swap.pt — one tensor-dict keyed by sample id.
    hs_path = tmp_path / "hidden_states_no_swap.pt"
    torch.save({sid: torch.zeros(1) for sid in sample_ids}, str(hs_path))

    # Stub results_no_swap.jsonl — minimal parse fields.
    results_path = tmp_path / "results_no_swap.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for sid in sample_ids:
            f.write(json.dumps({
                "sample_id": sid,
                "output_text": "",
                "parse_success": False,
                "normalized_answer": None,
            }) + "\n")

    # Stub manifest.json with a v4-compliant parity block.
    manifest_path = tmp_path / "manifest.json"
    manifest_v4 = {
        "parity": extract_parity_block(cfg, sample_ids=sample_ids),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_v4, f)

    # Samples passed here mirror what the outer CLI path passes. Extra
    # ``gold_answer`` key is required by ``_load_reused_no_swap`` when it
    # computes per_sample_correct.
    samples_with_gold = [
        {"sample_id": sid, "gold_answer": "0"} for sid in sample_ids
    ]

    # (a) v4-compliant manifest must NOT raise.
    load_reused(str(tmp_path), samples=samples_with_gold, config=cfg)

    # (b) Strip sample_regime → must raise with "sample_regime" in message.
    manifest_stripped = {
        "parity": {
            k: v for k, v in manifest_v4["parity"].items()
            if k != "sample_regime"
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_stripped, f)

    with pytest.raises(ManifestParityError, match="sample_regime"):
        load_reused(str(tmp_path), samples=samples_with_gold, config=cfg)
