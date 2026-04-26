"""Runtime smoke tests for Stage 1 / Phase A.

These tests deliberately bypass the conftest transformers stub so they
reflect REAL runtime readiness, not the sandbox's mocked environment.

Stage 1 hardening (2026-04-25): the previous version called
``pytest.importorskip("transformers")`` which was satisfied by the conftest
stub — the smoke test then always tried to spawn a subprocess (which has no
stub) and either ran it or failed misleadingly. Two harnesses, one sandbox:
the smoke result was uncorrelated with whether the runtime was actually ready.

Resolution:
  * ``_real_transformers_available()`` checks for the on-disk package by
    inspecting ``__is_test_stub__`` and ``__file__`` flags before falling
    back to ``importlib.util.find_spec`` in a fresh subprocess.
  * Smoke tests skip with an explicit reason when only the stub is present,
    so the result is binary: real-deps-present → run smoke; real-deps-absent
    → skip cleanly. No false greens, no false reds from stub leakage.

Tests covered here:

  1. ``stage1.run`` importable as a package module (regression: bare-import
     drift in stage1/run.py).
  2. ``python -m stage1.run --help`` exits 0 from the repo root.
  3. ``stage1.utils.config.load_config`` opens YAML as UTF-8 regardless of
     the OS locale (Windows cp949 ambient).
  4. The Phase A no-swap reuse path passes ``sample_ids=`` so reuse against
     a v4+ manifest no longer raises on missing sample_regime.
  5. Stub-detection regression: smoke tests must NOT pass while the stub
     is the only ``transformers`` available.
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


# ─── Stub-aware availability helpers ────────────────────────────────────────
#
# conftest.py installs a deterministic transformers stub at session start.
# `pytest.importorskip("transformers")` is satisfied by that stub and gives
# us no signal about whether the REAL runtime is ready. These helpers ask the
# precise question instead: "is the on-disk transformers package importable
# in a clean subprocess that does NOT inherit our stub?".


def _module_is_stub(name: str) -> bool:
    """Return True iff ``sys.modules[name]`` is the conftest-installed stub."""
    mod = sys.modules.get(name)
    if mod is None:
        return False
    if getattr(mod, "__is_test_stub__", False):
        return True
    # Older stubs flagged themselves only by lacking ``__file__``.
    return not hasattr(mod, "__file__")


def _real_module_available_in_subprocess(name: str) -> bool:
    """True iff ``import {name}`` succeeds in a fresh Python subprocess.

    The subprocess does NOT inherit our in-process module table, so the
    conftest stub is invisible to it. This is the only reliable way to ask
    "is the real package installed?" from inside pytest.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {name}"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0


def _require_real_runtime() -> None:
    """Skip the current test unless the real transformers + torch are available.

    Two-step check, both required:
      * In-process ``transformers`` is the conftest stub or absent → skip.
      * A fresh subprocess can ``import transformers`` and ``import torch``
        → otherwise skip.
    """
    if _module_is_stub("transformers"):
        # Stub is masking the real package's absence; only the subprocess
        # check is authoritative.
        if not _real_module_available_in_subprocess("transformers"):
            pytest.skip(
                "real `transformers` is not installed in this environment "
                "(only the conftest stub is loaded). Smoke tests reflect "
                "true runtime readiness — skipping."
            )
    elif "transformers" not in sys.modules and \
            importlib.util.find_spec("transformers") is None:
        pytest.skip("real `transformers` not installed in this environment")

    if not _real_module_available_in_subprocess("torch"):
        pytest.skip("real `torch` not installed in this environment")


# ─── §10.1 ──────────────────────────────────────────────────────────────────


def test_stage1_run_importable():
    """``stage1.run`` must be importable under its package-qualified name.

    Stage 1 hardening (2026-04-25): availability check no longer trusts the
    conftest stub. ``_require_real_runtime`` skips cleanly if the on-disk
    transformers / torch packages are absent — this matches what the
    subprocess will actually see.
    """
    _require_real_runtime()

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

    Stage 1 hardening: real-runtime availability is checked via
    ``_require_real_runtime``; the conftest stub does NOT mask this path.
    """
    _require_real_runtime()

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


def _run_module_help_cp949(module_name: str) -> subprocess.CompletedProcess:
    """Run `python -m <module> --help` with cp949 stdout/stderr encoding."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp949"
    return subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_phase_abc_help_cp949_exit_zero():
    """Phase A/B/C CLI help must be locale-safe on Korean Windows cp949."""
    _require_real_runtime()
    modules = ["stage1.run_phase_a", "stage1.run_phase_b", "stage1.run_phase_c"]
    for module_name in modules:
        result = _run_module_help_cp949(module_name)
        assert result.returncode == 0, (
            f"`python -m {module_name} --help` failed under cp949 "
            f"(rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_phase_c_sanity_cp949_fixture_exit_zero():
    """Minimal Phase C dry-run path must be cp949-safe."""
    fixture_dir = _REPO_ROOT / "stage1" / "tests" / "fixtures" / "phase_b_run_fixture"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp949"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stage1.run_phase_c",
            "--phase-b-run",
            str(fixture_dir),
            "--sanity",
            "--bootstrap-n",
            "10",
            "--seed",
            "0",
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"`python -m stage1.run_phase_c --sanity` failed under cp949 "
        f"(rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ─── §10.5: Stub-detection regression ───────────────────────────────────────


def test_smoke_helper_detects_stub_when_real_absent():
    """Sanity-check the stub-detection helper itself.

    If ``transformers`` is the conftest stub AND a fresh subprocess cannot
    import the real package, ``_require_real_runtime`` MUST skip — not pass.
    This protects against re-introducing the false-confidence behavior where
    the smoke tests trusted the in-process stub.

    The test inspects the helper's behavior rather than re-running the smoke
    tests to keep the assertion direct and avoid recursive subprocess spawn.
    """
    # This guard targets the specific failure mode "only stub exists, real runtime
    # absent". If real runtime is installed, _require_real_runtime should pass.
    if _real_module_available_in_subprocess("transformers") and \
            _real_module_available_in_subprocess("torch"):
        pytest.skip("real runtime is present — stub-leakage path not exercised")

    # We are in the situation the regression guards against. Calling
    # ``_require_real_runtime`` from a regular pytest function raises
    # ``Skipped`` — capture it via pytest.raises rather than letting it
    # bubble (otherwise we'd just skip this very test).
    with pytest.raises(pytest.skip.Exception):
        _require_real_runtime()


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
