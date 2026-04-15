"""Unit tests for Phase B patcher and wording gate.

Tests covering actual model forwards (identity-patch equivalence, no-patch
generate byte-equality, all-clean-patch recipient equivalence, state_dict
stability) require ``torch`` and ``transformers`` to be importable. They are
auto-skipped when those packages are not available in the sandbox.

The forbidden-phrases gate test is pure-Python and always runs.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import pytest

# Make ``stage1`` importable when pytest is invoked from repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The wording gate has no heavy dependencies, so import at module load.
from stage1.utils.wording import FORBIDDEN_PHRASES, check_artifacts_for_forbidden

# Try to import heavy deps; skip GPU/model tests cleanly if missing.
# conftest.py injects a ``MagicMock`` stub for ``transformers`` at collection
# time when the real package wasn't yet imported; detect that case and drop
# the stub so the real package (if installed) can be resolved.
_HAS_TORCH = True
try:
    import importlib
    import importlib.util
    import sys as _sys
    import unittest.mock as _mock

    _tf_mod = _sys.modules.get("transformers")
    # conftest.py installs ``types.ModuleType("transformers")`` with MagicMock
    # attributes but no ``__file__``; real transformers always has ``__file__``.
    if _tf_mod is not None and not hasattr(_tf_mod, "__file__"):
        for _k in list(_sys.modules):
            if _k == "transformers" or _k.startswith("transformers."):
                del _sys.modules[_k]
    # Check on-disk installation (not sys.modules).
    _spec = importlib.machinery.PathFinder.find_spec("transformers")
    if _spec is None:
        raise ImportError("transformers not installed on disk")
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from transformers.models.qwen2 import Qwen2Config as _Q  # noqa: F401
except Exception:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch/transformers not installed in this environment",
)


# ─── Forbidden-phrases gate (pure Python) ────────────────────────────────────

def test_forbidden_phrases_gate(tmp_path):
    """Every canonical phrase must be detected as a violation.

    Spec §10 test #5.
    """
    violations_count = 0
    for phrase in FORBIDDEN_PHRASES:
        p = tmp_path / f"artifact_{violations_count}.txt"
        p.write_text(f"This text contains the literal phrase {phrase} for testing.",
                     encoding="utf-8")
        violations_count += 1

    paths = [str(tmp_path / f"artifact_{i}.txt") for i in range(len(FORBIDDEN_PHRASES))]
    violations = check_artifacts_for_forbidden(paths)

    # Each file has exactly one forbidden phrase, so len(violations) should
    # match len(FORBIDDEN_PHRASES). Some phrases overlap as substrings (e.g.,
    # "proves the mechanism" contains "proves mechanism"? — no, the former has
    # "the" in it. But "restoration effect" vs "restoration proportion" are
    # disjoint.) So the one-file-one-phrase fixture yields exactly N matches
    # provided no two phrases are substrings of each other. Assert >= N to
    # tolerate any accidental overlap, then verify every phrase appears.
    assert len(violations) >= len(FORBIDDEN_PHRASES)
    for phrase in FORBIDDEN_PHRASES:
        assert any(phrase in v for v in violations), (
            f"phrase {phrase!r} not flagged in any violation string"
        )


def test_forbidden_phrases_gate_skips_missing(tmp_path):
    """Missing paths are skipped silently (spec §8)."""
    ghost = str(tmp_path / "does_not_exist.txt")
    real = tmp_path / "clean.txt"
    real.write_text("Nothing inflammatory here.", encoding="utf-8")
    violations = check_artifacts_for_forbidden([ghost, str(real)])
    assert violations == []


def test_forbidden_phrases_gate_utf8(tmp_path):
    """Gate reads UTF-8 (Windows default cp1252 would crash on MGSM-zh)."""
    p = tmp_path / "zh.txt"
    p.write_text("中文测试 — proves the mechanism — 更多中文。",
                 encoding="utf-8")
    violations = check_artifacts_for_forbidden([str(p)])
    assert any("proves the mechanism" in v for v in violations)


# ─── Identity / equivalence tests (require torch) ────────────────────────────

@requires_torch
def test_identity_patch_equivalence():
    """forward_with_patches with patch_states={} equals model(input_ids).logits.

    Spec §10 test #1. Uses a tiny ad-hoc Qwen2 config to avoid network access.
    """
    import torch
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    from stage1.intervention.patcher import forward_with_patches

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rope_theta=10000.0,
    )
    model = Qwen2ForCausalLM(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)

    with torch.no_grad():
        ref_logits = model(input_ids=input_ids).logits
        final_hidden, _, _ = forward_with_patches(model, input_ids, patch_states={})
        # lm_head over the patched path to get logits for the same positions.
        patched_logits = model.lm_head(final_hidden)

    # Spec §10 test #1: fp32 tolerance 1e-5 (Round 2 tightening).
    max_abs_diff = (ref_logits - patched_logits).abs().max().item()
    assert max_abs_diff < 1e-5, f"logits diverged: max_abs_diff={max_abs_diff}"


@requires_torch
def test_empty_patch_generate_bytewise_equal():
    """no_patch greedy decode matches model.generate token-for-token.

    Spec §10 test #2.
    """
    import torch
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    from stage1.intervention.patcher import PatchConfig, run_patched_inference_single

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rope_theta=10000.0,
    )
    model = Qwen2ForCausalLM(cfg).eval()

    class DummyTokenizer:
        """Minimal tokenizer shim: encodes via Python ``ord`` ≤ 255 bytes."""
        eos_token_id = None

        def __call__(self, text, return_tensors=None, padding=False):
            ids = [b % cfg.vocab_size for b in text.encode("utf-8")[:6]]

            class _O:
                pass

            o = _O()
            o.input_ids = torch.tensor([ids], dtype=torch.long)
            return o

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(i)) for i in ids)

    tok = DummyTokenizer()
    pc = PatchConfig("no_patch", [], "restoration")

    prompt = "abcdef"
    input_ids = tok(prompt).input_ids
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        ref_out = model.generate(
            input_ids, do_sample=False, max_new_tokens=16,
            pad_token_id=cfg.eos_token_id or 0,
        )
    ref_new = ref_out[0, prompt_len:].tolist()

    result = run_patched_inference_single(
        model=model,
        tokenizer=tok,
        prompt=prompt,
        patch_config=pc,
        generation_config={"do_sample": False, "temperature": 0.0,
                           "max_new_tokens": 16},
    )

    # run_patched_inference_single decodes; we need the raw IDs. Re-decode via
    # the model directly using the same path and compare the first K tokens
    # that both produced. The test contract is byte-equality on IDs, not text,
    # so we instead re-run the patched path but retrieve IDs through a side
    # channel: patched_logits equivalence (test #1) + greedy argmax is
    # deterministic ⇒ if logits match and decode is argmax, IDs match. The
    # text round-trip via the dummy tokenizer preserves IDs as whitespace-
    # separated strings, so parse back.
    patched_new = [int(x) for x in result["output_text"].split()]

    # Compare up to the shorter length (early-stop on eos is OK).
    k = min(len(ref_new), len(patched_new))
    assert k > 0, "no tokens generated"
    assert ref_new[:k] == patched_new[:k], (
        f"byte-equality failed: ref={ref_new[:k]}, patched={patched_new[:k]}"
    )


@requires_torch
def test_all_clean_patch_matches_recipient():
    """Patching all layers with clean recipient states recovers recipient output.

    Spec §10 test #3. Uses two Qwen2 instances with different random weights as
    the composed/recipient pair — an all-layers patch overwrites every residual
    stream value, so the final hidden should be determined by the "clean"
    states, modulo the last-layer norm that still uses composed weights.

    Because composed.norm weights differ from recipient.norm weights, exact
    match of final_hidden is NOT expected here; we relax the assertion to the
    structural invariant that the patched final hidden EQUALS norm(recipient's
    last layer output), which is what our patching pipeline should produce.
    """
    import torch
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    from stage1.intervention.patcher import (
        extract_all_layer_hidden_states,
        forward_with_patches,
    )

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, rope_theta=10000.0,
    )
    recipient = Qwen2ForCausalLM(cfg).eval()
    torch.manual_seed(1)
    composed = Qwen2ForCausalLM(cfg).eval()

    class DummyTokenizer:
        eos_token_id = None

        def __call__(self, text, return_tensors=None, padding=False):
            ids = [b % cfg.vocab_size for b in text.encode("utf-8")[:6]]

            class _O:
                pass

            o = _O()
            o.input_ids = torch.tensor([ids], dtype=torch.long)
            return o

    tok = DummyTokenizer()
    prompt = "abcdef"

    clean_states = extract_all_layer_hidden_states(recipient, tok, prompt)
    patch_states = {i: clean_states[i] for i in range(cfg.num_hidden_layers)}

    input_ids = tok(prompt).input_ids
    with torch.no_grad():
        final_patched, all_out, _ = forward_with_patches(
            composed, input_ids, patch_states=patch_states,
        )
        # Expected: composed.norm applied to the recipient's last layer output.
        expected_hidden = composed.model.norm(
            clean_states[-1].to(final_patched.device).to(final_patched.dtype)
        )

    max_diff = (final_patched - expected_hidden).abs().max().item()
    assert max_diff < 1e-4, f"final hidden diverged: max_diff={max_diff}"

    # Also verify the last element of all_outputs (after patch at layer L-1)
    # equals the clean state at that layer (within dtype cast).
    last_patched = all_out[-1]
    max_diff_last = (last_patched - clean_states[-1]).abs().max().item()
    assert max_diff_last < 1e-5, (
        f"last layer output not overwritten by patch: max_diff_last={max_diff_last}"
    )


@requires_torch
def test_state_dict_hash_stable():
    """run_patched_inference must not mutate composed model weights in place.

    Spec §10 test #4.
    """
    import torch
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    from stage1.intervention.patcher import (
        PatchConfig,
        run_patched_inference,
    )

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, rope_theta=10000.0,
    )
    recipient = Qwen2ForCausalLM(cfg).eval()
    torch.manual_seed(1)
    composed = Qwen2ForCausalLM(cfg).eval()

    class DummyTokenizer:
        eos_token_id = None

        def __call__(self, text, return_tensors=None, padding=False):
            ids = [b % cfg.vocab_size for b in text.encode("utf-8")[:6]]

            class _O:
                pass

            o = _O()
            o.input_ids = torch.tensor([ids], dtype=torch.long)
            return o

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(i)) for i in ids)

    tok = DummyTokenizer()

    def _sha(m):
        h = hashlib.sha256()
        for k in sorted(m.state_dict().keys()):
            t = m.state_dict()[k]
            h.update(k.encode("utf-8"))
            h.update(t.detach().to("cpu").contiguous().numpy().tobytes())
        return h.hexdigest()

    before = _sha(composed)

    samples = [
        {"sample_id": "s0", "prompt": "abcdef", "gold_answer": "0"},
        {"sample_id": "s1", "prompt": "ghijkl", "gold_answer": "0"},
    ]
    pc = PatchConfig("patch_final_only", [cfg.num_hidden_layers - 1], "restoration")
    _ = run_patched_inference(
        target_model=composed,
        recipient_model=recipient,
        composed_model=None,
        tokenizer=tok,
        samples=samples,
        patch_config=pc,
        generation_config={"do_sample": False, "temperature": 0.0,
                           "max_new_tokens": 4},
    )

    after = _sha(composed)
    assert before == after, "composed model weights mutated during run"


# ─── Round 2 regression guards (codex A1/A2/A3 fixes) ────────────────────────

def _has_gpu_and_weights() -> bool:
    """Return True iff CUDA is available and Qwen2.5-1.5B-Instruct is cached.

    Used to gate the real-weights smoke test (spec §10). In a sandbox with
    no GPU / no pretrained weights, the smoke test cannot run and must be
    skipped; in production CI with the weights cached, it runs automatically.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        from transformers import AutoConfig
        AutoConfig.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True,
        )
        return True
    except Exception:
        return False


@requires_torch
def test_run_phase_b_module_imports():
    """Guard against import-path regressions in ``stage1.run_phase_b``.

    Codex adversarial finding A1: R1 used bare ``from utils.config import ...``
    which raises ``ModuleNotFoundError`` under the spec §10 canonical
    invocation ``python -m stage1.run_phase_b``. This test imports the module
    via ``importlib`` so any future drift to unqualified imports is caught at
    pytest time rather than at smoke-test time.

    The test specifically guards against the **bare-package** ModuleNotFoundError
    (``No module named 'utils'|'data'|'models'|'inference'|'analysis'|
    'intervention'``) that would be raised under the spec-canonical
    ``python -m stage1.run_phase_b`` invocation when imports are unqualified.
    If some OTHER dependency is missing from the sandbox (e.g., ``scipy``),
    the test skips cleanly — the regression it guards against is orthogonal.
    """
    import importlib
    import sys as _sys
    _sys.modules.pop("stage1.run_phase_b", None)
    _BARE = {"utils", "data", "models", "inference", "analysis", "intervention"}
    try:
        mod = importlib.import_module("stage1.run_phase_b")
    except ModuleNotFoundError as e:
        # Distinguish the regression we care about (bare top-level import
        # escaping the stage1.* namespace) from unrelated missing deps.
        missing = getattr(e, "name", "") or ""
        if missing.split(".", 1)[0] in _BARE and not missing.startswith("stage1"):
            raise AssertionError(
                f"Import-path regression detected: stage1.run_phase_b raised "
                f"ModuleNotFoundError for bare top-level '{missing}'. All "
                f"imports must be fully qualified as 'from stage1.<...>'."
            ) from e
        pytest.skip(f"unrelated dep missing in sandbox: {missing!r}")
    # Sanity: the canonical public surface must be reachable.
    assert hasattr(mod, "run_phase_b")
    assert hasattr(mod, "main")
    assert hasattr(mod, "EPSILON_DELTA")
    assert hasattr(mod, "PHASE_A_CROSS_CHECK_TOL")
    assert hasattr(mod, "_load_latest_phase_a_summary")
    assert hasattr(mod, "_phase_a_outputs_dir")


def _import_run_phase_b_or_skip():
    """Import ``stage1.run_phase_b`` or ``pytest.skip`` if sandbox deps miss."""
    import importlib
    import sys as _sys
    _sys.modules.pop("stage1.run_phase_b", None)
    try:
        return importlib.import_module("stage1.run_phase_b")
    except ModuleNotFoundError as e:
        pytest.skip(f"cannot import stage1.run_phase_b: {e}")


@requires_torch
def test_phase_a_loader_no_match_returns_none(monkeypatch, tmp_path):
    """Codex adversarial finding A2: the loader must NOT silently pass when
    no Phase A summary exists. It must return ``(None, None)`` and the caller
    (sanity checks) is responsible for turning that into an explicit FAIL in
    non-sanity runs.

    The fix resolves the glob relative to the repo root, not CWD. This test
    monkeypatches the outputs dir to an empty tmp path (simulating a machine
    without any prior Phase A run) and confirms the loader reports no match
    regardless of what the CWD is.
    """
    mod = _import_run_phase_b_or_skip()

    # Point the resolved outputs dir at an empty tmp path.
    monkeypatch.setattr(mod, "_phase_a_outputs_dir", lambda: str(tmp_path))
    # Also cd elsewhere to prove the loader is CWD-invariant.
    monkeypatch.chdir(tmp_path)
    summary, path = mod._load_latest_phase_a_summary()
    assert summary is None
    assert path is None


@requires_torch
def test_phase_a_loader_resolves_repo_relative(monkeypatch, tmp_path):
    """The loader uses the repo-root-relative path, so CWD changes do not
    cause silent misses. Sibling test to the no-match case above.
    """
    import json as _json
    mod = _import_run_phase_b_or_skip()

    # Create a fake Phase A outputs tree under tmp_path.
    run_dir = tmp_path / "run_20991231_000000"
    run_dir.mkdir()
    summary_payload = {
        "all_conditions": [
            {"condition": "hard_swap_b8", "accuracy": 0.32},
        ],
        "baseline_accuracy": 0.80,
    }
    (run_dir / "phase_a_summary.json").write_text(
        _json.dumps(summary_payload), encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_phase_a_outputs_dir", lambda: str(tmp_path))
    # Change CWD to somewhere unrelated — must not affect resolution.
    other = tmp_path.parent
    monkeypatch.chdir(other)
    summary, path = mod._load_latest_phase_a_summary()
    assert summary is not None
    assert summary["baseline_accuracy"] == 0.80
    assert path is not None
    assert "run_20991231_000000" in path


@pytest.mark.skipif(
    not _has_gpu_and_weights(),
    reason=(
        "Real-weights smoke test (spec §10) requires CUDA + cached "
        "Qwen/Qwen2.5-1.5B-Instruct weights. Run manually on production box: "
        "python -m stage1.run_phase_b "
        "--config stage1/configs/stage2_confound.yaml --sanity --seed 42"
    ),
)
def test_smoke_marker():
    """Auto-run the real-weights sanity smoke when the environment permits.

    Spec §10 / codex adversarial finding A3: the writer round-2 sandbox has
    neither CUDA nor pretrained weights, so this test skips there. In a CI
    or dev-box with both present, it exercises the end-to-end CLI entrypoint
    in sanity mode and asserts the four canonical artifacts are produced.
    """
    import subprocess
    import sys as _sys
    import glob as _glob
    import os as _os

    repo_root = _os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    )
    cfg = _os.path.join(
        repo_root, "stage1", "configs", "stage2_confound.yaml",
    )
    proc = subprocess.run(
        [_sys.executable, "-m", "stage1.run_phase_b",
         "--config", cfg, "--sanity", "--seed", "42"],
        cwd=repo_root, capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, (
        f"smoke test exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
    )
    # Verify the four canonical artifacts exist in the latest run dir.
    run_dirs = sorted(_glob.glob(
        _os.path.join(repo_root, "stage1", "outputs", "phase_b", "run_*")
    ))
    assert run_dirs, "no phase_b run directory produced"
    latest = run_dirs[-1]
    for name in (
        "phase_b_summary.txt", "phase_b_summary.json",
        "restoration_table.csv", "corruption_table.csv",
    ):
        assert _os.path.isfile(_os.path.join(latest, name)), (
            f"missing artifact: {name} in {latest}"
        )
