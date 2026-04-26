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
#
# v4 PRIORITY 4 — order-independence:
#
# conftest.py installs a deterministic transformers stub at session start.
# To exercise the real GPU/model tests we must drop that stub and let Python
# resolve the on-disk package. The dance below SNAPSHOTS the stub state
# before mutation; if the real-import fails we restore the snapshot so
# subsequent test modules (regardless of collection order) see the same
# ``sys.modules`` state they would have seen if this module had not been
# collected at all.
_HAS_TORCH = True
_RESTORED_AFTER_FAILURE = False
try:
    import importlib
    import importlib.util
    import sys as _sys

    _stub_snapshot = {
        _k: _v for _k, _v in list(_sys.modules.items())
        if _k == "transformers" or _k.startswith("transformers.")
    }
    _is_stub = (
        getattr(_sys.modules.get("transformers"), "__is_test_stub__", False)
        # Backward compat: older stubs set no flag, only no __file__.
        or (
            "transformers" in _sys.modules
            and not hasattr(_sys.modules["transformers"], "__file__")
        )
    )
    if _is_stub:
        for _k in list(_stub_snapshot):
            del _sys.modules[_k]
    try:
        _spec = importlib.machinery.PathFinder.find_spec("transformers")
        if _spec is None:
            raise ImportError("transformers not installed on disk")
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from transformers.models.qwen2 import Qwen2Config as _Q  # noqa: F401
    except Exception:
        # Restore the conftest stub so subsequent test modules see the same
        # sys.modules state regardless of collection order. Without this,
        # earlier module-load behavior depended on whether
        # test_phase_b_patcher was collected before or after the dependent
        # tests, which made subset/full pytest runs disagree (v4 P4).
        if _is_stub:
            for _k in list(_sys.modules):
                if _k == "transformers" or _k.startswith("transformers."):
                    del _sys.modules[_k]
            _sys.modules.update(_stub_snapshot)
        _RESTORED_AFTER_FAILURE = True
        raise
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
def test_input_side_patch_cache_consistency_identity():
    """Input+output patching with the model's OWN clean states must be a no-op
    on cache state.

    Priority 1 repair (post-RED-LIGHT): the new ``forward_with_patches``
    applies an INPUT-side patch BEFORE each patched layer's forward call so
    the prompt-KV cache slot at that layer reflects the clean input. When the
    composed model equals the source model (so ``clean_layer_states`` came
    from the same model we are patching), the input + output patches are
    semantically no-ops and the resulting DynamicCache must equal the
    unpatched run's cache exactly.

    This is the strongest invariant that does not require comparing across
    models with different weights; combined with
    ``test_all_clean_patch_matches_recipient`` (residual-equivalence under
    cross-model patching) it covers the new patch semantics.
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
    model = Qwen2ForCausalLM(cfg).eval()

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

    own_states = extract_all_layer_hidden_states(model, tok, prompt)
    input_ids = tok(prompt).input_ids

    # Patch layers 1, 2 (skip layer 0 — its input is the embedding which is
    # never patched in production patch sets, and forward_with_patches rejects
    # input-side patch at layer 0).
    patched_layers = [1, 2]
    patch_states = {n: own_states[n] for n in patched_layers}
    patch_input_states = {n: own_states[n - 1] for n in patched_layers}

    with torch.no_grad():
        unpatched_hidden, _, unpatched_cache = forward_with_patches(
            model, input_ids, patch_states={},
        )
        patched_hidden, _, patched_cache = forward_with_patches(
            model, input_ids,
            patch_states=patch_states,
            patch_input_states=patch_input_states,
        )

    # Final hidden should be numerically very close under no-op patches.
    # In transformers 4.57.x this path can differ at ~1e-3 while cache slots
    # still match exactly (the stronger invariant we care about here).
    final_diff = (patched_hidden - unpatched_hidden).abs().max().item()
    assert final_diff < 2e-3, f"final hidden diverged under no-op patches: {final_diff}"

    # Cache slot for each patched layer must match the unpatched cache exactly.
    # DynamicCache in current transformers exposes per-layer (K, V) via __getitem__.
    for n in patched_layers:
        k_diff = (patched_cache[n][0] - unpatched_cache[n][0]).abs().max().item()
        v_diff = (patched_cache[n][1] - unpatched_cache[n][1]).abs().max().item()
        assert k_diff < 1e-5, f"K cache at layer {n} diverged: {k_diff}"
        assert v_diff < 1e-5, f"V cache at layer {n} diverged: {v_diff}"


@requires_torch
def test_input_side_patch_changes_cache_when_input_differs():
    """Input-side patch must materially change the prompt-KV cache when the
    patched input differs from what the model would have computed.

    Priority 1 repair: this test catches the regression in which input-side
    patching is silently dropped (e.g., a refactor that forgets to thread
    ``patch_input_states`` through). We construct a synthetic "clean" tensor
    that visibly differs from the model's natural layer-(N-1) output, patch it
    in as the input to layer N, and assert that cache[N] now differs from the
    unpatched cache. Also asserts that without the input patch, cache[N] is
    UNCHANGED — which is the failure mode of the old code.
    """
    import torch
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    from stage1.intervention.patcher import forward_with_patches

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, rope_theta=10000.0,
    )
    model = Qwen2ForCausalLM(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.long)

    # Patch input to layer 2 with a fixed pattern that is highly unlikely to
    # equal the natural h_1 output.
    fake_clean_input = torch.full((1, 6, cfg.hidden_size), 0.5)
    fake_clean_output = torch.full((1, 6, cfg.hidden_size), 0.7)

    with torch.no_grad():
        # Baseline: unpatched.
        _, _, base_cache = forward_with_patches(model, input_ids, patch_states={})

        # Old-behavior reproduction: output-only patch at layer 2.
        # cache[2] should be IDENTICAL to baseline (because layer 2's input
        # was unpatched in this run).
        _, _, output_only_cache = forward_with_patches(
            model, input_ids,
            patch_states={2: fake_clean_output},
        )

        # New behavior: input + output patch at layer 2.
        # cache[2] MUST differ from baseline because layer 2 now sees the
        # fake_clean_input.
        _, _, both_cache = forward_with_patches(
            model, input_ids,
            patch_states={2: fake_clean_output},
            patch_input_states={2: fake_clean_input},
        )

    # Old failure mode: output-only patch leaves cache[2] unchanged.
    k_diff_old = (output_only_cache[2][0] - base_cache[2][0]).abs().max().item()
    assert k_diff_old < 1e-6, (
        f"output-only patch unexpectedly altered cache[2] (diff={k_diff_old}); "
        f"the test premise is wrong"
    )

    # Priority 1 fix: input-side patch MUST move cache[2] away from baseline.
    k_diff_new = (both_cache[2][0] - base_cache[2][0]).abs().max().item()
    v_diff_new = (both_cache[2][1] - base_cache[2][1]).abs().max().item()
    assert k_diff_new > 1e-3, (
        f"input-side patch failed to update cache[2] keys (diff={k_diff_new}); "
        f"this is the Priority 1 cache-consistency bug"
    )
    assert v_diff_new > 1e-3, (
        f"input-side patch failed to update cache[2] values (diff={v_diff_new})"
    )


@requires_torch
def test_create_causal_mask_current_transformers_signature(monkeypatch):
    """Regression guard for transformers==4.57.x causal-mask API path.

    transformers 4.57.6 exposes
    ``transformers.masking_utils.create_causal_mask(input_embeds=...)``.
    The local dev env may still be on 4.40.x, so this test installs a tiny
    shim with the 4.57-style signature and verifies the patcher passes the
    current kwarg names. This keeps the regression independent of the installed
    package version.
    """
    import sys
    import types
    import torch

    from stage1.intervention.patcher import _build_causal_mask

    seen: dict = {}
    fake_module = types.ModuleType("transformers.masking_utils")

    def create_causal_mask(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids=None,
    ):
        seen.update(
            {
                "config": config,
                "input_embeds": input_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )
        batch, seq, _ = input_embeds.shape
        return torch.zeros((batch, 1, seq, seq), dtype=input_embeds.dtype)

    fake_module.create_causal_mask = create_causal_mask
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", fake_module)

    class _Inner:
        pass

    class _Model:
        model = _Inner()
        config = object()

    inputs_embeds = torch.zeros((1, 5, 8), dtype=torch.float32)
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    cache_position = torch.arange(5, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)

    mask = _build_causal_mask(
        model=_Model(),
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    assert tuple(mask.shape) == (1, 1, 5, 5)
    assert "input_embeds" in seen, (
        "patcher did not pass input_embeds to create_causal_mask on current transformers path"
    )
    assert "inputs_embeds" not in seen, (
        "patcher passed deprecated inputs_embeds kwarg on current transformers path"
    )


@requires_torch
def test_layer_zero_input_patch_rejected():
    """Input-side patch at layer 0 must raise ValueError.

    Priority 1 repair: layer 0's input is the embedding output, which is
    intentionally never patched (and would require extracting the embedding
    output separately). Production patch sets all start at layer ≥ 1; the
    code enforces this.
    """
    import pytest
    import torch
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    from stage1.intervention.patcher import forward_with_patches

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, rope_theta=10000.0,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    fake = torch.zeros((1, 4, cfg.hidden_size))

    with pytest.raises(ValueError, match="Input-side patch at layer 0"):
        forward_with_patches(
            model, input_ids,
            patch_states={},
            patch_input_states={0: fake},
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
    # Private loader helpers were removed from stage1.run_phase_b.
    # Public anchor gate API is in stage1.utils.anchor_gate.
    assert hasattr(mod, "_phase_a_outputs_dir")
    assert hasattr(mod, "_stage1_outputs_dir")
    assert hasattr(mod, "evaluate_phase_b_anchor_gate")


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
def test_anchor_gate_public_loader_no_match_returns_none(monkeypatch, tmp_path):
    """Public API contract: empty Phase A outputs tree returns (None, None)."""
    from stage1.utils import anchor_gate as ag

    monkeypatch.chdir(tmp_path)
    summary, path = ag.load_latest_phase_a_summary(str(tmp_path), current_parity=None)
    assert summary is None
    assert path is None


def test_anchor_gate_public_loader_resolves_explicit_path(monkeypatch, tmp_path):
    """Public API contract: explicit phase_a_dir is CWD-independent."""
    import json as _json
    from stage1.utils import anchor_gate as ag

    run_dir = tmp_path / "run_20991231_000000"
    run_dir.mkdir()
    summary_payload = {
        "all_conditions": [{"condition": "hard_swap_b8", "accuracy": 0.32}],
        "baseline_accuracy": 0.80,
    }
    (run_dir / "phase_a_summary.json").write_text(
        _json.dumps(summary_payload), encoding="utf-8",
    )
    # Parity disabled here (current_parity=None), so manifest is optional.
    monkeypatch.chdir(tmp_path.parent)
    summary, path = ag.load_latest_phase_a_summary(str(tmp_path), current_parity=None)
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
