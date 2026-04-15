"""Prompt-side hidden-state patching for restoration interventions (Phase B).

Strategy: manual per-layer Qwen2 prompt forward pass with per-layer patching,
seeding a ``DynamicCache`` from the patched residual stream; then greedy
continuation that reuses that cache so the patch effect propagates to every
autoregressively generated token.

IMPORTANT METHODOLOGICAL CONSTRAINT:
    Clean hidden states are available for prompt tokens only. Therefore patching
    applies only to prompt-side processing. All output summaries must explicitly
    state this is prompt-side restoration intervention, NOT full-sequence causal
    intervention.

Usage (Jupyter):
    from stage1.intervention.patcher import PatchConfig, run_patched_inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# Methodology string emitted into every Phase B CSV row (spec §11.3).
METHODOLOGY_TAG: str = (
    "prompt-side patching; patch at prompt tokens only; "
    "continuation via DynamicCache"
)


# ─── Patch condition definitions ─────────────────────────────────────────────

# Restoration patches: inject clean (no_swap) states into composed (hard_swap) model
RESTORATION_PATCHES = {
    "no_patch":              [],
    "patch_boundary_local":  [7, 8, 9],
    "patch_recovery_early":  [20, 21, 22],
    "patch_recovery_full":   list(range(20, 28)),
    "patch_final_only":      [27],
    "patch_all_downstream":  list(range(8, 28)),
}

# Reverse corruption patches mirrored to match restoration granularity per
# spec §7 Finding #11(a).
REVERSE_CORRUPTION_PATCHES = {
    "corrupt_boundary_local":  [7, 8, 9],
    "corrupt_recovery_early":  [20, 21, 22],
    "corrupt_recovery_full":   list(range(20, 28)),
    "corrupt_final_only":      [27],
}


@dataclass
class PatchConfig:
    """Configuration for a single patching run."""
    patch_name: str
    patch_layers: List[int]
    direction: str = "restoration"  # "restoration" or "corruption"

    def __post_init__(self):
        if self.direction not in ("restoration", "corruption"):
            raise ValueError(f"Invalid direction: {self.direction}")


def get_all_patch_configs() -> List[PatchConfig]:
    """Return all patch configurations (restoration + reverse corruption)."""
    configs = []
    for name, layers in RESTORATION_PATCHES.items():
        configs.append(PatchConfig(name, layers, "restoration"))
    for name, layers in REVERSE_CORRUPTION_PATCHES.items():
        configs.append(PatchConfig(name, layers, "corruption"))
    return configs


# ─── Qwen2 plumbing helpers ──────────────────────────────────────────────────

def _get_model_components(model: AutoModelForCausalLM):
    """Return (embed_tokens, layers, norm, rotary_emb) from a Qwen2-family model."""
    inner = model.model
    rotary = getattr(inner, "rotary_emb", None)
    if rotary is None:
        raise AttributeError(
            "Expected model.model.rotary_emb on a Qwen2-family model; "
            "transformers version may have moved the rotary embedding."
        )
    return inner.embed_tokens, inner.layers, inner.norm, rotary


def _build_causal_mask(
    model: AutoModelForCausalLM,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Build a 4D causal attention mask, accommodating both known transformers paths.

    Spec §12 R1: try ``model.model._update_causal_mask`` first (older path, 4.38–4.44),
    fall back to ``transformers.masking_utils.create_causal_mask`` (4.45+/5.x).
    If neither is available, fail-fast with a clear message.
    """
    inner = model.model
    if hasattr(inner, "_update_causal_mask"):
        return inner._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, False
        )
    try:
        from transformers.masking_utils import create_causal_mask  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Neither model.model._update_causal_mask nor "
            "transformers.masking_utils.create_causal_mask is available. "
            "Unsupported transformers version."
        ) from exc
    return create_causal_mask(
        config=model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


def _build_prompt_inputs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    past_key_values,
) -> Dict[str, torch.Tensor]:
    """Precompute the tensors every Qwen2 decoder layer needs for the prompt forward.

    Returns a dict with keys:
        - ``hidden``: embed_tokens(input_ids) output, [1, S, H]
        - ``attention_mask_4d``: 4D causal mask, broadcastable to [1, 1, S, S]
        - ``position_ids``: [1, S]
        - ``position_embeddings``: tuple ``(cos, sin)`` from ``rotary_emb``
        - ``cache_position``: [S]
    """
    embed_tokens, _, _, rotary_emb = _get_model_components(model)

    batch, seq = input_ids.shape
    device = input_ids.device

    hidden = embed_tokens(input_ids)

    attention_mask_2d = torch.ones((batch, seq), dtype=torch.long, device=device)
    cache_position = torch.arange(seq, device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0).expand(batch, -1)

    attention_mask_4d = _build_causal_mask(
        model=model,
        inputs_embeds=hidden,
        attention_mask=attention_mask_2d,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # rotary_emb signature stabilised on (hidden_states, position_ids) → (cos, sin).
    cos, sin = rotary_emb(hidden, position_ids)
    position_embeddings = (cos, sin)

    return {
        "hidden": hidden,
        "attention_mask_4d": attention_mask_4d,
        "position_ids": position_ids,
        "position_embeddings": position_embeddings,
        "cache_position": cache_position,
    }


# ─── Layer-by-layer forward with patching ────────────────────────────────────

def forward_with_patches(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    patch_states: Dict[int, torch.Tensor],
    *,
    return_cache: bool = True,
):
    """Manual per-layer Qwen2 prompt forward with hidden-state patching.

    Patches hidden states AFTER each specified layer; the patched hidden is
    passed to the next layer, so downstream KV cache entries are computed from
    the patched residual stream. The ``DynamicCache`` returned is populated by
    passing ``past_key_values=cache`` and ``use_cache=True`` to each layer call
    and is suitable for seeding a greedy continuation.

    Args:
        model: The model to run forward through (composed or recipient).
        input_ids: [1, S] input token IDs on the model's device.
        patch_states: {layer_index: tensor [1, S, H]} injected after that layer.
        return_cache: When False, returns a ``None`` in place of the cache tuple
            element (used by equivalence tests that only care about final hidden).

    Returns:
        ``(final_hidden, all_layer_outputs, cache)`` where:
          - ``final_hidden`` is ``norm(h_after_last_layer)`` shaped [1, S, H].
          - ``all_layer_outputs`` is a Python list of per-layer hidden states
            (detached, CPU) captured AFTER any patch at that layer.
          - ``cache`` is a ``DynamicCache`` populated with prompt KV, or None
            if ``return_cache=False``.
    """
    from transformers import DynamicCache  # local import for sandboxed envs

    _, layers, norm, _ = _get_model_components(model)

    cache = DynamicCache() if return_cache else None
    inputs = _build_prompt_inputs(model, input_ids, past_key_values=cache)
    hidden = inputs["hidden"]
    attention_mask_4d = inputs["attention_mask_4d"]
    position_ids = inputs["position_ids"]
    position_embeddings = inputs["position_embeddings"]
    cache_position = inputs["cache_position"]

    all_outputs: List[torch.Tensor] = []

    for layer_idx, layer in enumerate(layers):
        layer_output = layer(
            hidden,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=(cache is not None),
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        # Qwen2 layer returns hidden tensor directly in 4.45+/5.x; older versions
        # return a tuple. Handle both.
        if isinstance(layer_output, tuple):
            hidden = layer_output[0]
        else:
            hidden = layer_output

        if layer_idx in patch_states:
            patch = patch_states[layer_idx]
            if patch.shape != hidden.shape:
                raise ValueError(
                    f"Patch shape mismatch at layer {layer_idx}: "
                    f"expected {hidden.shape}, got {patch.shape}"
                )
            hidden = patch.to(device=hidden.device, dtype=hidden.dtype)

        all_outputs.append(hidden.detach().to("cpu"))

    final_hidden = norm(hidden)
    return final_hidden, all_outputs, cache


# ─── Extract per-layer prompt hidden states ──────────────────────────────────

def extract_all_layer_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device=None,
) -> List[torch.Tensor]:
    """Run a full prompt-only forward and return per-layer hidden states.

    Returns:
        List of [1, S, H] tensors (CPU), one per transformer layer.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer(prompt, return_tensors="pt", padding=False).input_ids.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    # outputs.hidden_states: tuple of (n_layers+1,) tensors [1, S, H].
    # Index 0 = embedding output; 1..n_layers = transformer layer outputs.
    n = model.config.num_hidden_layers
    return [outputs.hidden_states[i + 1].detach().to("cpu") for i in range(n)]


# ─── Main patched inference ──────────────────────────────────────────────────

def _greedy_continue_with_cache(
    model: AutoModelForCausalLM,
    first_token_id: torch.Tensor,
    cache,
    prompt_len: int,
    max_new_tokens: int,
    eos_token_id: Optional[int],
) -> torch.Tensor:
    """Manual greedy decode loop reusing a prompt-seeded ``DynamicCache``.

    Per the project constraint in ``CLAUDE.md``/spec: do NOT fall back to
    ``model.generate(current_ids, ...)`` after patching — that would re-run the
    prompt tokens through the unpatched forward and discard the patch.

    Args:
        model: The patched model.
        first_token_id: [1, 1] long tensor — the first generated token produced
            from the patched prompt's final hidden state.
        cache: ``DynamicCache`` populated with the prompt KV (from
            ``forward_with_patches``).
        prompt_len: Number of prompt tokens already in the cache.
        max_new_tokens: Total max new tokens INCLUDING ``first_token_id``.
        eos_token_id: Stop token. ``None`` disables early stop.

    Returns:
        LongTensor [K] of generated token IDs (``K <= max_new_tokens``), on CPU.
    """
    device = first_token_id.device
    generated = [first_token_id.view(1, 1)]
    current = first_token_id.view(1, 1)

    remaining = max_new_tokens - 1
    for step in range(remaining):
        if eos_token_id is not None and int(current.item()) == int(eos_token_id):
            break
        # The first generated token occupies absolute position `prompt_len`
        # (prompt tokens live at 0..prompt_len-1). At loop step=0 we are feeding
        # that first token into the model to compute its representation, so the
        # correct RoPE / cache index is `prompt_len + step`, NOT prompt_len+1+step.
        pos_idx = prompt_len + step
        cache_position = torch.tensor([pos_idx], device=device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                input_ids=current,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_ids=position_ids,
            )
        logits = outputs.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
        generated.append(next_id)
        current = next_id

    return torch.cat(generated, dim=1).squeeze(0).to("cpu")


def run_patched_inference_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    patch_config: PatchConfig,
    clean_layer_states: Optional[List[torch.Tensor]] = None,
    corrupt_layer_states: Optional[List[torch.Tensor]] = None,
    generation_config: Optional[dict] = None,
    device=None,
) -> Dict:
    """Run patched inference on a single sample.

    For ``no_patch`` (empty ``patch_layers``), the path is still manual:
    we call ``forward_with_patches`` with ``patch_states={}`` to build the
    ``DynamicCache``, then greedy-decode. This guarantees byte-equivalence with
    ``model.generate(input_ids)`` under greedy decoding (test #2) while keeping
    the code path uniform.

    Args:
        model: The model to run inference on.
        tokenizer: Tokenizer.
        prompt: Input prompt string.
        patch_config: PatchConfig specifying which layers to patch.
        clean_layer_states: Per-layer hidden states from recipient (no_swap).
            Required for restoration patches with non-empty layers.
        corrupt_layer_states: Per-layer hidden states from composed (hard_swap).
            Required for corruption patches with non-empty layers.
        generation_config: Generation parameters dict.
        device: Device override.

    Returns:
        Dict with ``output_text``, ``patched_layers``, ``methodology``,
        ``direction``, ``patch_name``.
    """
    if device is None:
        device = next(model.parameters()).device

    gen_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 512,
    }
    if generation_config:
        gen_kwargs["do_sample"] = generation_config.get("do_sample", False)
        gen_kwargs["temperature"] = generation_config.get("temperature", 0.0)
        gen_kwargs["max_new_tokens"] = generation_config.get("max_new_tokens", 512)

    if gen_kwargs["do_sample"]:
        raise ValueError(
            "Phase B greedy-only path does not support sampling "
            "(config.generation.do_sample must be False)."
        )

    input_ids = tokenizer(prompt, return_tensors="pt", padding=False).input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Build patch_states dict
    patch_states: Dict[int, torch.Tensor] = {}
    for layer_idx in patch_config.patch_layers:
        if patch_config.direction == "restoration":
            if clean_layer_states is None:
                raise ValueError("clean_layer_states required for restoration patches")
            patch_states[layer_idx] = clean_layer_states[layer_idx].to(device)
        else:  # corruption
            if corrupt_layer_states is None:
                raise ValueError("corrupt_layer_states required for corruption patches")
            patch_states[layer_idx] = corrupt_layer_states[layer_idx].to(device)

    model.eval()
    with torch.no_grad():
        final_hidden, _, cache = forward_with_patches(
            model, input_ids, patch_states, return_cache=True
        )
        # First token comes from lm_head applied to the patched final hidden.
        first_token_logits = model.lm_head(final_hidden[:, -1, :])
        first_token_id = torch.argmax(first_token_logits, dim=-1).view(1, 1)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        generated_ids = _greedy_continue_with_cache(
            model=model,
            first_token_id=first_token_id,
            cache=cache,
            prompt_len=prompt_len,
            max_new_tokens=gen_kwargs["max_new_tokens"],
            eos_token_id=eos_id,
        )

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "output_text": output_text,
        "patched_layers": list(patch_config.patch_layers),
        "patch_name": patch_config.patch_name,
        "direction": patch_config.direction,
        "methodology": METHODOLOGY_TAG,
    }


def run_patched_inference(
    target_model: AutoModelForCausalLM,
    recipient_model: AutoModelForCausalLM,
    composed_model: Optional[AutoModelForCausalLM],
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    patch_config: PatchConfig,
    generation_config: Optional[dict] = None,
) -> List[Dict]:
    """Run patched inference across all samples for a given patch condition.

    For restoration patches, ``target_model = composed_model``, patching with
    clean states from ``recipient_model``.
    For corruption patches, ``target_model = recipient_model``, patching with
    corrupt states from ``composed_model``.

    States are extracted ONCE per sample per direction and freed immediately
    after the patched forward (spec §12 R5 — no cross-sample state caching).

    INTENTIONAL DESIGN — why we recompute per sample instead of loading
    Stage-1-style artifacts from disk:
      * Stage 1 saves prompt-POOLED hidden states ``[n_layers, hidden_dim]``
        (one vector per layer, last-token anchor). Per-layer prompt-PATCHING
        needs full-sequence states ``[1, seq_len, hidden_dim]`` to rebuild
        the DynamicCache correctly.
      * Shape incompatibility means ``torch.load(hidden_states_no_swap.pt)``
        cannot be used as a drop-in source — it would silently degrade to
        broadcasting a single vector over the sequence axis.
      * Spec §12 R5 also forbids cross-sample caching to keep memory bounded
        for long runs. Recomputing per sample is therefore both correct
        (shape-faithful) and compliant (memory-bounded).
    If a future Phase-B-produced full-sequence artifact is ever persisted,
    add a loader here guarded by an explicit ``map_location="cpu"`` + shape
    assertion (``tensor.shape == (1, seq_len, hidden_size)``) and a per-
    sample alignment check by ``sample_id``.

    Args:
        target_model: The model receiving patches.
        recipient_model: Clean (no_swap) model source of clean states.
        composed_model: Composed (hard_swap) model source of corrupt states.
            Required only for corruption patches with non-empty layers.
        tokenizer: Tokenizer.
        samples: List of sample dicts (each with ``sample_id``, ``prompt``,
            ``gold_answer``).
        patch_config: PatchConfig for this run.
        generation_config: Generation parameters.

    Returns:
        List of per-sample result dicts in the same order as ``samples``.
    """
    target_model.eval()
    recipient_model.eval()
    if composed_model is not None:
        composed_model.eval()

    results: List[Dict] = []
    device = next(target_model.parameters()).device

    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]

        clean_states: Optional[List[torch.Tensor]] = None
        corrupt_states: Optional[List[torch.Tensor]] = None

        if patch_config.direction == "restoration" and patch_config.patch_layers:
            clean_states = extract_all_layer_hidden_states(
                recipient_model, tokenizer, prompt, device=device,
            )
        elif patch_config.direction == "corruption" and patch_config.patch_layers:
            if composed_model is None:
                raise ValueError("composed_model required for corruption patches")
            corrupt_states = extract_all_layer_hidden_states(
                composed_model, tokenizer, prompt, device=device,
            )

        result = run_patched_inference_single(
            model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            patch_config=patch_config,
            clean_layer_states=clean_states,
            corrupt_layer_states=corrupt_states,
            generation_config=generation_config,
            device=device,
        )

        # Free per-sample states immediately (spec §12 R5).
        del clean_states, corrupt_states

        result["sample_id"] = sample["sample_id"]
        result["patch_condition"] = patch_config.patch_name
        results.append(result)

        logger.info(
            "[%d/%d] %s | %s",
            idx + 1, len(samples), sample["sample_id"], patch_config.patch_name,
        )

    return results
