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
import inspect
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
    layers = inner.layers
    rotary = getattr(inner, "rotary_emb", None)
    if rotary is None and layers:
        # transformers 4.40.x stores rotary_emb under each attention module;
        # 4.45+/4.57.x exposes it as model.model.rotary_emb.
        rotary = getattr(getattr(layers[0], "self_attn", None), "rotary_emb", None)
    if rotary is None:
        raise AttributeError(
            "Expected Qwen2 rotary_emb on model.model or layer.self_attn; "
            "transformers version may have moved the rotary embedding."
        )
    return inner.embed_tokens, layers, inner.norm, rotary


def _past_length(past_key_values) -> int:
    """Best-effort cache length helper across transformers cache variants."""
    if past_key_values is None:
        return 0
    for method in ("get_seq_length", "get_usable_length"):
        fn = getattr(past_key_values, method, None)
        if fn is None:
            continue
        try:
            return int(fn())
        except TypeError:
            try:
                return int(fn(0))
            except Exception:
                continue
        except Exception:
            continue
    return 0


def _legacy_4d_causal_mask(
    model: AutoModelForCausalLM,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
) -> torch.Tensor:
    """Build a 4D causal mask when transformers.masking_utils is absent."""
    batch, seq, _ = inputs_embeds.shape
    past_len = _past_length(past_key_values)
    try:
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask,
        )

        return _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch, seq),
            inputs_embeds,
            past_len,
            sliding_window=getattr(model.config, "sliding_window", None),
        )
    except Exception:
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        target_len = past_len + seq
        q_positions = (
            torch.arange(seq, device=inputs_embeds.device).view(seq, 1) + past_len
        )
        k_positions = torch.arange(target_len, device=inputs_embeds.device).view(
            1, target_len
        )
        blocked = k_positions > q_positions
        mask = torch.zeros(
            (batch, 1, seq, target_len),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        return mask.masked_fill(blocked.view(1, 1, seq, target_len), min_dtype)


def _build_causal_mask(
    model: AutoModelForCausalLM,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Build a 4D causal attention mask across transformers API variants.

    First try ``model.model._update_causal_mask`` (older path). If unavailable,
    use ``transformers.masking_utils.create_causal_mask`` and adapt keyword names
    via signature introspection (for example ``input_embeds`` vs
    ``inputs_embeds`` across releases).
    """
    inner = model.model
    if hasattr(inner, "_update_causal_mask"):
        return inner._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, False
        )
    try:
        from transformers.masking_utils import create_causal_mask  # type: ignore
    except ImportError:
        return _legacy_4d_causal_mask(
            model=model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

    params = inspect.signature(create_causal_mask).parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    def accepts(name: str) -> bool:
        return has_var_kw or name in params

    kwargs = {}
    if accepts("config"):
        kwargs["config"] = model.config
    if accepts("attention_mask"):
        kwargs["attention_mask"] = attention_mask
    if accepts("cache_position"):
        kwargs["cache_position"] = cache_position
    if accepts("position_ids"):
        kwargs["position_ids"] = position_ids

    # transformers API drift: input_embeds (newer) vs inputs_embeds (older).
    if accepts("input_embeds"):
        kwargs["input_embeds"] = inputs_embeds
    elif accepts("inputs_embeds"):
        kwargs["inputs_embeds"] = inputs_embeds
    else:
        raise RuntimeError(
            "create_causal_mask signature is missing input_embeds/inputs_embeds"
        )

    # Keep backward compatibility if cache arg name changes.
    if accepts("past_key_values"):
        kwargs["past_key_values"] = past_key_values
    elif accepts("past_key_value"):
        kwargs["past_key_value"] = past_key_values
    elif accepts("cache"):
        kwargs["cache"] = past_key_values
    else:
        raise RuntimeError(
            "create_causal_mask signature is missing past_key_values-compatible parameter"
        )

    try:
        return create_causal_mask(**kwargs)
    except TypeError as exc:
        try:
            return create_causal_mask(
                model.config,
                inputs_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
            )
        except TypeError:
            raise RuntimeError(
                "Could not call transformers.masking_utils.create_causal_mask "
                f"with signature {inspect.signature(create_causal_mask)}"
            ) from exc


def _layer_forward_params(layer) -> Dict[str, inspect.Parameter]:
    return dict(inspect.signature(layer.forward).parameters)


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
    embed_tokens, layers, _, rotary_emb = _get_model_components(model)

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

    position_embeddings = None
    if layers and "position_embeddings" in _layer_forward_params(layers[0]):
        # transformers 4.45+/4.57.x computes RoPE once at model level and
        # passes (cos, sin) into each decoder layer. transformers 4.40.x
        # computes RoPE inside each attention module, so this remains None.
        try:
            position_embeddings = rotary_emb(hidden, position_ids)
        except TypeError:
            position_embeddings = rotary_emb(hidden, seq_len=seq)

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
    patch_input_states: Optional[Dict[int, torch.Tensor]] = None,
    return_cache: bool = True,
):
    """Manual per-layer Qwen2 prompt forward with hidden-state patching.

    PATCH SEMANTICS (post-RED-LIGHT repair, Priority 1):
        For each patched layer N we apply BOTH an INPUT-side and an OUTPUT-side
        patch. The input-side patch is what fixes the prompt-KV-cache
        inconsistency that the previous "output-only" implementation had. The
        output-side patch makes the residual flowing into layer N+1 explicitly
        equal to the clean residual at that boundary.

        Concretely, with ``hidden`` denoting the residual at the layer-N input:
            1. If ``layer_idx in patch_input_states``: replace ``hidden`` with
               ``patch_input_states[layer_idx]`` BEFORE calling layer N. This
               causes layer N to project K and V from clean input, so the
               ``DynamicCache`` slot for layer N stores keys/values that match
               what the clean model would have written. This is the fix for
               continuation-time attention (autoregressive tokens read prompt
               KVs, so cache consistency is required for the patch to persist
               beyond the first generated token).
            2. After layer N's forward, if ``layer_idx in patch_states``,
               replace the residual with ``patch_states[layer_idx]`` so the
               input to layer N+1 is exactly the clean residual at boundary N.

        For a patched layer set {N0, N1, ...} with N0 the smallest, only the
        input patch at N0 is strictly necessary for cache consistency (because
        the output patch at Nk-1 already provides the clean input to Nk for
        k≥1). We apply input patches at every patched layer for symmetry and
        to keep the implementation order-independent.

    OLD BUG (for the record):
        The previous implementation called layer N first, then overrode the
        hidden state. Cache[N] therefore stored K/V projected from the *input
        to* layer N, which was the unpatched residual. For ``patch_final_only
        [27]`` this meant the patch only altered the first-token logit (via
        lm_head on the patched final hidden); every subsequent autoregressive
        token attended to unpatched K_27 in the prompt cache, effectively
        nullifying the intervention beyond token 0. For multi-layer patches
        such as ``patch_recovery_full [20..27]`` the inconsistency was bounded
        to cache[20] (downstream cache entries became consistent because their
        layer inputs came from the patched residual chain).

    WHAT IS AND IS NOT PATCHED (under the new semantics):
        Patched per layer N in patch_states:
            - cache slot N (K/V projected from clean h_{N-1})
            - residual carried into layer N+1 (= clean h_N)
        NOT patched:
            - layer N's own weights (this is an activation patch, not a
              weight transplant)
            - autoregressive continuation tokens' own forward (the composed
              model's weights still process them; the patch propagates only
              through the prompt-KV cache they attend to)
            - the embedding lookup (patches starting at layer 0 are rejected
              upstream by ``run_patched_inference_single``)
        Limitations:
            - This remains a prompt-side intervention. Generation tokens are
              affected only via the patched prompt KVs they attend to.
            - Restoration interpretation is "patch the residual stream from
              boundary N onward to be clean, with cache state consistent with
              that residual". It is NOT "transplant clean weights into layer
              N" and is NOT a formal causal-mediation identification.

    Args:
        model: The model to run forward through (composed or recipient).
        input_ids: [1, S] input token IDs on the model's device.
        patch_states: {layer_index: tensor [1, S, H]} OUTPUT-side patch —
            replaces the residual AFTER layer N's forward.
        patch_input_states: {layer_index: tensor [1, S, H]} INPUT-side patch —
            replaces the residual BEFORE layer N's forward, so layer N writes
            consistent K/V to the cache. Must satisfy ``layer_index >= 1``
            (layer 0's input is the embedding output, which is never patched
            in any production patch set; ``run_patched_inference_single``
            rejects layer-0 patches upstream).
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
    if patch_input_states is None:
        patch_input_states = {}

    for layer_idx, layer in enumerate(layers):
        # INPUT-side patch (Priority 1 repair): replace residual BEFORE the
        # layer's forward so K/V written to cache reflect the clean input.
        if layer_idx in patch_input_states:
            if layer_idx == 0:
                raise ValueError(
                    "Input-side patch at layer 0 is not supported (the input "
                    "would be the embedding output, which is intentionally "
                    "never patched). All production patch sets exclude layer 0."
                )
            patch_in = patch_input_states[layer_idx]
            if patch_in.shape != hidden.shape:
                raise ValueError(
                    f"Input-side patch shape mismatch at layer {layer_idx}: "
                    f"expected {hidden.shape}, got {patch_in.shape}"
                )
            hidden = patch_in.to(device=hidden.device, dtype=hidden.dtype)

        params = _layer_forward_params(layer)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        layer_kwargs = {
            "attention_mask": attention_mask_4d,
            "position_ids": position_ids,
            "use_cache": (cache is not None),
        }
        if cache is not None:
            if "past_key_values" in params:
                layer_kwargs["past_key_values"] = cache
            elif "past_key_value" in params:
                layer_kwargs["past_key_value"] = cache
            elif has_var_kw:
                layer_kwargs["past_key_values"] = cache
        if position_embeddings is not None and (
            "position_embeddings" in params or has_var_kw
        ):
            layer_kwargs["position_embeddings"] = position_embeddings
        if "cache_position" in params or has_var_kw:
            layer_kwargs["cache_position"] = cache_position

        layer_output = layer(hidden, **layer_kwargs)
        # Qwen2 layer returns hidden tensor directly in 4.45+/5.x; older versions
        # return a tuple. Handle both.
        if isinstance(layer_output, tuple):
            hidden = layer_output[0]
        else:
            hidden = layer_output

        # OUTPUT-side patch: replace residual AFTER the layer's forward so the
        # input to layer N+1 is exactly the clean residual at boundary N.
        if layer_idx in patch_states:
            patch = patch_states[layer_idx]
            if patch.shape != hidden.shape:
                raise ValueError(
                    f"Output-side patch shape mismatch at layer {layer_idx}: "
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
            forward_params = inspect.signature(model.forward).parameters
            model_kwargs = {
                "input_ids": current,
                "past_key_values": cache,
                "use_cache": True,
            }
            if "cache_position" in forward_params:
                model_kwargs["cache_position"] = cache_position
            if "position_ids" in forward_params:
                model_kwargs["position_ids"] = position_ids
            outputs = model(**model_kwargs)
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

    # Build patch_states (output-side) AND patch_input_states (input-side) so
    # that for each patched layer N the prompt-KV cache slot is populated from
    # the clean (or corrupt) input — i.e., the residual that layer N would have
    # seen in the recipient (or composed) model. This is the Priority 1 fix
    # for the cache-consistency bug.
    patch_states: Dict[int, torch.Tensor] = {}
    patch_input_states: Dict[int, torch.Tensor] = {}
    for layer_idx in patch_config.patch_layers:
        if layer_idx == 0:
            raise ValueError(
                "Patch at layer 0 is not supported (input-side patch would be "
                "the embedding output, which is intentionally never patched). "
                "All production patch sets exclude layer 0."
            )
        if patch_config.direction == "restoration":
            if clean_layer_states is None:
                raise ValueError("clean_layer_states required for restoration patches")
            src = clean_layer_states
        else:  # corruption
            if corrupt_layer_states is None:
                raise ValueError("corrupt_layer_states required for corruption patches")
            src = corrupt_layer_states
        # Output-side: replace residual after layer N → clean h_N.
        patch_states[layer_idx] = src[layer_idx].to(device)
        # Input-side: replace residual before layer N → clean h_{N-1}, the
        # input layer N would have seen in the source model. This makes
        # cache[layer_idx] consistent with the patched residual.
        patch_input_states[layer_idx] = src[layer_idx - 1].to(device)

    model.eval()
    with torch.no_grad():
        final_hidden, _, cache = forward_with_patches(
            model, input_ids, patch_states,
            patch_input_states=patch_input_states,
            return_cache=True,
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
