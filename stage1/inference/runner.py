"""Greedy inference runner with hidden state extraction.

Hidden states are extracted from a prompt-only forward pass, giving a fixed
anchor (last token of the prompt) that is identical across all conditions.
Answer text is then generated in a separate generate() call.
"""

from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _extract_prompt_hidden_states(
    fwd_hidden_states,
    pooling: str,
) -> torch.Tensor:
    """
    Extract hidden states from a prompt-only forward pass.

    Args:
        fwd_hidden_states: tuple of (n_layers+1,) tensors, each [batch, seq_len, hidden_dim].
            Index 0 is the embedding layer output; indices 1..n_layers are transformer layers.
        pooling: "last_token" or "mean". Applied to prompt tokens only.

    Returns:
        Tensor of shape [n_layers, hidden_dim].
    """
    # Skip embedding layer (index 0); take transformer layer outputs
    layer_outputs = fwd_hidden_states[1:]  # (n_layers,) each [batch, seq_len, hidden_dim]

    result = []
    for layer_hs in layer_outputs:
        hs = layer_hs[0]  # [seq_len, hidden_dim], batch dim removed

        if pooling == "last_token":
            # Fixed anchor: last token of the prompt
            repr_vec = hs[-1]
        elif pooling == "mean":
            repr_vec = hs.mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling!r}. Use 'last_token' or 'mean'.")

        result.append(repr_vec)

    return torch.stack(result)  # [n_layers, hidden_dim]


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    generation_config: dict,
    pooling: str = "last_token",
    device=None,
) -> List[Dict]:
    """
    Run greedy inference on samples.

    For each sample:
    1. Prompt-only forward pass  → hidden states (fixed anchor, independent of generation length)
    2. generate()                → answer text

    Args:
        model: The model to run inference with.
        tokenizer: Tokenizer for the model.
        samples: List of dicts with keys: sample_id, prompt, gold_answer.
        generation_config: Dict with generation parameters.
        pooling: Hidden state pooling method ("last_token" or "mean").
        device: Device override (inferred from model if None).

    Returns:
        List of dicts with keys: sample_id, output_text, hidden_states.
        hidden_states shape: [n_layers, hidden_dim] per sample.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    gen_kwargs = {
        "do_sample": generation_config.get("do_sample", False),
        "max_new_tokens": generation_config.get("max_new_tokens", 256),
    }
    # Only pass temperature when sampling; avoids HF warning for greedy decoding
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = generation_config.get("temperature", 1.0)

    results = []

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            input_ids = tokenizer(
                sample["prompt"],
                return_tensors="pt",
                padding=False,
            ).input_ids.to(device)

            prompt_len = input_ids.shape[1]

            # ── Pass 1: prompt-only forward for hidden states ──────────────────
            fwd_outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
            )
            hidden_states = _extract_prompt_hidden_states(
                fwd_outputs.hidden_states,
                pooling=pooling,
            ).cpu()

            anchor_pos = prompt_len - 1
            print(
                f"  [{idx+1}/{len(samples)}] {sample['sample_id']} | "
                f"hidden state anchor: prompt {pooling}, position {anchor_pos}"
            )

            # ── Pass 2: generate for answer text ──────────────────────────────
            gen_outputs = model.generate(input_ids, **gen_kwargs)
            generated_ids = gen_outputs[0, prompt_len:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            results.append({
                "sample_id": sample["sample_id"],
                "output_text": output_text,
                "hidden_states": hidden_states,
            })

    return results
