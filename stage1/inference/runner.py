"""Greedy inference runner with hidden state extraction."""

from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_hidden_states(
    outputs,
    input_length: int,
    pooling: str = "last_token",
) -> torch.Tensor:
    """
    Extract hidden states from model outputs.

    Args:
        outputs: Model outputs with hidden_states.
        input_length: Length of the input tokens (to find generated positions).
        pooling: "last_token" or "mean".

    Returns:
        Tensor of shape [n_layers, hidden_dim].
    """
    # outputs.hidden_states is a tuple of (n_layers+1,) tensors of shape [batch, seq_len, hidden_dim]
    # We include all layers (skip the embedding layer at index 0)
    hidden_states = outputs.hidden_states[1:]  # skip embedding output
    n_layers = len(hidden_states)

    result = []
    for layer_idx in range(n_layers):
        hs = hidden_states[layer_idx][0]  # [seq_len, hidden_dim], batch=0

        if pooling == "last_token":
            # Last token of the full sequence (input + generated)
            layer_repr = hs[-1]
        elif pooling == "mean":
            layer_repr = hs.mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        result.append(layer_repr)

    return torch.stack(result)  # [n_layers, hidden_dim]


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    generation_config: dict,
    pooling: str = "last_token",
    device: str = None,
) -> List[Dict]:
    """
    Run greedy inference on samples and extract hidden states.

    Args:
        model: The model to run inference with.
        tokenizer: Tokenizer for the model.
        samples: List of dicts with keys: sample_id, prompt, gold_answer.
        generation_config: Dict with generation parameters.
        pooling: Hidden state pooling method ("last_token" or "mean").
        device: Device to run on (inferred from model if None).

    Returns:
        List of dicts with keys: sample_id, output_text, hidden_states.
        hidden_states shape: [n_layers, hidden_dim] per sample.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []

    gen_kwargs = {
        "do_sample": generation_config.get("do_sample", False),
        "temperature": generation_config.get("temperature", 0.0),
        "max_new_tokens": generation_config.get("max_new_tokens", 256),
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }
    # When do_sample=False, temperature is irrelevant; avoid HF warning
    if not gen_kwargs["do_sample"]:
        gen_kwargs.pop("temperature", None)

    with torch.no_grad():
        for sample in samples:
            input_ids = tokenizer(
                sample["prompt"],
                return_tensors="pt",
                padding=False,
            ).input_ids.to(device)

            input_length = input_ids.shape[1]

            outputs = model.generate(input_ids, **gen_kwargs)

            # Decode only the generated tokens
            generated_ids = outputs.sequences[0, input_length:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Extract hidden states from the last forward pass
            # outputs.hidden_states is a tuple of step-wise hidden states
            # Each step: tuple of (n_layers+1,) tensors of [batch, 1, hidden_dim]
            # We need hidden states from the final step for the full context
            # Use the last generation step's hidden states
            last_step_hidden = outputs.hidden_states[-1]

            # Build a pseudo-output object for extract_hidden_states
            class _HiddenStatesContainer:
                def __init__(self, hs):
                    self.hidden_states = hs

            full_seq_len = outputs.sequences.shape[1]
            hs_container = _HiddenStatesContainer(last_step_hidden)
            hidden_states = extract_hidden_states(
                hs_container,
                input_length=input_length,
                pooling=pooling,
            )

            results.append({
                "sample_id": sample["sample_id"],
                "output_text": output_text.strip(),
                "hidden_states": hidden_states.cpu(),
            })

    return results
