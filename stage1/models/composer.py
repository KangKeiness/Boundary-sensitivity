"""Model composer for two-cut hard swap layer interventions."""

import copy
import random
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models(
    recipient_name: str,
    donor_name: str,
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load recipient and donor models plus the recipient tokenizer."""
    recipient = AutoModelForCausalLM.from_pretrained(
        recipient_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    donor = AutoModelForCausalLM.from_pretrained(
        donor_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        recipient_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _validate_architecture(recipient, donor)
    return recipient, donor, tokenizer


def _validate_architecture(
    recipient: AutoModelForCausalLM,
    donor: AutoModelForCausalLM,
):
    """Fail-fast check that recipient and donor have compatible architectures."""
    r_cfg = recipient.config
    d_cfg = donor.config

    checks = [
        ("num_hidden_layers", r_cfg.num_hidden_layers, d_cfg.num_hidden_layers),
        ("hidden_size", r_cfg.hidden_size, d_cfg.hidden_size),
        ("num_attention_heads", r_cfg.num_attention_heads, d_cfg.num_attention_heads),
    ]
    for name, r_val, d_val in checks:
        if r_val != d_val:
            raise ValueError(
                f"Architecture mismatch on {name}: "
                f"recipient={r_val}, donor={d_val}"
            )


def _get_transformer_layers(model: AutoModelForCausalLM):
    """Get the list of transformer layers from the model."""
    if hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(
        f"Cannot find transformer layers in model architecture: {type(model)}"
    )


def compose_model(
    recipient: AutoModelForCausalLM,
    donor: AutoModelForCausalLM,
    b: int,
    t: int,
    condition: str = "hard_swap",
    seed: int = 42,
) -> Tuple[AutoModelForCausalLM, Dict]:
    """
    Create a composed model via two-cut hard swap.

    Bottom block (layers 0..b-1): recipient
    Middle block (layers b..t-1): donor
    Top block (layers t..L-1): recipient

    Non-transformer modules (embeddings, lm_head, final norm) always from recipient.

    Args:
        recipient: The recipient (instruct) model.
        donor: The donor (base) model.
        b: Lower boundary (start of donor block).
        t: Upper boundary (end of donor block, exclusive).
        condition: One of "hard_swap", "reference", "random_donor".
        seed: RNG seed for random_donor reproducibility.

    Returns:
        (composed_model, metadata) where metadata contains condition-specific
        information such as source_start for random_donor.
    """
    num_layers = recipient.config.num_hidden_layers
    if b < 0 or t > num_layers or b >= t:
        raise ValueError(
            f"Invalid boundaries: b={b}, t={t}, num_layers={num_layers}. "
            f"Need 0 <= b < t <= num_layers."
        )

    composed = copy.deepcopy(recipient)
    composed_layers = _get_transformer_layers(composed)
    donor_layers = _get_transformer_layers(donor)

    metadata: Dict = {}

    if condition == "random_donor":
        # Same-width block, random source position in donor.
        # The block width matches the main hard-swap condition (t - b).
        # source_start is randomly sampled but fixed by seed for reproducibility.
        block_width = t - b
        max_start = num_layers - block_width
        rng = random.Random(seed)
        source_start = rng.randint(0, max_start)
        metadata["source_start"] = source_start
        metadata["seed"] = seed
        for i in range(block_width):
            composed_layers[b + i].load_state_dict(
                donor_layers[source_start + i].state_dict()
            )
    else:
        # hard_swap or reference: direct positional swap b..t-1.
        #
        # Reference condition: same hard-swap mechanism as main conditions,
        # but boundary (b_ref, t_ref) is fixed a priori before any sweep results
        # are observed. This is a canonical fixed baseline — NOT a structurally
        # distinct composition and not a reproduction of Bandarkar-style recipes.
        for layer_idx in range(b, t):
            composed_layers[layer_idx].load_state_dict(
                donor_layers[layer_idx].state_dict()
            )

    return composed, metadata


def get_condition_model(
    recipient: AutoModelForCausalLM,
    donor: AutoModelForCausalLM,
    condition: str,
    b: Optional[int] = None,
    t: Optional[int] = None,
    b_ref: Optional[int] = None,
    t_ref: Optional[int] = None,
    random_donor_seed: int = 42,
) -> Tuple[AutoModelForCausalLM, Dict]:
    """
    Get the appropriate model for a given experimental condition.

    Args:
        recipient: Recipient (instruct) model.
        donor: Donor (base) model.
        condition: One of "no_swap", "hard_swap", "reference", "random_donor".
        b: Lower boundary for hard_swap / random_donor.
        t: Upper boundary for hard_swap / random_donor.
        b_ref: Reference lower boundary (for reference condition).
        t_ref: Reference upper boundary (for reference condition).
        random_donor_seed: Seed for random_donor source position.

    Returns:
        (model, metadata) — metadata is {} for no_swap / hard_swap / reference,
        and {"source_start": int, "seed": int} for random_donor.
    """
    if condition == "no_swap":
        return recipient, {}

    if condition == "hard_swap":
        if b is None or t is None:
            raise ValueError("hard_swap requires b and t parameters")
        return compose_model(recipient, donor, b, t, condition="hard_swap")

    if condition == "reference":
        if b_ref is None or t_ref is None:
            raise ValueError("reference requires b_ref and t_ref parameters")
        return compose_model(recipient, donor, b_ref, t_ref, condition="reference")

    if condition == "random_donor":
        if b is None or t is None:
            raise ValueError("random_donor requires b and t parameters")
        return compose_model(recipient, donor, b, t, condition="random_donor", seed=random_donor_seed)

    raise ValueError(f"Unknown condition: {condition!r}")
