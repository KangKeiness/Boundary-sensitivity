"""Boundary Disruption Score (BDS) calculation."""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def cosine_distance(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """Centered cosine distance between two vectors: 1 - cosine_similarity."""
    h1_centered = h1 - h1.mean()
    h2_centered = h2 - h2.mean()
    sim = F.cosine_similarity(h1_centered.unsqueeze(0), h2_centered.unsqueeze(0))
    return 1.0 - sim.squeeze()


def linear_cka(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """
    Linear CKA between two representation vectors.

    For single vectors, this reduces to squared cosine similarity
    of the centered representations.
    """
    h1_centered = h1 - h1.mean()
    h2_centered = h2 - h2.mean()
    dot = torch.dot(h1_centered, h2_centered)
    norm1 = torch.dot(h1_centered, h1_centered)
    norm2 = torch.dot(h2_centered, h2_centered)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return torch.tensor(0.0)
    return dot ** 2 / (norm1 * norm2)


def compute_bds_single_sample(
    h_rec: torch.Tensor,
    h_comp: torch.Tensor,
    b: int,
    t: int,
) -> Dict[str, float]:
    """
    Compute BDS for a single sample.

    Args:
        h_rec: Hidden states from recipient model, shape [n_layers, hidden_dim].
        h_comp: Hidden states from composed model, shape [n_layers, hidden_dim].
        b: Lower boundary index.
        t: Upper boundary index.

    Returns:
        Dict with BDS metrics.
    """
    # Lower boundary: transition at layer b (between b-1 and b)
    delta_rec_lower = cosine_distance(h_rec[b - 1], h_rec[b])
    delta_comp_lower = cosine_distance(h_comp[b - 1], h_comp[b])
    bds_lower = (delta_comp_lower - delta_rec_lower).item()

    # Upper boundary: transition at layer t (between t-1 and t)
    delta_rec_upper = cosine_distance(h_rec[t - 1], h_rec[t])
    delta_comp_upper = cosine_distance(h_comp[t - 1], h_comp[t])
    bds_upper = (delta_comp_upper - delta_rec_upper).item()

    bds_total = bds_lower + bds_upper

    # Secondary: linear CKA-based disruption
    cka_rec_lower = linear_cka(h_rec[b - 1], h_rec[b]).item()
    cka_comp_lower = linear_cka(h_comp[b - 1], h_comp[b]).item()
    cka_rec_upper = linear_cka(h_rec[t - 1], h_rec[t]).item()
    cka_comp_upper = linear_cka(h_comp[t - 1], h_comp[t]).item()

    return {
        "bds_lower": bds_lower,
        "bds_upper": bds_upper,
        "bds_total": bds_total,
        "delta_rec_lower": delta_rec_lower.item(),
        "delta_comp_lower": delta_comp_lower.item(),
        "delta_rec_upper": delta_rec_upper.item(),
        "delta_comp_upper": delta_comp_upper.item(),
        "cka_rec_lower": cka_rec_lower,
        "cka_comp_lower": cka_comp_lower,
        "cka_rec_upper": cka_rec_upper,
        "cka_comp_upper": cka_comp_upper,
    }


def compute_bds(
    hidden_states_recipient: List[Dict],
    hidden_states_composed: List[Dict],
    b: int,
    t: int,
) -> Dict:
    """
    Compute BDS across all samples, aligned by sample_id.

    Uses ALL samples regardless of parse success (hidden states are parse-independent).

    Args:
        hidden_states_recipient: List of {sample_id, hidden_states} from recipient.
        hidden_states_composed: List of {sample_id, hidden_states} from composed model.
        b: Lower boundary index.
        t: Upper boundary index.

    Returns:
        Dict with per-sample and aggregate BDS metrics.
    """
    # Build lookup by sample_id
    rec_by_id = {s["sample_id"]: s["hidden_states"] for s in hidden_states_recipient}
    comp_by_id = {s["sample_id"]: s["hidden_states"] for s in hidden_states_composed}

    # Validate alignment
    rec_ids = set(rec_by_id.keys())
    comp_ids = set(comp_by_id.keys())
    missing_in_comp = rec_ids - comp_ids
    missing_in_rec = comp_ids - rec_ids
    if missing_in_comp:
        raise ValueError(f"Samples missing in composed results: {missing_in_comp}")
    if missing_in_rec:
        raise ValueError(f"Samples missing in recipient results: {missing_in_rec}")

    common_ids = sorted(rec_ids & comp_ids)

    per_sample = []
    for sid in common_ids:
        h_rec = rec_by_id[sid]
        h_comp = comp_by_id[sid]
        bds_metrics = compute_bds_single_sample(h_rec, h_comp, b, t)
        bds_metrics["sample_id"] = sid
        per_sample.append(bds_metrics)

    # Aggregate: mean across samples
    metric_keys = [
        "bds_lower", "bds_upper", "bds_total",
        "delta_rec_lower", "delta_comp_lower",
        "delta_rec_upper", "delta_comp_upper",
        "cka_rec_lower", "cka_comp_lower",
        "cka_rec_upper", "cka_comp_upper",
    ]
    aggregate = {}
    for key in metric_keys:
        values = [s[key] for s in per_sample]
        aggregate[f"mean_{key}"] = sum(values) / len(values)

    return {
        "per_sample": per_sample,
        "aggregate": aggregate,
        "n_samples": len(per_sample),
        "b": b,
        "t": t,
    }
