"""Boundary Disruption Score (BDS) calculation.

Primary metric : centered cosine distance (per-sample, then aggregated)
Secondary metric: linear CKA over sample matrices (corpus-level robustness check)
"""

from typing import Dict, List

import torch
import torch.nn.functional as F


# ─── Distance / similarity primitives ────────────────────────────────────────

def cosine_distance(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """Centered cosine distance: 1 - cosine_similarity(h1 - mean, h2 - mean)."""
    h1c = h1 - h1.mean()
    h2c = h2 - h2.mean()
    sim = F.cosine_similarity(h1c.unsqueeze(0), h2c.unsqueeze(0))
    return 1.0 - sim.squeeze()


def linear_cka_matrix(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    True linear CKA over sample matrices.

    X, Y : [n_samples, hidden_dim]

    Computes HSIC(XX^T, YY^T) / sqrt(HSIC(XX^T,XX^T) * HSIC(YY^T,YY^T))
    using the unbiased, double-centered Gram matrix estimator.

    This is the correct aggregate metric; the old per-vector version was
    equivalent to squared centred cosine similarity and is NOT used here.
    """
    print(f"  True CKA uses sample matrices {list(X.shape)}")
    n = X.shape[0]
    if n < 2:
        return 0.0

    X = X.float()
    Y = Y.float()

    # Centre columns (subtract feature means)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices  [n, n]
    K = X @ X.T
    L = Y @ Y.T

    # Double-centre the Gram matrices
    H = torch.eye(n, dtype=X.dtype, device=X.device) - torch.ones(n, n, dtype=X.dtype, device=X.device) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    # HSIC via Frobenius inner product  (trace trick: tr(A@B) = (A*B.T).sum(), B sym → (A*B).sum())
    hsic_KL = (Kc * Lc).sum() / (n - 1) ** 2
    hsic_KK = (Kc * Kc).sum() / (n - 1) ** 2
    hsic_LL = (Lc * Lc).sum() / (n - 1) ** 2

    denom = torch.sqrt(hsic_KK * hsic_LL)
    if denom < 1e-10:
        return 0.0
    return float(hsic_KL / denom)


# ─── Per-sample BDS (primary metric) ─────────────────────────────────────────

def compute_bds_single_sample(
    h_rec: torch.Tensor,
    h_comp: torch.Tensor,
    b: int,
    t: int,
) -> Dict[str, float]:
    """
    Compute primary BDS metrics for a single sample using centered cosine distance.

    Args:
        h_rec  : [n_layers, hidden_dim] from recipient model.
        h_comp : [n_layers, hidden_dim] from composed model.
        b      : lower boundary index.
        t      : upper boundary index.

    Returns dict with per-boundary delta values and total BDS.
    """
    # Lower boundary: transition between layers b-1 and b
    delta_rec_lower  = cosine_distance(h_rec[b - 1],  h_rec[b])
    delta_comp_lower = cosine_distance(h_comp[b - 1], h_comp[b])
    bds_lower = (delta_comp_lower - delta_rec_lower).item()

    # Upper boundary: transition between layers t-1 and t
    delta_rec_upper  = cosine_distance(h_rec[t - 1],  h_rec[t])
    delta_comp_upper = cosine_distance(h_comp[t - 1], h_comp[t])
    bds_upper = (delta_comp_upper - delta_rec_upper).item()

    return {
        "bds_lower":        bds_lower,
        "bds_upper":        bds_upper,
        "bds_total":        bds_lower + bds_upper,
        "delta_rec_lower":  delta_rec_lower.item(),
        "delta_comp_lower": delta_comp_lower.item(),
        "delta_rec_upper":  delta_rec_upper.item(),
        "delta_comp_upper": delta_comp_upper.item(),
    }


# ─── Corpus-level BDS ─────────────────────────────────────────────────────────

def compute_bds(
    hidden_states_recipient: List[Dict],
    hidden_states_composed: List[Dict],
    b: int,
    t: int,
) -> Dict:
    """
    Compute BDS across all samples, aligned by sample_id.

    Primary metric  : centered cosine distance, averaged across samples.
    Secondary metric: linear CKA computed over sample matrices (not per-sample).

    BDS uses ALL samples regardless of parse success (hidden states are
    parse-independent). Missing samples raise an error.

    Args:
        hidden_states_recipient : List of {sample_id, hidden_states}.
        hidden_states_composed  : List of {sample_id, hidden_states}.
        b : lower boundary index.
        t : upper boundary index.

    Returns dict with per_sample list, aggregate dict, and CKA values.
    """
    # ── Align by sample_id ────────────────────────────────────────────────
    rec_by_id  = {s["sample_id"]: s["hidden_states"] for s in hidden_states_recipient}
    comp_by_id = {s["sample_id"]: s["hidden_states"] for s in hidden_states_composed}

    rec_ids  = set(rec_by_id)
    comp_ids = set(comp_by_id)
    missing_comp = rec_ids - comp_ids
    missing_rec  = comp_ids - rec_ids
    if missing_comp:
        raise ValueError(f"Samples missing in composed results: {missing_comp}")
    if missing_rec:
        raise ValueError(f"Samples missing in recipient results: {missing_rec}")

    common_ids = sorted(rec_ids & comp_ids)

    # ── Per-sample centered cosine distance (primary metric) ──────────────
    per_sample = []

    # Also collect layer vectors for CKA matrices
    h_rec_b1_list   = []   # h_rec[b-1]   for all samples
    h_rec_b_list    = []   # h_rec[b]
    h_rec_t1_list   = []   # h_rec[t-1]
    h_rec_t_list    = []   # h_rec[t]
    h_comp_b1_list  = []
    h_comp_b_list   = []
    h_comp_t1_list  = []
    h_comp_t_list   = []

    for sid in common_ids:
        h_rec  = rec_by_id[sid]
        h_comp = comp_by_id[sid]

        bds_metrics = compute_bds_single_sample(h_rec, h_comp, b, t)
        bds_metrics["sample_id"] = sid
        per_sample.append(bds_metrics)

        h_rec_b1_list.append(h_rec[b - 1])
        h_rec_b_list.append(h_rec[b])
        h_rec_t1_list.append(h_rec[t - 1])
        h_rec_t_list.append(h_rec[t])
        h_comp_b1_list.append(h_comp[b - 1])
        h_comp_b_list.append(h_comp[b])
        h_comp_t1_list.append(h_comp[t - 1])
        h_comp_t_list.append(h_comp[t])

    # ── True linear CKA over sample matrices (secondary metric) ──────────
    # CKA measures similarity between adjacent-layer representations.
    # Lower CKA for composed vs recipient indicates higher disruption.
    cka_rec_lower  = linear_cka_matrix(torch.stack(h_rec_b1_list),  torch.stack(h_rec_b_list))
    cka_comp_lower = linear_cka_matrix(torch.stack(h_comp_b1_list), torch.stack(h_comp_b_list))
    cka_rec_upper  = linear_cka_matrix(torch.stack(h_rec_t1_list),  torch.stack(h_rec_t_list))
    cka_comp_upper = linear_cka_matrix(torch.stack(h_comp_t1_list), torch.stack(h_comp_t_list))

    # CKA-based disruption: decrease in CKA = more disruption (negative = disrupted)
    cka_bds_lower = cka_comp_lower - cka_rec_lower
    cka_bds_upper = cka_comp_upper - cka_rec_upper
    cka_bds_total = cka_bds_lower + cka_bds_upper

    # ── Aggregate primary metric ──────────────────────────────────────────
    primary_keys = [
        "bds_lower", "bds_upper", "bds_total",
        "delta_rec_lower", "delta_comp_lower",
        "delta_rec_upper", "delta_comp_upper",
    ]
    aggregate = {
        f"mean_{k}": sum(s[k] for s in per_sample) / len(per_sample)
        for k in primary_keys
    }

    # Add CKA secondary metrics to aggregate
    aggregate.update({
        "cka_rec_lower":   cka_rec_lower,
        "cka_comp_lower":  cka_comp_lower,
        "cka_rec_upper":   cka_rec_upper,
        "cka_comp_upper":  cka_comp_upper,
        "cka_bds_lower":   cka_bds_lower,
        "cka_bds_upper":   cka_bds_upper,
        "cka_bds_total":   cka_bds_total,
    })

    return {
        "per_sample": per_sample,
        "aggregate":  aggregate,
        "n_samples":  len(per_sample),
        "b": b,
        "t": t,
    }


# ─── Sweep helper (notebook API) ─────────────────────────────────────────────

def compute_bds_sweep(
    hidden_no_swap: List[Dict],
    hidden_sweep: Dict[str, List[Dict]],
    config,
) -> Dict[str, Dict]:
    """
    Compute BDS for all sweep conditions at once.

    Args:
        hidden_no_swap : List of {sample_id, hidden_states} for no_swap baseline.
        hidden_sweep   : {condition_name: list of {sample_id, hidden_states}}.
        config         : Stage1Config.

    Returns:
        {condition_name: bds_result_dict}
    """
    results = {}
    for cond_name, hs_composed in hidden_sweep.items():
        if cond_name.startswith("hard_swap_b"):
            b = int(cond_name.split("_b")[1])
            t = config.t_fixed
        elif cond_name == "reference":
            b = config.reference.b_ref
            t = config.reference.t_ref
        elif cond_name == "random_donor":
            b = config.boundary_grid[0]
            t = config.t_fixed
        else:
            continue
        print(f"  Computing BDS for {cond_name} (b={b}, t={t})")
        results[cond_name] = compute_bds(hidden_no_swap, hs_composed, b, t)
    return results
