"""Post-hoc analysis script for Stage 1 completed runs.

Loads saved artifacts (hidden states, BDS results, evaluation, manifest)
from a run directory and computes additional metrics — specifically
Boundary-Propagated Drift (BPD) and Excess BPD (EBPD) — without
re-running any experiments.

Usage (CLI):
    python stage1/analysis/post_analysis.py --run_dir /workspace/outputs/run_20260405_130326

Usage (Jupyter):
    from stage1.analysis.post_analysis import (
        load_run, compute_bpd, compute_recovery_metrics,
        compute_all_metric_correlations, print_summary,
    )
"""

import argparse
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from scipy.stats import spearmanr

from stage1.analysis.bds import cosine_distance, linear_cka_matrix

logger = logging.getLogger(__name__)


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_run(run_dir: str) -> Dict:
    """
    Load all saved artifacts from a completed run directory.

    Expects the following files (created by stage1/utils/logger.py):
    - manifest.json
    - evaluation.json
    - hidden_states_{condition}.pt  (one per condition)
    - bds_{condition}.json          (one per swap/random_donor condition)

    Args:
        run_dir: Path to the run directory, e.g.
                 /workspace/outputs/run_20260405_130326/

    Returns:
        Dict with keys:
            hidden_states : {condition_name: {sample_id: tensor [n_layers, hidden_dim]}}
            evaluation    : full evaluation.json dict
            bds           : {condition_name: bds_json_dict}
            manifest      : full manifest.json dict
            boundary_grid : List[int] extracted from manifest
            t_fixed       : int extracted from manifest
    """
    run_dir = os.path.abspath(run_dir)

    # ── manifest ──────────────────────────────────────────────────────────
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    boundary_grid: List[int] = manifest["config"]["boundary_grid"]
    t_fixed: int = manifest["config"]["t_fixed"]

    # ── evaluation ────────────────────────────────────────────────────────
    eval_path = os.path.join(run_dir, "evaluation.json")
    with open(eval_path) as f:
        evaluation = json.load(f)

    # ── BDS results ───────────────────────────────────────────────────────
    bds: Dict[str, Dict] = {}
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith("bds_") and fname.endswith(".json"):
            cond = fname[len("bds_"):-len(".json")]
            with open(os.path.join(run_dir, fname)) as f:
                bds[cond] = json.load(f)

    # ── Hidden states ─────────────────────────────────────────────────────
    hidden_states: Dict[str, Dict[str, torch.Tensor]] = {}
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith("hidden_states_") and fname.endswith(".pt"):
            cond = fname[len("hidden_states_"):-len(".pt")]
            pt_path = os.path.join(run_dir, fname)
            try:
                hs_dict = torch.load(pt_path, map_location="cpu")
                hidden_states[cond] = hs_dict
                logger.info("Loaded hidden states for %s (%d samples)", cond, len(hs_dict))
            except Exception as exc:
                logger.warning("Could not load %s: %s — skipping condition %s", fname, exc, cond)

    return {
        "hidden_states": hidden_states,
        "evaluation":    evaluation,
        "bds":           bds,
        "manifest":      manifest,
        "boundary_grid": boundary_grid,
        "t_fixed":       t_fixed,
    }


# ─── BPD computation ─────────────────────────────────────────────────────────

def compute_bpd(
    hidden_states_recipient: Dict[str, torch.Tensor],
    hidden_states_composed: Dict[str, torch.Tensor],
    b: int,
    t_fixed: int,
    n_layers: int = 28,
) -> Dict:
    """
    Compute Boundary-Propagated Drift (BPD) for a single boundary condition.

    BPD measures how far the composed model's representations have drifted
    from the recipient's, averaged over all layers from the lower boundary
    to the last layer. Unlike BDS (which measures only at the two boundary
    points), BPD captures how the perturbation propagates downstream.

    Definitions:
        D_l   = centered_cosine_distance(h_comp[l], h_rec[l])
        BPD   = mean over l in [b, L-1] of D_l, averaged over samples.

        ExcessTransition_l = cosine_distance(h_comp[l-1], h_comp[l])
                           - cosine_distance(h_rec[l-1],  h_rec[l])
        EBPD  = mean over l in [b, L-1] of ExcessTransition_l,
                averaged over samples.  EBPD generalises BDS: BDS is the
                value at the single boundary point; EBPD averages it over
                all downstream layers.

    Note: per_layer_excess[0] (at l=b) corresponds to BDS_lower for that
    boundary, NOT BDS_total. BDS_total = BDS_lower + BDS_upper, where upper
    is computed at l=t. Do not compare EBPD values directly against
    mean_bds_total from bds_*.json.

    Args:
        hidden_states_recipient : {sample_id: tensor [n_layers, hidden_dim]}
        hidden_states_composed  : {sample_id: tensor [n_layers, hidden_dim]}
        b         : lower boundary index (first donor layer).
        t_fixed   : upper boundary index (not used in BPD, stored for reference).
        n_layers  : total number of transformer layers (default 28 for Qwen2.5-1.5B).

    Returns:
        Dict with keys:
            bpd_mean         : float  — average drift across downstream layers & samples
            ebpd_mean        : float  — average excess transition downstream
            per_layer_drift  : List[float]  — D_l averaged over samples, for l in [b, L-1]
            per_layer_excess : List[float]  — ExcessTransition_l avg over samples
            per_sample_bpd   : List[float]  — per-sample BPD (for bootstrap)
            n_samples        : int
            b                : int
            t_fixed          : int
            layers_range     : [int, int]   — [b, L-1] inclusive
    """
    # Align by sample_id
    common_ids = sorted(set(hidden_states_recipient) & set(hidden_states_composed))
    if not common_ids:
        raise ValueError("No common sample IDs between recipient and composed hidden states.")

    L = n_layers  # last layer index is L-1
    downstream_layers = list(range(b, L))  # b .. L-1 inclusive

    # Accumulators: one entry per downstream layer
    layer_drift_sums  = [0.0] * len(downstream_layers)
    layer_excess_sums = [0.0] * len(downstream_layers)
    per_sample_bpd: List[float] = []

    for sid in common_ids:
        h_rec  = hidden_states_recipient[sid].float()   # [n_layers, hidden_dim]
        h_comp = hidden_states_composed[sid].float()

        sample_drift_vals: List[float] = []

        for i, l in enumerate(downstream_layers):
            # D_l: positional drift at layer l
            d_l = cosine_distance(h_comp[l], h_rec[l]).item()
            layer_drift_sums[i] += d_l
            sample_drift_vals.append(d_l)

            # ExcessTransition_l: l vs l-1 (need l >= 1)
            if l >= 1:
                trans_comp = cosine_distance(h_comp[l - 1], h_comp[l]).item()
                trans_rec  = cosine_distance(h_rec[l - 1],  h_rec[l]).item()
                layer_excess_sums[i] += (trans_comp - trans_rec)
            # l == 0 is impossible here since b >= 1 in all configs, but guard anyway

        per_sample_bpd.append(float(sum(sample_drift_vals) / len(sample_drift_vals)))

    n = len(common_ids)
    per_layer_drift  = [s / n for s in layer_drift_sums]
    per_layer_excess = [s / n for s in layer_excess_sums]

    bpd_mean  = float(sum(per_layer_drift)  / len(per_layer_drift))
    ebpd_mean = float(sum(per_layer_excess) / len(per_layer_excess))

    return {
        "bpd_mean":         bpd_mean,
        "ebpd_mean":        ebpd_mean,
        "per_layer_drift":  per_layer_drift,
        "per_layer_excess": per_layer_excess,
        "per_sample_bpd":   per_sample_bpd,
        "n_samples":        n,
        "b":                b,
        "t_fixed":          t_fixed,
        "layers_range":     [b, L - 1],
    }


def compute_bpd_sweep(
    run_data: Dict,
) -> Dict[str, Dict]:
    """
    Compute BPD for every condition (hard_swap and random_donor) in a loaded run.

    Args:
        run_data: Output of load_run().

    Returns:
        {condition_name: bpd_result_dict}
    """
    hs = run_data["hidden_states"]
    boundary_grid = run_data["boundary_grid"]
    t_fixed = run_data["t_fixed"]
    n_layers = run_data["manifest"].get("hidden_state_layer_count", 28)

    if "no_swap" not in hs:
        raise ValueError("no_swap hidden states not found — required as recipient baseline.")

    hs_recipient = hs["no_swap"]
    results: Dict[str, Dict] = {}

    for b in boundary_grid:
        for prefix in ("hard_swap_b", "random_donor_b"):
            cond = f"{prefix}{b}"
            if cond not in hs:
                logger.warning("Hidden states missing for %s — skipping BPD.", cond)
                continue
            logger.info("Computing BPD for %s (b=%d)", cond, b)
            results[cond] = compute_bpd(hs_recipient, hs[cond], b, t_fixed, n_layers=n_layers)

    return results


# ─── Recovery-side metrics ───────────────────────────────────────────────────

def _pairwise_cosine_distance_matrix(H: torch.Tensor) -> torch.Tensor:
    """Pairwise centered cosine distance matrix for rows of H [n, d]."""
    H = H.float()
    n = H.shape[0]
    D = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(H[i], H[j]).item()
            D[i, j] = d
            D[j, i] = d
    return D


def compute_recovery_metrics(
    hidden_states_recipient: Dict[str, torch.Tensor],
    hidden_states_composed: Dict[str, torch.Tensor],
    t_fixed: int = 20,
    n_layers: int = 28,
) -> Dict:
    """
    Compute all recovery-zone metrics comparing composed vs recipient hidden states.

    Recovery zone: layers t_fixed to n_layers-1 inclusive.

    Metrics:
        RD_cos   — mean cosine distance across recovery layers, averaged over samples
        RD_l2    — mean L2 distance across recovery layers, averaged over samples
        FLD_cos  — cosine distance at the final layer, averaged over samples
        FLD_l2   — L2 distance at the final layer, averaged over samples
        Rec_CKA  — 1 - linear_CKA per recovery layer, averaged across layers
        Rec_RSA  — 1 - Spearman(RDM_comp, RDM_rec) per recovery layer, averaged

    Args:
        hidden_states_recipient : {sample_id: tensor [n_layers, hidden_dim]}
        hidden_states_composed  : {sample_id: tensor [n_layers, hidden_dim]}
        t_fixed  : first recovery layer (upper boundary).
        n_layers : total number of layers (default 28 for Qwen2.5-1.5B).

    Returns:
        Dict with per-metric averages, per-layer breakdowns, and per-sample values.
    """
    common_ids = sorted(set(hidden_states_recipient) & set(hidden_states_composed))
    if not common_ids:
        raise ValueError("No common sample IDs between recipient and composed hidden states.")

    L = n_layers
    recovery_layers = list(range(t_fixed, L))  # t_fixed .. L-1
    n_recovery = len(recovery_layers)
    n_samples = len(common_ids)

    # ── Per-sample, per-layer accumulators for RD and FLD ────────────────
    layer_drift_cos_sums = [0.0] * n_recovery
    layer_drift_l2_sums = [0.0] * n_recovery
    per_sample_rd_cos: List[float] = []
    per_sample_fld_cos: List[float] = []

    # ── Per-layer sample matrices for CKA and RSA ────────────────────────
    # H_comp_per_layer[i] and H_rec_per_layer[i] will be [n_samples, hidden_dim]
    H_comp_per_layer: List[List[torch.Tensor]] = [[] for _ in range(n_recovery)]
    H_rec_per_layer: List[List[torch.Tensor]] = [[] for _ in range(n_recovery)]

    for sid in common_ids:
        h_rec = hidden_states_recipient[sid].float()
        h_comp = hidden_states_composed[sid].float()

        sample_drift_cos: List[float] = []

        for i, l in enumerate(recovery_layers):
            d_cos = cosine_distance(h_comp[l], h_rec[l]).item()
            d_l2 = torch.norm(h_comp[l] - h_rec[l], p=2).item()

            layer_drift_cos_sums[i] += d_cos
            layer_drift_l2_sums[i] += d_l2
            sample_drift_cos.append(d_cos)

            H_comp_per_layer[i].append(h_comp[l])
            H_rec_per_layer[i].append(h_rec[l])

        per_sample_rd_cos.append(float(sum(sample_drift_cos) / len(sample_drift_cos)))
        # FLD: last recovery layer (L-1)
        fld_cos = cosine_distance(h_comp[L - 1], h_rec[L - 1]).item()
        per_sample_fld_cos.append(fld_cos)

    # ── Per-layer averages ───────────────────────────────────────────────
    per_layer_drift_cos = [s / n_samples for s in layer_drift_cos_sums]
    per_layer_drift_l2 = [s / n_samples for s in layer_drift_l2_sums]

    rd_cos = float(sum(per_layer_drift_cos) / n_recovery)
    rd_l2 = float(sum(per_layer_drift_l2) / n_recovery)
    fld_cos = float(sum(per_sample_fld_cos) / n_samples)

    # FLD L2: recompute from final layer across samples
    fld_l2_sum = 0.0
    for sid in common_ids:
        h_rec = hidden_states_recipient[sid].float()
        h_comp = hidden_states_composed[sid].float()
        fld_l2_sum += torch.norm(h_comp[L - 1] - h_rec[L - 1], p=2).item()
    fld_l2 = float(fld_l2_sum / n_samples)

    # ── CKA per recovery layer ───────────────────────────────────────────
    per_layer_cka_shift: List[float] = []
    for i in range(n_recovery):
        H_comp_mat = torch.stack(H_comp_per_layer[i])  # [n_samples, hidden_dim]
        H_rec_mat = torch.stack(H_rec_per_layer[i])
        cka_val = linear_cka_matrix(H_comp_mat, H_rec_mat)
        per_layer_cka_shift.append(1.0 - cka_val)

    recovery_cka = float(sum(per_layer_cka_shift) / n_recovery)

    # ── RSA per recovery layer ───────────────────────────────────────────
    per_layer_rsa_shift: List[float] = []
    for i in range(n_recovery):
        H_comp_mat = torch.stack(H_comp_per_layer[i])
        H_rec_mat = torch.stack(H_rec_per_layer[i])

        rdm_comp = _pairwise_cosine_distance_matrix(H_comp_mat)
        rdm_rec = _pairwise_cosine_distance_matrix(H_rec_mat)

        # Extract upper triangles
        idx = torch.triu_indices(n_samples, n_samples, offset=1)
        upper_comp = rdm_comp[idx[0], idx[1]].numpy()
        upper_rec = rdm_rec[idx[0], idx[1]].numpy()

        if len(upper_comp) < 2:
            per_layer_rsa_shift.append(0.0)
            continue

        rho, _ = spearmanr(upper_comp, upper_rec)
        if math.isnan(rho):
            logger.warning("RSA returned NaN at recovery layer %d — treating as 0 correlation", recovery_layers[i])
            per_layer_rsa_shift.append(1.0)
        else:
            per_layer_rsa_shift.append(1.0 - float(rho))

    recovery_rsa = float(sum(per_layer_rsa_shift) / n_recovery)

    return {
        "rd_cosine": rd_cos,
        "rd_l2": rd_l2,
        "fld_cosine": fld_cos,
        "fld_l2": fld_l2,
        "recovery_cka": recovery_cka,
        "recovery_rsa": recovery_rsa,
        "per_layer_drift_cosine": per_layer_drift_cos,
        "per_layer_drift_l2": per_layer_drift_l2,
        "per_layer_cka_shift": per_layer_cka_shift,
        "per_layer_rsa_shift": per_layer_rsa_shift,
        "per_sample_rd_cosine": per_sample_rd_cos,
        "per_sample_fld_cosine": per_sample_fld_cos,
        "n_samples": n_samples,
        "recovery_layers": [t_fixed, L - 1],
    }


def compute_recovery_sweep(run_data: Dict) -> Dict[str, Dict]:
    """
    Compute recovery metrics for every condition in a loaded run.

    Args:
        run_data: Output of load_run().

    Returns:
        {condition_name: recovery_metrics_dict}
    """
    hs = run_data["hidden_states"]
    boundary_grid = run_data["boundary_grid"]
    t_fixed = run_data["t_fixed"]
    n_layers = run_data["manifest"].get("hidden_state_layer_count", 28)

    if "no_swap" not in hs:
        raise ValueError("no_swap hidden states not found — required as recipient baseline.")

    hs_recipient = hs["no_swap"]
    results: Dict[str, Dict] = {}

    for b in boundary_grid:
        for prefix in ("hard_swap_b", "random_donor_b"):
            cond = f"{prefix}{b}"
            if cond not in hs:
                logger.warning("Hidden states missing for %s — skipping recovery metrics.", cond)
                continue
            logger.info("Computing recovery metrics for %s", cond)
            results[cond] = compute_recovery_metrics(
                hs_recipient, hs[cond], t_fixed=t_fixed, n_layers=n_layers,
            )

    return results


# ─── All-metric correlation ─────────────────────────────────────────────────

def compute_all_metric_correlations(
    boundary_grid: List[int],
    recovery_results: Dict[str, Dict],
    evaluation: Dict,
) -> Dict:
    """
    Compute Spearman correlation of each recovery metric with accuracy degradation.

    Only hard_swap conditions are used; random_donor is excluded.

    Args:
        boundary_grid    : List of boundary values tested.
        recovery_results : {condition_name: recovery_metrics_dict} from compute_recovery_sweep.
        evaluation       : evaluation.json dict.

    Returns:
        Dict keyed by metric name, each with {rho, p, status}.
    """
    baseline_acc: float = evaluation["baseline_accuracy"]
    accuracies: Dict = evaluation["accuracies"]

    metric_keys = ["rd_cosine", "rd_l2", "fld_cosine", "fld_l2", "recovery_cka", "recovery_rsa"]
    metric_vals: Dict[str, List[float]] = {k: [] for k in metric_keys}
    degs: List[float] = []

    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        if cond not in recovery_results:
            continue
        cond_acc = accuracies.get(cond, {}).get("accuracy")
        if cond_acc is None:
            continue

        degradation = max(0.0, baseline_acc - cond_acc)
        degs.append(degradation)
        for k in metric_keys:
            metric_vals[k].append(recovery_results[cond][k])

    correlations: Dict[str, Dict] = {}
    for k in metric_keys:
        if len(degs) < 2:
            correlations[k] = {"rho": None, "p": None, "status": "insufficient_data"}
            continue

        rho_val, p_val = spearmanr(metric_vals[k], degs)
        if math.isnan(rho_val) or math.isnan(p_val):
            logger.warning(
                "Spearman correlation returned NaN for %s — likely constant input values", k
            )
            correlations[k] = {"rho": None, "p": None, "status": "degenerate"}
        else:
            correlations[k] = {"rho": float(rho_val), "p": float(p_val), "status": "valid"}

    return correlations


# ─── BPD-degradation correlation ─────────────────────────────────────────────

def compute_bpd_degradation_correlation(
    boundary_grid: List[int],
    bpd_results: Dict[str, Dict],
    evaluation: Dict,
) -> Dict:
    """
    Compute Spearman rank correlation between BPD/EBPD and accuracy degradation.

    Degradation per boundary = max(0, baseline_accuracy - condition_accuracy),
    consistent with evaluator.py H1 definition.

    Args:
        boundary_grid : List of boundary values tested.
        bpd_results   : Output of compute_bpd_sweep() or similar.
        evaluation    : evaluation.json dict (output of evaluate_experiment()).

    Returns:
        Dict with keys:
            rho_bpd, p_bpd     : Spearman rho and p-value for BPD vs degradation
                                  (None if NaN or insufficient data)
            rho_ebpd, p_ebpd   : Spearman rho and p-value for EBPD vs degradation
                                  (None if NaN or insufficient data)
            status             : "valid" | "degenerate" | "insufficient_data"
            boundary_table     : List of per-boundary dicts for inspection
    """
    baseline_acc: float = evaluation["baseline_accuracy"]
    accuracies: Dict = evaluation["accuracies"]

    bpd_vals:   List[float] = []
    ebpd_vals:  List[float] = []
    degs:       List[float] = []
    table:      List[Dict]  = []

    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        if cond not in bpd_results:
            logger.warning("BPD missing for %s — excluded from correlation.", cond)
            continue
        cond_acc = accuracies.get(cond, {}).get("accuracy")
        if cond_acc is None:
            logger.warning("Accuracy missing for %s — excluded from correlation.", cond)
            continue

        degradation = max(0.0, baseline_acc - cond_acc)
        bpd_vals.append(bpd_results[cond]["bpd_mean"])
        ebpd_vals.append(bpd_results[cond]["ebpd_mean"])
        degs.append(degradation)
        table.append({
            "boundary":    b,
            "condition":   cond,
            "accuracy":    cond_acc,
            "degradation": degradation,
            "bpd_mean":    bpd_results[cond]["bpd_mean"],
            "ebpd_mean":   bpd_results[cond]["ebpd_mean"],
        })

    rho_bpd = p_bpd = rho_ebpd = p_ebpd = None
    status = "insufficient_data"

    if len(bpd_vals) >= 2:
        status = "valid"

        _rho_bpd, _p_bpd = spearmanr(bpd_vals, degs)
        if math.isnan(_rho_bpd) or math.isnan(_p_bpd):
            logger.warning(
                "Spearman correlation returned NaN for bpd — "
                "likely due to near-constant input values"
            )
            status = "degenerate"
        else:
            rho_bpd = float(_rho_bpd)
            p_bpd   = float(_p_bpd)

        _rho_ebpd, _p_ebpd = spearmanr(ebpd_vals, degs)
        if math.isnan(_rho_ebpd) or math.isnan(_p_ebpd):
            logger.warning(
                "Spearman correlation returned NaN for ebpd — "
                "likely due to near-constant input values"
            )
            status = "degenerate"
        else:
            rho_ebpd = float(_rho_ebpd)
            p_ebpd   = float(_p_ebpd)

    return {
        "rho_bpd":        rho_bpd,
        "p_bpd":          p_bpd,
        "rho_ebpd":       rho_ebpd,
        "p_ebpd":         p_ebpd,
        "status":         status,
        "boundary_table": table,
    }


# ─── Summary report ──────────────────────────────────────────────────────────

def print_summary(run_dir: str) -> None:
    """
    Load a completed run and print a comparison table of all metrics.

    Includes per-boundary accuracy, BDS, BPD, EBPD, and random_donor
    comparison rows. Also prints Spearman correlations for all metric
    families vs. accuracy degradation, including recovery-zone metrics
    (RD cosine/L2, FLD cosine/L2, Recovery CKA, Recovery RSA).

    Args:
        run_dir: Path to the run directory.
    """
    print(f"\n=== Post-Analysis Summary ===")
    print(f"Run: {os.path.abspath(run_dir)}")

    run_data = load_run(run_dir)
    bpd_results = compute_bpd_sweep(run_data)
    corr = compute_bpd_degradation_correlation(
        run_data["boundary_grid"], bpd_results, run_data["evaluation"]
    )
    recovery_results = compute_recovery_sweep(run_data)
    recovery_corr = compute_all_metric_correlations(
        run_data["boundary_grid"], recovery_results, run_data["evaluation"]
    )

    evaluation   = run_data["evaluation"]
    bds_results  = run_data["bds"]
    boundary_grid = run_data["boundary_grid"]
    baseline_acc  = evaluation["baseline_accuracy"]
    accuracies    = evaluation["accuracies"]

    print(f"Baseline accuracy: {baseline_acc:.4f}\n")

    # ── Main boundary table ───────────────────────────────────────────────
    header = f"{'Boundary':>8}  {'Accuracy':>9}  {'Degradation':>11}  {'BDS_total':>9}  {'BPD_mean':>9}  {'EBPD_mean':>9}"
    print(header)
    print("-" * len(header))

    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        acc      = accuracies.get(cond, {}).get("accuracy")
        deg      = max(0.0, baseline_acc - acc) if acc is not None else float("nan")
        bds_tot  = bds_results.get(cond, {}).get("aggregate", {}).get("mean_bds_total")
        bpd_mean = bpd_results.get(cond, {}).get("bpd_mean")
        ebpd_mean = bpd_results.get(cond, {}).get("ebpd_mean")

        def _fmt(v: Optional[float], w: int = 9, d: int = 4) -> str:
            return f"{v:{w}.{d}f}" if v is not None else f"{'N/A':>{w}}"

        print(
            f"{b:>8}  {_fmt(acc)}  {_fmt(deg, 11)}  "
            f"{_fmt(bds_tot)}  {_fmt(bpd_mean)}  {_fmt(ebpd_mean)}"
        )

    # ── Correlation summary ───────────────────────────────────────────────
    print()
    bds_rho = evaluation.get("bds_delta_rho")
    bds_p   = evaluation.get("bds_delta_p")
    _r = lambda v: f"{v:.4f}" if v is not None else "N/A"

    print(f"BDS-degradation correlation:  rho={_r(bds_rho)}  p={_r(bds_p)}")
    print(f"BPD-degradation correlation:  rho={_r(corr['rho_bpd'])}  p={_r(corr['p_bpd'])}")
    print(f"EBPD-degradation correlation: rho={_r(corr['rho_ebpd'])}  p={_r(corr['p_ebpd'])}")
    if corr["status"] != "valid":
        print(f"  [WARNING] Correlation status: {corr['status']}")

    # ── Random donor comparison ───────────────────────────────────────────
    print("\nRandom donor comparison:")
    rd_header = f"{'Boundary':>8}  {'HD_acc':>7}  {'RD_acc':>7}  {'HD_BPD':>7}  {'RD_BPD':>7}"
    print(rd_header)
    print("-" * len(rd_header))

    for b in boundary_grid:
        hd_cond = f"hard_swap_b{b}"
        rd_cond = f"random_donor_b{b}"

        hd_acc  = accuracies.get(hd_cond, {}).get("accuracy")
        rd_acc  = accuracies.get(rd_cond, {}).get("accuracy")
        hd_bpd  = bpd_results.get(hd_cond, {}).get("bpd_mean")
        rd_bpd  = bpd_results.get(rd_cond, {}).get("bpd_mean")

        def _fs(v: Optional[float]) -> str:
            return f"{v:7.4f}" if v is not None else f"{'N/A':>7}"

        print(f"{b:>8}  {_fs(hd_acc)}  {_fs(rd_acc)}  {_fs(hd_bpd)}  {_fs(rd_bpd)}")

    print()

    # ── EBPD-BDS consistency check ────────────────────────────────────────
    # per_layer_excess[0] at l=b should match mean_bds_lower from bds_*.json.
    # (BDS_total = BDS_lower + BDS_upper; do NOT compare against mean_bds_total.)
    _TOL = 1e-5
    mismatches: List[str] = []
    checked = 0
    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        bpd_r = bpd_results.get(cond)
        bds_r = bds_results.get(cond)
        if bpd_r is None or bds_r is None:
            continue
        per_layer_excess = bpd_r.get("per_layer_excess", [])
        if not per_layer_excess:
            continue
        ebpd_lower = per_layer_excess[0]
        bds_lower  = bds_r.get("aggregate", {}).get("mean_bds_lower")
        if bds_lower is None:
            continue
        checked += 1
        if abs(ebpd_lower - bds_lower) > _TOL:
            mismatches.append(
                f"  b={b}: per_layer_excess[0]={ebpd_lower:.6f}  "
                f"mean_bds_lower={bds_lower:.6f}  diff={ebpd_lower - bds_lower:.2e}"
            )

    if checked == 0:
        print("EBPD-BDS consistency check: SKIPPED (no overlapping data)")
    elif mismatches:
        print("EBPD-BDS consistency check: FAILED")
        for msg in mismatches:
            print(msg)
    else:
        print(f"EBPD-BDS consistency check: PASSED ({checked} boundaries checked)")

    # ── Recovery Metric Comparison ────────────────────────────────────────
    print(f"\n{'=' * 67}")
    print("=== Recovery Metric Comparison ===")
    print(f"{'Metric':<18}{'Mean Value':>12}{'Corr w/ Degradation':>21}{'p-value':>10}{'Status':>8}")
    print("\u2500" * 67)

    # Recovery metrics
    recovery_rows = [
        ("RD (cosine)",    "rd_cosine"),
        ("RD (L2)",        "rd_l2"),
        ("FLD (cosine)",   "fld_cosine"),
        ("FLD (L2)",       "fld_l2"),
        ("Recovery CKA",   "recovery_cka"),
        ("Recovery RSA",   "recovery_rsa"),
    ]

    # Compute mean of each recovery metric across hard_swap conditions
    for label, key in recovery_rows:
        vals = [
            recovery_results[f"hard_swap_b{b}"][key]
            for b in boundary_grid
            if f"hard_swap_b{b}" in recovery_results
        ]
        mean_val = sum(vals) / len(vals) if vals else float("nan")
        rc = recovery_corr.get(key, {})
        rho = rc.get("rho")
        p = rc.get("p")
        status = rc.get("status", "N/A")
        print(
            f"{label:<18}{mean_val:>12.4f}"
            f"{_r(rho):>21}{_r(p):>10}"
            f"{status:>8}"
        )

    # BDS row (existing)
    bds_mean_vals = [
        bds_results.get(f"hard_swap_b{b}", {}).get("aggregate", {}).get("mean_bds_total")
        for b in boundary_grid
    ]
    bds_mean_vals = [v for v in bds_mean_vals if v is not None]
    bds_mean = sum(bds_mean_vals) / len(bds_mean_vals) if bds_mean_vals else float("nan")
    print(
        f"{'BDS (existing)':<18}{bds_mean:>12.4f}"
        f"{_r(bds_rho):>21}{_r(bds_p):>10}"
        f"{'valid' if bds_rho is not None else 'N/A':>8}"
    )

    # BPD row (existing)
    bpd_mean_vals = [
        bpd_results.get(f"hard_swap_b{b}", {}).get("bpd_mean")
        for b in boundary_grid
    ]
    bpd_mean_vals = [v for v in bpd_mean_vals if v is not None]
    bpd_mean = sum(bpd_mean_vals) / len(bpd_mean_vals) if bpd_mean_vals else float("nan")
    print(
        f"{'BPD (existing)':<18}{bpd_mean:>12.4f}"
        f"{_r(corr['rho_bpd']):>21}{_r(corr['p_bpd']):>10}"
        f"{corr['status']:>8}"
    )

    print()


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc BPD analysis for a completed Stage 1 run."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the completed run directory (contains manifest.json, evaluation.json, etc.)",
    )
    parser.add_argument(
        "--log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    print_summary(args.run_dir)


if __name__ == "__main__":
    main()
