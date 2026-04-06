"""Post-hoc analysis script for Stage 1 completed runs.

Loads saved artifacts (hidden states, BDS results, evaluation, manifest)
from a run directory and computes additional metrics — specifically
Boundary-Propagated Drift (BPD) and Excess BPD (EBPD) — without
re-running any experiments.

Usage (CLI):
    python stage1/analysis/post_analysis.py --run_dir /workspace/outputs/run_20260405_130326

Usage (Jupyter):
    from stage1.analysis.post_analysis import load_run, compute_bpd, print_summary
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from scipy.stats import spearmanr

from stage1.analysis.bds import cosine_distance

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

    import math

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
    comparison rows. Also prints Spearman correlations for all three
    metric families vs. accuracy degradation.

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
