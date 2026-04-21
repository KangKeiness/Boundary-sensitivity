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
import sys
from typing import Dict, List, Optional, Tuple  # noqa: F401 — Tuple used below

import torch
from scipy.stats import spearmanr

from stage1.analysis.bds import cosine_distance, linear_cka_matrix

logger = logging.getLogger(__name__)


# ─── Condition-name enumeration ──────────────────────────────────────────────
#
# Phase C spec §8: extend the set of recognized condition-name prefixes used by
# ``compute_bpd_sweep`` to include Phase A width-confound grid families
# (``fixed_w4_``, ``fixed_b8_``, ``random_fixed_``) and Phase B restoration /
# corruption conditions (``patch_``, ``corrupt_``), in addition to the legacy
# Phase A / Stage 2 prefixes (``hard_swap_b``, ``random_donor_b``).
#
# The extension is ADDITIVE. The iteration order for the legacy subset is
# preserved: for every ``b`` in ``boundary_grid``, ``hard_swap_b{b}`` is
# followed by ``random_donor_b{b}``. Only the NEW families are discovered via
# ``sorted(hs.keys())`` filtering, and they are appended AFTER the legacy
# subset so that every pre-existing callsite sees byte-identical output when
# only legacy keys are present.

CONDITION_NAME_PREFIXES: Tuple[str, ...] = (
    "hard_swap_b",       # Phase A / Stage 2 (pre-existing)
    "random_donor_b",    # Phase A / Stage 2 (pre-existing)
    "fixed_w4_",         # Phase A width-confound grid
    "fixed_b8_",         # Phase A width-confound grid
    "random_fixed_",     # Phase A random-donor variant
    "patch_",            # Phase B restoration conditions
    "corrupt_",          # Phase B reverse-corruption conditions
)

_LEGACY_B_SUFFIX_PREFIXES: Tuple[str, ...] = ("hard_swap_b", "random_donor_b")


def _enumerate_conditions(hs: Dict, boundary_grid: List[int]) -> List[str]:
    """Return condition-name keys of ``hs`` that match known prefix families.

    Iteration order:
        1. For each ``b`` in ``boundary_grid``, emit ``hard_swap_b{b}`` then
           ``random_donor_b{b}`` (byte-identical to the pre-change loop).
        2. Then append keys whose prefix is in the new families
           (``fixed_w4_``, ``fixed_b8_``, ``random_fixed_``, ``patch_``,
           ``corrupt_``) in ``sorted(hs.keys())`` order.

    Keys absent from ``hs`` are skipped (with a WARNING log for the legacy
    prefixes to preserve the pre-change warning behaviour).
    """
    out: List[str] = []
    # Legacy prefixes: preserve pre-change order and warning behaviour.
    for b in boundary_grid:
        for prefix in _LEGACY_B_SUFFIX_PREFIXES:
            cond = f"{prefix}{b}"
            if cond not in hs:
                logger.warning("Hidden states missing for %s — skipping BPD.", cond)
                continue
            out.append(cond)

    # New families: sorted, filtered, de-duplicated against already-emitted.
    new_prefixes = tuple(
        p for p in CONDITION_NAME_PREFIXES if p not in _LEGACY_B_SUFFIX_PREFIXES
    )
    emitted = set(out)
    for key in sorted(hs.keys()):
        if key in emitted:
            continue
        for p in new_prefixes:
            if key.startswith(p):
                out.append(key)
                emitted.add(key)
                break
    return out


# v4 P3 — static Phase A grid mirror.
# Mirrors ``stage1.models.composer.PHASE_A_GRID`` byte-for-byte. The two tables
# MUST stay in sync; ``test_post_analysis_static_grid_matches_composer``
# imports composer (when torch is available) and asserts equality. The static
# copy here lets analysis run in environments without torch / transformers.
_STATIC_FIXED_W4_GRID: Dict[str, Tuple[int, int]] = {
    "fixed_w4_pos1": (4, 8),
    "fixed_w4_pos2": (8, 12),
    "fixed_w4_pos3": (12, 16),
    "fixed_w4_pos4": (16, 20),
}
_STATIC_FIXED_B8_GRID: Dict[str, Tuple[int, int]] = {
    "fixed_b8_w2": (8, 10),
    "fixed_b8_w4": (8, 12),
    "fixed_b8_w6": (8, 14),
    "fixed_b8_w8": (8, 16),
}
_STATIC_PHASE_A_GRID: Dict[str, Tuple[int, int]] = {
    **_STATIC_FIXED_W4_GRID,
    **_STATIC_FIXED_B8_GRID,
    **{f"random_{k}": v for k, v in _STATIC_FIXED_W4_GRID.items()},
    **{f"random_{k}": v for k, v in _STATIC_FIXED_B8_GRID.items()},
}


def _infer_b_for_condition(cond: str, run_data: Dict) -> int:
    """Infer the boundary value ``b`` for a non-legacy condition name.

    For legacy ``hard_swap_b{b}`` / ``random_donor_b{b}`` names this helper
    is not used (the caller iterates ``boundary_grid`` directly). For new
    families, ``b`` is resolved from a static grid that mirrors
    ``composer.PHASE_A_GRID``.

    v4 P3: the static grid below is the *primary* source — analysis must not
    depend on whether ``composer`` (which transitively requires torch /
    transformers) is importable. The canonical ``composer.PHASE_A_GRID`` is
    consulted only as an upgrade path *if* it imports successfully, to catch
    drift between the two grids in environments that have both. A drift would
    hard-fail with a ``ValueError`` at first use, not silently disagree.

    Cross-checked at test time by
    ``test_post_analysis_static_grid_matches_composer``.
    """
    if cond in _STATIC_PHASE_A_GRID:
        b_val, _t_val = _STATIC_PHASE_A_GRID[cond]
        # Drift-detection upgrade path — only when composer is importable.
        try:
            from stage1.models.composer import PHASE_A_GRID
        except Exception:
            PHASE_A_GRID = None  # noqa: N806 — torch/transformers absent
        if PHASE_A_GRID is not None and cond in PHASE_A_GRID:
            heavy_b, heavy_t = PHASE_A_GRID[cond]
            if (heavy_b, heavy_t) != (b_val, _t_val):
                raise ValueError(
                    f"Phase A grid drift: composer.PHASE_A_GRID[{cond!r}]="
                    f"{(heavy_b, heavy_t)} but post_analysis._STATIC_PHASE_A_GRID"
                    f"[{cond!r}]={(b_val, _t_val)}. Re-sync the two tables."
                )
        return b_val

    if cond.startswith("fixed_b"):
        # fixed_b8_xxx → 8
        suffix = cond[len("fixed_b"):]
        num = []
        for ch in suffix:
            if ch.isdigit():
                num.append(ch)
            else:
                break
        if num:
            return int("".join(num))
    if cond.startswith("fixed_w"):
        # If we reached here the static grid did not contain ``cond`` — that
        # means an unknown fixed_w* family was added without updating the
        # static table. Refuse rather than silently returning the wrong value.
        raise ValueError(
            f"Cannot infer b for {cond!r}: condition is in the fixed_w family "
            f"but not in post_analysis._STATIC_PHASE_A_GRID. Add it to the "
            f"static table (and composer.PHASE_A_GRID) before running analysis."
        )
    if cond.startswith("patch_") or cond.startswith("corrupt_"):
        # Phase B compose_meta.b == 8 by construction.
        return 8
    grid = run_data.get("boundary_grid") or []
    if grid:
        return int(grid[-1])
    return 8


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

    # Legacy emission first (byte-identical to the pre-Phase-C loop).
    enumerated = _enumerate_conditions(hs, boundary_grid)
    for cond in enumerated:
        # Resolve ``b`` for this condition.
        if cond.startswith("hard_swap_b") or cond.startswith("random_donor_b"):
            # Parse trailing integer after the shared prefix.
            suffix = cond.split("_b")[-1]
            try:
                b = int(suffix)
            except ValueError:
                logger.warning("Unable to parse boundary from %s — skipping.", cond)
                continue
        else:
            b = _infer_b_for_condition(cond, run_data)
        logger.info("Computing BPD for %s (b=%d)", cond, b)
        results[cond] = compute_bpd(hs_recipient, hs[cond], b, t_fixed, n_layers=n_layers)

    return results


# ─── Progress helper ─────────────────────────────────────────────────────────

def _progress(current: int, total: int, label: str, bar_len: int = 30) -> None:
    """Print an inline progress bar to stderr (no newline until 100%)."""
    frac = current / total if total > 0 else 1.0
    filled = int(bar_len * frac)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = frac * 100
    end = "\n" if current == total else ""
    sys.stderr.write(f"\r  {label} |{bar}| {pct:5.1f}% ({current}/{total}){end}")
    sys.stderr.flush()


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

    for s_idx, sid in enumerate(common_ids):
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
        _progress(s_idx + 1, n_samples, "RD/FLD")

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
        _progress(i + 1, n_recovery, "CKA   ")

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
            _progress(i + 1, n_recovery, "RSA   ")
            continue

        rho, _ = spearmanr(upper_comp, upper_rec)
        if math.isnan(rho):
            logger.warning("RSA returned NaN at recovery layer %d — treating as 0 correlation", recovery_layers[i])
            per_layer_rsa_shift.append(1.0)
        else:
            per_layer_rsa_shift.append(1.0 - float(rho))
        _progress(i + 1, n_recovery, "RSA   ")

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

    # Build condition list for progress tracking
    conditions = []
    for b in boundary_grid:
        for prefix in ("hard_swap_b", "random_donor_b"):
            cond = f"{prefix}{b}"
            if cond in hs:
                conditions.append((b, cond))

    for c_idx, (b, cond) in enumerate(conditions):
        sys.stderr.write(f"\n[{c_idx + 1}/{len(conditions)}] {cond}\n")
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


# ─── Phase A analysis functions (additive — D8) ──────────────────────────────

def load_phase_a_run(run_dir: str) -> Dict:
    """
    Load all Phase A artifacts from a completed phase_a run directory.

    Expects the following files (written by stage1/run_phase_a.py):
      - manifest.json
      - phase_a_summary.json
      - phase_a_all_conditions.csv
      - grid1_position_effect.csv
      - grid2_width_effect.csv
      - phase_a_summary.txt

    Args:
        run_dir: Absolute path to the Phase A run directory.

    Returns:
        Dict with keys:
            manifest              : full manifest.json dict
            summary_json          : full phase_a_summary.json dict
            all_conditions_csv    : List[Dict] rows from phase_a_all_conditions.csv
            grid1_csv             : List[Dict] rows from grid1_position_effect.csv
            grid2_csv             : List[Dict] rows from grid2_width_effect.csv
            summary_txt           : str content of phase_a_summary.txt
    """
    import csv as _csv

    run_dir = os.path.abspath(run_dir)

    def _load_json(fname: str) -> Dict:
        with open(os.path.join(run_dir, fname)) as f:
            return json.load(f)

    def _load_csv(fname: str) -> List[Dict]:
        path = os.path.join(run_dir, fname)
        with open(path, newline="") as f:
            return list(_csv.DictReader(f))

    def _load_txt(fname: str) -> str:
        with open(os.path.join(run_dir, fname), encoding="utf-8") as f:
            return f.read()

    return {
        "manifest":           _load_json("manifest.json"),
        "summary_json":       _load_json("phase_a_summary.json"),
        "all_conditions_csv": _load_csv("phase_a_all_conditions.csv"),
        "grid1_csv":          _load_csv("grid1_position_effect.csv"),
        "grid2_csv":          _load_csv("grid2_width_effect.csv"),
        "summary_txt":        _load_txt("phase_a_summary.txt"),
    }


def compute_phase_a_primary_table(run_data: Dict, grid: str) -> List[Dict]:
    """
    Return the primary-metric table for a Phase A grid.

    PRIMARY metrics (directly comparable across all Phase A conditions):
      condition, b, t, width, accuracy, degradation, fld_cos, fld_l2

    SECONDARY/EXPLORATORY metrics (not included — recovery-zone length varies):
      RD_cos, RD_l2, BPD_mean, EBPD_mean, CKA, RSA

    Args:
        run_data: Output of load_phase_a_run().
        grid: One of "grid1" (position-effect) or "grid2" (width-effect).

    Returns:
        List of dicts with PRIMARY metric columns only.
    """
    _VALID_GRIDS = {"grid1", "grid2"}
    if grid not in _VALID_GRIDS:
        raise ValueError(f"grid must be one of {_VALID_GRIDS}, got {grid!r}")

    key = "grid1_csv" if grid == "grid1" else "grid2_csv"
    raw_rows = run_data[key]

    PRIMARY_COLS = ["condition", "b", "t", "width", "accuracy", "degradation", "fld_cos", "fld_l2"]
    result: List[Dict] = []
    for row in raw_rows:
        filtered = {col: row[col] for col in PRIMARY_COLS if col in row}
        # Cast numeric strings back to float/int
        for num_col in ("b", "t", "width"):
            if num_col in filtered and filtered[num_col] not in (None, "", "None"):
                try:
                    filtered[num_col] = int(filtered[num_col])
                except (ValueError, TypeError):
                    pass
        for float_col in ("accuracy", "degradation", "fld_cos", "fld_l2"):
            if float_col in filtered and filtered[float_col] not in (None, "", "None"):
                try:
                    filtered[float_col] = float(filtered[float_col])
                except (ValueError, TypeError):
                    pass
        result.append(filtered)
    return result


def print_phase_a_summary(run_dir: str) -> None:
    """
    Load a completed Phase A run and print PRIMARY two-grid tables.

    Output includes:
      - PRIMARY metrics header (A9)
      - Grid 1: position-effect table (fixed_w4_pos* + random_fixed_w4_pos*)
      - Grid 2: width-effect table (fixed_b8_w* + random_fixed_b8_w*)
      - PRIMARY vs SECONDARY classification (A9 / 10.3)
      - primary_metrics_note from JSON (A5)

    Does NOT touch Stage 1 entry points (print_summary, compute_recovery_sweep).

    Args:
        run_dir: Path to the Phase A run directory.
    """
    run_data = load_phase_a_run(run_dir)
    summary_json = run_data["summary_json"]
    manifest = run_data["manifest"]

    print()
    print("=" * 70)
    print("PHASE A — WIDTH CONFOUND SEPARATION GRID  [post_analysis]")
    print("=" * 70)
    print(f"Run: {os.path.abspath(run_dir)}")
    print(f"Baseline accuracy (no_swap): {summary_json.get('baseline_accuracy', 'N/A')}")
    print(f"N samples: {summary_json.get('n_samples', 'N/A')}")
    print(f"Sanity mode: {summary_json.get('sanity_mode', 'N/A')}")
    print()

    # PRIMARY / SECONDARY classification note (A5 / A9)
    note = summary_json.get("primary_metrics_note", "")
    print("METRIC CLASSIFICATION:")
    print(f"  PRIMARY metrics (directly comparable): accuracy, degradation, fld_cos, fld_l2")
    print(f"  SECONDARY/EXPLORATORY metrics (NOT directly comparable across Grid 1 conditions,")
    print(f"    because recovery-zone length = L - t varies when t varies):")
    print(f"    RD_cos, RD_l2, BPD_mean, EBPD_mean, CKA, RSA")
    print()
    print(f"[primary_metrics_note]: {note}")
    print()

    _COL_FMT = f"{'Condition':<25} {'b':>3} {'t':>3} {'w':>3} {'Accuracy':>9} {'Degradation':>12} {'FLD_cos':>9} {'FLD_l2':>9}"
    _DIV = "-" * len(_COL_FMT)

    def _fmt_row(r: Dict) -> str:
        def _s(v, w: int = 3) -> str:
            return f"{v!s:>{w}}" if v not in (None, "", "None") else f"{'N/A':>{w}}"
        def _f(v, w: int = 9, d: int = 4) -> str:
            try:
                return f"{float(v):{w}.{d}f}"
            except (TypeError, ValueError):
                return f"{'N/A':>{w}}"
        return (
            f"{r.get('condition',''):<25} {_s(r.get('b'))} {_s(r.get('t'))} "
            f"{_s(r.get('width'))} {_f(r.get('accuracy'), 9, 4)} "
            f"{_f(r.get('degradation'), 12, 4)} {_f(r.get('fld_cos'), 9, 6)} "
            f"{_f(r.get('fld_l2'), 9, 4)}"
        )

    # Grid 1 table
    print("-" * 70)
    print("Grid 1: Position effect  (fixed width = 4)")
    print("PRIMARY metrics — condition, b, t, width, accuracy, degradation, fld_cos, fld_l2")
    print("-" * 70)
    g1_rows = compute_phase_a_primary_table(run_data, "grid1")
    if g1_rows:
        print(_COL_FMT)
        print(_DIV)
        for r in g1_rows:
            print(_fmt_row(r))
    else:
        print("  (no data — sanity run may have been limited to 1 non-baseline condition)")
    print()

    # Grid 2 table
    print("-" * 70)
    print("Grid 2: Width effect  (fixed boundary b = 8)")
    print("PRIMARY metrics — condition, b, t, width, accuracy, degradation, fld_cos, fld_l2")
    print("-" * 70)
    g2_rows = compute_phase_a_primary_table(run_data, "grid2")
    if g2_rows:
        print(_COL_FMT)
        print(_DIV)
        for r in g2_rows:
            print(_fmt_row(r))
    else:
        print("  (no data — sanity run may not include Grid 2 conditions)")
    print()

    # Random donor seeds verification note
    rds = manifest.get("random_donor_seeds", {})
    if rds:
        print(f"Random donor seeds (A6): {len(rds)} condition(s) logged.")
        for cname, cseed in sorted(rds.items()):
            print(f"  {cname}: {cseed}")
    print()

    print("SECONDARY/EXPLORATORY metrics (not shown — comparability across Grid 1 conditions")
    print("is compromised because recovery-zone length L - t differs when t varies):")
    print("  RD_cos, RD_l2, BPD_mean, EBPD_mean, CKA, RSA")
    print("  Use stage1.analysis.post_analysis.print_summary for Stage 1 (boundary-sweep) runs.")
    print()


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis for a completed Stage 1 or Phase A run."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the completed run directory (contains manifest.json, evaluation.json, etc.)",
    )
    parser.add_argument(
        "--phase_a",
        action="store_true",
        help="Treat run_dir as a Phase A run and print primary two-grid tables.",
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
    if args.phase_a:
        print_phase_a_summary(args.run_dir)
    else:
        print_summary(args.run_dir)


if __name__ == "__main__":
    main()
