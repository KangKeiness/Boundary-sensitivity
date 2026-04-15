"""Phase B — Restoration Intervention (prompt-side hidden-state patching).

Tests whether recovery-side disruption is causally relevant to degradation
via prompt-side hidden-state patching. See
``notes/specs/phase_b_rewrite.md`` for the authoritative spec.

IMPORTANT METHODOLOGICAL CONSTRAINT:
    Patching applies only to prompt-side hidden-state processing. Clean hidden
    states are available for prompt tokens only. This is prompt-side
    restoration intervention, NOT full-sequence causal intervention. All
    summaries state this explicitly.

Phase C terminology ("restoration effect", "residual effect", "restoration
proportion") is reserved and MUST NOT appear in Phase B artifacts; this is
enforced by the FORBIDDEN_PHRASES gate in ``utils.wording``.

Usage (CLI):
    python stage1/run_phase_b.py --config stage1/configs/stage2_confound.yaml
    python stage1/run_phase_b.py --config stage1/configs/stage2_confound.yaml --sanity

Usage (Jupyter):
    from stage1.run_phase_b import run_phase_b
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import hashlib
import json
import logging
import os
import random
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers

from stage1.utils.config import load_config, setup_logging
from stage1.utils.logger import create_run_dir
from stage1.utils.wording import FORBIDDEN_PHRASES, check_artifacts_for_forbidden
from stage1.data.loader import load_mgsm
from stage1.models.composer import load_models, compose_model
from stage1.inference.parser import parse_answer
from stage1.analysis.evaluator import exact_match
from stage1.intervention.patcher import (
    METHODOLOGY_TAG,
    PatchConfig,
    RESTORATION_PATCHES,
    REVERSE_CORRUPTION_PATCHES,
    run_patched_inference,
)

logger = logging.getLogger(__name__)

# Effect-size threshold for the comparative sentence gate (spec §7, §11.10).
EPSILON_DELTA: float = 0.02

# Cross-phase accuracy tolerance (2/250 samples).
PHASE_A_CROSS_CHECK_TOL: float = 0.008

# Canonical methodological-constraint string (contains "prompt-side" per spec §11.2).
METHODOLOGICAL_CONSTRAINT: str = (
    "Patching applies only to prompt-side hidden-state processing. "
    "Clean hidden states are available for prompt tokens only. "
    "This is prompt-side restoration intervention, NOT full-sequence "
    "causal intervention. Claims remain intervention-based and conservative."
)


# ─── Determinism / seeding helpers ───────────────────────────────────────────

def _apply_determinism(seed: int) -> List[str]:
    """Seed all RNGs and enable deterministic algorithms.

    Spec §6. Returns a list of warning strings captured from
    ``torch.use_deterministic_algorithms`` (``warn_only=True``).
    """
    # Must be set BEFORE any CUDA op.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    warnings: List[str] = []
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # pragma: no cover — environment-dependent
        warnings.append(f"use_deterministic_algorithms raised: {exc!r}")
    return warnings


def _git_sha() -> str:
    try:
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_repo_root
        ).decode().strip()
    except Exception:
        return "unknown"


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """Deterministic SHA-256 over the model's state_dict tensor bytes."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for key in sorted(sd.keys()):
        t = sd[key]
        h.update(key.encode("utf-8"))
        # Always hash on CPU in contiguous float32 to be deterministic across devices.
        h.update(t.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


# ─── Phase A cross-check loader ──────────────────────────────────────────────

def _phase_a_outputs_dir() -> str:
    """Resolve the Phase A outputs directory relative to the repo root.

    Uses ``pathlib`` resolution from this file's location rather than CWD, so
    the spec §11.7 cross-check is invariant to how the entrypoint is launched
    (``python -m stage1.run_phase_b`` vs. ``python stage1/run_phase_b.py`` vs.
    scheduled job with arbitrary CWD). Spec §11.7.
    """
    import pathlib
    return str(
        pathlib.Path(__file__).resolve().parents[1] / "stage1" / "outputs" / "phase_a"
    )


def _load_latest_phase_a_summary() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return (summary_dict, resolved_path) or (None, None) if no match.

    The dataset manifest block and ``hard_swap_b8`` / ``no_swap`` accuracies
    are read from this artifact (spec §4, §7). The path is resolved relative
    to the repo root (NOT CWD) so the §11.7 acceptance check is deterministic.
    """
    pattern = os.path.join(_phase_a_outputs_dir(), "run_*", "phase_a_summary.json")
    candidates = sorted(glob.glob(pattern), reverse=True)
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f), path
        except Exception:
            continue
    return None, None


def _stage1_outputs_dir() -> str:
    """Resolve the Stage 1 sweep outputs dir (``stage1/outputs/run_*``).

    Stage 1 runs include the original ``hard_swap_b8`` condition at (b=8, t=20),
    which Phase A's confound grid does NOT contain (Phase A uses fixed_w4_posN /
    fixed_b8_wN). The treatment-parity cross-check (no_patch ~ hard_swap_b8)
    therefore must fall back to Stage 1 artifacts when Phase A lacks the anchor.
    """
    import pathlib
    return str(pathlib.Path(__file__).resolve().parents[1] / "stage1" / "outputs")


def _load_latest_stage1_hard_swap_b8() -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Return ``(hard_swap_b8_accuracy, no_swap_accuracy, evaluation_path)``.

    Searches ``stage1/outputs/run_*/evaluation.json`` in reverse-chronological
    order (newest first) for a run containing a ``hard_swap_b8`` row. Returns
    ``(None, None, None)`` if no matching run exists.
    """
    pattern = os.path.join(_stage1_outputs_dir(), "run_*", "evaluation.json")
    candidates = sorted(glob.glob(pattern), reverse=True)
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                eval_dict = json.load(f)
        except Exception:
            continue
        accs = eval_dict.get("accuracies") or {}
        hs = accs.get("hard_swap_b8")
        ns = accs.get("no_swap")
        hs_acc = float(hs["accuracy"]) if isinstance(hs, dict) and hs.get("accuracy") is not None else None
        ns_acc = (
            float(ns["accuracy"]) if isinstance(ns, dict) and ns.get("accuracy") is not None
            else (float(eval_dict["baseline_accuracy"]) if eval_dict.get("baseline_accuracy") is not None else None)
        )
        if hs_acc is not None:
            return hs_acc, ns_acc, path
    return None, None, None


def _phase_a_condition_accuracy(summary: Dict[str, Any], condition: str) -> Optional[float]:
    """Pull accuracy for a named condition from a Phase A summary dict."""
    # Phase A emits per-condition rows under ``all_conditions``.
    rows = summary.get("all_conditions") or []
    for row in rows:
        if row.get("condition") == condition:
            val = row.get("accuracy")
            if val is None:
                continue
            return float(val)
    # Baseline fallback: ``baseline_accuracy`` is no_swap.
    if condition == "no_swap":
        val = summary.get("baseline_accuracy")
        if val is not None:
            return float(val)
    return None


# ─── Paired bootstrap for comparative-sentence gate ──────────────────────────

def _paired_bootstrap_diff_ci(
    corrects_a: List[int], corrects_b: List[int],
    *, n_resamples: int = 1000, seed: int = 0, ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi) of mean(a) - mean(b) via paired bootstrap.

    ``corrects_a``/``corrects_b`` are 0/1 integer lists of equal length (per-sample
    correctness). Sampling is paired over indices.
    """
    n = len(corrects_a)
    if n == 0 or n != len(corrects_b):
        raise ValueError("bootstrap requires equal-length non-empty vectors")
    a = np.asarray(corrects_a, dtype=np.float64)
    b = np.asarray(corrects_b, dtype=np.float64)
    point = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(diffs, alpha))
    hi = float(np.quantile(diffs, 1.0 - alpha))
    return point, lo, hi


# ─── IO helpers ──────────────────────────────────────────────────────────────

def _save_condition_results(run_dir: str, name: str, results: List[Dict]) -> None:
    """Save per-condition results as JSONL (UTF-8)."""
    path = os.path.join(run_dir, f"results_{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = {k: v for k, v in r.items() if k != "hidden_states"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: str, fieldnames: List[str], data: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(data)


def _compute_acc(results: List[Dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("correct", False)) / len(results)


def _annotate_correct(results: List[Dict], samples: List[Dict]) -> None:
    """In-place: parse answers and attach ``correct`` (bool) to each result.

    O(n) gold-answer lookup — Finding #12.
    """
    gold_by_id = {s["sample_id"]: s["gold_answer"] for s in samples}
    for r in results:
        parsed = parse_answer(r["output_text"])
        r.update(parsed)
        r["correct"] = exact_match(
            gold_by_id[r["sample_id"]],
            r.get("normalized_answer"),
        )


# ─── Main entrypoint ─────────────────────────────────────────────────────────

def run_phase_b(
    config_path: str,
    sanity: bool = False,
    seed: int = 42,
    run_name: Optional[str] = None,
) -> str:
    """Run Phase B restoration intervention experiment.

    Primary treatment: ``hard_swap_b8``. Primary reference: ``no_swap``.

    Returns:
        Path to the run directory.
    """
    setup_logging()

    # (1) Determinism flags FIRST (before any CUDA op via model load).
    determinism_warnings = _apply_determinism(seed)

    config = load_config(config_path)

    if sanity:
        config.dataset.debug_n = 5

    print("Phase B — Restoration Intervention (prompt-side)")
    print(f"  Config     : {config_path}")
    print(f"  Sanity mode: {sanity}")
    print(f"  Seed       : {seed}")
    print(f"  Run name   : {run_name}")

    run_dir = create_run_dir(base_dir="stage1/outputs/phase_b")
    print(f"  Run dir    : {run_dir}")

    # (2) Load data
    samples = load_mgsm(config)
    print(f"  Samples    : {len(samples)}")

    # (3) Load models
    print("Loading models...")
    recipient, donor, tokenizer = load_models(
        recipient_name=config.models.recipient,
        donor_name=config.models.donor,
        recipient_revision=config.models.recipient_revision,
        donor_revision=config.models.donor_revision,
    )
    n_layers = recipient.config.num_hidden_layers
    hidden_size = recipient.config.hidden_size
    print(f"  Layers     : {n_layers}, hidden_size: {hidden_size}")
    if n_layers != 28 or hidden_size != 1536:
        raise RuntimeError(
            f"Architecture assertion failed: expected num_hidden_layers=28, "
            f"hidden_size=1536; got {n_layers}, {hidden_size}."
        )

    # (4) Compose hard_swap_b8 (b=8, t=20).
    t_fixed = 20
    print(f"Composing hard_swap_b8 (b=8, t={t_fixed})...")
    composed, compose_meta = compose_model(
        recipient, donor, b=8, t=t_fixed, condition="hard_swap",
    )
    compose_meta = dict(compose_meta)  # copy; augment with hashes
    compose_meta["b"] = 8
    compose_meta["t"] = t_fixed
    compose_meta["condition"] = "hard_swap_b8"

    # Record state_dict SHA-256 before any inference (spec §10 test #4).
    sd_sha_before = _state_dict_sha256(composed)
    compose_meta["state_dict_sha256_before"] = sd_sha_before

    # (5) Free donor immediately after compose (spec §12 R6, memory rule).
    del donor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gen_config = {
        "do_sample":      config.generation.do_sample,
        "temperature":    config.generation.temperature,
        "max_new_tokens": config.generation.max_new_tokens,
    }

    # (6) Build patch configs.
    restoration_configs = [
        PatchConfig(name, layers, "restoration")
        for name, layers in RESTORATION_PATCHES.items()
    ]
    corruption_configs = [
        PatchConfig(name, layers, "corruption")
        for name, layers in REVERSE_CORRUPTION_PATCHES.items()
    ]

    if sanity:
        # Sanity-mode conditions per spec §7:
        # {no_patch, patch_recovery_full, clean_no_patch (recipient),
        #  corrupt_recovery_full} × 5 samples.
        keep_r = {"no_patch", "patch_recovery_full"}
        restoration_configs = [pc for pc in restoration_configs if pc.patch_name in keep_r]
        keep_c = {"corrupt_recovery_full"}
        corruption_configs = [pc for pc in corruption_configs if pc.patch_name in keep_c]

    # (7) Restoration interventions on composed.
    print(f"\n{'=' * 60}")
    print("RESTORATION INTERVENTIONS (composed ← clean patches)")
    print(f"{'=' * 60}")

    restoration_results: Dict[str, List[Dict]] = {}
    for pc in restoration_configs:
        print(f"\n--- {pc.patch_name} (layers: {pc.patch_layers}) ---")
        results = run_patched_inference(
            target_model=composed,
            recipient_model=recipient,
            composed_model=None,
            tokenizer=tokenizer,
            samples=samples,
            patch_config=pc,
            generation_config=gen_config,
        )
        _annotate_correct(results, samples)
        restoration_results[pc.patch_name] = results
        _save_condition_results(run_dir, f"restoration_{pc.patch_name}", results)

    # (8) Clean baseline: no_patch on recipient (spec §7).
    print(f"\n{'=' * 60}")
    print("CLEAN BASELINE (recipient, no intervention)")
    print(f"{'=' * 60}")
    clean_baseline_results = run_patched_inference(
        target_model=recipient,
        recipient_model=recipient,
        composed_model=None,
        tokenizer=tokenizer,
        samples=samples,
        patch_config=PatchConfig("clean_no_patch", [], "restoration"),
        generation_config=gen_config,
    )
    _annotate_correct(clean_baseline_results, samples)
    _save_condition_results(run_dir, "clean_no_patch", clean_baseline_results)

    # (9) Reverse corruption interventions on recipient.
    print(f"\n{'=' * 60}")
    print("REVERSE CORRUPTION (recipient ← corrupt patches)")
    print(f"{'=' * 60}")
    corruption_results: Dict[str, List[Dict]] = {}
    for pc in corruption_configs:
        print(f"\n--- {pc.patch_name} (layers: {pc.patch_layers}) ---")
        results = run_patched_inference(
            target_model=recipient,
            recipient_model=recipient,
            composed_model=composed,
            tokenizer=tokenizer,
            samples=samples,
            patch_config=pc,
            generation_config=gen_config,
        )
        _annotate_correct(results, samples)
        corruption_results[pc.patch_name] = results
        _save_condition_results(run_dir, f"corruption_{pc.patch_name}", results)

    # (10) Accuracy tables.
    no_patch_acc = _compute_acc(restoration_results.get("no_patch", []))
    clean_baseline_acc = _compute_acc(clean_baseline_results)

    restoration_table: List[Dict[str, Any]] = []
    for name in RESTORATION_PATCHES:
        if name not in restoration_results:
            continue
        acc = _compute_acc(restoration_results[name])
        restoration_table.append({
            "condition": name,
            "accuracy": round(acc, 4),
            "delta_from_no_patch": round(acc - no_patch_acc, 4),
            "delta_from_clean_baseline": round(acc - clean_baseline_acc, 4),
            "methodology": METHODOLOGY_TAG,
        })

    corruption_table: List[Dict[str, Any]] = []
    for name in REVERSE_CORRUPTION_PATCHES:
        if name not in corruption_results:
            continue
        acc = _compute_acc(corruption_results[name])
        corruption_table.append({
            "condition": name,
            "accuracy": round(acc, 4),
            "delta_from_clean_baseline": round(acc - clean_baseline_acc, 4),
            "methodology": METHODOLOGY_TAG,
        })

    # (11) Write CSV tables.
    _write_csv(
        os.path.join(run_dir, "restoration_table.csv"),
        ["condition", "accuracy", "delta_from_no_patch",
         "delta_from_clean_baseline", "methodology"],
        restoration_table,
    )
    _write_csv(
        os.path.join(run_dir, "corruption_table.csv"),
        ["condition", "accuracy", "delta_from_clean_baseline", "methodology"],
        corruption_table,
    )

    # (12) Cross-phase check: treatment parity + baseline parity.
    # Phase A's confound grid does NOT contain hard_swap_b8 (it uses
    # fixed_w4_posN / fixed_b8_wN — the original b=8,t=20 anchor is absent).
    # We therefore source hard_swap_b8 from the latest Stage 1 sweep run
    # (stage1/outputs/run_*/evaluation.json) and fall back to Phase A only for
    # the no_swap baseline.
    phase_a, phase_a_path = _load_latest_phase_a_summary()
    phase_a_hard_swap_acc: Optional[float] = None
    phase_a_no_swap_acc: Optional[float] = None
    if phase_a is not None:
        phase_a_hard_swap_acc = _phase_a_condition_accuracy(phase_a, "hard_swap_b8")
        phase_a_no_swap_acc = _phase_a_condition_accuracy(phase_a, "no_swap")

    stage1_hs_acc, stage1_ns_acc, stage1_eval_path = _load_latest_stage1_hard_swap_b8()

    # Prefer Stage 1 anchor for hard_swap_b8 (Phase A normally lacks it).
    anchor_hard_swap_acc: Optional[float] = (
        phase_a_hard_swap_acc if phase_a_hard_swap_acc is not None else stage1_hs_acc
    )
    anchor_no_swap_acc: Optional[float] = (
        phase_a_no_swap_acc if phase_a_no_swap_acc is not None else stage1_ns_acc
    )
    anchor_hard_swap_source: Optional[str] = (
        "phase_a" if phase_a_hard_swap_acc is not None
        else ("stage1" if stage1_hs_acc is not None else None)
    )
    anchor_no_swap_source: Optional[str] = (
        "phase_a" if phase_a_no_swap_acc is not None
        else ("stage1" if stage1_ns_acc is not None else None)
    )

    cross_check_passed: Optional[bool] = None
    deltas_ok: List[bool] = []
    if anchor_hard_swap_acc is not None:
        deltas_ok.append(
            abs(no_patch_acc - anchor_hard_swap_acc) <= PHASE_A_CROSS_CHECK_TOL
        )
    if anchor_no_swap_acc is not None:
        deltas_ok.append(
            abs(clean_baseline_acc - anchor_no_swap_acc) <= PHASE_A_CROSS_CHECK_TOL
        )
    if deltas_ok:
        cross_check_passed = all(deltas_ok)

    # (13) Environment block.
    env_block: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(next(composed.parameters()).device),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "git_sha": _git_sha(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "deterministic_algorithms_enabled": True,
        "determinism_warnings": determinism_warnings,
        "seed": seed,
    }

    # (14) Build summary JSON.
    dataset_block: Dict[str, Any] = {
        "name": config.dataset.name,
        "lang": config.dataset.lang,
        "split": config.dataset.split,
        "n_samples": len(samples),
    }
    # Mirror the Phase A dataset/manifest entry verbatim when available.
    if phase_a is not None and "dataset" in phase_a:
        dataset_block["phase_a_dataset_manifest"] = phase_a["dataset"]

    summary: Dict[str, Any] = {
        "phase": "B",
        "description": "Restoration intervention — prompt-side hidden-state patching",
        "treatment_condition": "hard_swap_b8",
        "reference_condition": "no_swap",
        "t_fixed": t_fixed,
        "epsilon_delta": EPSILON_DELTA,
        "sanity_mode": sanity,
        "run_name": run_name,
        "seed": seed,
        "no_patch_accuracy": round(no_patch_acc, 4),
        "clean_baseline_accuracy": round(clean_baseline_acc, 4),
        "restoration_table": restoration_table,
        "corruption_table": corruption_table,
        "methodological_constraint": METHODOLOGICAL_CONSTRAINT,
        "phase_a_cross_check": {
            "phase_a_summary_path": phase_a_path,
            "phase_a_outputs_dir": _phase_a_outputs_dir(),
            "phase_a_hard_swap_b8_accuracy": phase_a_hard_swap_acc,
            "phase_a_no_swap_accuracy": phase_a_no_swap_acc,
            "stage1_evaluation_path": stage1_eval_path,
            "stage1_hard_swap_b8_accuracy": stage1_hs_acc,
            "stage1_no_swap_accuracy": stage1_ns_acc,
            "anchor_hard_swap_b8_accuracy": anchor_hard_swap_acc,
            "anchor_hard_swap_source": anchor_hard_swap_source,
            "anchor_no_swap_accuracy": anchor_no_swap_acc,
            "anchor_no_swap_source": anchor_no_swap_source,
            "tolerance": PHASE_A_CROSS_CHECK_TOL,
            "passed": cross_check_passed,
            "note": (
                "Treatment parity (no_patch ≈ hard_swap_b8) is cross-checked "
                "against the newest Stage 1 sweep run since Phase A's confound "
                "grid does not contain hard_swap_b8 at (b=8, t=20)."
            ),
        },
        "compose_meta": compose_meta,
        "dataset": dataset_block,
        "environment": env_block,
    }

    # (15) Comparative-sentence gate (spec §7, §11.10).
    non_nopatch = [r for r in restoration_table if r["condition"] != "no_patch"]
    # Claims restricted to the 4 corruption-mirrored conditions (§11).
    claim_eligible = [r for r in non_nopatch if r["condition"] in (
        "patch_boundary_local", "patch_recovery_early",
        "patch_recovery_full", "patch_final_only",
    )]
    comparative_sentence: str
    comparative_block: Dict[str, Any] = {"fired": False}
    if len(claim_eligible) >= 2 and all(
        r["condition"] in restoration_results for r in claim_eligible
    ):
        best = max(claim_eligible, key=lambda r: r["delta_from_no_patch"])
        boundary_local = next(
            (r for r in claim_eligible if r["condition"] == "patch_boundary_local"), None,
        )
        if boundary_local is not None and best["condition"] != "patch_boundary_local":
            best_corr = [int(x.get("correct", False))
                         for x in restoration_results[best["condition"]]]
            bl_corr = [int(x.get("correct", False))
                       for x in restoration_results["patch_boundary_local"]]
            # Pair on sample ordering, which matches across conditions by construction.
            point, ci_lo, ci_hi = _paired_bootstrap_diff_ci(
                best_corr, bl_corr, n_resamples=1000, seed=0, ci=0.95,
            )
            both_positive = (best["delta_from_no_patch"] > 0.0
                             and boundary_local["delta_from_no_patch"] > 0.0)
            gate = both_positive and point > EPSILON_DELTA and ci_lo > 0.0
            comparative_block = {
                "fired": gate,
                "best_condition": best["condition"],
                "best_delta": best["delta_from_no_patch"],
                "boundary_local_delta": boundary_local["delta_from_no_patch"],
                "point_estimate_diff": point,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "epsilon_delta": EPSILON_DELTA,
            }
            if gate:
                comparative_sentence = (
                    f"Recovery-side intervention at {best['condition']} yields a larger "
                    f"prompt-side accuracy delta than patch_boundary_local "
                    f"(point={point:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
                    f"eps={EPSILON_DELTA})."
                )
            else:
                comparative_sentence = (
                    "Restoration deltas do not meet the effect-size threshold for a "
                    "directional claim "
                    f"(point={point:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
                    f"eps={EPSILON_DELTA})."
                )
        else:
            comparative_sentence = (
                "Restoration deltas do not meet the effect-size threshold for a "
                "directional claim (insufficient eligible conditions)."
            )
    else:
        comparative_sentence = (
            "Restoration deltas do not meet the effect-size threshold for a "
            "directional claim (insufficient eligible conditions)."
        )
    summary["comparative_claim"] = comparative_block

    # (16) Human-readable summary TXT (contains t=20 header and exactly one
    #      comparative sentence).
    lines: List[str] = [
        "=" * 60,
        "PHASE B — RESTORATION INTERVENTION RESULTS",
        "=" * 60,
        "",
        f"recovery-zone layers [20..27] defined at fixed t={t_fixed}",
        "",
        "Treatment condition: hard_swap_b8",
        "Reference condition: no_swap",
        f"Samples: {len(samples)}",
        f"Sanity mode: {sanity}",
        f"Seed: {seed}",
        "",
        "This is prompt-side restoration intervention only.",
        "Clean hidden states are available for prompt tokens only.",
        "",
        "-" * 60,
        "Restoration table (clean states into composed model)",
        "-" * 60,
        f"{'Condition':<25} {'Accuracy':>8} {'dNoPatch':>10} {'dClean':>10}",
    ]
    for r in restoration_table:
        lines.append(
            f"{r['condition']:<25} {r['accuracy']:>8.4f} "
            f"{r['delta_from_no_patch']:>+10.4f} {r['delta_from_clean_baseline']:>+10.4f}"
        )
    lines += [
        "",
        "-" * 60,
        "Reverse corruption table (corrupt states into recipient)",
        "-" * 60,
        f"{'Condition':<25} {'Accuracy':>8} {'dClean':>10}",
    ]
    for r in corruption_table:
        lines.append(
            f"{r['condition']:<25} {r['accuracy']:>8.4f} "
            f"{r['delta_from_clean_baseline']:>+10.4f}"
        )
    lines += [
        "",
        "-" * 60,
        "Interpretation",
        "-" * 60,
        comparative_sentence,
        "",
        "Note: These results are intervention-based evidence under the "
        "prompt-side constraint described above. They do NOT by themselves "
        "establish a complete mechanism; scientific claims remain conservative.",
        "",
    ]
    summary_text = "\n".join(lines)
    print(summary_text)

    # (17) Record state_dict hash AFTER all inference passes.
    sd_sha_after = _state_dict_sha256(composed)
    summary["compose_meta"]["state_dict_sha256_after"] = sd_sha_after
    if sd_sha_after != sd_sha_before:
        raise RuntimeError(
            "Composed model state_dict SHA-256 changed across the run "
            f"(before={sd_sha_before[:12]}..., after={sd_sha_after[:12]}...). "
            "In-place weight mutation is forbidden."
        )

    # (18) Write JSON + TXT.
    with open(os.path.join(run_dir, "phase_b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "phase_b_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    # (19) Conservative-wording gate (spec §11.5). Fail hard on any violation.
    wording_artifacts = [
        os.path.join(run_dir, "phase_b_summary.txt"),
        os.path.join(run_dir, "phase_b_summary.json"),
        os.path.join(run_dir, "restoration_table.csv"),
        os.path.join(run_dir, "corruption_table.csv"),
    ]
    wording_violations = check_artifacts_for_forbidden(wording_artifacts)
    if wording_violations:
        print("\nFATAL: Conservative-wording gate FAILED:")
        for v in wording_violations:
            print(f"  {v}")
        raise RuntimeError(
            "Phase B FAILED: forbidden phrases found in summary artifacts."
        )

    # (20) Eval-sanity checks (real, not vacuous).
    print(f"\n{'=' * 60}\nPHASE B SANITY CHECKS\n{'=' * 60}")
    checks: List[Tuple[str, bool]] = []

    # no_patch present
    checks.append(("no_patch results exist", "no_patch" in restoration_results))

    # No NaN in any accuracy / delta cell.
    def _no_nan(rows: List[Dict]) -> bool:
        for r in rows:
            for k, v in r.items():
                if isinstance(v, float) and v != v:
                    return False
        return True

    checks.append(("No NaN in restoration table", _no_nan(restoration_table)))
    checks.append(("No NaN in corruption table", _no_nan(corruption_table)))
    checks.append(("clean_baseline_accuracy is finite",
                   clean_baseline_acc == clean_baseline_acc))
    checks.append((f"t_fixed == {t_fixed} in summary", summary["t_fixed"] == t_fixed))
    checks.append(("Conservative-wording gate clean", len(wording_violations) == 0))
    checks.append(("state_dict SHA-256 stable across run",
                   sd_sha_before == sd_sha_after))

    # Cross-phase accuracy check — spec §11.7 acceptance criterion.
    # In sanity mode, the prior Phase A summary may legitimately be absent
    # (dev-box bootstrap); we allow it there. In a full run, a missing Phase A
    # summary is an explicit FAIL: we refuse to silently pass an acceptance
    # criterion by absence.
    if cross_check_passed is None:
        if sanity:
            checks.append((
                "Phase A cross-check skipped (sanity mode, no prior summary)",
                True,
            ))
        else:
            checks.append((
                f"Phase A cross-check FAILED: no phase_a_summary.json found under "
                f"{_phase_a_outputs_dir()} (spec §11.7 requires a prior Phase A run)",
                False,
            ))
    else:
        checks.append((
            f"Phase A cross-check within |Δ| ≤ {PHASE_A_CROSS_CHECK_TOL}",
            cross_check_passed,
        ))

    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")

    if not all(ok for _, ok in checks):
        raise RuntimeError("Phase B FAILED sanity checks (see PASS/FAIL list above).")

    print(f"\nPhase B PASSED all sanity checks.")
    print(f"Outputs saved to: {run_dir}")

    # Cleanup
    del composed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_dir


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B: Restoration intervention")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to Phase B config YAML (required).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global seed for all RNGs and CLI traceability (default: 42).",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Human-readable label for this run (logged in summary JSON).",
    )
    parser.add_argument(
        "--sanity", action="store_true",
        help="Sanity mode: 5 samples x {no_patch, patch_recovery_full, "
             "clean_no_patch, corrupt_recovery_full}.",
    )
    args = parser.parse_args()
    run_phase_b(
        config_path=args.config,
        sanity=args.sanity,
        seed=args.seed,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

# Note: the authoritative FORBIDDEN_PHRASES list lives in
# ``stage1/utils/wording.py`` and is invariant-tested in
# ``stage1/tests/test_phase_b_patcher.py::test_forbidden_phrases_gate``.
