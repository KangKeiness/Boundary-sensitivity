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
from stage1.utils.manifest_parity import extract_parity_block
from stage1.utils.anchor_gate import (
    ANCHOR_WORKFLOW_DOC,
    PHASE_A_CROSS_CHECK_TOL,
    evaluate_phase_b_anchor_gate,
    render_anchor_gate_diagnostic,
)
from stage1.utils.run_status import (
    RUN_STATUS_FAILED,
    RUN_STATUS_PASSED,
    RUN_STATUS_PENDING,
    write_phase_b_status_artifacts,
)
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

# Cross-phase accuracy tolerance (2/250 samples). Authoritative definition lives
# in ``stage1/utils/anchor_gate.py``; re-exported here for backward-compat with
# tests and call sites that imported it from this module.
# (PHASE_A_CROSS_CHECK_TOL is imported above.)

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
# The actual loader / parity-filter / decision logic lives in
# ``stage1/utils/anchor_gate.py`` (torch-free) so that integration-level
# regression tests can exercise it end-to-end. The thin wrappers below preserve
# the original module-level CWD-invariant resolution for callers that imported
# them by name.


def _phase_a_outputs_dir() -> str:
    """Resolve the Phase A outputs directory relative to the repo root.

    CWD-invariant per spec §11.7. Delegates to the canonical resolver in
    ``stage1.utils.anchor_gate``.
    """
    from stage1.utils.anchor_gate import default_phase_a_outputs_dir
    return default_phase_a_outputs_dir()


def _stage1_outputs_dir() -> str:
    """Resolve the Stage 1 sweep outputs dir (``stage1/outputs/run_*``)."""
    from stage1.utils.anchor_gate import default_stage1_outputs_dir
    return default_stage1_outputs_dir()


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
    # RED LIGHT P2: anchor selection filtered by manifest parity.
    # YELLOW-LIGHT v3 P1: gate decision delegated to ``utils.anchor_gate`` so
    # full integration-level regression tests can exercise it end-to-end.
    # v4 P2: include sample_regime in the parity block so anchors with a
    # different debug/full mode, sample count, or sample ordering are rejected.
    current_parity = extract_parity_block(
        config, sample_ids=[s["sample_id"] for s in samples],
    )
    gate = evaluate_phase_b_anchor_gate(
        no_patch_acc=no_patch_acc,
        clean_baseline_acc=clean_baseline_acc,
        sanity=sanity,
        current_parity=current_parity,
        tolerance=PHASE_A_CROSS_CHECK_TOL,
        phase_a_dir=_phase_a_outputs_dir(),
        stage1_dir=_stage1_outputs_dir(),
    )
    # Local aliases mirror the legacy variables so the rest of this function
    # (summary block, sanity-check labels) reads identically to before.
    phase_a_path = gate.phase_a_summary_path
    phase_a_hard_swap_acc = gate.phase_a_hard_swap_b8_accuracy
    phase_a_no_swap_acc = gate.phase_a_no_swap_accuracy
    stage1_eval_path = gate.stage1_evaluation_path
    stage1_hs_acc = gate.stage1_hard_swap_b8_accuracy
    stage1_ns_acc = gate.stage1_no_swap_accuracy
    anchor_hard_swap_acc = gate.anchor_hard_swap_b8_accuracy
    anchor_no_swap_acc = gate.anchor_no_swap_accuracy
    anchor_hard_swap_source = gate.anchor_hard_swap_source
    anchor_no_swap_source = gate.anchor_no_swap_source
    cross_check_passed = gate.passed
    cross_check_missing_anchors = list(gate.missing_anchors)
    cross_check_failed_anchors = list(gate.failed_anchors)
    # Re-load the Phase A summary dict for the dataset-mirror block below.
    # (We don't reuse the loader; the path is enough.)
    phase_a: Optional[Dict[str, Any]] = None
    if phase_a_path is not None:
        try:
            with open(phase_a_path, encoding="utf-8") as _f:
                phase_a = json.load(_f)
        except Exception:
            phase_a = None

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
        # RED LIGHT P4: record generation config for decode-budget traceability.
        "generation_config": gen_config,
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
        "phase_a_cross_check": gate.to_summary_dict(
            phase_a_outputs_dir=_phase_a_outputs_dir(),
        ),
        "compose_meta": compose_meta,
        "dataset": dataset_block,
        "environment": env_block,
        # RED LIGHT P2: embed parity block for downstream manifest checks.
        "parity": current_parity,
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

    # YELLOW-LIGHT v3 P3: explicit run-status field. Initialised as "pending"
    # and updated to "passed" / "failed" by every ``_persist_summary`` call.
    summary["run_status"] = RUN_STATUS_PENDING
    summary["failure_reason"] = None

    # (16) Human-readable summary TXT body (contains t=20 header and exactly one
    #      comparative sentence). The leading status banner is prepended by
    #      ``write_phase_b_status_artifacts`` on each persist, so a failed run
    #      cannot be mistaken for a passed one by a human glancing at the file.
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
    body_lines = list(lines)  # banner-free TXT body, used by every persist call
    summary_text = "\n".join(body_lines)
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

    # YELLOW-LIGHT v3 P3: every persist of phase_b_summary.{json,txt} +
    # RUN_STATUS.txt goes through the centralised helper. This is the ONLY
    # path that writes those files — there is no flow where a failed run
    # can leave status="pending"/"passed" artifacts on disk.
    def _persist_summary(status: str, failure_reason: Optional[str] = None) -> None:
        write_phase_b_status_artifacts(
            run_dir, summary, body_lines, status,
            failure_reason=failure_reason,
        )

    # (18) Initial write with status="pending" so the wording gate can scan files.
    _persist_summary(RUN_STATUS_PENDING)

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
        wording_reason = (
            "wording_violations: " + " | ".join(wording_violations[:5])
            + (" (+more)" if len(wording_violations) > 5 else "")
        )
        _persist_summary(RUN_STATUS_FAILED, failure_reason=wording_reason)
        raise RuntimeError(
            "Phase B FAILED: forbidden phrases found in summary artifacts. "
            f"Artifacts updated with run_status='failed'. See {run_dir}/RUN_STATUS.txt."
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
    # RED LIGHT Fix A: full runs require BOTH anchors present and passing.
    if cross_check_passed is None:
        if sanity:
            checks.append((
                "Phase A cross-check skipped (sanity mode, no prior anchors required)",
                True,
            ))
        else:
            # Full run with missing anchors: hard fail.
            missing_str = ", ".join(cross_check_missing_anchors)
            checks.append((
                f"Cross-check FAILED: missing anchor(s) [{missing_str}]. "
                f"Full runs require BOTH hard_swap_b8 and no_swap anchors from "
                f"a parity-compatible prior run. Searched: "
                f"{_phase_a_outputs_dir()}, {_stage1_outputs_dir()}",
                False,
            ))
    elif cross_check_passed:
        checks.append((
            f"Cross-check PASSED: BOTH anchors within |Δ| ≤ {PHASE_A_CROSS_CHECK_TOL} "
            f"(hard_swap_b8 from {anchor_hard_swap_source}, "
            f"no_swap from {anchor_no_swap_source})",
            True,
        ))
    else:
        # Both present but at least one failed tolerance.
        failed_str = "; ".join(cross_check_failed_anchors)
        checks.append((
            f"Cross-check FAILED: {failed_str}",
            False,
        ))

    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")

    failed_labels = [label for label, ok in checks if not ok]
    if failed_labels:
        # YELLOW-LIGHT v3 P3: rewrite artifacts with explicit failure status
        # BEFORE raising, so a downstream user inspecting the run dir cannot
        # mistake a failed run for a valid one. Includes the actionable anchor
        # diagnostic when the failure was a cross-check miss.
        diag = render_anchor_gate_diagnostic(
            gate,
            phase_a_dir=_phase_a_outputs_dir(),
            stage1_dir=_stage1_outputs_dir(),
            workflow_doc=ANCHOR_WORKFLOW_DOC,
        )
        sanity_reason = (
            "sanity_check_failed: " + " | ".join(failed_labels[:3])
            + (f" (+{len(failed_labels) - 3} more)" if len(failed_labels) > 3 else "")
        )
        if diag:
            sanity_reason = f"{sanity_reason}. {diag}"
        _persist_summary(RUN_STATUS_FAILED, failure_reason=sanity_reason)
        raise RuntimeError(
            "Phase B FAILED sanity checks (see PASS/FAIL list above). "
            f"Artifacts updated with run_status='failed'. See {run_dir}/RUN_STATUS.txt."
        )

    # All gates passed — finalise artifacts with status="passed".
    _persist_summary(RUN_STATUS_PASSED)
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
    parser = argparse.ArgumentParser(
        description=(
            "Phase B: Restoration intervention. "
            "Full runs hard-fail unless BOTH the hard_swap_b8 anchor (from a "
            "parity-compatible Stage 1 run) AND the no_swap anchor (from a "
            "parity-compatible Phase A or Stage 1 run) are available within "
            f"tolerance {PHASE_A_CROSS_CHECK_TOL}. See {ANCHOR_WORKFLOW_DOC} "
            "for the parity contract and the recipe to generate compatible "
            "anchors."
        ),
        epilog=(
            "Anchor workflow: " + ANCHOR_WORKFLOW_DOC + "\n"
            "Common cause of 'missing anchor(s)' failure: Stage 1 was run with "
            "stage1_main.yaml (max_new_tokens=256) instead of stage2_confound.yaml "
            "(max_new_tokens=512), so the parity filter rejects it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to Phase B config YAML (required). Must match the Stage 1 / "
             "Phase A anchors' configs in models, dataset, generation, and "
             "hidden-state pooling — see notes/anchors_workflow.md.",
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
        help="Sanity mode (development only): 5 samples × "
             "{no_patch, patch_recovery_full, clean_no_patch, "
             "corrupt_recovery_full}. Relaxes the anchor gate to best-effort. "
             "Never use for review or final results.",
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
