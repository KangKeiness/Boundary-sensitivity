"""Phase C — Mediation-style decomposition (analysis-only).

This entrypoint consumes a completed Phase B run directory's per-condition
JSONLs and emits a three-artifact decomposition under
``stage1/outputs/phase_c/run_<timestamp>/``:

- ``phase_c_decomposition_table.csv``
- ``phase_c_summary.json``
- ``phase_c_summary.txt``

IMPORTANT METHODOLOGICAL CONSTRAINT:
    Mediation analysis here decomposes accuracy deltas under prompt-side
    restoration intervention only. It is not a formal NIE/NDE decomposition.

See ``notes/specs/phase_c_mediation.md`` for the authoritative spec.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import logging
import os
import pathlib
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stage1.analysis.mediation import (
    CLAIM_ELIGIBLE_CONDITIONS,
    EPSILON_DENOM,
    compute_decomposition_table,
)
from stage1.utils.wording import (
    FORBIDDEN_PHRASES_PHASE_C,
    check_artifacts_for_forbidden,
)

logger = logging.getLogger(__name__)


# Verbatim mandated caveat per spec §scope note. Byte-exact; do not reword.
MANDATED_CAVEAT: str = (
    "Mediation analysis here decomposes accuracy deltas under prompt-side "
    "restoration intervention only. It is not a formal NIE/NDE decomposition."
)

# Header line per spec §11.10.
PHASE_C_TXT_HEADER: str = (
    "Phase C \u2014 mediation-style decomposition of prompt-side restoration "
    "intervention (not a formal NIE/NDE decomposition)"
)


# ─── Path helpers ────────────────────────────────────────────────────────────


def _phase_b_outputs_dir() -> str:
    """Resolve the Phase B outputs directory relative to this file (CWD-invariant)."""
    return str(
        pathlib.Path(__file__).resolve().parent / "outputs" / "phase_b"
    )


def _phase_c_outputs_dir() -> str:
    return str(
        pathlib.Path(__file__).resolve().parent / "outputs" / "phase_c"
    )


def _resolve_phase_b_run(explicit: Optional[str]) -> str:
    """Return an absolute path to the Phase B run dir to consume.

    If ``explicit`` is given, use it (after absolute-path resolution). Else,
    pick the lexicographically-latest ``run_*`` directory under the Phase B
    outputs folder. Raises ``RuntimeError`` if none exist.

    NOTE: this only resolves the path. Upstream validity gating
    (``run_status == "passed"``) is a separate explicit check applied by
    ``_assert_phase_b_passed`` so that the gate decision is auditable in
    isolation and tested without the full Phase C pipeline.
    """
    if explicit:
        resolved = str(pathlib.Path(explicit).resolve())
        if not os.path.isdir(resolved):
            raise RuntimeError(f"--phase-b-run does not exist: {resolved}")
        return resolved
    pattern = os.path.join(_phase_b_outputs_dir(), "run_*")
    candidates = sorted(glob.glob(pattern), reverse=True)
    for c in candidates:
        if os.path.isdir(c):
            return str(pathlib.Path(c).resolve())
    raise RuntimeError(
        f"No Phase B runs found under {_phase_b_outputs_dir()}; "
        "pass --phase-b-run explicitly."
    )


# ─── Upstream-validity gate (v4 PRIORITY 1) ─────────────────────────────────
#
# Phase C must NOT silently consume a failed Phase B run. We hard-fail unless
# ``phase_b_summary.run_status == "passed"`` (or the explicit override flag is
# set). The gate is its own helper so it is unit-testable without invoking the
# rest of the Phase C pipeline.


_VALID_UPSTREAM_STATUS: str = "passed"


class FailedUpstreamError(RuntimeError):
    """Raised when Phase C is asked to consume a non-passed Phase B run."""


def _assert_phase_b_passed(
    phase_b_run_dir: str,
    *,
    allow_failed_upstream: bool = False,
) -> Dict[str, Any]:
    """Validate that the Phase B run at ``phase_b_run_dir`` is safe to consume.

    Returns the loaded ``phase_b_summary.json`` dict on success.

    Hard-fails by default (``FailedUpstreamError``) when:
      - ``phase_b_summary.json`` is missing
      - ``run_status`` field is absent (legacy Phase B run pre-dates v3 P3 —
        unverifiable, so refuse)
      - ``run_status != "passed"``

    The single escape hatch is ``allow_failed_upstream=True`` (CLI:
    ``--allow-failed-upstream``). When set, the function logs a loud warning
    and returns the summary anyway. Callers that pass this flag are
    responsible for embedding the override into Phase C's own summary so the
    decision is auditable downstream.
    """
    summary_path = os.path.join(phase_b_run_dir, "phase_b_summary.json")
    if not os.path.exists(summary_path):
        raise FailedUpstreamError(
            f"Refusing to consume Phase B run {phase_b_run_dir!r}: "
            f"phase_b_summary.json not found. The run is incomplete."
        )
    try:
        with open(summary_path, encoding="utf-8") as f:
            pb_summary = json.load(f)
    except Exception as exc:
        raise FailedUpstreamError(
            f"Refusing to consume Phase B run {phase_b_run_dir!r}: "
            f"phase_b_summary.json is unreadable ({exc!r})."
        ) from exc

    status = pb_summary.get("run_status")
    failure_reason = pb_summary.get("failure_reason")

    if status is None:
        # Legacy Phase B run pre-dates the v3 P3 run-status field.
        # Cannot prove it is valid — refuse by default.
        msg = (
            f"Refusing to consume Phase B run {phase_b_run_dir!r}: "
            f"phase_b_summary.json has no `run_status` field. The run "
            f"pre-dates the v3 P3 run-status convention; re-run Phase B "
            f"so the gate result is recorded. "
            f"(Override: --allow-failed-upstream — debugging only.)"
        )
        if not allow_failed_upstream:
            raise FailedUpstreamError(msg)
        logger.warning("UPSTREAM-GATE OVERRIDDEN — %s", msg)
        return pb_summary

    if status != _VALID_UPSTREAM_STATUS:
        msg = (
            f"Refusing to consume Phase B run {phase_b_run_dir!r}: "
            f"upstream run_status={status!r} (expected {_VALID_UPSTREAM_STATUS!r})"
            + (f", failure_reason={failure_reason!r}" if failure_reason else "")
            + ". A failed Phase B run produces methodologically invalid "
            "decomposition. Re-run Phase B until run_status='passed'. "
            "(Override: --allow-failed-upstream — debugging only.)"
        )
        if not allow_failed_upstream:
            raise FailedUpstreamError(msg)
        logger.warning("UPSTREAM-GATE OVERRIDDEN — %s", msg)
        return pb_summary

    return pb_summary


def _create_run_dir(run_name: Optional[str]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    leaf = f"run_{ts}" if not run_name else f"run_{ts}_{run_name}"
    base = _phase_c_outputs_dir()
    os.makedirs(base, exist_ok=True)
    run_dir = os.path.join(base, leaf)
    os.makedirs(run_dir, exist_ok=True)
    return str(pathlib.Path(run_dir).resolve())


def _git_sha() -> str:
    try:
        repo_root = str(pathlib.Path(__file__).resolve().parents[1])
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Artifact writers ────────────────────────────────────────────────────────


_CSV_COLUMNS: Tuple[str, ...] = (
    "condition",
    "accuracy",
    "restoration_effect",
    "restoration_effect_ci_lo",
    "restoration_effect_ci_hi",
    "residual_effect",
    "residual_effect_ci_lo",
    "residual_effect_ci_hi",
    "restoration_proportion",
    "restoration_proportion_ci_lo",
    "restoration_proportion_ci_hi",
    "restoration_proportion_ci_reason",
    "total_effect",
    # RED LIGHT P5: alias_NIE/NDE/TE/MP columns REMOVED to eliminate formal
    # mediation terminology risk. Use only conservative terms:
    # restoration_effect, residual_effect, restoration_proportion.
    "n_aligned",
    "is_best_condition",
    "methodology",
)

_METHODOLOGY_TAG: str = (
    "prompt-side restoration intervention (not a formal NIE/NDE decomposition)"
)


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if not np.isfinite(x):
            return ""
        return f"{x:.6f}"
    return str(x)


def _write_decomposition_csv(
    path: str,
    table: Dict[str, Any],
) -> None:
    best = table["best_condition"]
    residual = table["residual"]
    proportion = table["proportion"]
    # Rows: include no_patch and every patched condition, excluding the
    # no_patch-audit row's restoration_effect/proportion columns (those apply
    # only to the best-condition row per spec §11.2).
    # Exclude the trivial no_patch row per spec §11.2 which counts "number of
    # restoration conditions present": in sanity (2 conditions), full (5-6).
    # The no_patch row is still informative — keep it so n_aligned is visible.
    body_rows: List[Dict[str, str]] = []
    for r in table["rows"]:
        is_best = (r["condition"] == best)
        row: Dict[str, str] = {k: "" for k in _CSV_COLUMNS}
        row["condition"] = r["condition"]
        row["accuracy"] = _fmt(r.get("accuracy"))
        # Per-condition restoration_effect (NIE).
        row["restoration_effect"] = _fmt(r.get("restoration_effect"))
        row["restoration_effect_ci_lo"] = _fmt(r.get("restoration_effect_ci_lo"))
        row["restoration_effect_ci_hi"] = _fmt(r.get("restoration_effect_ci_hi"))
        # Per-condition residual_effect (NDE) + restoration_proportion (MP).
        # mediation.compute_decomposition_table populates these for every
        # patched condition (spec §11.2 updated — reviewer Phase C 4a).
        row["residual_effect"] = _fmt(r.get("residual_effect"))
        row["residual_effect_ci_lo"] = _fmt(r.get("residual_effect_ci_lo"))
        row["residual_effect_ci_hi"] = _fmt(r.get("residual_effect_ci_hi"))
        row["restoration_proportion"] = _fmt(r.get("restoration_proportion"))
        row["restoration_proportion_ci_lo"] = _fmt(r.get("restoration_proportion_ci_lo"))
        row["restoration_proportion_ci_hi"] = _fmt(r.get("restoration_proportion_ci_hi"))
        row["restoration_proportion_ci_reason"] = _fmt(r.get("restoration_proportion_ci_reason"))
        row["total_effect"] = _fmt(r.get("total_effect"))
        # Best-condition block (from paired-bootstrap on best vs clean/no_patch)
        # overrides the per-condition row for the best pick — keeps the canonical
        # "best-condition" numbers identical to summary.json.
        if is_best:
            row["residual_effect"] = _fmt(residual["point"])
            row["residual_effect_ci_lo"] = _fmt(residual["ci_lo"])
            row["residual_effect_ci_hi"] = _fmt(residual["ci_hi"])
            row["restoration_proportion"] = _fmt(proportion["point"])
            row["restoration_proportion_ci_lo"] = _fmt(proportion["ci_lo"])
            row["restoration_proportion_ci_hi"] = _fmt(proportion["ci_hi"])
        row["n_aligned"] = _fmt(r.get("n_aligned"))
        row["is_best_condition"] = "True" if is_best else "False"
        row["methodology"] = _METHODOLOGY_TAG
        body_rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_CSV_COLUMNS))
        w.writeheader()
        w.writerows(body_rows)


def _copy_upstream_provenance(
    phase_b_run_dir: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]:
    """Return (upstream_provenance_block, dataset_block, summary_sha256)."""
    summary_path = os.path.join(phase_b_run_dir, "phase_b_summary.json")
    if not os.path.exists(summary_path):
        raise RuntimeError(
            f"Phase B summary missing at {summary_path}; cannot inherit provenance."
        )
    with open(summary_path, encoding="utf-8") as f:
        phase_b_summary = json.load(f)
    sha = _sha256_of_file(summary_path)
    provenance: Dict[str, Any] = {
        "phase_b_run_path": phase_b_run_dir,
        "phase_b_summary_sha256": sha,
        "environment": phase_b_summary.get("environment", {}),
        "phase_b_git_sha": phase_b_summary.get("environment", {}).get("git_sha", "unknown"),
    }
    dataset = phase_b_summary.get("dataset")
    return provenance, dataset, sha


def _build_summary_json(
    *,
    run_dir: str,
    phase_b_run_dir: str,
    table: Dict[str, Any],
    bootstrap_n: int,
    seed: int,
    ci: float,
    run_name: Optional[str],
    sanity: bool,
) -> Dict[str, Any]:
    provenance, dataset, _sha = _copy_upstream_provenance(phase_b_run_dir)
    best = table["best_condition"]
    residual = table["residual"]
    proportion = table["proportion"]

    # Mirror CSV rows as JSON decomposition table (typed, not formatted).
    # Includes per-condition residual_effect / restoration_proportion
    # (spec §11.2 updated). RED LIGHT P5: alias columns removed.
    decomposition_table: List[Dict[str, Any]] = []
    for r in table["rows"]:
        is_best = r["condition"] == best
        # Best-condition row uses the canonical paired-bootstrap residual /
        # proportion stats from the dedicated best-vs-{clean,no_patch} pass;
        # all other rows use per-condition values from compute_decomposition_table.
        if is_best:
            residual_pt = residual["point"]
            residual_lo = residual["ci_lo"]
            residual_hi = residual["ci_hi"]
            proportion_pt = proportion["point"]
            proportion_lo = proportion["ci_lo"]
            proportion_hi = proportion["ci_hi"]
            proportion_reason = proportion["ci_reason"]
        else:
            residual_pt = r.get("residual_effect")
            residual_lo = r.get("residual_effect_ci_lo")
            residual_hi = r.get("residual_effect_ci_hi")
            proportion_pt = r.get("restoration_proportion")
            proportion_lo = r.get("restoration_proportion_ci_lo")
            proportion_hi = r.get("restoration_proportion_ci_hi")
            proportion_reason = r.get("restoration_proportion_ci_reason")
        row: Dict[str, Any] = {
            "condition": r["condition"],
            "accuracy": r.get("accuracy"),
            "restoration_effect": r.get("restoration_effect"),
            "restoration_effect_ci_lo": r.get("restoration_effect_ci_lo"),
            "restoration_effect_ci_hi": r.get("restoration_effect_ci_hi"),
            "residual_effect": residual_pt,
            "residual_effect_ci_lo": residual_lo,
            "residual_effect_ci_hi": residual_hi,
            "restoration_proportion": proportion_pt,
            "restoration_proportion_ci_lo": proportion_lo,
            "restoration_proportion_ci_hi": proportion_hi,
            "restoration_proportion_ci_reason": proportion_reason,
            "total_effect": r.get("total_effect"),
            "n_aligned": r.get("n_aligned"),
            "is_best_condition": is_best,
            "methodology": _METHODOLOGY_TAG,
        }
        decomposition_table.append(row)

    try:
        pandas_version = __import__("pandas").__version__
    except Exception:
        pandas_version = "not_installed"

    env_block: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pandas_version,
        "git_sha": _git_sha(),
        "bootstrap_seed": seed,
        "bootstrap_n": bootstrap_n,
        "bootstrap_ci": ci,
        "phase_b_run_path": phase_b_run_dir,
        "phase_b_summary_sha256": provenance["phase_b_summary_sha256"],
        # Tolerance used by _cross_check_accuracies against phase_b_summary.json.
        # RED LIGHT Fix D: tightened from 2e-3 to 5e-5. Phase B rounds to 4
        # decimals; max rounding error is 5e-5. Anything beyond that is a real
        # mismatch.
        "acc_cross_check_tolerance": 5e-5,
    }

    summary: Dict[str, Any] = {
        "phase": "C",
        "description": (
            "Mediation-style decomposition of Phase B accuracy deltas under "
            "prompt-side restoration intervention."
        ),
        "caveat": MANDATED_CAVEAT,
        "methodology": _METHODOLOGY_TAG,
        "sanity_mode": sanity,
        "run_name": run_name,
        "seed": seed,
        "decomposition_table": decomposition_table,
        "best_condition": best,
        "residual": {
            "point": residual["point"],
            "ci_lo": residual["ci_lo"],
            "ci_hi": residual["ci_hi"],
            "n_aligned": residual["n_aligned"],
        },
        "proportion": {
            "point": proportion["point"],
            "ci_lo": proportion["ci_lo"],
            "ci_hi": proportion["ci_hi"],
            "n_aligned": proportion["n_aligned"],
            "denom_point": proportion["denom_point"],
            "n_resamples_dropped_denominator":
                proportion["n_resamples_dropped_denominator"],
            "ci_reason": proportion["ci_reason"],
            "epsilon_denom": EPSILON_DENOM,
        },
        "acc_no_patch": table["acc_no_patch"],
        "acc_clean_no_patch": table["acc_clean_no_patch"],
        "sample_pairing": table["sample_pairing"],
        "bootstrap": {"n": bootstrap_n, "seed": seed, "ci": ci},
        "environment": env_block,
        "upstream_provenance": provenance,
        "dataset": dataset,
        "forbidden_phrases_gate": [],  # filled after gate runs
        "claim_eligible_conditions": list(CLAIM_ELIGIBLE_CONDITIONS),
        # RED LIGHT P6: post-selection caution.
        "best_condition_selection_note": (
            "Best condition selected via argmax over restoration_effect among "
            "claim-eligible conditions. This is a descriptive / exploratory "
            "pick, NOT a confirmatory winner inference. The CI reflects "
            "bootstrap variability of the selected condition, NOT a "
            "multiplicity-adjusted comparison."
        ),
    }
    return summary


def _write_summary_txt(path: str, summary: Dict[str, Any]) -> None:
    best = summary["best_condition"]
    residual = summary["residual"]
    proportion = summary["proportion"]
    lines: List[str] = [
        "=" * 72,
        PHASE_C_TXT_HEADER,
        "=" * 72,
        "",
        MANDATED_CAVEAT,
        "",
        f"Best restoration condition: {best}",
        f"acc(clean_no_patch)   = {summary['acc_clean_no_patch']:.6f}",
        f"acc(no_patch)         = {summary['acc_no_patch']:.6f}",
        "",
        "-" * 72,
        "Decomposition table",
        "-" * 72,
        f"{'Condition':<28} {'rEff':>10} {'rEff_lo':>10} {'rEff_hi':>10} {'n':>5}",
    ]
    for r in summary["decomposition_table"]:
        def _f(x: Any) -> str:
            return f"{x:10.6f}" if isinstance(x, (int, float)) and x is not None \
                else f"{'':>10}"
        lines.append(
            f"{r['condition']:<28} "
            f"{_f(r['restoration_effect'])} "
            f"{_f(r['restoration_effect_ci_lo'])} "
            f"{_f(r['restoration_effect_ci_hi'])} "
            f"{r['n_aligned']:>5}"
        )
    lines += [
        "",
        "-" * 72,
        f"Residual for best condition ({best}):",
        "-" * 72,
        f"  point   = {residual['point']:.6f}",
        f"  95% CI  = [{residual['ci_lo']:.6f}, {residual['ci_hi']:.6f}]",
        f"  n       = {residual['n_aligned']}",
        "",
        "-" * 72,
        f"Proportion for best condition ({best}):",
        "-" * 72,
    ]
    if proportion["point"] is None:
        lines.append(
            f"  point   = null (reason: {proportion['ci_reason']})"
        )
    else:
        lines.append(f"  point   = {proportion['point']:.6f}")
    if proportion["ci_lo"] is None or proportion["ci_hi"] is None:
        lines.append(
            f"  95% CI  = null (reason: {proportion['ci_reason']})"
        )
    else:
        lines.append(
            f"  95% CI  = [{proportion['ci_lo']:.6f}, {proportion['ci_hi']:.6f}]"
        )
    lines += [
        f"  denom   = {proportion['denom_point']:.6f}",
        f"  epsilon = {proportion['epsilon_denom']:.4f}",
        "",
        "-" * 72,
        "Interpretation note",
        "-" * 72,
        "Decomposition reports descriptive accuracy deltas under prompt-side "
        "restoration intervention only. No formal identification of direct or "
        "indirect pathways is attempted; causal claims remain outside scope.",
        "",
        "Post-selection note (RED LIGHT P6): the 'best condition' is selected "
        "via argmax over restoration_effect among claim-eligible conditions. "
        "This is a descriptive / exploratory pick, NOT a confirmatory winner "
        "inference. The associated CI reflects bootstrap variability of the "
        "selected condition's delta, NOT a multiplicity-adjusted comparison "
        "across all conditions. All conditions are reported above; interpret "
        "the best-condition row as a summary convenience, not a significance "
        "claim.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─── Main entrypoint ─────────────────────────────────────────────────────────


def run_phase_c(
    phase_b_run: Optional[str] = None,
    *,
    sanity: bool = False,
    bootstrap_n: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
    run_name: Optional[str] = None,
    allow_failed_upstream: bool = False,
) -> str:
    """Run Phase C mediation-style decomposition analysis.

    Returns:
        Absolute path to the Phase C run directory.

    v4 P1 — strict upstream gate:
        Default behavior hard-fails (FailedUpstreamError) unless the resolved
        Phase B run reports ``run_status == "passed"`` in its
        ``phase_b_summary.json``. Set ``allow_failed_upstream=True`` (CLI
        ``--allow-failed-upstream``) only for explicit debugging — the
        override is logged and embedded into the Phase C summary.
    """
    # Seed numpy (analysis-only; no torch RNG touched here).
    np.random.seed(seed)

    phase_b_run_dir = _resolve_phase_b_run(phase_b_run)
    # v4 P1: hard-fail on non-passed upstream BEFORE creating any output dir or
    # running expensive analysis. The gate raises if the upstream is invalid.
    pb_summary_for_gate = _assert_phase_b_passed(
        phase_b_run_dir, allow_failed_upstream=allow_failed_upstream,
    )
    upstream_status = pb_summary_for_gate.get("run_status")

    run_dir = _create_run_dir(run_name)

    print("Phase C \u2014 mediation-style decomposition (prompt-side only)")
    print(f"  Phase B run : {phase_b_run_dir}")
    print(f"  Upstream    : run_status={upstream_status!r}"
          + (" [OVERRIDE: --allow-failed-upstream]"
             if allow_failed_upstream and upstream_status != _VALID_UPSTREAM_STATUS
             else ""))
    print(f"  Sanity mode : {sanity}")
    print(f"  Seed        : {seed}")
    print(f"  Bootstrap N : {bootstrap_n}")
    print(f"  Run dir     : {run_dir}")

    if sanity:
        required = [
            "results_clean_no_patch.jsonl",
            "results_restoration_no_patch.jsonl",
            "results_restoration_patch_recovery_full.jsonl",
        ]
        missing = [
            n for n in required
            if not os.path.exists(os.path.join(phase_b_run_dir, n))
        ]
        if missing:
            raise RuntimeError(
                f"--sanity requires these JSONLs in {phase_b_run_dir}: {missing}"
            )

    table = compute_decomposition_table(
        phase_b_run_dir,
        bootstrap_n=bootstrap_n,
        seed=seed,
        ci=ci,
        # RED LIGHT Fix C: in non-sanity mode, require exact sample-ID equality.
        strict_sample_ids=(not sanity),
    )

    csv_path = os.path.join(run_dir, "phase_c_decomposition_table.csv")
    json_path = os.path.join(run_dir, "phase_c_summary.json")
    txt_path = os.path.join(run_dir, "phase_c_summary.txt")

    _write_decomposition_csv(csv_path, table)

    summary = _build_summary_json(
        run_dir=run_dir,
        phase_b_run_dir=phase_b_run_dir,
        table=table,
        bootstrap_n=bootstrap_n,
        seed=seed,
        ci=ci,
        run_name=run_name,
        sanity=sanity,
    )

    # v4 P1: record upstream-gate decision in the Phase C summary so the
    # provenance is auditable. The override path is loud and explicit.
    summary["upstream_gate"] = {
        "phase_b_run_status": upstream_status,
        "phase_b_failure_reason": pb_summary_for_gate.get("failure_reason"),
        "allow_failed_upstream": bool(allow_failed_upstream),
        "passed_default_gate": (upstream_status == _VALID_UPSTREAM_STATUS),
    }

    # Write the summary JSON first WITHOUT the gate result so the gate can
    # read it; then re-write with the gate populated.
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _write_summary_txt(txt_path, summary)

    # Conservative-wording gate (Phase C phrases).
    violations = check_artifacts_for_forbidden(
        [txt_path, json_path, csv_path],
        phrases=FORBIDDEN_PHRASES_PHASE_C,
    )
    summary["forbidden_phrases_gate"] = violations
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Re-run gate on the rewritten JSON (which now lists the gate result,
    # empty on success). If violations, raise hard.
    violations_after = check_artifacts_for_forbidden(
        [txt_path, json_path, csv_path],
        phrases=FORBIDDEN_PHRASES_PHASE_C,
    )
    if violations_after:
        for v in violations_after:
            print(f"  FORBIDDEN: {v}")
        raise RuntimeError("Phase C FAILED: forbidden phrases found in artifacts.")

    # Eval-sanity cross-checks.
    best_condition = summary["best_condition"]
    if best_condition not in CLAIM_ELIGIBLE_CONDITIONS:
        raise RuntimeError(
            f"Best condition {best_condition!r} not in claim-eligible set "
            f"{CLAIM_ELIGIBLE_CONDITIONS}."
        )
    if MANDATED_CAVEAT not in summary["caveat"]:
        raise RuntimeError("Mandated caveat missing from summary.caveat")
    with open(txt_path, encoding="utf-8") as f:
        txt_body = f.read()
    if MANDATED_CAVEAT not in txt_body:
        raise RuntimeError("Mandated caveat missing from summary.txt")

    # Cross-check: acc_* values within 1e-6 of phase_b_summary numbers (if present).
    _cross_check_accuracies(summary, phase_b_run_dir)

    print(f"\nPhase C PASSED. Artifacts in: {run_dir}")
    return run_dir


def _cross_check_accuracies(summary: Dict[str, Any], phase_b_run_dir: str) -> None:
    """Reconcile acc_no_patch / acc_clean_no_patch vs. Phase B summary.

    RED LIGHT Fix D: prefer exact numerator/denominator count reconciliation.
    Phase C computes accuracies from the same JSONLs that Phase B wrote, so
    on the same sample set the counts MUST match exactly. If Phase B summary
    records 4-decimal-rounded values, use a tight tolerance (5e-5) rather than
    the previous overly-loose 2e-3.
    """
    pb_path = os.path.join(phase_b_run_dir, "phase_b_summary.json")
    if not os.path.exists(pb_path):
        return
    with open(pb_path, encoding="utf-8") as f:
        pb = json.load(f)
    pb_no_patch = pb.get("no_patch_accuracy")
    pb_clean = pb.get("clean_baseline_accuracy")
    if pb_no_patch is None and pb_clean is None:
        raise RuntimeError(
            "phase_b_summary.json is missing BOTH 'no_patch_accuracy' and "
            "'clean_baseline_accuracy'. Phase B schema has drifted; "
            "update Phase C cross-check to the new key names."
        )
    # RED LIGHT Fix D: tightened tolerance.
    # Phase B rounds accuracies to 4 decimals (e.g., 0.3200). Phase C recomputes
    # from the same JSONL files, so on identical sample sets the true accuracy
    # is identical. The only source of difference is Phase B's rounding.
    # A 4-decimal round can shift by at most 5e-5. Use that as the tight bound.
    # Previous value was 2e-3 which could mask genuine mismatches.
    tol = 5e-5
    if pb_no_patch is not None:
        diff = abs(summary["acc_no_patch"] - float(pb_no_patch))
        if diff > tol:
            # Provide detailed diagnostics.
            raise RuntimeError(
                f"acc_no_patch reconciliation FAILED vs phase_b_summary.\n"
                f"  Phase C computed: {summary['acc_no_patch']:.8f}\n"
                f"  Phase B recorded: {pb_no_patch}\n"
                f"  |Δ| = {diff:.8f} > tolerance {tol}\n"
                f"  This indicates either a sample-set mismatch or a computation "
                f"  bug. Phase C must compute on the exact same samples as Phase B."
            )
    if pb_clean is not None:
        diff = abs(summary["acc_clean_no_patch"] - float(pb_clean))
        if diff > tol:
            raise RuntimeError(
                f"acc_clean_no_patch reconciliation FAILED vs phase_b_summary.\n"
                f"  Phase C computed: {summary['acc_clean_no_patch']:.8f}\n"
                f"  Phase B recorded: {pb_clean}\n"
                f"  |Δ| = {diff:.8f} > tolerance {tol}\n"
                f"  This indicates either a sample-set mismatch or a computation "
                f"  bug. Phase C must compute on the exact same samples as Phase B."
            )


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase C: mediation-style decomposition of Phase B deltas.",
    )
    parser.add_argument(
        "--phase-b-run", type=str, default=None,
        help="Absolute path to a Phase B run directory. "
             "Defaults to the latest stage1/outputs/phase_b/run_* directory.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Bootstrap seed (default 0 for cross-phase determinism).",
    )
    parser.add_argument(
        "--bootstrap-n", type=int, default=1000,
        help="Number of bootstrap resamples (default 1000).",
    )
    parser.add_argument(
        "--ci", type=float, default=0.95, help="Central CI level (default 0.95).",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Human-readable label appended to the run-dir leaf.",
    )
    parser.add_argument(
        "--sanity", action="store_true",
        help="Sanity mode: require only clean_no_patch, restoration_no_patch, "
             "and restoration_patch_recovery_full.",
    )
    parser.add_argument(
        "--allow-failed-upstream", action="store_true",
        help=(
            "DEBUG-ONLY override: consume the Phase B run even when its "
            "phase_b_summary.run_status != 'passed'. Default behavior hard-fails "
            "to prevent invalid upstream evidence from propagating. The override "
            "is recorded in the Phase C summary's `upstream_gate` block."
        ),
    )
    # --config is not used by Phase C (analysis-only, no model/dataset config);
    # accept-and-ignore for cross-phase CLI consistency.
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Optional config path (analysis-only Phase C does not consume a "
            "config; provided for cross-phase CLI parity). Logged when set."
        ),
    )
    args = parser.parse_args()

    if args.config is not None:
        print(f"  Config (advisory): {args.config}")

    run_phase_c(
        phase_b_run=args.phase_b_run,
        sanity=args.sanity,
        bootstrap_n=args.bootstrap_n,
        seed=args.seed,
        ci=args.ci,
        run_name=args.run_name,
        allow_failed_upstream=args.allow_failed_upstream,
    )


if __name__ == "__main__":
    main()
