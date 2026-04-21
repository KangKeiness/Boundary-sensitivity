"""Phase B anchor-gate decision logic — torch-free for testability.

Decoupled from torch / transformers / numpy so that integration-level
regression tests can exercise the full gate (manifest parity filter, anchor
selection precedence, missing-anchor detection, tolerance check) end-to-end
in environments without GPU dependencies.

The gate decides whether a Phase B run's measured no_patch / clean_baseline
accuracies match prior parity-compatible Phase A and Stage 1 anchors within
``PHASE_A_CROSS_CHECK_TOL``. Full mode requires BOTH anchors; sanity mode is
best-effort.

Anchor source precedence:
    1. Phase A ``phase_a_summary.json`` (preferred for ``no_swap``).
    2. Stage 1 ``evaluation.json`` (required for ``hard_swap_b8`` because
       Phase A's confound grid does not contain hard_swap_b8 by construction).

Both candidate run directories are filtered by ``manifest_parity`` against
the current Phase B parity block before being accepted.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from stage1.utils.manifest_parity import check_manifest_parity

logger = logging.getLogger(__name__)


# Cross-phase accuracy tolerance (2/250 samples).
PHASE_A_CROSS_CHECK_TOL: float = 0.008

# Operator-facing workflow doc; failures point here so users have a concrete
# next step rather than guessing at parity rules.
ANCHOR_WORKFLOW_DOC: str = "notes/anchors_workflow.md"


def default_phase_a_outputs_dir() -> str:
    """Resolve the Phase A outputs directory relative to the repo root."""
    return str(
        pathlib.Path(__file__).resolve().parents[2]
        / "stage1" / "outputs" / "phase_a"
    )


def default_stage1_outputs_dir() -> str:
    """Resolve the Stage 1 sweep outputs directory relative to the repo root."""
    return str(
        pathlib.Path(__file__).resolve().parents[2] / "stage1" / "outputs"
    )


def _read_manifest(manifest_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, encoding="utf-8") as mf:
            return json.load(mf)
    except Exception:
        logger.warning("Cannot read manifest %s — skipping.", manifest_path)
        return None


def _parity_compatible(
    manifest_path: str,
    current_parity: Optional[Dict[str, Any]],
    *,
    run_label: str,
) -> bool:
    """Return True iff the manifest at ``manifest_path`` is parity-compatible.

    When ``current_parity`` is None, parity filtering is disabled (used only
    by callers that have already validated parity upstream). When the manifest
    file is missing, the run is rejected — manifest-less anchors cannot be
    proven compatible.
    """
    if current_parity is None:
        return True
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        logger.warning(
            "%s rejected — no readable manifest.json (P2 parity requires manifest).",
            run_label,
        )
        return False
    mismatches = check_manifest_parity(
        manifest, current_parity,
        source_path=manifest_path,
        target_desc="current Phase B config",
    )
    if mismatches:
        logger.warning(
            "%s rejected — manifest parity mismatch: %s",
            run_label, "; ".join(mismatches),
        )
        return False
    return True


def load_latest_phase_a_summary(
    phase_a_dir: str,
    current_parity: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return (summary_dict, resolved_path) for the newest parity-compatible
    Phase A run, or (None, None) if no candidate matches.
    """
    pattern = os.path.join(phase_a_dir, "run_*", "phase_a_summary.json")
    candidates = sorted(glob.glob(pattern), reverse=True)
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue
        manifest_path = os.path.join(os.path.dirname(path), "manifest.json")
        if not _parity_compatible(
            manifest_path, current_parity,
            run_label=f"Phase A run {path}",
        ):
            continue
        return summary, path
    return None, None


def phase_a_condition_accuracy(
    summary: Dict[str, Any], condition: str,
) -> Optional[float]:
    """Pull accuracy for ``condition`` from a Phase A summary dict.

    Falls back to ``baseline_accuracy`` when ``condition == "no_swap"``.
    """
    rows = summary.get("all_conditions") or []
    for row in rows:
        if row.get("condition") == condition:
            val = row.get("accuracy")
            if val is None:
                continue
            return float(val)
    if condition == "no_swap":
        val = summary.get("baseline_accuracy")
        if val is not None:
            return float(val)
    return None


def load_latest_stage1_hard_swap_b8(
    stage1_dir: str,
    current_parity: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Return (hard_swap_b8_accuracy, no_swap_accuracy, evaluation_path) for the
    newest parity-compatible Stage 1 run, or (None, None, None) on no match.
    """
    pattern = os.path.join(stage1_dir, "run_*", "evaluation.json")
    candidates = sorted(glob.glob(pattern), reverse=True)
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                eval_dict = json.load(f)
        except Exception:
            continue
        manifest_path = os.path.join(os.path.dirname(path), "manifest.json")
        if not _parity_compatible(
            manifest_path, current_parity,
            run_label=f"Stage 1 run {path}",
        ):
            continue
        accs = eval_dict.get("accuracies") or {}
        hs = accs.get("hard_swap_b8")
        ns = accs.get("no_swap")
        hs_acc = (
            float(hs["accuracy"])
            if isinstance(hs, dict) and hs.get("accuracy") is not None
            else None
        )
        ns_acc = (
            float(ns["accuracy"])
            if isinstance(ns, dict) and ns.get("accuracy") is not None
            else (
                float(eval_dict["baseline_accuracy"])
                if eval_dict.get("baseline_accuracy") is not None
                else None
            )
        )
        if hs_acc is not None:
            return hs_acc, ns_acc, path
    return None, None, None


@dataclass
class AnchorGateResult:
    """Structured result of the cross-phase anchor gate.

    ``passed``:
        - ``True``  → all required anchors present and within tolerance.
        - ``False`` → all required anchors present but at least one outside tolerance.
        - ``None``  → "missing anchors" signal.
            * Full mode: hard fail (one or both anchors absent).
            * Sanity mode: skipped (no anchors available at all — acceptable).
    """

    passed: Optional[bool]
    sanity_mode: bool
    missing_anchors: List[str] = field(default_factory=list)
    failed_anchors: List[str] = field(default_factory=list)
    anchor_hard_swap_b8_accuracy: Optional[float] = None
    anchor_no_swap_accuracy: Optional[float] = None
    anchor_hard_swap_source: Optional[str] = None
    anchor_no_swap_source: Optional[str] = None
    phase_a_summary_path: Optional[str] = None
    phase_a_hard_swap_b8_accuracy: Optional[float] = None
    phase_a_no_swap_accuracy: Optional[float] = None
    stage1_evaluation_path: Optional[str] = None
    stage1_hard_swap_b8_accuracy: Optional[float] = None
    stage1_no_swap_accuracy: Optional[float] = None
    tolerance: float = PHASE_A_CROSS_CHECK_TOL

    def to_summary_dict(
        self,
        *,
        phase_a_outputs_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the ``phase_a_cross_check`` block embedded in phase_b_summary.json."""
        return {
            "phase_a_summary_path": self.phase_a_summary_path,
            "phase_a_outputs_dir": (
                phase_a_outputs_dir or default_phase_a_outputs_dir()
            ),
            "phase_a_hard_swap_b8_accuracy": self.phase_a_hard_swap_b8_accuracy,
            "phase_a_no_swap_accuracy": self.phase_a_no_swap_accuracy,
            "stage1_evaluation_path": self.stage1_evaluation_path,
            "stage1_hard_swap_b8_accuracy": self.stage1_hard_swap_b8_accuracy,
            "stage1_no_swap_accuracy": self.stage1_no_swap_accuracy,
            "anchor_hard_swap_b8_accuracy": self.anchor_hard_swap_b8_accuracy,
            "anchor_hard_swap_source": self.anchor_hard_swap_source,
            "anchor_no_swap_accuracy": self.anchor_no_swap_accuracy,
            "anchor_no_swap_source": self.anchor_no_swap_source,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "missing_anchors": list(self.missing_anchors),
            "failed_anchors": list(self.failed_anchors),
            "note": (
                "Treatment parity (no_patch ≈ hard_swap_b8) is cross-checked "
                "against the newest Stage 1 sweep run since Phase A's confound "
                "grid does not contain hard_swap_b8 at (b=8, t=20). Anchor "
                "runs are filtered by manifest parity (model IDs, dataset, "
                "generation config including max_new_tokens). See "
                f"{ANCHOR_WORKFLOW_DOC} for the parity-compatible recipe."
            ),
        }


def evaluate_phase_b_anchor_gate(
    no_patch_acc: float,
    clean_baseline_acc: float,
    *,
    sanity: bool,
    current_parity: Optional[Dict[str, Any]] = None,
    tolerance: float = PHASE_A_CROSS_CHECK_TOL,
    phase_a_dir: Optional[str] = None,
    stage1_dir: Optional[str] = None,
) -> AnchorGateResult:
    """Look up parity-compatible anchors and apply the Phase B cross-check.

    Decision rule:
        - Full mode (``sanity=False``): BOTH ``hard_swap_b8`` and ``no_swap``
          anchors must be present (from a parity-compatible Phase A or Stage 1
          run) AND both must be within ``tolerance``. Any missing or failing
          anchor → hard fail.
        - Sanity mode: best-effort — pass with whatever anchors are available;
          ``passed=None`` if no anchors are available at all (treated by callers
          as "skipped, acceptable").

    Anchor source precedence: Phase A first, then Stage 1 fallback (Phase A's
    grid does not contain ``hard_swap_b8`` by construction; Stage 1's
    ``evaluation.json`` does).
    """
    pa_dir = phase_a_dir or default_phase_a_outputs_dir()
    s1_dir = stage1_dir or default_stage1_outputs_dir()

    phase_a, phase_a_path = load_latest_phase_a_summary(pa_dir, current_parity)
    pa_hs: Optional[float] = None
    pa_ns: Optional[float] = None
    if phase_a is not None:
        pa_hs = phase_a_condition_accuracy(phase_a, "hard_swap_b8")
        pa_ns = phase_a_condition_accuracy(phase_a, "no_swap")

    s1_hs, s1_ns, s1_path = load_latest_stage1_hard_swap_b8(s1_dir, current_parity)

    anchor_hs = pa_hs if pa_hs is not None else s1_hs
    anchor_ns = pa_ns if pa_ns is not None else s1_ns
    anchor_hs_src = (
        "phase_a" if pa_hs is not None else ("stage1" if s1_hs is not None else None)
    )
    anchor_ns_src = (
        "phase_a" if pa_ns is not None else ("stage1" if s1_ns is not None else None)
    )

    result = AnchorGateResult(
        passed=None,
        sanity_mode=sanity,
        anchor_hard_swap_b8_accuracy=anchor_hs,
        anchor_no_swap_accuracy=anchor_ns,
        anchor_hard_swap_source=anchor_hs_src,
        anchor_no_swap_source=anchor_ns_src,
        phase_a_summary_path=phase_a_path,
        phase_a_hard_swap_b8_accuracy=pa_hs,
        phase_a_no_swap_accuracy=pa_ns,
        stage1_evaluation_path=s1_path,
        stage1_hard_swap_b8_accuracy=s1_hs,
        stage1_no_swap_accuracy=s1_ns,
        tolerance=tolerance,
    )

    if not sanity:
        if anchor_hs is None:
            result.missing_anchors.append("hard_swap_b8")
        if anchor_ns is None:
            result.missing_anchors.append("no_swap")
        if result.missing_anchors:
            result.passed = None
            return result
        hs_ok = abs(no_patch_acc - anchor_hs) <= tolerance
        ns_ok = abs(clean_baseline_acc - anchor_ns) <= tolerance
        if not hs_ok:
            result.failed_anchors.append(
                f"hard_swap_b8: |no_patch({no_patch_acc:.4f}) - anchor({anchor_hs:.4f})| "
                f"= {abs(no_patch_acc - anchor_hs):.6f} > tol {tolerance}"
            )
        if not ns_ok:
            result.failed_anchors.append(
                f"no_swap: |clean_baseline({clean_baseline_acc:.4f}) - anchor({anchor_ns:.4f})| "
                f"= {abs(clean_baseline_acc - anchor_ns):.6f} > tol {tolerance}"
            )
        result.passed = (hs_ok and ns_ok)
    else:
        deltas_ok: List[bool] = []
        if anchor_hs is not None:
            deltas_ok.append(abs(no_patch_acc - anchor_hs) <= tolerance)
        if anchor_ns is not None:
            deltas_ok.append(abs(clean_baseline_acc - anchor_ns) <= tolerance)
        result.passed = all(deltas_ok) if deltas_ok else None

    return result


def render_anchor_gate_diagnostic(
    result: AnchorGateResult,
    *,
    phase_a_dir: Optional[str] = None,
    stage1_dir: Optional[str] = None,
    workflow_doc: Optional[str] = ANCHOR_WORKFLOW_DOC,
) -> str:
    """Build a human-readable, operator-actionable diagnostic for a gate failure.

    Returns the empty string when the gate passed or was skipped without
    failures.
    """
    lines: List[str] = []
    if result.missing_anchors and not result.sanity_mode:
        lines.append(
            f"Cross-check FAILED: missing anchor(s) "
            f"[{', '.join(result.missing_anchors)}]. "
            "Full Phase B runs require BOTH hard_swap_b8 and no_swap anchors "
            "from a parity-compatible prior run."
        )
        lines.append(
            f"Searched: {phase_a_dir or default_phase_a_outputs_dir()}, "
            f"{stage1_dir or default_stage1_outputs_dir()}"
        )
        lines.append(
            "Common causes: (1) no Phase A / Stage 1 run exists yet; "
            "(2) all candidate runs were rejected by manifest parity "
            "(check warnings above for the first mismatched field — the most "
            "common offender is generation.max_new_tokens differing between "
            "stage1_main.yaml (256) and stage2_confound.yaml (512))."
        )
    if result.failed_anchors:
        lines.append("Cross-check FAILED — anchor(s) outside tolerance:")
        for f in result.failed_anchors:
            lines.append(f"  - {f}")
        lines.append(
            "This means the current Phase B no_patch / clean_baseline accuracy "
            "does not reproduce a prior parity-compatible run within tolerance "
            f"{result.tolerance}. Investigate seed / decode / model-revision drift."
        )
    if not lines:
        return ""
    if workflow_doc:
        lines.append(f"See {workflow_doc} for the parity-compatible anchor recipe.")
    return "\n".join(lines)
