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
the current Phase B parity block and by upstream validity metadata before
being accepted.
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
_VALID_UPSTREAM_STATUS: str = "passed"
_STAGE1_PHASE_NAME: str = "Stage1"
_PHASE_A_PHASE_NAMES = {"A", "PhaseA", "phase_a", "Phase A"}

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


def _record_rejection(rejections: Optional[List[str]], message: str) -> None:
    logger.warning(message)
    if rejections is not None:
        rejections.append(message)


def _parity_compatible(
    manifest_path: str,
    current_parity: Optional[Dict[str, Any]],
    *,
    run_label: str,
    rejections: Optional[List[str]] = None,
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
        _record_rejection(
            rejections,
            f"{run_label} rejected: no readable manifest.json "
            "(parity requires manifest).",
        )
        return False
    mismatches = check_manifest_parity(
        manifest, current_parity,
        source_path=manifest_path,
        target_desc="current Phase B config",
    )
    if mismatches:
        _record_rejection(
            rejections,
            f"{run_label} rejected: manifest parity mismatch: "
            + "; ".join(mismatches),
        )
        return False
    return True


def _load_parity_compatible_manifest(
    manifest_path: str,
    current_parity: Optional[Dict[str, Any]],
    *,
    run_label: str,
    rejections: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load a manifest and return it only when manifest parity passes."""
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        _record_rejection(
            rejections,
            f"{run_label} rejected: no readable manifest.json "
            "(parity requires manifest).",
        )
        return None
    if not _parity_compatible(
        manifest_path,
        current_parity,
        run_label=run_label,
        rejections=rejections,
    ):
        return None
    return manifest


def _require_passed_status(
    obj: Dict[str, Any],
    key: str,
    *,
    run_label: str,
    field_label: str,
    allow_missing: bool,
    rejections: Optional[List[str]],
) -> bool:
    """Require ``obj[key] == 'passed'``; optionally tolerate missing only."""
    if key not in obj:
        if allow_missing:
            _record_rejection(
                rejections,
                f"{run_label}: missing {field_label}; accepted only because "
                "allow_unverified_upstream=True.",
            )
            return True
        _record_rejection(
            rejections,
            f"{run_label} rejected: missing required validity field "
            f"{field_label}.",
        )
        return False
    status = obj.get(key)
    if status != _VALID_UPSTREAM_STATUS:
        _record_rejection(
            rejections,
            f"{run_label} rejected: {field_label}={status!r} "
            f"(expected {_VALID_UPSTREAM_STATUS!r}).",
        )
        return False
    return True


def _hidden_verification_valid(
    manifest: Dict[str, Any],
    *,
    run_label: str,
    required_conditions: Tuple[str, ...],
    allow_missing: bool,
    rejections: Optional[List[str]],
) -> bool:
    hsv = manifest.get("hidden_state_verification")
    if not isinstance(hsv, dict):
        if allow_missing:
            _record_rejection(
                rejections,
                f"{run_label}: missing hidden_state_verification; accepted only "
                "because allow_unverified_upstream=True.",
            )
            return True
        _record_rejection(
            rejections,
            f"{run_label} rejected: missing required hidden_state_verification.",
        )
        return False
    if hsv.get("all_ok") is not True:
        _record_rejection(
            rejections,
            f"{run_label} rejected: hidden_state_verification.all_ok="
            f"{hsv.get('all_ok')!r}.",
        )
        return False

    artifacts = hsv.get("artifacts")
    if isinstance(artifacts, list):
        ok_conditions = {
            str(a.get("condition"))
            for a in artifacts
            if isinstance(a, dict) and a.get("ok") is True
        }
        missing = sorted(set(required_conditions) - ok_conditions)
        if missing:
            _record_rejection(
                rejections,
                f"{run_label} rejected: hidden_state_verification lacks passed "
                f"entries for required condition(s): {missing}.",
            )
            return False
    elif not allow_missing:
        _record_rejection(
            rejections,
            f"{run_label} rejected: hidden_state_verification.artifacts missing "
            "or malformed.",
        )
        return False

    return True


def _phase_a_upstream_valid(
    summary: Dict[str, Any],
    manifest: Dict[str, Any],
    *,
    run_label: str,
    allow_unverified_upstream: bool,
    rejections: Optional[List[str]],
) -> bool:
    phase = manifest.get("phase")
    if phase not in _PHASE_A_PHASE_NAMES:
        _record_rejection(
            rejections,
            f"{run_label} rejected: manifest phase={phase!r}; expected Phase A.",
        )
        return False

    ok = _require_passed_status(
        manifest,
        "run_status",
        run_label=run_label,
        field_label="manifest.run_status",
        allow_missing=allow_unverified_upstream,
        rejections=rejections,
    )
    summary_status = summary.get("run_status")
    if summary_status is not None and summary_status != _VALID_UPSTREAM_STATUS:
        _record_rejection(
            rejections,
            f"{run_label} rejected: phase_a_summary.run_status={summary_status!r} "
            f"(expected {_VALID_UPSTREAM_STATUS!r}).",
        )
        ok = False
    ok = _hidden_verification_valid(
        manifest,
        run_label=run_label,
        required_conditions=("no_swap",),
        allow_missing=allow_unverified_upstream,
        rejections=rejections,
    ) and ok
    return ok


def _stage1_upstream_valid(
    run_dir: str,
    eval_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    *,
    run_label: str,
    allow_unverified_upstream: bool,
    rejections: Optional[List[str]],
) -> bool:
    phase = manifest.get("phase")
    if phase != _STAGE1_PHASE_NAME:
        _record_rejection(
            rejections,
            f"{run_label} rejected: manifest phase={phase!r}; expected "
            f"{_STAGE1_PHASE_NAME!r} canonical Stage1 source.",
        )
        return False

    ok = _require_passed_status(
        manifest,
        "self_verification",
        run_label=run_label,
        field_label="manifest.self_verification",
        allow_missing=allow_unverified_upstream,
        rejections=rejections,
    )
    ok = _require_passed_status(
        manifest,
        "run_status",
        run_label=run_label,
        field_label="manifest.run_status",
        allow_missing=allow_unverified_upstream,
        rejections=rejections,
    ) and ok

    eval_status = eval_dict.get("run_status") or eval_dict.get("evaluation_status")
    if eval_status is not None and eval_status != _VALID_UPSTREAM_STATUS:
        _record_rejection(
            rejections,
            f"{run_label} rejected: evaluation status={eval_status!r} "
            f"(expected {_VALID_UPSTREAM_STATUS!r}).",
        )
        ok = False

    conditions = manifest.get("conditions")
    if not isinstance(conditions, list):
        if allow_unverified_upstream:
            _record_rejection(
                rejections,
                f"{run_label}: missing manifest.conditions; accepted only "
                "because allow_unverified_upstream=True.",
            )
        else:
            _record_rejection(
                rejections,
                f"{run_label} rejected: missing manifest.conditions needed to "
                "prove hard_swap_b8 is canonical.",
            )
            ok = False
    elif "hard_swap_b8" not in conditions or "no_swap" not in conditions:
        _record_rejection(
            rejections,
            f"{run_label} rejected: manifest.conditions must include both "
            "hard_swap_b8 and no_swap.",
        )
        ok = False

    bds_path = os.path.join(run_dir, "bds_hard_swap_b8.json")
    if os.path.exists(bds_path):
        try:
            with open(bds_path, encoding="utf-8") as f:
                bds = json.load(f)
        except Exception as exc:
            _record_rejection(
                rejections,
                f"{run_label} rejected: unreadable bds_hard_swap_b8.json "
                f"({exc!r}).",
            )
            ok = False
        else:
            if bds.get("b") != 8 or bds.get("t") != 20:
                _record_rejection(
                    rejections,
                    f"{run_label} rejected: hard_swap_b8 metadata has "
                    f"b={bds.get('b')!r}, t={bds.get('t')!r}; expected b=8, t=20.",
                )
                ok = False

    ok = _hidden_verification_valid(
        manifest,
        run_label=run_label,
        required_conditions=("no_swap", "hard_swap_b8"),
        allow_missing=allow_unverified_upstream,
        rejections=rejections,
    ) and ok
    return ok


def load_latest_phase_a_summary(
    phase_a_dir: str,
    current_parity: Optional[Dict[str, Any]] = None,
    *,
    allow_unverified_upstream: bool = False,
    rejections: Optional[List[str]] = None,
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
        manifest = _load_parity_compatible_manifest(
            manifest_path, current_parity,
            run_label=f"Phase A run {path}",
            rejections=rejections,
        )
        if manifest is None:
            continue
        if not _phase_a_upstream_valid(
            summary,
            manifest,
            run_label=f"Phase A run {path}",
            allow_unverified_upstream=allow_unverified_upstream,
            rejections=rejections,
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
    *,
    allow_unverified_upstream: bool = False,
    rejections: Optional[List[str]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[bool]]:
    """Return (hard_swap_b8_accuracy, no_swap_accuracy, evaluation_path,
    criteria_passed) for the newest parity-compatible Stage 1 run, or
    (None, None, None, None) on no match.

    ``criteria_passed`` mirrors ``evaluation.criteria.passed`` — Stage 1's
    operational 2-of-3 criteria stability indicator. Surfaced for diagnostics
    only; the anchor gate does NOT hard-fail on ``criteria_passed=False``
    because parity + tolerance are the actual reproducibility contract for
    cross-phase anchor reuse.
    """
    pattern = os.path.join(stage1_dir, "run_*", "evaluation.json")
    candidates = sorted(glob.glob(pattern), reverse=True)
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                eval_dict = json.load(f)
        except Exception:
            continue
        run_dir = os.path.dirname(path)
        manifest_path = os.path.join(os.path.dirname(path), "manifest.json")
        manifest = _load_parity_compatible_manifest(
            manifest_path, current_parity,
            run_label=f"Stage 1 run {path}",
            rejections=rejections,
        )
        if manifest is None:
            continue
        if not _stage1_upstream_valid(
            run_dir,
            eval_dict,
            manifest,
            run_label=f"Stage 1 run {path}",
            allow_unverified_upstream=allow_unverified_upstream,
            rejections=rejections,
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
        criteria = eval_dict.get("criteria") or {}
        criteria_passed: Optional[bool] = (
            bool(criteria["passed"]) if "passed" in criteria else None
        )
        if hs_acc is not None:
            return hs_acc, ns_acc, path, criteria_passed
    return None, None, None, None


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
    # Advisory only: Stage 1 sweep's 2-of-3 criteria.passed flag from the run
    # that supplied the hard_swap_b8 anchor. Surfaced in summary diagnostics
    # so an operator can see whether the anchor came from a sweep that itself
    # met the operational stability criteria. NOT used as a hard gate —
    # parity + cross-check tolerance remain the actual reproducibility
    # contract.
    stage1_criteria_passed: Optional[bool] = None
    tolerance: float = PHASE_A_CROSS_CHECK_TOL
    anchor_rejections: List[str] = field(default_factory=list)

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
            "stage1_criteria_passed": self.stage1_criteria_passed,
            "anchor_hard_swap_b8_accuracy": self.anchor_hard_swap_b8_accuracy,
            "anchor_hard_swap_source": self.anchor_hard_swap_source,
            "anchor_no_swap_accuracy": self.anchor_no_swap_accuracy,
            "anchor_no_swap_source": self.anchor_no_swap_source,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "missing_anchors": list(self.missing_anchors),
            "failed_anchors": list(self.failed_anchors),
            "anchor_rejections": list(self.anchor_rejections),
            "note": (
                "Treatment parity (no_patch ≈ hard_swap_b8) is cross-checked "
                "against the newest valid canonical Stage1 sweep run. Phase A "
                "is allowed as a no_swap source only; it is never allowed to "
                "supply the hard_swap_b8 treatment anchor. Anchor runs are "
                "filtered by manifest parity (model IDs, dataset, generation "
                "config including max_new_tokens), sample regime parity, and "
                "passed upstream verification metadata. "
                "stage1_criteria_passed is advisory only — it reports the "
                "Stage 1 sweep's own 2-of-3 criteria.passed flag and is NOT "
                "a Phase B gate; parity + cross-check tolerance are the gate. "
                f"See {ANCHOR_WORKFLOW_DOC} for the parity-compatible recipe."
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
    allow_unverified_upstream: bool = False,
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

    Anchor source rule: ``hard_swap_b8`` must come from a valid canonical
    Stage1 run. Phase A can supply ``no_swap`` only.
    """
    pa_dir = phase_a_dir or default_phase_a_outputs_dir()
    s1_dir = stage1_dir or default_stage1_outputs_dir()
    anchor_rejections: List[str] = []

    phase_a, phase_a_path = load_latest_phase_a_summary(
        pa_dir,
        current_parity,
        allow_unverified_upstream=allow_unverified_upstream,
        rejections=anchor_rejections,
    )
    pa_hs: Optional[float] = None
    pa_ns: Optional[float] = None
    if phase_a is not None:
        pa_hs = phase_a_condition_accuracy(phase_a, "hard_swap_b8")
        pa_ns = phase_a_condition_accuracy(phase_a, "no_swap")
        if pa_hs is not None:
            _record_rejection(
                anchor_rejections,
                "Phase A hard_swap_b8-like row ignored: Phase B treatment "
                "anchor must come from canonical Stage1 b=8,t=20.",
            )

    s1_hs, s1_ns, s1_path, s1_criteria_passed = load_latest_stage1_hard_swap_b8(
        s1_dir,
        current_parity,
        allow_unverified_upstream=allow_unverified_upstream,
        rejections=anchor_rejections,
    )

    anchor_hs = s1_hs
    anchor_ns = pa_ns if pa_ns is not None else s1_ns
    anchor_hs_src = "stage1" if s1_hs is not None else None
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
        stage1_criteria_passed=s1_criteria_passed,
        tolerance=tolerance,
        anchor_rejections=anchor_rejections,
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
            "(2) all candidate runs were rejected by manifest parity or "
            "upstream-validity checks (check warnings above for the first "
            "mismatched field — common offenders are generation.max_new_tokens "
            "differing between stage1_main.yaml (256) and stage2_confound.yaml "
            "(512), missing run_status/self_verification metadata, or a "
            "non-Stage1 hard_swap_b8 source)."
        )
    if result.anchor_rejections:
        lines.append("Rejected anchor candidates:")
        for msg in result.anchor_rejections[:10]:
            lines.append(f"  - {msg}")
        if len(result.anchor_rejections) > 10:
            lines.append(f"  - ... {len(result.anchor_rejections) - 10} more")
    if result.failed_anchors:
        lines.append("Cross-check FAILED — anchor(s) outside tolerance:")
        for f in result.failed_anchors:
            lines.append(f"  - {f}")
        lines.append(
            "This means the current Phase B no_patch / clean_baseline accuracy "
            "does not reproduce a prior parity-compatible run within tolerance "
            f"{result.tolerance}. Investigate seed / decode / model-revision drift."
        )
    # Advisory note: surface stage1_criteria_passed=False so an operator
    # investigating an anchor failure can see whether the Stage 1 sweep that
    # supplied the hard_swap_b8 anchor itself failed its 2-of-3 criteria.
    # This is informational; it never causes the gate itself to fail.
    if (
        result.stage1_criteria_passed is False
        and result.anchor_hard_swap_source == "stage1"
    ):
        lines.append(
            "Advisory: the Stage 1 run supplying the hard_swap_b8 anchor has "
            "criteria.passed=False (operational stability indicator). The "
            "anchor's accuracy is still reproducible against parity, so the "
            "cross-check above is the relevant gate; consider whether to "
            "regenerate the Stage 1 sweep before publishing claims."
        )
    if not lines:
        return ""
    if workflow_doc:
        lines.append(f"See {workflow_doc} for the parity-compatible anchor recipe.")
    return "\n".join(lines)
