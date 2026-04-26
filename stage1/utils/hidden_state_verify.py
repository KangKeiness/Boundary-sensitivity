"""Self-verification for hidden_states_*.pt artifacts.

Stage 1 hardening (2026-04-25): a Stage 1 / Phase A run is only trustworthy if
its hidden-state artifacts can be reloaded from disk and their structure
matches expectations. The structural checks are deliberately strict — anything
that drifts (sample IDs missing, shape changed, dtype changed, layer count
changed) collapses the foundation.

Public API:
    verify_hidden_state_artifacts(run_dir, expected_sample_ids, *,
                                  expected_condition_names,
                                  expected_layer_count, expected_hidden_size)

Returns a structured report (List[Dict]) describing every artifact checked.
Raises HiddenStateVerificationError on any inconsistency.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


class HiddenStateVerificationError(RuntimeError):
    """Raised when a hidden-state artifact fails self-verification."""


def _torch_load(path: str) -> Any:
    """Lazy import for torch.load so this module is import-safe without torch."""
    import torch  # noqa: F401  (import-time check)

    return torch.load(path, map_location="cpu")


def _list_hidden_state_files(run_dir: str) -> List[str]:
    out: List[str] = []
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith("hidden_states_") and fname.endswith(".pt"):
            out.append(os.path.join(run_dir, fname))
    return out


def verify_hidden_state_artifacts(
    run_dir: str,
    expected_sample_ids: Iterable[str],
    *,
    expected_condition_names: Optional[Iterable[str]] = None,
    allow_unexpected_conditions: bool = False,
    expected_layer_count: Optional[int] = None,
    expected_hidden_size: Optional[int] = None,
    raise_on_error: bool = True,
) -> List[Dict[str, Any]]:
    """Verify every ``hidden_states_*.pt`` in ``run_dir``.

    Checks (per file):
      0. If ``expected_condition_names`` is supplied, the set of
         ``hidden_states_*.pt`` files exactly covers that condition set.
      1. File loads with ``torch.load(map_location="cpu")``.
      2. Top-level object is a dict keyed by sample_id (strings).
      3. Sample-ID set is exactly equal to ``expected_sample_ids``.
      4. Every value is a 2D tensor (layers, hidden_size) — Stage 1's
         ``last_token`` / ``mean`` pooling produces this rank.
      5. Layer count is consistent across all sample tensors in the file
         and equals ``expected_layer_count`` when provided.
      6. Hidden size is consistent and equals ``expected_hidden_size`` when
         provided.
      7. Dtype is consistent across the file.

    Args:
        run_dir: Path to the run directory.
        expected_sample_ids: Iterable of strings — the canonical sample-ID
            set that should be present in every artifact. Order is irrelevant.
        expected_condition_names: Optional iterable of condition names that
            must have matching ``hidden_states_{condition}.pt`` artifacts.
            When supplied, missing expected artifacts fail verification.
        allow_unexpected_conditions: When False (default), condition files not
            listed in ``expected_condition_names`` also fail verification.
            Set True only for explicit debugging / exploratory analysis.
        expected_layer_count: If set, every artifact's per-sample tensor must
            have first-dim == this value.
        expected_hidden_size: If set, every artifact's per-sample tensor must
            have second-dim == this value.
        raise_on_error: When True (default), the function raises
            ``HiddenStateVerificationError`` on the first failure. Set False
            to collect all failures and inspect the returned report.

    Returns:
        A list of per-artifact reports. Each report dict contains:
            condition, path, n_samples, layer_count, hidden_size, dtype,
            ok (bool), errors (List[str]).

    Raises:
        HiddenStateVerificationError: when any artifact fails a check and
            raise_on_error is True.
        FileNotFoundError: when ``run_dir`` does not exist or contains no
            ``hidden_states_*.pt`` files at all (a successful Stage 1 / Phase
            A run must produce at least one).
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    expected_set: Set[str] = set(expected_sample_ids)
    if not expected_set:
        raise ValueError("expected_sample_ids must be non-empty")

    files = _list_hidden_state_files(run_dir)
    if not files:
        raise FileNotFoundError(
            f"No hidden_states_*.pt files in {run_dir}. A successful Stage 1 / "
            f"Phase A run must persist at least one hidden-state artifact."
        )

    reports: List[Dict[str, Any]] = []
    aggregate_failures: List[str] = []

    observed_conditions = {
        os.path.basename(path)[len("hidden_states_"):-len(".pt")]
        for path in files
    }
    if expected_condition_names is not None:
        expected_conditions = {str(c) for c in expected_condition_names}
        missing_conditions = sorted(expected_conditions - observed_conditions)
        unexpected_conditions = sorted(observed_conditions - expected_conditions)

        coverage_errors: List[str] = []
        if missing_conditions:
            coverage_errors.append(
                "missing expected condition artifacts: "
                + ", ".join(missing_conditions[:10])
                + ("..." if len(missing_conditions) > 10 else "")
            )
        if unexpected_conditions and not allow_unexpected_conditions:
            coverage_errors.append(
                "unexpected condition artifacts: "
                + ", ".join(unexpected_conditions[:10])
                + ("..." if len(unexpected_conditions) > 10 else "")
            )

        if coverage_errors:
            reports.append({
                "condition": "__coverage__",
                "path": run_dir,
                "n_samples": None,
                "layer_count": None,
                "hidden_size": None,
                "dtype": None,
                "ok": False,
                "errors": coverage_errors,
            })
            aggregate_failures.extend(
                f"condition coverage: {err}" for err in coverage_errors
            )

    for path in files:
        condition = os.path.basename(path)[len("hidden_states_"):-len(".pt")]
        report: Dict[str, Any] = {
            "condition": condition,
            "path": path,
            "n_samples": None,
            "layer_count": None,
            "hidden_size": None,
            "dtype": None,
            "ok": False,
            "errors": [],
        }

        try:
            hs_dict = _torch_load(path)
        except Exception as exc:
            report["errors"].append(f"torch.load failed: {exc!r}")
            reports.append(report)
            aggregate_failures.append(f"{condition}: torch.load failed: {exc!r}")
            continue

        if not isinstance(hs_dict, dict):
            report["errors"].append(
                f"top-level object is {type(hs_dict).__name__}, expected dict"
            )
            reports.append(report)
            aggregate_failures.append(
                f"{condition}: top-level object is not a dict"
            )
            continue

        observed_ids = set(hs_dict.keys())
        report["n_samples"] = len(observed_ids)

        missing = expected_set - observed_ids
        unexpected = observed_ids - expected_set
        if missing:
            report["errors"].append(
                f"missing sample_ids ({len(missing)}): "
                + ", ".join(sorted(missing)[:5])
                + ("..." if len(missing) > 5 else "")
            )
        if unexpected:
            report["errors"].append(
                f"unexpected sample_ids ({len(unexpected)}): "
                + ", ".join(sorted(unexpected)[:5])
                + ("..." if len(unexpected) > 5 else "")
            )

        # Per-tensor structural checks.
        observed_layers: Set[int] = set()
        observed_hidden: Set[int] = set()
        observed_dtypes: Set[str] = set()
        rank_issues: List[str] = []
        for sid in sorted(observed_ids):
            t = hs_dict[sid]
            shape = tuple(getattr(t, "shape", ()))
            if len(shape) != 2:
                rank_issues.append(f"{sid}: rank={len(shape)} shape={shape}")
                continue
            observed_layers.add(int(shape[0]))
            observed_hidden.add(int(shape[1]))
            observed_dtypes.add(str(getattr(t, "dtype", "unknown")))

        if rank_issues:
            report["errors"].append(
                f"non-2D tensors ({len(rank_issues)}): "
                + " | ".join(rank_issues[:3])
                + ("..." if len(rank_issues) > 3 else "")
            )

        if len(observed_layers) > 1:
            report["errors"].append(
                f"inconsistent layer count across samples: {sorted(observed_layers)}"
            )
        if len(observed_hidden) > 1:
            report["errors"].append(
                f"inconsistent hidden size across samples: {sorted(observed_hidden)}"
            )
        if len(observed_dtypes) > 1:
            report["errors"].append(
                f"inconsistent dtype across samples: {sorted(observed_dtypes)}"
            )

        report["layer_count"] = (
            next(iter(observed_layers)) if len(observed_layers) == 1 else None
        )
        report["hidden_size"] = (
            next(iter(observed_hidden)) if len(observed_hidden) == 1 else None
        )
        report["dtype"] = (
            next(iter(observed_dtypes)) if len(observed_dtypes) == 1 else None
        )

        if expected_layer_count is not None and report["layer_count"] is not None:
            if report["layer_count"] != expected_layer_count:
                report["errors"].append(
                    f"layer_count={report['layer_count']} != "
                    f"expected {expected_layer_count}"
                )
        if expected_hidden_size is not None and report["hidden_size"] is not None:
            if report["hidden_size"] != expected_hidden_size:
                report["errors"].append(
                    f"hidden_size={report['hidden_size']} != "
                    f"expected {expected_hidden_size}"
                )

        report["ok"] = len(report["errors"]) == 0
        if not report["ok"]:
            aggregate_failures.append(
                f"{condition}: " + " | ".join(report["errors"])
            )
        reports.append(report)

    if aggregate_failures and raise_on_error:
        raise HiddenStateVerificationError(
            "Hidden-state artifact verification FAILED:\n  "
            + "\n  ".join(aggregate_failures)
        )

    return reports


def summarise_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a manifest-friendly summary from per-artifact reports."""
    return {
        "n_artifacts": len(reports),
        "all_ok": all(r["ok"] for r in reports),
        "artifacts": [
            {
                "condition": r["condition"],
                "n_samples": r["n_samples"],
                "layer_count": r["layer_count"],
                "hidden_size": r["hidden_size"],
                "dtype": r["dtype"],
                "ok": r["ok"],
                "errors": r["errors"],
            }
            for r in reports
        ],
    }
