"""Phase C — mediation-style decomposition of Phase B accuracy deltas.

This module is analysis-only. It consumes Phase B per-condition sample-level
JSONLs and produces ``restoration_effect``, ``residual_effect``, and
``restoration_proportion`` with 95% paired-bootstrap CIs keyed by
``sample_id``.

IMPORTANT METHODOLOGICAL CONSTRAINT:
    Mediation analysis here decomposes accuracy deltas under prompt-side
    restoration intervention only. It is not a formal NIE/NDE decomposition.

See ``notes/specs/phase_c_mediation.md`` for the authoritative spec.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level constants ───────────────────────────────────────────────────

CLAIM_ELIGIBLE_CONDITIONS: Tuple[str, ...] = (
    "patch_boundary_local",
    "patch_recovery_early",
    "patch_recovery_full",
    "patch_final_only",
)

# Below this |acc(clean) - acc(no_patch)|, restoration_proportion is null.
EPSILON_DENOM: float = 0.005

# Default bootstrap knobs (spec §6).
_DEFAULT_BOOTSTRAP_N: int = 1000
_DEFAULT_SEED: int = 0
_DEFAULT_CI: float = 0.95

# Fraction of bootstrap resamples that may hit a near-zero denominator before
# we downgrade the proportion CI to ``null`` with reason ``unstable_denominator``.
_UNSTABLE_DENOM_DROP_FRACTION: float = 0.05


# ── Data container ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConditionCorrectness:
    """Immutable per-condition correctness vector aligned with sample_ids."""

    condition: str
    sample_ids: Tuple[str, ...]
    correct: Tuple[bool, ...]


# ── IO ──────────────────────────────────────────────────────────────────────


def load_condition_correctness(jsonl_path: str) -> ConditionCorrectness:
    """Read a Phase B ``results_*.jsonl`` file and return a correctness vector.

    The file must be UTF-8 encoded. Each line must be a JSON object with at
    least ``sample_id`` (str) and ``correct`` (bool). The condition name is
    derived from the basename (``results_<name>.jsonl`` → ``<name>``).

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: on missing fields or duplicate ``sample_id``.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Phase B results file not found: {jsonl_path}")

    base = os.path.basename(jsonl_path)
    if not base.startswith("results_") or not base.endswith(".jsonl"):
        raise ValueError(
            f"Unexpected filename shape {base!r}; expected 'results_<name>.jsonl'."
        )
    condition = base[len("results_"):-len(".jsonl")]

    sample_ids: List[str] = []
    correct_vals: List[bool] = []
    seen: set = set()

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if "sample_id" not in row:
                raise ValueError(
                    f"{jsonl_path}:{line_no} missing 'sample_id' field"
                )
            if "correct" not in row:
                raise ValueError(
                    f"{jsonl_path}:{line_no} missing 'correct' field"
                )
            sid = str(row["sample_id"])
            if sid in seen:
                raise ValueError(
                    f"{jsonl_path}: duplicate sample_id {sid!r} at line {line_no}"
                )
            seen.add(sid)
            sample_ids.append(sid)
            correct_vals.append(bool(row["correct"]))

    return ConditionCorrectness(
        condition=condition,
        sample_ids=tuple(sample_ids),
        correct=tuple(correct_vals),
    )


# ── Alignment ───────────────────────────────────────────────────────────────


def align_by_sample_id(
    *conditions: ConditionCorrectness,
) -> Tuple[List[str], List[np.ndarray]]:
    """Intersect sample_ids across all inputs, returning aligned arrays.

    Returns:
        ``(aligned_ids, arrays)`` where ``aligned_ids`` is the sorted
        intersection of every input's ``sample_ids`` and ``arrays`` is a list
        of ``np.ndarray`` (dtype=int8, 0/1) in the same order as the inputs,
        each of length ``len(aligned_ids)``.

    Side effect:
        For every input whose ``sample_ids`` is a strict superset of the
        aligned set, a WARNING is logged naming the condition and the dropped
        count (substring ``"dropped"``).
    """
    if not conditions:
        raise ValueError("align_by_sample_id requires at least one ConditionCorrectness")

    id_sets = [set(c.sample_ids) for c in conditions]
    common = set.intersection(*id_sets)
    aligned_ids: List[str] = sorted(common)

    arrays: List[np.ndarray] = []
    for c in conditions:
        dropped = len(c.sample_ids) - len(aligned_ids)
        if dropped > 0:
            logger.warning(
                "Condition %s: dropped %d sample_id(s) not in the aligned "
                "intersection (|aligned|=%d, |condition|=%d).",
                c.condition, dropped, len(aligned_ids), len(c.sample_ids),
            )
        lookup = dict(zip(c.sample_ids, c.correct))
        arr = np.fromiter(
            (1 if lookup[sid] else 0 for sid in aligned_ids),
            dtype=np.int8,
            count=len(aligned_ids),
        )
        arrays.append(arr)
    return aligned_ids, arrays


# ── Paired bootstrap ────────────────────────────────────────────────────────


def _paired_bootstrap(
    fn: Callable[[List[np.ndarray]], float],
    arrays: Sequence[np.ndarray],
    *,
    n_resamples: int = _DEFAULT_BOOTSTRAP_N,
    seed: int = _DEFAULT_SEED,
    ci: float = _DEFAULT_CI,
) -> Tuple[float, float, float, int]:
    """Paired bootstrap over the shared index set.

    Args:
        fn: callable that maps a list of resampled arrays (same order and
            shape as ``arrays``) to a scalar float. May return ``np.nan`` to
            signal an invalid resample (e.g., unstable denominator); such
            resamples are dropped from the percentile computation.
        arrays: sequence of equal-length float32 arrays. The resample is
            index-paired across all arrays.
        n_resamples: number of bootstrap iterations.
        seed: ``numpy.random.default_rng`` seed; 0 by default for
            cross-phase determinism.
        ci: central confidence level (0.95 → 2.5 / 97.5 percentiles).

    Returns:
        ``(point, ci_lo, ci_hi, n_dropped)``. ``point`` is ``fn(arrays)``
        itself (not a mean of resamples).
    """
    if not arrays:
        raise ValueError("_paired_bootstrap requires at least one array")
    n = arrays[0].shape[0]
    for a in arrays:
        if a.shape[0] != n:
            raise ValueError("_paired_bootstrap: arrays must share length")
    if n == 0:
        raise ValueError("_paired_bootstrap: empty arrays")

    arrays_f32: List[np.ndarray] = [a.astype(np.float32, copy=False) for a in arrays]
    point = float(fn(arrays_f32))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples: List[float] = []
    dropped = 0
    for r in range(n_resamples):
        resampled = [a[idx[r]] for a in arrays_f32]
        val = fn(resampled)
        if not np.isfinite(val):
            dropped += 1
            continue
        samples.append(float(val))

    if not samples:
        return point, float("nan"), float("nan"), dropped

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(np.asarray(samples, dtype=np.float64), alpha))
    hi = float(np.quantile(np.asarray(samples, dtype=np.float64), 1.0 - alpha))
    return point, lo, hi, dropped


# ── Metric implementations ──────────────────────────────────────────────────


def _acc(arr: np.ndarray) -> float:
    return float(arr.astype(np.float32).mean())


def restoration_effect(
    patched: ConditionCorrectness,
    no_patch: ConditionCorrectness,
    *,
    bootstrap_n: int = _DEFAULT_BOOTSTRAP_N,
    seed: int = _DEFAULT_SEED,
    ci: float = _DEFAULT_CI,
) -> Dict[str, Any]:
    """Point estimate ``acc(patched) - acc(no_patch)`` on aligned sample_ids."""
    aligned_ids, (a, b) = _pair_align(patched, no_patch)
    n_aligned = len(aligned_ids)
    n_dropped = max(len(patched.sample_ids), len(no_patch.sample_ids)) - n_aligned

    point, lo, hi, _ = _paired_bootstrap(
        lambda xs: _acc(xs[0]) - _acc(xs[1]),
        [a, b],
        n_resamples=bootstrap_n, seed=seed, ci=ci,
    )
    return {
        "condition": patched.condition,
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_aligned": n_aligned,
        "n_dropped_ids": n_dropped,
    }


def residual_effect(
    clean: ConditionCorrectness,
    best_patched: ConditionCorrectness,
    *,
    bootstrap_n: int = _DEFAULT_BOOTSTRAP_N,
    seed: int = _DEFAULT_SEED,
    ci: float = _DEFAULT_CI,
) -> Dict[str, Any]:
    """Point estimate ``acc(clean) - acc(best_patched)`` on aligned ids."""
    aligned_ids, (a, b) = _pair_align(clean, best_patched)
    n_aligned = len(aligned_ids)
    n_dropped = max(len(clean.sample_ids), len(best_patched.sample_ids)) - n_aligned

    point, lo, hi, _ = _paired_bootstrap(
        lambda xs: _acc(xs[0]) - _acc(xs[1]),
        [a, b],
        n_resamples=bootstrap_n, seed=seed, ci=ci,
    )
    return {
        "condition": best_patched.condition,
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_aligned": n_aligned,
        "n_dropped_ids": n_dropped,
    }


def restoration_proportion(
    clean: ConditionCorrectness,
    no_patch: ConditionCorrectness,
    best_patched: ConditionCorrectness,
    *,
    bootstrap_n: int = _DEFAULT_BOOTSTRAP_N,
    seed: int = _DEFAULT_SEED,
    ci: float = _DEFAULT_CI,
    epsilon_denom: float = EPSILON_DENOM,
) -> Dict[str, Any]:
    """``(acc(best_patched) - acc(no_patch)) / (acc(clean) - acc(no_patch))``.

    Returns a dict with ``point`` (``None`` when the denominator is below
    ``epsilon_denom``), ``ci_lo`` / ``ci_hi`` (both ``None`` when unstable),
    ``n_aligned``, ``denom_point``, ``n_resamples_dropped_denominator``, and
    ``ci_reason`` ∈ ``{None, 'denominator_below_epsilon', 'unstable_denominator'}``.
    """
    aligned_ids, (clean_arr, no_patch_arr, best_arr) = _pair_align(
        clean, no_patch, best_patched,
    )
    n_aligned = len(aligned_ids)
    acc_clean = _acc(clean_arr)
    acc_no_patch = _acc(no_patch_arr)
    acc_best = _acc(best_arr)
    denom_point = float(acc_clean - acc_no_patch)
    numer_point = float(acc_best - acc_no_patch)

    if abs(denom_point) < epsilon_denom:
        return {
            "condition": best_patched.condition,
            "point": None,
            "ci_lo": None,
            "ci_hi": None,
            "n_aligned": n_aligned,
            "denom_point": denom_point,
            "n_resamples_dropped_denominator": 0,
            "ci_reason": "denominator_below_epsilon",
        }

    point_value = numer_point / denom_point

    def _ratio(xs: List[np.ndarray]) -> float:
        ac = _acc(xs[0])
        an = _acc(xs[1])
        ab = _acc(xs[2])
        denom = ac - an
        if abs(denom) < epsilon_denom:
            return float("nan")
        return (ab - an) / denom

    _, lo, hi, n_dropped = _paired_bootstrap(
        _ratio,
        [clean_arr, no_patch_arr, best_arr],
        n_resamples=bootstrap_n, seed=seed, ci=ci,
    )

    drop_frac = n_dropped / float(bootstrap_n) if bootstrap_n > 0 else 0.0
    if drop_frac > _UNSTABLE_DENOM_DROP_FRACTION:
        return {
            "condition": best_patched.condition,
            "point": point_value,
            "ci_lo": None,
            "ci_hi": None,
            "n_aligned": n_aligned,
            "denom_point": denom_point,
            "n_resamples_dropped_denominator": n_dropped,
            "ci_reason": "unstable_denominator",
        }

    return {
        "condition": best_patched.condition,
        "point": point_value,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_aligned": n_aligned,
        "denom_point": denom_point,
        "n_resamples_dropped_denominator": n_dropped,
        "ci_reason": None,
    }


def _pair_align(
    *conditions: ConditionCorrectness,
) -> Tuple[List[str], List[np.ndarray]]:
    """Thin wrapper over ``align_by_sample_id`` that raises on empty intersection."""
    aligned_ids, arrays = align_by_sample_id(*conditions)
    if not aligned_ids:
        names = ",".join(c.condition for c in conditions)
        raise ValueError(
            f"No sample_ids in common across conditions: {names}"
        )
    return aligned_ids, arrays


# ── Table orchestration ─────────────────────────────────────────────────────


_CLEAN_NO_PATCH_FILE = "results_clean_no_patch.jsonl"
_NO_PATCH_FILE = "results_restoration_no_patch.jsonl"
_RESTORATION_PATCH_PREFIX = "results_restoration_patch_"


def _list_restoration_patch_files(phase_b_run_dir: str) -> List[str]:
    """Return sorted absolute paths of every ``results_restoration_patch_*.jsonl``."""
    if not os.path.isdir(phase_b_run_dir):
        raise FileNotFoundError(
            f"Phase B run directory does not exist: {phase_b_run_dir}"
        )
    entries = sorted(os.listdir(phase_b_run_dir))
    out: List[str] = []
    for name in entries:
        if name.startswith(_RESTORATION_PATCH_PREFIX) and name.endswith(".jsonl"):
            out.append(os.path.join(phase_b_run_dir, name))
    return out


def _strip_restoration_prefix(condition: str) -> str:
    """``restoration_patch_boundary_local`` → ``patch_boundary_local``."""
    if condition.startswith("restoration_"):
        return condition[len("restoration_"):]
    return condition


def compute_decomposition_table(
    phase_b_run_dir: str,
    *,
    bootstrap_n: int = _DEFAULT_BOOTSTRAP_N,
    seed: int = _DEFAULT_SEED,
    ci: float = _DEFAULT_CI,
    epsilon_denom: float = EPSILON_DENOM,
) -> Dict[str, Any]:
    """Orchestrate Phase C decomposition from a Phase B run directory.

    Loads ``results_clean_no_patch.jsonl``, ``results_restoration_no_patch.jsonl``,
    and every ``results_restoration_patch_*.jsonl``. Computes
    ``restoration_effect`` per restoration condition; picks ``C_best`` as
    ``argmax`` among the claim-eligible conditions
    (``CLAIM_ELIGIBLE_CONDITIONS``), breaking ties alphabetically; computes
    ``residual_effect`` and ``restoration_proportion`` against ``C_best``.

    Returns a dict with ``rows`` (list of per-condition dicts),
    ``best_condition``, ``residual``, ``proportion``, ``sample_pairing``,
    ``acc_no_patch``, ``acc_clean_no_patch``.
    """
    clean_path = os.path.join(phase_b_run_dir, _CLEAN_NO_PATCH_FILE)
    no_patch_path = os.path.join(phase_b_run_dir, _NO_PATCH_FILE)
    patch_paths = _list_restoration_patch_files(phase_b_run_dir)

    if not patch_paths:
        raise RuntimeError(
            f"No restoration_patch_* JSONLs found under {phase_b_run_dir}"
        )

    clean = load_condition_correctness(clean_path)
    no_patch_raw = load_condition_correctness(no_patch_path)
    # Rename to drop the restoration_ prefix for display.
    no_patch = ConditionCorrectness(
        condition=_strip_restoration_prefix(no_patch_raw.condition),
        sample_ids=no_patch_raw.sample_ids,
        correct=no_patch_raw.correct,
    )

    patched_by_name: Dict[str, ConditionCorrectness] = {}
    for path in patch_paths:
        raw = load_condition_correctness(path)
        short = _strip_restoration_prefix(raw.condition)
        patched_by_name[short] = ConditionCorrectness(
            condition=short,
            sample_ids=raw.sample_ids,
            correct=raw.correct,
        )

    # Per-condition restoration_effect (+ per-condition residual_effect and
    # restoration_proportion — the reviewer requested NIE/NDE/MP columns per
    # patch condition, not only for the best pick). Alias columns map:
    #   restoration_effect       ≡ NIE   (acc(patched)  - acc(no_patch))
    #   residual_effect          ≡ NDE   (acc(clean)    - acc(patched))
    #   total_effect             ≡ TE    (acc(clean)    - acc(no_patch))
    #   restoration_proportion   ≡ MP    (NIE / TE; null when |TE| < EPSILON_DENOM)
    # Phase C wording MUST avoid formal NIE/NDE/MP/causal-mediation terminology
    # in prose; the aliases exist only in machine-readable columns for audit.
    rows: List[Dict[str, Any]] = []
    # no_patch is included for audit (trivially 0, 0, 0, n, 0).
    ids_all, (np_vec,) = align_by_sample_id(no_patch)
    _, (clean_full_arr,) = align_by_sample_id(clean)
    total_effect_point = float(_acc(clean_full_arr) - _acc(np_vec)) if len(clean_full_arr) == len(np_vec) else None
    # Proper TE on the paired (no_patch ∩ clean) intersection:
    _, (clean_aligned, np_aligned) = _pair_align(clean, no_patch)
    total_effect_aligned = float(_acc(clean_aligned) - _acc(np_aligned))

    rows.append({
        "condition": "no_patch",
        "accuracy": _acc(np_vec),
        "restoration_effect": 0.0,
        "restoration_effect_ci_lo": 0.0,
        "restoration_effect_ci_hi": 0.0,
        "residual_effect": total_effect_aligned,
        "residual_effect_ci_lo": None,
        "residual_effect_ci_hi": None,
        "restoration_proportion": 0.0,
        "restoration_proportion_ci_lo": None,
        "restoration_proportion_ci_hi": None,
        "total_effect": total_effect_aligned,
        # NIE/NDE/MP aliases for downstream cross-audience tools.
        "alias_NIE": 0.0,
        "alias_NDE": total_effect_aligned,
        "alias_TE": total_effect_aligned,
        "alias_MP": 0.0,
        "n_aligned": len(ids_all),
    })

    dropped_per_cond: Dict[str, int] = {}
    for name in sorted(patched_by_name):
        patched = patched_by_name[name]
        re_res = restoration_effect(
            patched, no_patch,
            bootstrap_n=bootstrap_n, seed=seed, ci=ci,
        )
        dropped_per_cond[name] = re_res["n_dropped_ids"]

        # Per-condition residual_effect (NDE) = acc(clean) - acc(patched).
        nde_res = residual_effect(
            clean, patched,
            bootstrap_n=bootstrap_n, seed=seed, ci=ci,
        )
        # Per-condition restoration_proportion (MP) = NIE / TE.
        mp_res = restoration_proportion(
            clean, no_patch, patched,
            bootstrap_n=bootstrap_n, seed=seed, ci=ci,
            epsilon_denom=epsilon_denom,
        )

        _, (arr,) = align_by_sample_id(patched)
        rows.append({
            "condition": name,
            "accuracy": _acc(arr),
            "restoration_effect": re_res["point"],
            "restoration_effect_ci_lo": re_res["ci_lo"],
            "restoration_effect_ci_hi": re_res["ci_hi"],
            "residual_effect": nde_res["point"],
            "residual_effect_ci_lo": nde_res["ci_lo"],
            "residual_effect_ci_hi": nde_res["ci_hi"],
            "restoration_proportion": mp_res["point"],
            "restoration_proportion_ci_lo": mp_res["ci_lo"],
            "restoration_proportion_ci_hi": mp_res["ci_hi"],
            "restoration_proportion_ci_reason": mp_res["ci_reason"],
            "total_effect": total_effect_aligned,
            # NIE/NDE/MP aliases — audit-only; prose uses mapped terminology.
            "alias_NIE": re_res["point"],
            "alias_NDE": nde_res["point"],
            "alias_TE": total_effect_aligned,
            "alias_MP": mp_res["point"],
            "n_aligned": re_res["n_aligned"],
        })

    # Claim-eligible subset and best-condition pick.
    eligible: List[Tuple[str, float]] = [
        (r["condition"], r["restoration_effect"])
        for r in rows
        if r["condition"] in CLAIM_ELIGIBLE_CONDITIONS
    ]
    if not eligible:
        raise RuntimeError(
            "No claim-eligible restoration conditions present in Phase B run "
            f"(expected at least one of {CLAIM_ELIGIBLE_CONDITIONS})."
        )
    # argmax with alphabetical tie-break. sorted() by (-effect, name) is the
    # deterministic pick.
    eligible_sorted = sorted(eligible, key=lambda t: (-t[1], t[0]))
    best_name = eligible_sorted[0][0]
    best_cond = patched_by_name[best_name]

    residual = residual_effect(
        clean, best_cond,
        bootstrap_n=bootstrap_n, seed=seed, ci=ci,
    )
    proportion = restoration_proportion(
        clean, no_patch, best_cond,
        bootstrap_n=bootstrap_n, seed=seed, ci=ci,
        epsilon_denom=epsilon_denom,
    )

    # Accuracies on the aligned-to-clean set for the summary.
    _, (clean_arr,) = align_by_sample_id(clean)
    acc_clean_no_patch = _acc(clean_arr)
    _, (np_arr,) = align_by_sample_id(no_patch)
    acc_no_patch = _acc(np_arr)

    return {
        "rows": rows,
        "best_condition": best_name,
        "residual": residual,
        "proportion": proportion,
        "sample_pairing": {
            "aligned_n_best_vs_clean": residual["n_aligned"],
            "aligned_n_best_vs_no_patch": next(
                (r["n_aligned"] for r in rows if r["condition"] == best_name), None,
            ),
            "dropped_ids_per_condition": dropped_per_cond,
            "n_clean_no_patch": len(clean.sample_ids),
            "n_no_patch": len(no_patch.sample_ids),
        },
        "acc_no_patch": acc_no_patch,
        "acc_clean_no_patch": acc_clean_no_patch,
    }
