"""Evaluation: accuracy metrics, BDS-delta correlation, systematic criteria checks.

Key design decisions (paper-grade validity):
- H1 uses degradation = max(0, baseline - condition), NOT abs(delta)
- Criterion 1: bootstrapped CI on accuracy delta (vs no-swap) excludes 0
- Criterion 2: bootstrap distribution of BDS-degradation rank correlation
               must be >80% positive across resamples
- Criterion 3: ordering stable between full-set and valid-only accuracy
"""

import types
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr


# ─── Basic helpers ────────────────────────────────────────────────────────────

def exact_match(gold: str, predicted: Optional[str]) -> bool:
    """Check if predicted answer exactly matches gold after normalization."""
    if predicted is None:
        return False
    return str(gold).strip() == str(predicted).strip()


def get_per_sample_correct(
    samples: List[Dict],
    parsed_results: List[Dict],
) -> List[int]:
    """
    Return a list of 0/1 (in samples order) indicating correct/incorrect.
    Parse failures are treated as incorrect.
    """
    parsed_by_id = {r["sample_id"]: r for r in parsed_results}
    result = []
    for s in samples:
        pr = parsed_by_id.get(s["sample_id"])
        if pr is None or not pr.get("parse_success", False):
            result.append(0)
        elif exact_match(s["gold_answer"], pr.get("normalized_answer")):
            result.append(1)
        else:
            result.append(0)
    return result


def compute_accuracy(
    samples: List[Dict],
    parsed_results: List[Dict],
) -> Dict:
    """
    Compute accuracy metrics for a single condition.

    Returns dict with: accuracy, valid_accuracy, valid_answer_rate,
    parse_failure_rate, n_correct, n_total, n_valid, n_parse_fail.
    """
    parsed_by_id = {r["sample_id"]: r for r in parsed_results}
    n_total = len(samples)
    n_correct = 0
    n_valid = 0
    n_parse_fail = 0

    for s in samples:
        pr = parsed_by_id.get(s["sample_id"])
        if pr is None or not pr.get("parse_success", False):
            n_parse_fail += 1
            continue
        n_valid += 1
        if exact_match(s["gold_answer"], pr.get("normalized_answer")):
            n_correct += 1

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    valid_accuracy = n_correct / n_valid if n_valid > 0 else 0.0

    return {
        "accuracy": accuracy,
        "valid_accuracy": valid_accuracy,
        "valid_answer_rate": n_valid / n_total if n_total > 0 else 0.0,
        "parse_failure_rate": n_parse_fail / n_total if n_total > 0 else 0.0,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_parse_fail": n_parse_fail,
    }


def compute_delta_vs_baseline(
    condition_accuracy: float,
    baseline_accuracy: float,
) -> float:
    """Signed accuracy delta (condition - baseline). Negative = degradation."""
    return condition_accuracy - baseline_accuracy


# ─── Bootstrap helpers ────────────────────────────────────────────────────────

def bootstrap_accuracy_delta_ci(
    baseline_correct: List[int],
    condition_correct: List[int],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap CI on accuracy delta: mean(condition) - mean(baseline).

    Resamples at the per-sample level, giving a CI that reflects both
    accuracy difference AND sample-level uncertainty.

    Returns:
        (ci_lower, ci_upper)
    """
    n = len(baseline_correct)
    bc = np.array(baseline_correct, dtype=float)
    cc = np.array(condition_correct, dtype=float)
    rng = np.random.RandomState(seed)

    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas.append(cc[idx].mean() - bc[idx].mean())

    deltas = np.sort(deltas)
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(deltas, alpha * 100)),
        float(np.percentile(deltas, (1 - alpha) * 100)),
    )


def _rank(values: List[float]) -> List[float]:
    """Assign ranks (1-based, average ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def rank_correlation(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation between two equal-length lists."""
    n = len(x)
    if n < 2:
        return 0.0
    rank_x = _rank(x)
    rank_y = _rank(y)
    d_sq = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def bootstrap_bds_degradation_correlation(
    boundary_grid: List[int],
    baseline_correct: List[int],
    per_cond_correct: Dict[str, List[int]],
    per_cond_bds: Dict[str, List[float]],
    n_bootstrap: int = 1000,
    # Pre-registered operational heuristic: >80% of bootstrap iterations must
    # yield positive BDS-degradation rank correlation for criterion 2 to pass.
    positive_threshold: float = 0.8,
    seed: int = 42,
) -> Tuple[bool, float, float]:
    """
    Bootstrap over samples, recompute BDS-degradation rank correlation each time.

    For each resample:
    - Compute degradation per boundary = max(0, resampled_baseline_acc - resampled_cond_acc)
    - Compute mean BDS per boundary = mean of per-sample BDS on resampled indices
    - Compute rank correlation across boundaries

    Note:
        positive_threshold is a heuristic, not a formal statistical cutoff.
        Treat the result as an operational stability indicator for Stage 1,
        not as a formal statistical test.

    Returns:
        (passes_threshold, positive_rate, mean_rho)
    """
    n = len(baseline_correct)
    bc = np.array(baseline_correct, dtype=float)
    rng = np.random.RandomState(seed)

    correlations = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)

        boot_deg = []
        boot_bds = []
        for b in boundary_grid:
            cond = f"hard_swap_b{b}"
            if cond not in per_cond_correct or cond not in per_cond_bds:
                continue
            cc = np.array(per_cond_correct[cond], dtype=float)
            bds_arr = np.array(per_cond_bds[cond], dtype=float)

            # Degradation: non-negative; positive means the swap hurt accuracy
            deg = max(0.0, bc[idx].mean() - cc[idx].mean())
            mean_bds = bds_arr[idx].mean()

            boot_deg.append(deg)
            boot_bds.append(mean_bds)

        if len(boot_deg) >= 2:
            rho = rank_correlation(boot_bds, boot_deg)
            correlations.append(rho)

    positive_rate = float(np.mean([c > 0 for c in correlations])) if correlations else 0.0
    mean_rho = float(np.mean(correlations)) if correlations else 0.0
    return positive_rate > positive_threshold, positive_rate, mean_rho


# ─── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_experiment(
    samples: List[Dict],
    condition_results: Dict[str, List[Dict]],
    bds_results: Dict[str, Dict],
    boundary_grid: List[int],
    config: dict,
) -> Dict:
    """
    Run full evaluation across all conditions.

    Args:
        samples: Original samples with gold answers.
        condition_results: {condition_name: list of parsed results}.
        bds_results: {condition_name: BDS results dict}.
        boundary_grid: List of boundary values tested.
        config: Dict with keys bootstrap_n, bootstrap_ci, criteria_threshold.

    Returns:
        Comprehensive evaluation results dict.
    """
    bootstrap_n = config.get("bootstrap_n", 1000)
    bootstrap_ci_level = config.get("bootstrap_ci", 0.95)
    criteria_threshold = config.get("criteria_threshold", 2)

    # ── Per-condition accuracy ─────────────────────────────────────────────
    accuracies = {}
    per_sample_correct = {}
    for cond, results in condition_results.items():
        accuracies[cond] = compute_accuracy(samples, results)
        per_sample_correct[cond] = get_per_sample_correct(samples, results)

    baseline_acc = accuracies.get("no_swap", {}).get("accuracy", 0.0)
    baseline_correct = per_sample_correct.get("no_swap", [0] * len(samples))

    # ── Signed deltas (condition - baseline) ──────────────────────────────
    deltas = {}
    for cond, acc in accuracies.items():
        deltas[cond] = compute_delta_vs_baseline(acc["accuracy"], baseline_acc)

    # ── H1: BDS-degradation rank correlation (P0-2: use degradation, not abs) ──
    boundary_degradations = []
    boundary_bds_totals = []
    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        if cond in accuracies and cond in bds_results:
            # Degradation = max(0, baseline - condition). Excludes improvements.
            degradation = max(0.0, baseline_acc - accuracies[cond]["accuracy"])
            boundary_degradations.append(degradation)
            boundary_bds_totals.append(bds_results[cond]["aggregate"]["mean_bds_total"])

    print(f"H1 degradation values: {[round(d, 4) for d in boundary_degradations]}")

    bds_delta_rho = None
    bds_delta_p = None
    if len(boundary_bds_totals) >= 2:
        bds_delta_rho, bds_delta_p = spearmanr(boundary_bds_totals, boundary_degradations)
        bds_delta_rho = float(bds_delta_rho)
        bds_delta_p = float(bds_delta_p)

    # ── Per-condition per-sample BDS (aligned with samples ordering) ──────
    sample_ids = [s["sample_id"] for s in samples]
    per_cond_bds_per_sample: Dict[str, List[float]] = {}
    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        if cond in bds_results:
            bds_by_id = {r["sample_id"]: r["bds_total"] for r in bds_results[cond]["per_sample"]}
            per_cond_bds_per_sample[cond] = [bds_by_id.get(sid, 0.0) for sid in sample_ids]

    # ── Criterion 1: bootstrapped delta CI excludes 0 for ≥1 boundary ────
    delta_cis: Dict[str, Tuple[float, float]] = {}
    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        if cond in per_sample_correct:
            ci_lo, ci_hi = bootstrap_accuracy_delta_ci(
                baseline_correct,
                per_sample_correct[cond],
                n_bootstrap=bootstrap_n,
                ci=bootstrap_ci_level,
                seed=42,
            )
            delta_cis[cond] = (ci_lo, ci_hi)

    print(
        f"Criterion 1 (degradation CI): "
        f"{ {c: f'[{ci[0]:.4f}, {ci[1]:.4f}]' for c, ci in delta_cis.items()} }"
    )

    # H1 is a degradation hypothesis: CI must be entirely negative (ci_hi < 0),
    # meaning the swap consistently hurt accuracy. A CI entirely above 0 would
    # indicate improvement, which is not evidence for H1.
    c1 = any(
        ci_hi < 0
        for (_, ci_hi) in delta_cis.values()
    )
    print(f"  → passes: {c1} (at least one boundary has ci_hi < 0)")

    # ── Criterion 2: bootstrap BDS-degradation rank correlation ───────────
    c2_passes, c2_positive_rate, c2_mean_rho = bootstrap_bds_degradation_correlation(
        boundary_grid=boundary_grid,
        baseline_correct=baseline_correct,
        per_cond_correct=per_sample_correct,
        per_cond_bds=per_cond_bds_per_sample,
        n_bootstrap=bootstrap_n,
        positive_threshold=0.8,
        seed=42,
    )
    print(f"Criterion 2: positive rate = {c2_positive_rate:.3f}")

    # ── Criterion 3: ordering consistent between full and valid-only ──────
    full_ordering = sorted(
        boundary_grid,
        key=lambda b: accuracies.get(f"hard_swap_b{b}", {}).get("accuracy", 0.0),
    )
    valid_ordering = sorted(
        boundary_grid,
        key=lambda b: accuracies.get(f"hard_swap_b{b}", {}).get("valid_accuracy", 0.0),
    )
    c3 = full_ordering == valid_ordering

    n_criteria_met = sum([c1, c2_passes, c3])
    passed = n_criteria_met >= criteria_threshold

    criteria = {
        "criterion_1_delta_ci_excludes_zero": c1,
        "criterion_2_bootstrap_positive": c2_passes,
        "criterion_2_positive_rate": c2_positive_rate,
        "criterion_2_mean_rho": c2_mean_rho,
        "criterion_3_ordering_consistent": c3,
        "n_criteria_met": n_criteria_met,
        "threshold": criteria_threshold,
        "passed": passed,
    }

    # ── Bootstrap CI on BDS total per condition (saved for reference) ────
    bds_bootstrap_ci: Dict[str, Dict] = {}
    for cond, bds_vals in per_cond_bds_per_sample.items():
        arr = np.array(bds_vals)
        rng = np.random.RandomState(42)
        n = len(arr)
        boot_means = [arr[rng.randint(0, n, size=n)].mean() for _ in range(bootstrap_n)]
        alpha = (1 - bootstrap_ci_level) / 2
        bds_bootstrap_ci[cond] = {
            "mean": float(arr.mean()),
            "ci_lower": float(np.percentile(boot_means, alpha * 100)),
            "ci_upper": float(np.percentile(boot_means, (1 - alpha) * 100)),
        }

    # ── Bootstrap CI on global BDS-delta correlation ──────────────────────
    bds_delta_ci = None
    if len(boundary_bds_totals) >= 2 and per_cond_bds_per_sample:
        rng2 = np.random.RandomState(42)
        n = len(samples)
        bc_arr = np.array(baseline_correct, dtype=float)
        boot_rhos = []
        for _ in range(bootstrap_n):
            idx = rng2.randint(0, n, size=n)
            boot_bds_agg = []
            boot_deg_agg = []
            for b in boundary_grid:
                cond = f"hard_swap_b{b}"
                if cond not in per_cond_bds_per_sample or cond not in per_sample_correct:
                    continue
                bds_arr = np.array(per_cond_bds_per_sample[cond])
                cc = np.array(per_sample_correct[cond], dtype=float)
                boot_bds_agg.append(bds_arr[idx].mean())
                boot_deg_agg.append(max(0.0, bc_arr[idx].mean() - cc[idx].mean()))
            if len(boot_bds_agg) >= 2:
                rho, _ = spearmanr(boot_bds_agg, boot_deg_agg)
                if not np.isnan(rho):
                    boot_rhos.append(float(rho))
        if boot_rhos:
            bds_delta_ci = [
                float(np.percentile(boot_rhos, 2.5)),
                float(np.percentile(boot_rhos, 97.5)),
            ]

    # ── Boundary table ────────────────────────────────────────────────────
    boundary_table = []
    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        row = {
            "boundary": b,
            "condition": cond,
            "accuracy": accuracies.get(cond, {}).get("accuracy"),
            "valid_accuracy": accuracies.get(cond, {}).get("valid_accuracy"),
            "delta_vs_baseline": deltas.get(cond),
            "degradation": max(0.0, baseline_acc - (accuracies.get(cond, {}).get("accuracy") or 0.0)),
        }
        if cond in delta_cis:
            row["delta_ci_lower"], row["delta_ci_upper"] = delta_cis[cond]
        if cond in bds_results:
            row["bds_lower"] = bds_results[cond]["aggregate"]["mean_bds_lower"]
            row["bds_upper"] = bds_results[cond]["aggregate"]["mean_bds_upper"]
            row["bds_total"] = bds_results[cond]["aggregate"]["mean_bds_total"]
        boundary_table.append(row)

    return {
        "accuracies": accuracies,
        "deltas": deltas,
        "baseline_accuracy": baseline_acc,
        "bds_delta_rho": bds_delta_rho,
        "bds_delta_p": bds_delta_p,
        "bds_delta_ci": bds_delta_ci,
        "bds_bootstrap_ci": bds_bootstrap_ci,
        "delta_cis": {c: list(ci) for c, ci in delta_cis.items()},
        "boundary_table": boundary_table,
        "ordering_consistent": c3,
        "criteria": criteria,
        # Convenience top-level fields for notebook
        "criterion_1": c1,
        "criterion_2": c2_passes,
        "criterion_2_rate": c2_positive_rate,
        "criterion_3": c3,
    }


def evaluate_all(
    no_swap: List[Dict],
    sweep: Dict[str, List[Dict]],
    reference: List[Dict],
    control: List[Dict],
    bds: Dict[str, Dict],
    config,
    samples: List[Dict] = None,
) -> types.SimpleNamespace:
    """
    Public API for notebook use. Calls evaluate_experiment and returns a
    SimpleNamespace for attribute-style access.

    Args:
        no_swap: Parsed results for the no_swap condition.
        sweep: {condition_name: parsed_results} for hard_swap_b* conditions.
        reference: Parsed results for reference condition.
        control: Parsed results for random_donor condition.
        bds: BDS results dict keyed by condition name.
        config: Stage1Config object.
        samples: Original dataset samples (required for per-sample correctness).

    Returns:
        SimpleNamespace with all evaluation fields accessible as attributes.
    """
    all_conditions = {"no_swap": no_swap, **sweep, "reference": reference, "random_donor": control}

    eval_config = {
        "bootstrap_n": config.evaluation.bootstrap_n,
        "bootstrap_ci": config.evaluation.bootstrap_ci,
        "criteria_threshold": config.evaluation.criteria_threshold,
    }

    result = evaluate_experiment(
        samples=samples,
        condition_results=all_conditions,
        bds_results=bds,
        boundary_grid=config.boundary_grid,
        config=eval_config,
    )

    return types.SimpleNamespace(**result)
