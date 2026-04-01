"""Evaluation: accuracy metrics, BDS-delta correlation, systematic criteria check."""

import random as pyrandom
from typing import Dict, List, Optional, Tuple

import numpy as np


def exact_match(gold: str, predicted: Optional[str]) -> bool:
    """Check if predicted answer exactly matches gold after normalization."""
    if predicted is None:
        return False
    return str(gold).strip() == str(predicted).strip()


def compute_accuracy(
    samples: List[Dict],
    parsed_results: List[Dict],
) -> Dict:
    """
    Compute accuracy metrics for a single condition.

    Args:
        samples: List of {sample_id, prompt, gold_answer}.
        parsed_results: List of {sample_id, output_text, parsed_answer, parse_success, normalized_answer}.

    Returns:
        Dict with accuracy, valid_answer_rate, parse_failure_rate, n_correct, n_total, n_valid, n_parse_fail.
    """
    parsed_by_id = {r["sample_id"]: r for r in parsed_results}
    n_total = len(samples)
    n_correct = 0
    n_valid = 0
    n_parse_fail = 0

    for s in samples:
        sid = s["sample_id"]
        pr = parsed_by_id.get(sid)
        if pr is None:
            n_parse_fail += 1
            continue
        if not pr["parse_success"]:
            n_parse_fail += 1
            continue
        n_valid += 1
        if exact_match(s["gold_answer"], pr["normalized_answer"]):
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
    """Compute accuracy delta vs no-swap baseline."""
    return condition_accuracy - baseline_accuracy


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the mean.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)
    means = []
    for _ in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        means.append(sample.mean())
    means = np.sort(means)
    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return float(arr.mean()), float(lower), float(upper)


def rank_correlation(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation between two lists."""
    n = len(x)
    if n < 2:
        return 0.0
    rank_x = _rank(x)
    rank_y = _rank(y)
    d_sq = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def _rank(values: List[float]) -> List[float]:
    """Assign ranks (1-based, average ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


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
        config: Evaluation config dict.

    Returns:
        Comprehensive evaluation results dict.
    """
    bootstrap_n = config.get("bootstrap_n", 1000)
    bootstrap_ci_level = config.get("bootstrap_ci", 0.95)

    # Compute per-condition accuracy
    accuracies = {}
    for cond, results in condition_results.items():
        accuracies[cond] = compute_accuracy(samples, results)

    baseline_acc = accuracies.get("no_swap", {}).get("accuracy", 0.0)

    # Compute deltas
    deltas = {}
    for cond, acc in accuracies.items():
        deltas[cond] = compute_delta_vs_baseline(acc["accuracy"], baseline_acc)

    # BDS-delta rank correlation
    boundary_deltas = []
    boundary_bds_totals = []
    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        if cond in deltas and cond in bds_results:
            boundary_deltas.append(abs(deltas[cond]))
            boundary_bds_totals.append(bds_results[cond]["aggregate"]["mean_bds_total"])

    bds_delta_corr = rank_correlation(boundary_bds_totals, boundary_deltas) if len(boundary_deltas) >= 2 else None

    # Bootstrap CI on BDS-delta correlation
    bds_delta_bootstrap = None
    if bds_results:
        per_sample_bds = {}
        for b in boundary_grid:
            cond = f"hard_swap_b{b}"
            if cond in bds_results:
                per_sample_bds[cond] = [
                    s["bds_total"] for s in bds_results[cond]["per_sample"]
                ]

        if per_sample_bds:
            # Bootstrap CI on mean BDS total for each condition
            bds_delta_bootstrap = {}
            for cond, values in per_sample_bds.items():
                mean, ci_lo, ci_hi = bootstrap_ci(
                    values, n_bootstrap=bootstrap_n, ci=bootstrap_ci_level
                )
                bds_delta_bootstrap[cond] = {
                    "mean": mean,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                }

    # Boundary delta table
    boundary_table = []
    for b in boundary_grid:
        cond = f"hard_swap_b{b}"
        row = {
            "boundary": b,
            "condition": cond,
            "accuracy": accuracies.get(cond, {}).get("accuracy"),
            "valid_accuracy": accuracies.get(cond, {}).get("valid_accuracy"),
            "delta_vs_baseline": deltas.get(cond),
        }
        if cond in bds_results:
            row["bds_lower"] = bds_results[cond]["aggregate"]["mean_bds_lower"]
            row["bds_upper"] = bds_results[cond]["aggregate"]["mean_bds_upper"]
            row["bds_total"] = bds_results[cond]["aggregate"]["mean_bds_total"]
        boundary_table.append(row)

    # Ordering consistency: full-set vs valid-only accuracy ordering
    full_ordering = sorted(boundary_grid, key=lambda b: accuracies.get(f"hard_swap_b{b}", {}).get("accuracy", 0.0))
    valid_ordering = sorted(boundary_grid, key=lambda b: accuracies.get(f"hard_swap_b{b}", {}).get("valid_accuracy", 0.0))
    ordering_consistent = full_ordering == valid_ordering

    # Systematic criteria check (2/3)
    criteria = _check_criteria(
        bds_delta_bootstrap=bds_delta_bootstrap,
        bds_delta_corr=bds_delta_corr,
        ordering_consistent=ordering_consistent,
        threshold=config.get("criteria_threshold", 2),
    )

    return {
        "accuracies": accuracies,
        "deltas": deltas,
        "baseline_accuracy": baseline_acc,
        "bds_delta_rank_correlation": bds_delta_corr,
        "bds_bootstrap_ci": bds_delta_bootstrap,
        "boundary_table": boundary_table,
        "ordering_consistent": ordering_consistent,
        "criteria": criteria,
    }


def _check_criteria(
    bds_delta_bootstrap: Optional[Dict],
    bds_delta_corr: Optional[float],
    ordering_consistent: bool,
    threshold: int = 2,
) -> Dict:
    """
    Check the 3 systematic criteria (need 2/3 to pass).

    1. At least one boundary has BDS delta with bootstrap CI excluding 0
    2. Boundary rank correlation is consistently positive
    3. Ordering stable between full-set and valid-only accuracy
    """
    criteria_met = []

    # Criterion 1: at least one boundary CI excludes 0
    c1 = False
    if bds_delta_bootstrap:
        for cond, ci_info in bds_delta_bootstrap.items():
            if ci_info["ci_lower"] > 0 or ci_info["ci_upper"] < 0:
                c1 = True
                break
    criteria_met.append(c1)

    # Criterion 2: rank correlation consistently positive
    c2 = bds_delta_corr is not None and bds_delta_corr > 0
    criteria_met.append(c2)

    # Criterion 3: ordering consistency
    c3 = ordering_consistent
    criteria_met.append(c3)

    n_met = sum(criteria_met)
    passed = n_met >= threshold

    return {
        "criterion_1_ci_excludes_zero": c1,
        "criterion_2_positive_correlation": c2,
        "criterion_3_ordering_consistent": c3,
        "n_criteria_met": n_met,
        "threshold": threshold,
        "passed": passed,
    }
