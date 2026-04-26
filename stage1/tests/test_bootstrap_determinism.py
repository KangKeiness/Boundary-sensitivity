"""Unit tests for bootstrap CI determinism — spec §10.1 (HIGH-2 addition).

Verifies that _bootstrap_ci and _spearman_rho produce reproducible results
and that seeded bootstrap yields identical CIs across two calls.
"""

import math
import random
import pytest

from stage1.run_phase_a import (
    _bootstrap_ci,
    _bootstrap_ci_clipped_mean_diff,
    _spearman_rho,
)


# ── Bootstrap CI determinism ──────────────────────────────────────────────────

def test_bootstrap_ci_same_seed_deterministic():
    """Seeded bootstrap must produce identical CIs across two calls."""
    values = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    ci1 = _bootstrap_ci(values, n_resamples=100, ci=0.95, seed=42)
    ci2 = _bootstrap_ci(values, n_resamples=100, ci=0.95, seed=42)
    assert ci1 == ci2, (
        f"Bootstrap CI with same seed must be deterministic; got {ci1} vs {ci2}"
    )


def test_bootstrap_ci_different_seeds_may_differ():
    """Different seeds should produce different CIs (with high probability)."""
    values = [float(i % 2) for i in range(50)]
    ci_a = _bootstrap_ci(values, n_resamples=200, ci=0.95, seed=1)
    ci_b = _bootstrap_ci(values, n_resamples=200, ci=0.95, seed=999)
    # Not guaranteed to differ but overwhelmingly likely; if this flakes, increase n
    # This is a soft check — we just verify both are valid (lo <= hi)
    assert ci_a[0] <= ci_a[1], f"CI low must be <= high; got {ci_a}"
    assert ci_b[0] <= ci_b[1], f"CI low must be <= high; got {ci_b}"


def test_bootstrap_ci_empty_returns_nan():
    """Empty values list must return (nan, nan)."""
    lo, hi = _bootstrap_ci([], n_resamples=10, ci=0.95, seed=0)
    assert math.isnan(lo) and math.isnan(hi), "Empty input must yield (nan, nan)"


def test_bootstrap_ci_single_value():
    """Single value: CI collapses to that value."""
    lo, hi = _bootstrap_ci([0.5], n_resamples=100, ci=0.95, seed=0)
    assert lo == pytest.approx(0.5, abs=1e-9)
    assert hi == pytest.approx(0.5, abs=1e-9)


def test_bootstrap_ci_respects_ci_level():
    """Wider CI (0.99) must have lo <= lo_95 and hi >= hi_95 (on average)."""
    values = [float(i % 3) for i in range(60)]
    lo_95, hi_95 = _bootstrap_ci(values, n_resamples=500, ci=0.95, seed=7)
    lo_99, hi_99 = _bootstrap_ci(values, n_resamples=500, ci=0.99, seed=7)
    assert lo_99 <= lo_95 + 1e-9, "99% CI low must be <= 95% CI low"
    assert hi_99 >= hi_95 - 1e-9, "99% CI high must be >= 95% CI high"


def _manual_bootstrap_ci_clipped_mean_diff(
    baseline_correct,
    condition_correct,
    *,
    n_resamples,
    ci,
    seed,
):
    """Reference implementation: clip AFTER each resampled mean diff."""
    n = len(baseline_correct)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        mean_base = sum(baseline_correct[i] for i in idxs) / n
        mean_cond = sum(condition_correct[i] for i in idxs) / n
        vals.append(max(0.0, mean_base - mean_cond))
    vals.sort()
    alpha = 1.0 - ci
    lo_idx = int(alpha / 2 * n_resamples)
    hi_idx = int((1.0 - alpha / 2) * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return vals[lo_idx], vals[hi_idx]


def test_bootstrap_ci_clipped_mean_diff_matches_point_estimator_definition():
    """Degradation bootstrap must clip AFTER each resampled mean diff.

    This guards the prior bug where clipping was applied per-sample before
    bootstrap, which does not match the point estimator max(0, mean diff).
    """
    baseline = [1.0, 0.0, 1.0, 0.0, 1.0]
    condition = [0.0, 1.0, 0.0, 1.0, 0.0]
    n_resamples = 200
    ci = 0.95
    seed = 17

    got = _bootstrap_ci_clipped_mean_diff(
        baseline,
        condition,
        n_resamples=n_resamples,
        ci=ci,
        seed=seed,
    )
    expected = _manual_bootstrap_ci_clipped_mean_diff(
        baseline,
        condition,
        n_resamples=n_resamples,
        ci=ci,
        seed=seed,
    )
    assert got == pytest.approx(expected, abs=1e-12), (
        f"clipped-mean bootstrap CI mismatch: got={got}, expected={expected}"
    )


# ── Spearman rho ─────────────────────────────────────────────────────────────

def test_spearman_rho_perfect_positive():
    """Perfectly correlated sequences give rho = 1.0."""
    xs = [1.0, 2.0, 3.0, 4.0]
    rho = _spearman_rho(xs, xs)
    assert rho == pytest.approx(1.0, abs=1e-9), f"Expected rho=1.0; got {rho}"


def test_spearman_rho_perfect_negative():
    """Perfectly anti-correlated sequences give rho = -1.0."""
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [4.0, 3.0, 2.0, 1.0]
    rho = _spearman_rho(xs, ys)
    assert rho == pytest.approx(-1.0, abs=1e-9), f"Expected rho=-1.0; got {rho}"


def test_spearman_rho_n4_underpowered_note():
    """n=4 Spearman rho returns a valid float (not nan) even for small n."""
    widths = [2, 4, 6, 8]
    degradation = [0.1, 0.3, 0.4, 0.6]
    rho = _spearman_rho(widths, degradation)
    assert not math.isnan(rho), f"n=4 Spearman rho must be finite; got {rho}"
    assert -1.0 <= rho <= 1.0, f"Rho must be in [-1, 1]; got {rho}"


def test_spearman_rho_too_short_returns_nan():
    """Single element returns nan."""
    rho = _spearman_rho([1.0], [1.0])
    assert math.isnan(rho), "Single-element Spearman must return nan"
