"""Unit tests for compute_random_donor_seed — 3 cases from spec §10.1."""

from stage1.models.composer import compute_random_donor_seed


def test_seed_42_8_12():
    # (42, 8, 12) == 42*1000 + 8*100 + 12 = 42000 + 800 + 12 = 42812
    assert compute_random_donor_seed(42, 8, 12) == 42812


def test_seed_42_12_16():
    # (42, 12, 16) == 42000 + 1200 + 16 = 43216
    assert compute_random_donor_seed(42, 12, 16) == 43216


def test_seed_42_16_20():
    # (42, 16, 20) == 42000 + 1600 + 20 = 43620
    assert compute_random_donor_seed(42, 16, 20) == 43620


def test_formula_structure():
    """Confirm the formula is seed_base*1000 + b*100 + t (not further encoded)."""
    seed_base, b, t = 0, 0, 0
    assert compute_random_donor_seed(seed_base, b, t) == 0
    assert compute_random_donor_seed(1, 0, 0) == 1000
    assert compute_random_donor_seed(0, 1, 0) == 100
    assert compute_random_donor_seed(0, 0, 1) == 1
