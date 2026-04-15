"""Unit tests for parse_condition_bt — all 13 grid mappings + unknown raises ValueError."""

import pytest
from stage1.models.composer import parse_condition_bt


class _FakeConfig:
    """Minimal config stub with t_fixed for Stage 1 style conditions."""
    t_fixed = 20


_CFG = _FakeConfig()


# ── Spec §10.1 — 13 mappings ──────────────────────────────────────────────────

def test_no_swap():
    assert parse_condition_bt("no_swap") == ("no_swap", None, None)


def test_hard_swap_b8():
    assert parse_condition_bt("hard_swap_b8", _CFG) == ("hard_swap", 8, 20)


def test_random_donor_b8():
    assert parse_condition_bt("random_donor_b8", _CFG) == ("random_donor", 8, 20)


def test_fixed_w4_pos1():
    assert parse_condition_bt("fixed_w4_pos1") == ("hard_swap", 4, 8)


def test_fixed_w4_pos2():
    assert parse_condition_bt("fixed_w4_pos2") == ("hard_swap", 8, 12)


def test_fixed_w4_pos3():
    assert parse_condition_bt("fixed_w4_pos3") == ("hard_swap", 12, 16)


def test_fixed_w4_pos4():
    assert parse_condition_bt("fixed_w4_pos4") == ("hard_swap", 16, 20)


def test_fixed_b8_w2():
    assert parse_condition_bt("fixed_b8_w2") == ("hard_swap", 8, 10)


def test_fixed_b8_w4():
    assert parse_condition_bt("fixed_b8_w4") == ("hard_swap", 8, 12)


def test_fixed_b8_w6():
    assert parse_condition_bt("fixed_b8_w6") == ("hard_swap", 8, 14)


def test_fixed_b8_w8():
    assert parse_condition_bt("fixed_b8_w8") == ("hard_swap", 8, 16)


def test_random_fixed_w4_pos3():
    assert parse_condition_bt("random_fixed_w4_pos3") == ("random_donor", 12, 16)


def test_random_fixed_b8_w6():
    assert parse_condition_bt("random_fixed_b8_w6") == ("random_donor", 8, 14)


# ── Unknown condition raises ValueError ──────────────────────────────────────

def test_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Cannot parse condition name"):
        parse_condition_bt("totally_unknown_condition_xyz")


def test_hard_swap_without_config_raises():
    """hard_swap_b{X} without config (t_fixed=None) must raise ValueError."""
    with pytest.raises(ValueError):
        parse_condition_bt("hard_swap_b8", None)
