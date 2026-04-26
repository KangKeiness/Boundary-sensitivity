"""Stage 1 hardening — boundary_grid duplicate validation.

Stage1Config.validate() must reject duplicate (b) values in boundary_grid.
Fail-closed; a duplicate would silently double-count a condition in BDS,
bootstrap, and ordering metrics.
"""

from __future__ import annotations

import pytest

from stage1.utils.config import (
    DatasetConfig,
    EvaluationConfig,
    GenerationConfig,
    HiddenStateConfig,
    ModelsConfig,
    RandomDonorConfig,
    ReferenceConfig,
    Stage1Config,
)


def _make(boundary_grid):
    return Stage1Config(
        models=ModelsConfig(recipient="r", donor="d"),
        boundary_grid=boundary_grid,
        t_fixed=20,
        reference=ReferenceConfig(b_ref=8, t_ref=20),
        hidden_state=HiddenStateConfig(),
        random_donor=RandomDonorConfig(),
        dataset=DatasetConfig(),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(),
    )


def test_unique_grid_passes():
    cfg = _make([2, 4, 6, 8, 10])
    cfg.validate()  # no raise


def test_duplicate_grid_raises():
    cfg = _make([2, 4, 4, 6, 8])
    with pytest.raises(ValueError, match="duplicate"):
        cfg.validate()


def test_duplicate_repeated_value_listed():
    cfg = _make([4, 4, 4])
    with pytest.raises(ValueError) as excinfo:
        cfg.validate()
    assert "[4]" in str(excinfo.value)


def test_multiple_duplicates_listed():
    cfg = _make([2, 2, 4, 4, 6])
    with pytest.raises(ValueError) as excinfo:
        cfg.validate()
    msg = str(excinfo.value)
    assert "2" in msg and "4" in msg


def test_duplicate_check_runs_before_range_check():
    """Duplicate detection must NOT be masked by other validation failures."""
    # b=20 also fails ``b >= t_fixed`` — but the dup check runs first.
    cfg = _make([8, 8])
    with pytest.raises(ValueError, match="duplicate"):
        cfg.validate()


# ─── Pre-main-run hardening: dataset.lang whitelist ─────────────────────────


def _make_with_lang(lang: str):
    return Stage1Config(
        models=ModelsConfig(recipient="r", donor="d"),
        boundary_grid=[2, 4, 6],
        t_fixed=20,
        reference=ReferenceConfig(b_ref=8, t_ref=20),
        hidden_state=HiddenStateConfig(),
        random_donor=RandomDonorConfig(),
        dataset=DatasetConfig(lang=lang),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(),
    )


def test_dataset_lang_zh_passes():
    """Production lang for the project: must validate."""
    _make_with_lang("zh").validate()


def test_dataset_lang_default_te_passes():
    """The dataclass default lang must remain in the whitelist."""
    cfg = Stage1Config(
        models=ModelsConfig(recipient="r", donor="d"),
        boundary_grid=[2, 4, 6],
        t_fixed=20,
        reference=ReferenceConfig(b_ref=8, t_ref=20),
        hidden_state=HiddenStateConfig(),
        random_donor=RandomDonorConfig(),
        dataset=DatasetConfig(),  # default lang
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(),
    )
    cfg.validate()  # no raise


def test_dataset_lang_unknown_raises():
    """Unknown / mistyped lang must be rejected before the loader is invoked."""
    with pytest.raises(ValueError, match="not a known MGSM language"):
        _make_with_lang("xx").validate()


def test_dataset_lang_empty_string_raises():
    with pytest.raises(ValueError, match="not a known MGSM language"):
        _make_with_lang("").validate()
