"""Configuration loading and validation for Stage 1 pipeline."""

import logging
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format for the Stage 1 pipeline."""
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        level=level,
        force=True,
    )


@dataclass
class ModelsConfig:
    recipient: str
    donor: str
    recipient_revision: Optional[str] = None
    donor_revision: Optional[str] = None


@dataclass
class ReferenceConfig:
    b_ref: int
    t_ref: int


@dataclass
class HiddenStateConfig:
    pooling: str = "last_token"

    def __post_init__(self):
        if self.pooling not in ("last_token", "mean"):
            raise ValueError(f"Invalid pooling method: {self.pooling}. Must be 'last_token' or 'mean'.")


@dataclass
class RandomDonorConfig:
    mode: str = "same_width_random_source"
    seed: int = 42


# MGSM upstream covers these 11 languages. Any value outside this set will be
# rejected by Stage1Config.validate() to defend against typos / missing
# ``dataset.lang`` keys silently falling back to the dataclass default.
_KNOWN_MGSM_LANGS: set = {
    "bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh",
}


@dataclass
class DatasetConfig:
    name: str = "mgsm"
    lang: str = "te"
    split: str = "test"
    debug_n: Optional[int] = None
    # Stage 1 hardening (2026-04-25): pin dataset to a stable snapshot.
    # ``revision`` controls the HuggingFace ref (commit SHA preferred).
    # ``expected_sha256`` locks the file bytes; mismatch aborts the run.
    revision: Optional[str] = None
    expected_sha256: Optional[str] = None


@dataclass
class GenerationConfig:
    do_sample: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 256


@dataclass
class EvaluationConfig:
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95
    criteria_threshold: int = 2


@dataclass
class Stage1Config:
    models: ModelsConfig
    boundary_grid: List[int]
    t_fixed: int
    reference: ReferenceConfig
    hidden_state: HiddenStateConfig
    random_donor: RandomDonorConfig
    dataset: DatasetConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig

    def validate(self):
        """Run validation checks on the config. Fail-closed."""
        # Stage 1 hardening: reject duplicate boundary values up front. A
        # duplicate would silently double a condition's contribution to the
        # sweep and corrupt downstream aggregates (BDS, bootstrap CI). There
        # is no legitimate reason to declare the same boundary twice.
        if len(set(self.boundary_grid)) != len(self.boundary_grid):
            seen = set()
            dups = []
            for b in self.boundary_grid:
                if b in seen and b not in dups:
                    dups.append(b)
                seen.add(b)
            raise ValueError(
                f"boundary_grid contains duplicate values: {dups}. "
                f"Each boundary must appear exactly once. Got {self.boundary_grid}."
            )
        for b in self.boundary_grid:
            if b >= self.t_fixed:
                raise ValueError(f"Boundary b={b} must be less than t_fixed={self.t_fixed}")
            if b <= 0:
                raise ValueError(f"Boundary b={b} must be positive")
        if self.reference.b_ref >= self.reference.t_ref:
            raise ValueError("Reference b_ref must be less than t_ref")
        if self.dataset.debug_n is not None and self.dataset.debug_n <= 0:
            raise ValueError("debug_n must be positive or null")
        # Pre-main-run hardening: MGSM lang must be one of the 11 upstream
        # languages. Catches typos and missing ``dataset.lang`` keys before
        # the loader tries to fetch a non-existent TSV.
        if self.dataset.lang not in _KNOWN_MGSM_LANGS:
            raise ValueError(
                f"dataset.lang={self.dataset.lang!r} is not a known MGSM "
                f"language. Expected one of {sorted(_KNOWN_MGSM_LANGS)}."
            )


def load_config(config_path: str) -> Stage1Config:
    """Load and validate a Stage 1 config from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # UTF-8 enforced: config loading must not depend on OS locale (Windows cp949).
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = Stage1Config(
        models=ModelsConfig(**raw["models"]),
        boundary_grid=raw["boundary_grid"],
        t_fixed=raw["t_fixed"],
        reference=ReferenceConfig(**raw["reference"]),
        hidden_state=HiddenStateConfig(**raw.get("hidden_state", {})),
        random_donor=RandomDonorConfig(**raw.get("random_donor", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )
    config.validate()
    return config
