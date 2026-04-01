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


@dataclass
class DatasetConfig:
    name: str = "juletxara/mgsm"
    lang: str = "te"
    split: str = "test"
    debug_n: Optional[int] = None


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
        """Run validation checks on the config."""
        for b in self.boundary_grid:
            if b >= self.t_fixed:
                raise ValueError(f"Boundary b={b} must be less than t_fixed={self.t_fixed}")
            if b <= 0:
                raise ValueError(f"Boundary b={b} must be positive")
        if self.reference.b_ref >= self.reference.t_ref:
            raise ValueError("Reference b_ref must be less than t_ref")
        if self.dataset.debug_n is not None and self.dataset.debug_n <= 0:
            raise ValueError("debug_n must be positive or null")


def load_config(config_path: str) -> Stage1Config:
    """Load and validate a Stage 1 config from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
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
