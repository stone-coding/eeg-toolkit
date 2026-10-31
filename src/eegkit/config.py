"""Pipeline configuration, loaded from YAML and validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Band = tuple[float, float]

# Medial sensorimotor channels: used for artifact rejection and CSP decoding. Frontal/lateral
# channels carry blink and muscle artifacts that do not bear on C3/C4 mu/beta dynamics.
SENSORIMOTOR = [
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Model):
    raw_dir: Path = Path("data/raw")
    bids_root: Path = Path("data/bids")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("outputs/cache")
    catalog_db: Path = Path("outputs/catalog.sqlite")


class PreprocessConfig(_Model):
    l_freq: float = Field(1.0, ge=0)
    h_freq: float = Field(40.0, gt=0)
    notch_freqs: list[float] = [60.0]
    flat_std_v: float = Field(1e-7, gt=0, description="channels with std below this are flat")
    noisy_z: float = Field(4.0, gt=0, description="robust z-score of log-variance for noisy")
    interpolate_bads: bool = True

    @model_validator(mode="after")
    def _check_band(self) -> PreprocessConfig:
        if self.l_freq >= self.h_freq:
            raise ValueError("l_freq must be smaller than h_freq")
        return self


class EpochConfig(_Model):
    tmin: float = -1.5
    tmax: float = 4.5
    baseline: tuple[float, float] | None = (-1.5, 0.0)
    reject_uv: float = Field(300.0, gt=0, description="peak-to-peak rejection threshold, µV")
    reject_channels: list[str] = SENSORIMOTOR

    @model_validator(mode="after")
    def _check_window(self) -> EpochConfig:
        if self.tmin >= self.tmax:
            raise ValueError("tmin must be smaller than tmax")
        if self.baseline and not (self.tmin <= self.baseline[0] < self.baseline[1] <= self.tmax):
            raise ValueError("baseline must lie within [tmin, tmax] and be ordered")
        return self


class DecodingConfig(_Model):
    tmin: float = 0.5
    tmax: float = 2.5
    fmin: float = 8.0
    fmax: float = 30.0
    channels: list[str] = SENSORIMOTOR
    n_components: int = Field(4, ge=2)
    n_splits: int = Field(5, ge=2)
    seed: int = 42


class AnalysisConfig(_Model):
    bands: dict[str, Band] = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
    band_window: tuple[float, float] = (0.5, 3.5)
    erders_channels: list[str] = ["C3", "C4"]
    erders_freqs: tuple[float, float] = (6.0, 30.0)
    erders_baseline: tuple[float, float] = (-1.0, 0.0)
    erders_crop: tuple[float, float] = (-1.0, 4.0)
    decoding: DecodingConfig = DecodingConfig()

    @field_validator("bands")
    @classmethod
    def _bands_ordered(cls, v: dict[str, Band]) -> dict[str, Band]:
        for name, (lo, hi) in v.items():
            if not 0 <= lo < hi:
                raise ValueError(f"band {name!r} must satisfy 0 <= low < high")
        return v


class Config(_Model):
    subjects: list[int] = list(range(1, 11))
    runs: list[int] = [4, 8, 12]
    paths: PathsConfig = PathsConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    epochs: EpochConfig = EpochConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    log_level: str = "INFO"

    @field_validator("subjects")
    @classmethod
    def _subjects_valid(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("at least one subject is required")
        if any(not 1 <= s <= 109 for s in v) or len(set(v)) != len(v):
            raise ValueError("subjects must be unique integers in 1..109")
        return v

    @field_validator("runs")
    @classmethod
    def _runs_valid(cls, v: list[int]) -> list[int]:
        if not v or any(r not in (4, 8, 12) for r in v):
            raise ValueError("runs must be a non-empty subset of {4, 8, 12} (left/right imagery)")
        return v


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load and validate a YAML config (defaults when ``path`` is None)."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    data.update(overrides)
    return Config.model_validate(data)
