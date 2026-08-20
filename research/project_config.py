"""Shared paths and immutable experiment defaults.

Changing values in this module changes where data or outputs are read and
written, not the modeling algorithm. Keep ``SEED``, split settings, and model
scripts unchanged when reproducing the historical experiment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

SEED = 1
TEST_FRAC = 0.15
N_SPLITS = 3
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_NAMES = ("final_data_100k_64.parquet", "final_data_64.parquet")


def resolve_data_dir() -> Path:
    """Return the first available data directory without changing data."""
    candidates = [
        os.environ.get("MFA_DATA_DIR"),
        "/mnt/d/LJH/data",
        r"D:\LJH\data",
        str(PROJECT_ROOT),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return PROJECT_ROOT


def resolve_parquet_path(names: Iterable[str] = DEFAULT_DATASET_NAMES) -> Path:
    """Resolve an explicit data path or a known historical dataset filename."""
    explicit = os.environ.get("MFA_DATASET_PATH")
    if explicit:
        return Path(explicit)

    data_dir = resolve_data_dir()
    candidates = [data_dir / name for name in names]
    candidates.extend(PROJECT_ROOT / name for name in names)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_source_path(name: str) -> Path:
    """Resolve a raw input file from the configured data location."""
    data_candidate = resolve_data_dir() / name
    project_candidate = PROJECT_ROOT / name
    return data_candidate if data_candidate.exists() or not project_candidate.exists() else project_candidate


def resolve_results_dir(script_path: str | Path) -> Path:
    """Use an explicit result directory or a repository-local default."""
    configured = os.environ.get("MFA_RESULTS_DIR")
    return Path(configured) if configured else Path(script_path).resolve().parent / "predict"
