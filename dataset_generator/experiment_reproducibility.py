"""Phase 1 — Reproducibility.

A single place that does the four concrete things a reproducible
experiment report needs: seed everything, record the exact configuration
used, record the software environment, and know where artifacts were
saved. Kept as one flat module (not a package) since it's four small,
unrelated-to-each-other utility functions, not a subsystem.
"""

from __future__ import annotations

import platform
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

TRACKED_PACKAGES = (
    "pydantic", "numpy", "pandas", "scipy", "pyarrow", "scikit-learn",
    "joblib", "langgraph", "matplotlib", "pytest",
)


def set_global_seed(seed: int) -> None:
    """Seeds every RNG this repository's *evaluation/experiment* code reads
    from a global source. Note this is deliberately separate from the core
    pipeline's own determinism mechanism (`RNGStreams`, seeded per-concern
    from `GeneratorConfig.seed` — see `docs/DESIGN_DECISIONS.md`), which
    remains the source of truth for dataset generation itself. This
    function covers the two *global* RNGs (`random`, NumPy's legacy global
    generator) that ad hoc experiment/report code might otherwise touch
    without a seed.
    """

    random.seed(seed)
    np.random.seed(seed)


def collect_environment_info() -> dict[str, Any]:
    """Package versions + timestamp + interpreter info — Phase 1's
    "log software versions and timestamps," so an experiment report is
    reproducible against the exact environment that produced it, not just
    the config.
    """

    package_versions: dict[str, str] = {}
    for package_name in TRACKED_PACKAGES:
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = "not installed"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions,
    }


def build_reproducibility_report(seed: int, config_summary: dict[str, Any]) -> dict[str, Any]:
    """Everything Phase 1 asks to be recorded, in one JSON-serializable dict —
    callers write this alongside whatever else an experiment report contains.
    """

    return {
        "seed": seed,
        "config": config_summary,
        "environment": collect_environment_info(),
    }
