"""Small validation helpers shared across config modules (not part of the public API)."""

from __future__ import annotations


def validate_probability_mapping(values: dict, label: str, tolerance: float = 1e-3) -> None:
    """Raise `ValueError` unless `values` sums to ~1.0 and every value is in [0, 1]."""

    total = sum(values.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{label} must sum to 1.0 (got {total:.4f})")
    for key, value in values.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{label}[{key!r}] must be within [0, 1] (got {value})")
