"""Shared plumbing for the three offline-RL trainers (CQL, IQL, BCQ) —
just the fingerprint convention every config in this repository uses.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel


def compute_config_fingerprint(config: BaseModel) -> str:
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()
