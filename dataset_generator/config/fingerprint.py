"""SHA-256 fingerprinting of a `GeneratorConfig`.

Stored in `metadata.json` at export time (Step 9) so any generated dataset
can later be tied back to the exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json

from dataset_generator.config.schema import GeneratorConfig

# Purely descriptive fields that must NOT affect the fingerprint: two configs
# that differ only in who ran them / why still produce byte-identical data.
_EXCLUDED_FIELDS = {"experiment"}


def compute_fingerprint(config: GeneratorConfig) -> str:
    """Return a stable SHA-256 hex digest of `config`'s generation-determining fields."""

    data = config.model_dump(mode="json", exclude=_EXCLUDED_FIELDS)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
