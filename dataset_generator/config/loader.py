"""Load/save `GeneratorConfig` to and from YAML or JSON files.

Keeps the config system usable both programmatically (`default_config()`)
and from a config file for reproducible, reviewable experiment runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dataset_generator.config.defaults import default_config
from dataset_generator.config.schema import GeneratorConfig

_YAML_SUFFIXES = {".yaml", ".yml"}
_JSON_SUFFIXES = {".json"}


def load_config(path: str | Path | None = None) -> GeneratorConfig:
    """Load a `GeneratorConfig`.

    If `path` is None, returns the Stage 2 reference default configuration.
    Otherwise reads a YAML or JSON file (by extension) and validates it into
    a `GeneratorConfig`, raising `pydantic.ValidationError` on malformed or
    inconsistent configuration.
    """

    if path is None:
        return default_config()

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"config file not found: {file_path}")

    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")

    if suffix in _YAML_SUFFIXES:
        data = yaml.safe_load(text)
    elif suffix in _JSON_SUFFIXES:
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported config file extension {suffix!r} (use .yaml/.yml/.json)")

    return GeneratorConfig.model_validate(data)


def save_config(config: GeneratorConfig, path: str | Path) -> None:
    """Serialize a `GeneratorConfig` to YAML or JSON, inferred from `path`'s extension."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json")

    suffix = file_path.suffix.lower()
    if suffix in _YAML_SUFFIXES:
        file_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    elif suffix in _JSON_SUFFIXES:
        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"unsupported config file extension {suffix!r} (use .yaml/.yml/.json)")
