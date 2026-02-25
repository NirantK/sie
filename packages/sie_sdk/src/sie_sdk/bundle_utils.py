from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def match_bundle_models(bundle_path: Path, models_dir: Path) -> list[str]:
    """Match models to a bundle by adapter module paths.

    Loads the bundle YAML to get its adapter module list, then scans
    model config YAMLs to find models whose adapter_path module matches.

    Args:
        bundle_path: Path to the bundle YAML file.
        models_dir: Path to the models directory containing *.yaml configs.

    Returns:
        List of model names (sie_id or derived from filename) whose adapters
        match the bundle's adapter list.
    """
    with bundle_path.open() as f:
        data = yaml.safe_load(f) or {}

    adapter_modules = set(data.get("adapters", []))
    if not adapter_modules:
        return []

    if not models_dir.exists():
        return []

    matched_models: list[str] = []
    for model_path in sorted(models_dir.glob("*.yaml")):
        try:
            model_data = yaml.safe_load(model_path.read_text()) or {}
        except Exception:  # noqa: BLE001
            logger.warning("Failed to parse model config %s", model_path.name, exc_info=True)
            continue
        profiles = model_data.get("profiles", {})
        for profile in profiles.values():
            adapter_path = profile.get("adapter_path", "")
            module_path = adapter_path.split(":", maxsplit=1)[0]
            if module_path in adapter_modules:
                model_name = model_data.get("sie_id", model_path.stem.replace("__", "/"))
                matched_models.append(model_name)
                break

    return matched_models
