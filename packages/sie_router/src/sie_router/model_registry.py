"""Router ModelRegistry - Source of truth for model→bundle mappings.

This module provides the ModelRegistry class which:
- Loads all bundle configs (TOML files) and model configs (YAML files)
- Computes model→bundle mappings based on bundle.models lists
- Resolves which bundle to use for a model request (priority-based or explicit override)
- Provides complete model catalog for /v1/models endpoint (even with zero workers)

See product/design.md Section 10.8 and tmp/sie_router_modelregistry_design_v3.md.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Model not found in any bundle (HTTP 404)."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model not found: {model}")


class BundleConflictError(Exception):
    """Bundle override incompatible with model (HTTP 409)."""

    def __init__(self, model: str, bundle: str, compatible_bundles: list[str]) -> None:
        self.model = model
        self.bundle = bundle
        self.compatible_bundles = compatible_bundles
        super().__init__(
            f"Bundle '{bundle}' does not support model '{model}'. Compatible bundles: {compatible_bundles}"
        )


@dataclass
class BundleInfo:
    """Information about a bundle."""

    name: str
    priority: int
    models: list[str] = field(default_factory=list)
    default: bool = False


@dataclass
class ModelInfo:
    """Information about a model and its compatible bundles."""

    name: str
    bundles: list[str] = field(default_factory=list)  # Ordered by priority (best first)


class ModelRegistry:
    """Source of truth for model→bundle mappings.

    Thread-safe registry that loads bundle and model configurations
    and provides bundle resolution for routing decisions.

    Attributes:
        bundles_dir: Path to bundles directory.
        models_dir: Path to models directory.
    """

    def __init__(
        self,
        bundles_dir: Path | str,
        models_dir: Path | str,
        *,
        auto_load: bool = True,
    ) -> None:
        """Initialize ModelRegistry.

        Args:
            bundles_dir: Path to directory containing bundle TOML files.
            models_dir: Path to directory containing model configs.
            auto_load: If True, load configs immediately. Set False for testing.
        """
        self._bundles_dir = Path(bundles_dir) if isinstance(bundles_dir, str) else bundles_dir
        self._models_dir = Path(models_dir) if isinstance(models_dir, str) else models_dir

        # Protected by lock for thread-safe reload
        self._lock = threading.RLock()
        self._bundles: dict[str, BundleInfo] = {}
        self._models: dict[str, ModelInfo] = {}
        self._model_names_lower: dict[str, str] = {}  # lowercase → canonical

        if auto_load:
            self.reload()

    @property
    def bundles_dir(self) -> Path:
        """Path to bundles directory."""
        return self._bundles_dir

    @property
    def models_dir(self) -> Path:
        """Path to models directory."""
        return self._models_dir

    def reload(self) -> None:
        """Reload all configs from disk.

        Thread-safe: acquires lock before modifying state.
        """
        with self._lock:
            self._bundles.clear()
            self._models.clear()
            self._model_names_lower.clear()

            self._load_bundles()
            self._load_models()
            self._compute_mappings()

            logger.info(
                "ModelRegistry loaded: %d bundles, %d models",
                len(self._bundles),
                len(self._models),
            )

    def _load_bundles(self) -> None:
        """Load all bundle TOML files from bundles directory."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        if not self._bundles_dir.exists():
            logger.warning("Bundles directory not found: %s", self._bundles_dir)
            return

        for bundle_path in self._bundles_dir.glob("*.toml"):
            try:
                with bundle_path.open("rb") as f:
                    data = tomllib.load(f)

                bundle_data = data.get("bundle", {})
                name = bundle_data.get("name", bundle_path.stem)
                priority = bundle_data.get("priority", 100)  # Default high priority
                models = bundle_data.get("models", [])
                default = bundle_data.get("default", False)

                self._bundles[name] = BundleInfo(
                    name=name,
                    priority=priority,
                    models=models,
                    default=default,
                )
                logger.debug("Loaded bundle '%s': priority=%d, models=%d", name, priority, len(models))

            except Exception:
                logger.exception("Failed to load bundle: %s", bundle_path)

    def _load_models(self) -> None:
        """Load model names from model config YAML files.

        This discovers what models exist by scanning *.yaml files.
        We don't need full ModelConfig - just the model name.
        """
        if not self._models_dir.exists():
            logger.warning("Models directory not found: %s", self._models_dir)
            return

        for config_path in self._models_dir.glob("*.yaml"):
            if not config_path.is_file():
                continue

            try:
                with config_path.open() as f:
                    config = yaml.safe_load(f)

                model_name = config.get("sie_id")
                if model_name:
                    # Initialize with empty bundles list - will be populated by _compute_mappings
                    self._models[model_name] = ModelInfo(name=model_name)
                    # Also store lowercase mapping for case-insensitive lookup
                    self._model_names_lower[model_name.lower()] = model_name
                    logger.debug("Discovered model: %s", model_name)

            except Exception:
                logger.exception("Failed to load model config: %s", config_path)

    def _compute_mappings(self) -> None:
        """Compute model→bundle mappings based on bundle.models lists.

        For each model mentioned in any bundle, record which bundles include it,
        sorted by priority (ascending = preferred first).
        """
        # Build model→bundles mapping from bundle configs
        model_to_bundles: dict[str, list[tuple[int, str]]] = {}  # model → [(priority, bundle)]

        for bundle in self._bundles.values():
            for model_name in bundle.models:
                if model_name not in model_to_bundles:
                    model_to_bundles[model_name] = []
                model_to_bundles[model_name].append((bundle.priority, bundle.name))

        # For each model, sort bundles by priority and store
        for model_name, bundle_list in model_to_bundles.items():
            # Sort by priority (ascending)
            bundle_list.sort(key=lambda x: x[0])
            bundles_sorted = [b[1] for b in bundle_list]

            if model_name in self._models:
                self._models[model_name].bundles = bundles_sorted
            else:
                # Model in bundle but not in models directory - still track it
                self._models[model_name] = ModelInfo(name=model_name, bundles=bundles_sorted)
                self._model_names_lower[model_name.lower()] = model_name
                logger.debug("Model '%s' from bundle not in models directory", model_name)

    def resolve_bundle(self, model: str, bundle_override: str | None = None) -> str:
        """Resolve which bundle to use for a model.

        Args:
            model: Model name (e.g., "BAAI/bge-m3").
            bundle_override: Optional explicit bundle (e.g., "sglang").

        Returns:
            Bundle name to use.

        Raises:
            ModelNotFoundError: Model not in any bundle (404).
            BundleConflictError: Override bundle doesn't support model (409).
        """
        with self._lock:
            # Normalize model name (case-insensitive lookup)
            canonical_model = self._model_names_lower.get(model.lower())
            if canonical_model is None:
                # Try exact match
                if model not in self._models:
                    raise ModelNotFoundError(model)
                canonical_model = model

            model_info = self._models.get(canonical_model)
            if model_info is None or not model_info.bundles:
                raise ModelNotFoundError(model)

            if bundle_override is not None:
                # Validate override is compatible
                if bundle_override not in model_info.bundles:
                    raise BundleConflictError(
                        model=model,
                        bundle=bundle_override,
                        compatible_bundles=model_info.bundles,
                    )
                return bundle_override

            # Return highest priority (first in sorted list)
            return model_info.bundles[0]

    def get_model_info(self, model: str) -> ModelInfo | None:
        """Get model info including compatible bundles.

        Args:
            model: Model name.

        Returns:
            ModelInfo if found, None otherwise.
        """
        with self._lock:
            # Try case-insensitive lookup first
            canonical = self._model_names_lower.get(model.lower())
            if canonical:
                return self._models.get(canonical)
            return self._models.get(model)

    def list_models(self) -> list[str]:
        """List all known model names.

        Returns:
            Sorted list of model names.
        """
        with self._lock:
            return sorted(self._models.keys())

    def list_bundles(self) -> list[str]:
        """List all bundle names.

        Returns:
            List of bundle names sorted by priority.
        """
        with self._lock:
            bundles_sorted = sorted(self._bundles.values(), key=lambda b: b.priority)
            return [b.name for b in bundles_sorted]

    def get_bundle_info(self, bundle: str) -> BundleInfo | None:
        """Get bundle info.

        Args:
            bundle: Bundle name.

        Returns:
            BundleInfo if found, None otherwise.
        """
        with self._lock:
            return self._bundles.get(bundle)

    def get_models_for_bundle(self, bundle: str) -> list[str]:
        """Get all models that can be served by a bundle.

        Args:
            bundle: Bundle name.

        Returns:
            List of model names.
        """
        with self._lock:
            bundle_info = self._bundles.get(bundle)
            if bundle_info is None:
                return []
            return list(bundle_info.models)

    def model_exists(self, model: str) -> bool:
        """Check if a model exists in the registry.

        Args:
            model: Model name.

        Returns:
            True if model is known, False otherwise.
        """
        with self._lock:
            if model.lower() in self._model_names_lower:
                return True
            return model in self._models


def parse_model_spec(model_spec: str) -> tuple[str | None, str]:
    """Parse model spec into (bundle_override, model_name).

    Format: [bundle:/]org/model[:variant]

    The separator is ":/" to distinguish bundle prefix from variant suffix.

    Examples:
        "BAAI/bge-m3" → (None, "BAAI/bge-m3")
        "sglang:/BAAI/bge-m3" → ("sglang", "BAAI/bge-m3")
        "BAAI/bge-m3:variant" → (None, "BAAI/bge-m3:variant")

    Args:
        model_spec: Model specification string.

    Returns:
        Tuple of (bundle_override, model_name).
    """
    if ":/" in model_spec:
        parts = model_spec.split(":/", 1)
        bundle = parts[0].lower()
        model = parts[1]
        return bundle, model
    return None, model_spec
