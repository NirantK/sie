"""Dependency checking for model adapters.

This module handles:
- Reading adapter dependencies from pyproject.toml files
- Reading model-specific dependencies from model YAML config files
- Checking installed package versions against requirements
- Reporting dependency conflicts before model loading
- Resolving bundle dependencies for startup sync
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig

logger = logging.getLogger(__name__)

# Note: flash-attn dependencies in adapter pyproject.toml files use platform markers
# (e.g., "; sys_platform == 'linux'") so pip/uv will only install them on Linux.
# This allows cross-platform development (macOS/Windows) while ensuring CUDA images get flash-attn.

# Path to the adapters directory (relative to this module)
_ADAPTERS_DIR = Path(__file__).parent.parent / "adapters"


def model_name_to_folder(model_name: str) -> str:
    """Convert a model name (org/model format) to folder name (org__model format).

    The folder naming convention uses '__' as separator because '/' is not allowed
    in directory names. Variant suffixes using ':' are also converted to '__'.

    Args:
        model_name: Model name in org/model or org/model:variant format.

    Returns:
        Folder name in org__model or org__model__variant format.

    Examples:
        >>> model_name_to_folder("BAAI/bge-m3")
        'BAAI__bge-m3'
        >>> model_name_to_folder("BAAI/bge-m3:FlagEmbedding")
        'BAAI__bge-m3__FlagEmbedding'
        >>> model_name_to_folder("simple-model")
        'simple-model'
    """
    return model_name.replace("/", "__").replace(":", "__")


def discover_model_configs(models_dir: Path) -> dict[str, Path]:
    """Discover all model configs by scanning YAML files in models directory.

    Models are stored as flat YAML files (e.g., baai-bge-m3.yaml) directly
    in the models directory. This function reads each YAML file and maps
    model names to their config file paths.

    Args:
        models_dir: Path to the models directory.

    Returns:
        Dict mapping model name (from config) to config file path.
    """
    model_configs: dict[str, Path] = {}

    if not models_dir.exists():
        return model_configs

    for config_path in models_dir.glob("*.yaml"):
        try:
            with config_path.open() as f:
                config = yaml.safe_load(f)
            model_name = config.get("sie_id")
            if model_name:
                model_configs[model_name] = config_path
        except (OSError, yaml.YAMLError, AttributeError):
            logger.debug("Failed to read config %s", config_path)
            continue

    return model_configs


@dataclass
class DepConflict:
    """A dependency conflict between required and installed versions."""

    package: str
    required: str
    installed: str | None
    reason: str

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dict for JSON serialization."""
        return {
            "package": self.package,
            "required": self.required,
            "installed": self.installed,
            "reason": self.reason,
        }


class DependencyConflictError(Exception):
    """Raised when model dependencies conflict with installed packages."""

    def __init__(self, model_name: str, conflicts: list[DepConflict]) -> None:
        self.model_name = model_name
        self.conflicts = conflicts

        # Build message
        conflict_strs = [
            f"  - {c.package}: requires {c.required}, installed {c.installed or 'none'}" for c in conflicts
        ]
        message = f"Model {model_name} has dependency conflicts:\n" + "\n".join(conflict_strs)
        super().__init__(message)


def get_adapter_deps(adapter_path: str) -> dict[str, str]:
    """Get dependencies for an adapter from its pyproject.toml.

    Args:
        adapter_path: Adapter path like "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"

    Returns:
        Dict mapping package names to version constraints.
    """
    if not adapter_path.startswith("sie_server.adapters."):
        # Custom adapter - no standard deps to check
        return {}

    # Extract adapter module name
    # e.g., "sie_server.adapters.sentence_transformer:Class" -> "sentence_transformer"
    module_path = adapter_path.split(":", maxsplit=1)[0]  # "sie_server.adapters.sentence_transformer"
    parts = module_path.split(".")
    if len(parts) < 3:
        return {}

    adapter_name = parts[2]  # "sentence_transformer"

    # Look for pyproject.toml in the adapter directory
    pyproject_path = _ADAPTERS_DIR / adapter_name / "pyproject.toml"

    if not pyproject_path.exists():
        logger.debug("No pyproject.toml found for adapter %s", adapter_name)
        return {}

    deps = _parse_pyproject_deps(pyproject_path)

    # Note: flash-attn deps have "; sys_platform == 'linux'" markers in adapter pyproject.toml
    # files, so pip/uv will skip them on macOS/Windows automatically. No exclusion needed here.

    return deps


def get_model_deps(config_path: Path) -> dict[str, str]:
    """Get model-specific dependencies from its YAML config.

    Args:
        config_path: Path to the model config YAML file.

    Returns:
        Dict mapping package names to version constraints.
    """
    if not config_path.exists():
        return {}

    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
        deps_list = config.get("dependencies", [])
        if not deps_list:
            return {}

        deps: dict[str, str] = {}
        for dep_str in deps_list:
            name, constraint = _parse_dep_string(dep_str)
            if name:
                deps[name] = constraint
        return deps
    except (OSError, yaml.YAMLError, AttributeError):
        logger.debug("Failed to read dependencies from %s", config_path)
        return {}


def _parse_pyproject_deps(pyproject_path: Path) -> dict[str, str]:
    """Parse dependencies from a pyproject.toml file.

    Args:
        pyproject_path: Path to pyproject.toml

    Returns:
        Dict mapping package names to version constraints.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found]

    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.exception("Failed to parse %s", pyproject_path)
        return {}

    deps: dict[str, str] = {}

    # Parse dependencies from [project.dependencies]
    project = data.get("project", {})
    for dep_str in project.get("dependencies", []):
        name, constraint = _parse_dep_string(dep_str)
        if name:
            deps[name] = constraint

    return deps


def _parse_dep_string(dep_str: str) -> tuple[str | None, str]:
    """Parse a PEP 508 dependency string.

    Args:
        dep_str: e.g., "transformers>=4.45,<4.56" or "flash-attn @ https://..."
                 or "sie-server[flash-attn]"

    Returns:
        Tuple of (package_name_with_extras, version_constraint_or_full_dep) or (None, "") if invalid.
        Package name includes extras if present (e.g., "sie-server[flash-attn]").
        For URL dependencies, returns (name, full_dep_string) so the URL is preserved.
    """
    try:
        req = Requirement(dep_str)
        # For URL dependencies (like "pkg @ https://..."), preserve the full string
        # because the URL is the constraint, not a version specifier
        if req.url:
            return req.name.lower(), dep_str.strip()

        # Build package name with extras if present
        pkg_name = req.name.lower()
        if req.extras:
            extras_str = ",".join(sorted(req.extras))
            pkg_name = f"{pkg_name}[{extras_str}]"

        # Convert specifier to string, or empty if no version specified
        constraint = str(req.specifier) if req.specifier else ""
        return pkg_name, constraint
    except InvalidRequirement:
        logger.debug("Failed to parse dependency: %s", dep_str)
        return None, ""


def get_installed_packages() -> dict[str, str]:
    """Get currently installed packages and their versions.

    Uses importlib.metadata which correctly sees packages from uv's
    ephemeral overlay environment (--with-requirements) because
    importlib.metadata uses sys.path which is modified by uv.

    Note: We collect unique package names first, then use version() to get
    the correct version. This is because distributions() returns ALL distributions
    (including duplicates from overlay and base venv), but version() respects
    sys.path order and returns the overlay version first.

    Returns:
        Dict mapping package names (lowercase) to version strings.
    """
    packages: dict[str, str] = {}

    # First collect all unique package names
    seen_names: set[str] = set()
    for dist in importlib.metadata.distributions():
        try:
            name = dist.metadata["Name"]
            if name:
                seen_names.add(name)
        except (KeyError, TypeError):
            continue

    # Then get version for each using version() which respects sys.path order
    for name in seen_names:
        try:
            version = importlib.metadata.version(name)
            normalized_name = re.sub(r"[-_.]+", "-", name.lower())
            packages[normalized_name] = version
        except importlib.metadata.PackageNotFoundError:
            continue
        except (KeyError, TypeError, ValueError):
            continue

    return packages


def version_satisfies(installed: str, constraint: str) -> bool:
    """Check if an installed version satisfies a constraint.

    Args:
        installed: Installed version string (e.g., "4.57.3").
        constraint: Version constraint (e.g., ">=4.45,<4.56").

    Returns:
        True if the version satisfies the constraint.
    """
    if not constraint:
        # No constraint = any version is fine
        return True

    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion

        spec = SpecifierSet(constraint)
        version = Version(installed)
        return version in spec
    except (InvalidSpecifier, InvalidVersion):
        logger.debug("Failed to check version %s against %s", installed, constraint)
        # If we can't parse, assume it's fine
        return True


def check_model_dependencies(
    config: ModelConfig,
    model_dir: Path,
) -> list[DepConflict]:
    """Check if model's dependencies are satisfied by current environment.

    Args:
        config: The model configuration.
        model_dir: Path to the model directory.

    Returns:
        List of dependency conflicts (empty if all satisfied).
    """
    from sie_server.core.loader import resolve_adapter_path

    # Resolve the full adapter path
    try:
        adapter_path = resolve_adapter_path(config, model_dir)
    except ValueError:
        # If we can't resolve the adapter, we can't check deps
        return []

    # Collect required deps from adapter
    required = get_adapter_deps(adapter_path)

    # Add model-specific deps from the already-parsed config
    for dep_str in config.dependencies:
        name, constraint = _parse_dep_string(dep_str)
        if name:
            required[name] = constraint

    if not required:
        return []

    # Get installed packages
    installed = get_installed_packages()

    # Check each requirement
    conflicts: list[DepConflict] = []
    for pkg, constraint in required.items():
        # Normalize package name
        normalized_pkg = re.sub(r"[-_.]+", "-", pkg.lower())

        if normalized_pkg not in installed:
            conflicts.append(
                DepConflict(
                    package=pkg,
                    required=constraint or "any",
                    installed=None,
                    reason="not installed",
                )
            )
        elif constraint and not version_satisfies(installed[normalized_pkg], constraint):
            conflicts.append(
                DepConflict(
                    package=pkg,
                    required=constraint,
                    installed=installed[normalized_pkg],
                    reason="version mismatch",
                )
            )

    return conflicts


# =============================================================================
# Bundle Dependency Resolution
# =============================================================================

# CUDA-only package names (normalized) - these are excluded when building CPU images
_CUDA_ONLY_PACKAGES = frozenset(
    {
        "flash-attn",
        "xformers",
    }
)


@dataclass
class BundleDepResult:
    """Result of resolving bundle dependencies."""

    requirements: list[str]  # PEP 508 requirement strings
    conflicts: list[str]  # Human-readable conflict messages
    models: list[str]  # Model names in the bundle

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "requirements": self.requirements,
            "conflicts": self.conflicts,
            "models": self.models,
        }


def load_bundle(bundle_path: Path) -> dict:
    """Load a bundle TOML file.

    Args:
        bundle_path: Path to the bundle TOML file.

    Returns:
        Parsed bundle config dict with 'name' and 'models' keys.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found]

    with bundle_path.open("rb") as f:
        data = tomllib.load(f)

    return data.get("bundle", {})


def get_model_adapter_paths(config_path: Path) -> list[str]:
    """Get all unique adapter paths from a model's config YAML.

    Scans all profiles (not just default) because different profiles may use
    different adapters with different dependencies.

    Args:
        config_path: Path to the model config YAML file.

    Returns:
        List of unique adapter path strings found across all profiles.
    """
    if not config_path.exists():
        return []

    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
        profiles = config.get("profiles", {})
        adapter_paths: list[str] = []
        seen: set[str] = set()
        for profile in profiles.values():
            adapter_path = profile.get("adapter_path")
            if adapter_path and adapter_path not in seen:
                seen.add(adapter_path)
                adapter_paths.append(adapter_path)
        return adapter_paths
    except Exception:
        logger.exception("Failed to load config for %s", config_path)
        return []


def collect_bundle_deps(
    bundle_name: str,
    bundles_dir: Path,
    models_dir: Path,
    *,
    exclude_cuda: bool = False,
) -> BundleDepResult:
    """Collect all dependencies for models in a bundle.

    Args:
        bundle_name: Name of the bundle (without .toml extension).
        bundles_dir: Path to the bundles directory.
        models_dir: Path to the models directory.
        exclude_cuda: If True, exclude CUDA-only packages (flash-attn, xformers).
            Use this when building CPU-only images.

    Returns:
        BundleDepResult with requirements, conflicts, and model names.
    """
    bundle_path = bundles_dir / f"{bundle_name}.toml"
    if not bundle_path.exists():
        return BundleDepResult(
            requirements=[],
            conflicts=[f"Bundle '{bundle_name}' not found at {bundle_path}"],
            models=[],
        )

    bundle = load_bundle(bundle_path)
    model_names = bundle.get("models", [])
    bundle_deps = bundle.get("deps", {})  # Optional [bundle.deps] section

    if not model_names:
        return BundleDepResult(
            requirements=[],
            conflicts=[f"Bundle '{bundle_name}' has no models defined"],
            models=[],
        )

    # Discover all model configs by reading YAML files
    known_model_configs = discover_model_configs(models_dir)

    # Collect deps from all models
    # Key: package name (normalized), Value: list of (constraint, source)
    all_deps: dict[str, list[tuple[str, str]]] = {}

    # Add bundle-level deps first (e.g., timm for Florence-2)
    for pkg, constraint in bundle_deps.items():
        normalized = re.sub(r"[-_.]+", "-", pkg.lower())
        if normalized not in all_deps:
            all_deps[normalized] = []
        all_deps[normalized].append((constraint, f"bundle:{bundle_name}"))

    for model_name in model_names:
        # Look up model config from discovered configs
        if model_name not in known_model_configs:
            logger.warning("Model '%s' not found in %s", model_name, models_dir)
            continue

        config_path = known_model_configs[model_name]

        # Get adapter paths from all profiles
        for adapter_path in get_model_adapter_paths(config_path):
            adapter_deps = get_adapter_deps(adapter_path)
            for pkg, constraint in adapter_deps.items():
                normalized = re.sub(r"[-_.]+", "-", pkg.lower())
                if normalized not in all_deps:
                    all_deps[normalized] = []
                all_deps[normalized].append((constraint, f"adapter:{adapter_path}"))

        # Get model-specific deps
        model_deps = get_model_deps(config_path)
        for pkg, constraint in model_deps.items():
            normalized = re.sub(r"[-_.]+", "-", pkg.lower())
            if normalized not in all_deps:
                all_deps[normalized] = []
            all_deps[normalized].append((constraint, f"model:{model_name}"))

    # Now merge constraints and check for conflicts
    requirements: list[str] = []
    conflicts: list[str] = []

    for pkg, constraints in all_deps.items():
        # Skip CUDA-only packages when building CPU images
        if exclude_cuda and pkg in _CUDA_ONLY_PACKAGES:
            logger.debug("Excluding CUDA-only package: %s", pkg)
            continue

        merged = _merge_constraints(pkg, constraints)
        if merged.startswith("CONFLICT:"):
            conflicts.append(merged[9:])  # Strip "CONFLICT:" prefix
        else:
            requirements.append(merged)

    return BundleDepResult(
        requirements=requirements,
        conflicts=conflicts,
        models=model_names,
    )


def _merge_constraints(pkg: str, constraints: list[tuple[str, str]]) -> str:
    """Merge multiple constraints for a package into a single requirement.

    Args:
        pkg: Package name (normalized).
        constraints: List of (constraint, source) tuples.
            For URL deps, constraint is the full PEP 508 string (e.g., "pkg @ https://...")

    Returns:
        PEP 508 requirement string, or "CONFLICT:..." if incompatible.
    """
    if not constraints:
        return pkg

    # If all constraints are empty, just return package name
    non_empty = [(c, s) for c, s in constraints if c]
    if not non_empty:
        return pkg

    # Check for URL dependencies (constraint contains " @ ")
    # URL deps can't be merged with version specifiers
    url_deps = [(c, s) for c, s in non_empty if " @ " in c]
    version_deps = [(c, s) for c, s in non_empty if " @ " not in c]

    if url_deps:
        # If we have URL deps, use the first one (they should all be the same)
        # URL deps take precedence over version specifiers
        if len(url_deps) > 1:
            # Multiple URL deps - check if they're the same
            urls = [c for c, _ in url_deps]
            if len(set(urls)) > 1:
                return f"CONFLICT: {pkg} has conflicting URL dependencies: {urls}"
        return url_deps[0][0]

    # No URL deps, merge version specifiers as before
    combined_specs: list[str] = []
    sources: list[str] = []

    for constraint, source in version_deps:
        combined_specs.append(constraint)
        sources.append(source)

    # Deduplicate individual specifiers (e.g. ">=0.2,<1" and "<1,>=0.2"
    # both contain ">=0.2" and "<1") while preserving order
    seen: set[str] = set()
    unique_specs: list[str] = []
    for constraint_str in combined_specs:
        for spec_part in constraint_str.split(","):
            spec_part = spec_part.strip()
            if spec_part and spec_part not in seen:
                seen.add(spec_part)
                unique_specs.append(spec_part)

    # Join all specifiers
    combined = ",".join(unique_specs)

    # Try to parse and validate the combined specifier
    try:
        spec = SpecifierSet(combined)
        # Check if the specifier is satisfiable (has any valid versions)
        # We do this by checking common version patterns
        if not _is_specifier_satisfiable(spec):
            return f"CONFLICT: {pkg} has conflicting constraints: {combined} (from {', '.join(sources)})"
        return f"{pkg}{combined}"
    except InvalidSpecifier as e:
        return f"CONFLICT: {pkg} has invalid constraints: {combined} ({e})"


def _is_specifier_satisfiable(spec: SpecifierSet) -> bool:
    """Check if a specifier set can be satisfied.

    This is a heuristic check - we test a range of versions.

    Args:
        spec: SpecifierSet to check.

    Returns:
        True if likely satisfiable, False if clearly conflicting.
    """
    # Test common version patterns (must cover ranges used by adapter deps)
    test_versions = [
        "0.1.0",
        "0.2.0",
        "0.3.0",
        "0.5.0",
        "0.7.0",
        "0.9.0",
        "1.0.0",
        "1.5.0",
        "2.0.0",
        "2.5.0",
        "2.8.0",
        "2.9.0",
        "2.9.1",
        "2.10.0",
        "3.0.0",
        "4.0.0",
        "4.45.0",
        "4.50.0",
        "4.51.3",
        "4.55.0",
        "4.56.0",
        "4.57.0",
        "4.57.3",
        "5.0.0",
        "6.0.0",
    ]

    from packaging.version import InvalidVersion

    for v in test_versions:
        try:
            if Version(v) in spec:
                return True
        except InvalidVersion:
            continue

    return False


def resolve_bundle_deps_cli(
    bundle_name: str | None,
    model_names: list[str] | None,
    bundles_dir: Path,
    models_dir: Path,
) -> int:
    """CLI entry point for resolving bundle dependencies.

    Prints requirements to stdout (one per line) or errors to stderr.

    Args:
        bundle_name: Name of the bundle to resolve.
        model_names: Alternative: explicit list of model names.
        bundles_dir: Path to the bundles directory.
        models_dir: Path to the models directory.

    Returns:
        Exit code (0 for success, 1 for conflicts/errors).
    """
    if bundle_name:
        result = collect_bundle_deps(bundle_name, bundles_dir, models_dir)
    elif model_names:
        # Create a synthetic bundle from the model list
        result = _collect_model_list_deps(model_names, models_dir)
    else:
        return 1

    if result.conflicts:
        for conflict in result.conflicts:
            print(conflict, file=sys.stderr)
        return 1

    # Print requirements to stdout for shell scripts to consume
    for req in result.requirements:
        print(req)

    return 0


def _collect_model_list_deps(
    model_names: list[str],
    models_dir: Path,
    *,
    exclude_cuda: bool = False,
) -> BundleDepResult:
    """Collect deps for an explicit list of models.

    Args:
        model_names: List of model names.
        models_dir: Path to the models directory.
        exclude_cuda: If True, exclude CUDA-only packages (flash-attn, xformers).
            Use this when building CPU-only images.

    Returns:
        BundleDepResult with requirements, conflicts, and model names.
    """
    # Discover all model configs by reading YAML files
    known_model_configs = discover_model_configs(models_dir)

    # Same logic as collect_bundle_deps but without loading bundle file
    all_deps: dict[str, list[tuple[str, str]]] = {}

    for model_name in model_names:
        # Look up model config from discovered configs
        if model_name not in known_model_configs:
            logger.warning("Model '%s' not found in %s", model_name, models_dir)
            continue

        config_path = known_model_configs[model_name]

        for adapter_path in get_model_adapter_paths(config_path):
            adapter_deps = get_adapter_deps(adapter_path)
            for pkg, constraint in adapter_deps.items():
                normalized = re.sub(r"[-_.]+", "-", pkg.lower())
                if normalized not in all_deps:
                    all_deps[normalized] = []
                all_deps[normalized].append((constraint, f"adapter:{adapter_path}"))

        model_deps = get_model_deps(config_path)
        for pkg, constraint in model_deps.items():
            normalized = re.sub(r"[-_.]+", "-", pkg.lower())
            if normalized not in all_deps:
                all_deps[normalized] = []
            all_deps[normalized].append((constraint, f"model:{model_name}"))

    requirements: list[str] = []
    conflicts: list[str] = []

    for pkg, constraints in all_deps.items():
        # Skip CUDA-only packages when building CPU images
        if exclude_cuda and pkg in _CUDA_ONLY_PACKAGES:
            logger.debug("Excluding CUDA-only package: %s", pkg)
            continue

        merged = _merge_constraints(pkg, constraints)
        if merged.startswith("CONFLICT:"):
            conflicts.append(merged[9:])
        else:
            requirements.append(merged)

    return BundleDepResult(
        requirements=requirements,
        conflicts=conflicts,
        models=model_names,
    )
