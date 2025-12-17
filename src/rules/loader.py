"""YAML ruleset loading and management.

Provides functions to:
- List available rulesets
- Load rulesets from YAML files
- Merge overrides into rulesets
"""

import logging
from pathlib import Path

import yaml

from ..models.rules import RuleSet

logger = logging.getLogger(__name__)

# Default rulesets directory (inside src/ package for proper wheel packaging)
RULESETS_DIR = Path(__file__).parent.parent / "rulesets"


def get_ruleset_path(name: str = "default") -> Path:
    """Get the path to a ruleset file.

    Args:
        name: Ruleset name (without .yaml extension)

    Returns:
        Path to the ruleset YAML file

    Raises:
        FileNotFoundError: If ruleset doesn't exist
    """
    path = RULESETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Ruleset '{name}' not found at {path}")
    return path


def list_rulesets() -> list[dict[str, str]]:
    """List all available rulesets.

    Returns:
        List of dicts with 'name' and 'description' keys
    """
    rulesets = []

    if not RULESETS_DIR.exists():
        logger.warning(f"Rulesets directory not found: {RULESETS_DIR}")
        return rulesets

    for yaml_file in RULESETS_DIR.glob("*.yaml"):
        name = yaml_file.stem
        description = _extract_description(yaml_file)
        rulesets.append({
            "name": name,
            "description": description,
        })

    return sorted(rulesets, key=lambda r: r["name"])


def _extract_description(yaml_path: Path) -> str:
    """Extract description from first comment line of YAML file."""
    try:
        with open(yaml_path) as f:
            first_line = f.readline().strip()
            if first_line.startswith("#"):
                return first_line.lstrip("#").strip()
    except Exception:
        pass
    return f"Rules from {yaml_path.name}"


def load_ruleset(
    name: str = "default",
    override: dict | None = None,
) -> RuleSet:
    """Load a ruleset from YAML file with optional overrides.

    Args:
        name: Ruleset name (without .yaml extension)
        override: Optional dict of values to override

    Returns:
        RuleSet instance with merged overrides
    """
    path = get_ruleset_path(name)

    with open(path) as f:
        yaml_content = f.read()

    ruleset = RuleSet.from_yaml(yaml_content)

    if override:
        ruleset = ruleset.merge_override(override)
        logger.debug(f"Applied overrides to ruleset '{name}'")

    return ruleset


def save_ruleset(ruleset: RuleSet, name: str) -> Path:
    """Save a ruleset to YAML file.

    Args:
        ruleset: RuleSet instance to save
        name: Filename (without .yaml extension)

    Returns:
        Path to saved file
    """
    path = RULESETS_DIR / f"{name}.yaml"

    # Convert to dict and write as YAML
    data = ruleset.model_dump(exclude_none=True)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved ruleset to {path}")
    return path


def validate_ruleset_yaml(yaml_content: str) -> tuple[bool, str | None]:
    """Validate YAML content as a valid ruleset.

    Args:
        yaml_content: YAML string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        RuleSet.from_yaml(yaml_content)
        return True, None
    except Exception as e:
        return False, str(e)
