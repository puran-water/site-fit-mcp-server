"""Spatial Contract validation and migration.

Provides validation against the v1.0 JSON schema and migration
from legacy unversioned contracts.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Load schema at module level
SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "spatial_contract_v1.json"
SPATIAL_CONTRACT_V1_SCHEMA: dict[str, Any] | None = None

CURRENT_VERSION = "1.0.0"
SUPPORTED_VERSIONS = ["1.0.0", "0.9"]  # 0.9 = legacy unversioned


def _load_schema() -> dict[str, Any]:
    """Load the JSON schema, caching it for reuse."""
    global SPATIAL_CONTRACT_V1_SCHEMA
    if SPATIAL_CONTRACT_V1_SCHEMA is None:
        if SCHEMA_PATH.exists():
            with open(SCHEMA_PATH) as f:
                SPATIAL_CONTRACT_V1_SCHEMA = json.load(f)
        else:
            logger.warning("schema_file_not_found", path=str(SCHEMA_PATH))
            SPATIAL_CONTRACT_V1_SCHEMA = {}
    return SPATIAL_CONTRACT_V1_SCHEMA


def validate_contract(
    contract: dict[str, Any],
    strict: bool = False,
) -> dict[str, Any]:
    """Validate and optionally migrate a spatial contract.

    Args:
        contract: The contract dictionary to validate
        strict: If True, reject non-1.0.0 versions. If False, attempt migration.

    Returns:
        Validated/migrated contract (always 1.0.0 format)

    Raises:
        ValueError: If contract is invalid or migration fails
        ImportError: If jsonschema is not installed (only for full validation)

    Examples:
        >>> contract = {"placements": [{"id": "TK-101", "x": 50, "y": 30}]}
        >>> validated = validate_contract(contract)  # Migrates to 1.0.0
        >>> validated["contract_version"]
        '1.0.0'
    """
    version = contract.get("contract_version", "0.9")  # Assume legacy if missing

    if version == CURRENT_VERSION:
        # Already current version, validate against schema
        _validate_against_schema(contract)
        logger.info("contract_validated", version=version)
        return contract

    if strict:
        raise ValueError(
            f"Contract version '{version}' not supported in strict mode. "
            f"Expected '{CURRENT_VERSION}'. Re-export from site-fit."
        )

    # Migration path
    logger.warning("contract_migration_needed", from_version=version, to_version=CURRENT_VERSION)
    migrated = _migrate_contract(contract, version)
    _validate_against_schema(migrated)
    logger.info("contract_migrated", from_version=version, to_version=CURRENT_VERSION)
    return migrated


def _validate_against_schema(contract: dict[str, Any]) -> None:
    """Validate contract against JSON schema.

    Attempts to use jsonschema library, falls back to basic validation.
    """
    try:
        import jsonschema
    except ImportError:
        # Fall back to basic validation
        _basic_validation(contract)
        return

    schema = _load_schema()
    if not schema:
        logger.warning("schema_validation_skipped", reason="schema not loaded")
        return

    try:
        jsonschema.validate(contract, schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Contract validation failed: {e.message}") from e


def _basic_validation(contract: dict[str, Any]) -> None:
    """Basic validation without jsonschema library.

    Checks only the essential required fields.
    """
    required_fields = ["contract_version", "project", "site", "placements"]
    missing = [f for f in required_fields if f not in contract]
    if missing:
        raise ValueError(f"Contract missing required fields: {missing}")

    if "project" in contract and "id" not in contract["project"]:
        raise ValueError("Contract project must have 'id' field")

    if "site" in contract:
        site = contract["site"]
        if "boundary" not in site:
            raise ValueError("Contract site must have 'boundary' field")
        if "units" not in site:
            raise ValueError("Contract site must have 'units' field")

    if "placements" in contract:
        for i, p in enumerate(contract["placements"]):
            if "id" not in p or "x" not in p or "y" not in p:
                raise ValueError(f"Placement {i} missing required fields (id, x, y)")


def _migrate_contract(contract: dict[str, Any], from_version: str) -> dict[str, Any]:
    """Migrate old contract format to v1.0.0.

    Args:
        contract: The contract to migrate
        from_version: Source version string

    Returns:
        Migrated contract in v1.0.0 format

    Raises:
        ValueError: If no migration path exists for the version
    """
    if from_version == "0.9":
        return _migrate_from_legacy(contract)

    raise ValueError(f"No migration path from version '{from_version}'")


def _migrate_from_legacy(contract: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy unversioned contract to v1.0.0.

    Legacy format characteristics:
    - No contract_version field
    - May use structure_id instead of id in placements
    - May have flat structure (no program wrapper)
    - May be missing units field
    """
    migrated: dict[str, Any] = {
        "contract_version": CURRENT_VERSION,
    }

    # Project metadata
    if "project" in contract:
        migrated["project"] = contract["project"]
    else:
        migrated["project"] = {
            "id": contract.get("project_id", contract.get("id", "unknown")),
            "name": contract.get("project_name", ""),
            "revision": contract.get("revision", "A"),
        }

    # Ensure project.id exists
    if "id" not in migrated["project"]:
        migrated["project"]["id"] = "unknown"

    # Site data
    if "site" in contract:
        migrated["site"] = contract["site"].copy()
    else:
        migrated["site"] = {}

    # Ensure required site fields
    if "boundary" not in migrated["site"]:
        # Try to extract from top-level
        if "boundary" in contract:
            migrated["site"]["boundary"] = contract["boundary"]
        elif "site_boundary" in contract:
            migrated["site"]["boundary"] = contract["site_boundary"]
        else:
            migrated["site"]["boundary"] = []

    if "units" not in migrated["site"]:
        migrated["site"]["units"] = contract.get("units", "meters")

    if "crs" not in migrated["site"]:
        migrated["site"]["crs"] = contract.get("crs", "local")

    # Migrate keepouts
    if "keepouts" in contract and "keepouts" not in migrated["site"]:
        migrated["site"]["keepouts"] = contract["keepouts"]

    # Migrate entrances
    if "entrances" in contract and "entrances" not in migrated["site"]:
        migrated["site"]["entrances"] = contract["entrances"]

    # Program (structures)
    if "program" in contract:
        migrated["program"] = contract["program"]
    elif "structures" in contract:
        migrated["program"] = {"structures": contract["structures"]}
    elif "equipment" in contract:
        # Very old format used 'equipment' instead of 'structures'
        migrated["program"] = {"structures": contract["equipment"]}
    else:
        migrated["program"] = {"structures": []}

    # Placements - handle structure_id -> id migration
    placements = contract.get("placements", [])
    migrated["placements"] = []
    for p in placements:
        new_p = p.copy()
        # Migrate structure_id to id
        if "structure_id" in new_p and "id" not in new_p:
            new_p["id"] = new_p.pop("structure_id")
        # Ensure rotation_deg exists
        if "rotation_deg" not in new_p:
            new_p["rotation_deg"] = new_p.get("rotation", 0)
        migrated["placements"].append(new_p)

    # Road network
    if "road_network" in contract:
        migrated["road_network"] = contract["road_network"]
    elif "roads" in contract:
        # Legacy format
        migrated["road_network"] = {"segments": contract["roads"]}

    # Metrics
    if "metrics" in contract:
        migrated["metrics"] = contract["metrics"]

    # Provenance
    if "provenance" in contract:
        migrated["provenance"] = contract["provenance"]
    else:
        # Generate minimal provenance
        migrated["provenance"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "solver_version": "unknown",
            "solution_id": contract.get("solution_id", ""),
            "job_id": contract.get("job_id", ""),
        }

    return migrated


def create_contract(
    project_id: str,
    project_name: str,
    boundary: list[list[float]],
    structures: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    road_network: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    units: str = "meters",
    crs: str = "local",
    revision: str = "A",
    solution_id: str = "",
    job_id: str = "",
    solver_seed: int | None = None,
    ruleset_id: str = "",
) -> dict[str, Any]:
    """Create a new v1.0.0 spatial contract.

    Helper function to create contracts with proper structure.

    Args:
        project_id: Unique project identifier
        project_name: Human-readable project name
        boundary: Site boundary polygon [[x,y], ...]
        structures: List of structure definitions
        placements: List of placement positions
        road_network: Optional road network data
        metrics: Optional solution metrics
        units: Coordinate units ('meters' or 'feet')
        crs: Coordinate reference system
        revision: Document revision
        solution_id: Solution identifier
        job_id: Job identifier
        solver_seed: Random seed for reproducibility
        ruleset_id: Ruleset used for generation

    Returns:
        Valid v1.0.0 contract dictionary
    """
    contract: dict[str, Any] = {
        "contract_version": CURRENT_VERSION,
        "project": {
            "id": project_id,
            "name": project_name,
            "revision": revision,
        },
        "site": {
            "boundary": boundary,
            "units": units,
            "crs": crs,
        },
        "program": {
            "structures": structures,
        },
        "placements": [
            {
                "id": p.get("id", p.get("structure_id", "")),
                "x": p.get("x", 0),
                "y": p.get("y", 0),
                "rotation_deg": p.get("rotation_deg", p.get("rotation", 0)),
            }
            for p in placements
        ],
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "solver_version": "1.0.0",
            "solution_id": solution_id,
            "job_id": job_id,
        },
    }

    if road_network:
        contract["road_network"] = road_network

    if metrics:
        contract["metrics"] = metrics

    if solver_seed is not None:
        contract["provenance"]["solver_seed"] = solver_seed

    if ruleset_id:
        contract["provenance"]["ruleset_id"] = ruleset_id

    return contract
