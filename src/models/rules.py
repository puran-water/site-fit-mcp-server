"""Engineering rules and constraints for site layout."""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class AccessRules(BaseModel):
    """Rules for site access roads and vehicle circulation."""

    road_width: float = Field(default=6.0, ge=3.0, le=15.0, description="Road width in meters")
    min_turning_radius: float = Field(
        default=12.0, ge=6.0, le=25.0, description="Minimum turning radius in meters"
    )
    dock_depth: float = Field(
        default=15.0, ge=10.0, le=30.0, description="Default dock/apron depth in meters"
    )
    fire_lane_width: float = Field(
        default=6.0, ge=4.0, le=8.0, description="Fire lane width in meters"
    )
    max_dead_end_length: float = Field(
        default=150.0, ge=50.0, description="Maximum dead-end road length before turnaround"
    )


class NFPA820ZoneConfig(BaseModel):
    """NFPA 820 hazardous area zone configuration for an equipment type."""

    class_i_div_1_radius: float = Field(
        default=0.0, ge=0, description="Class I Division 1 zone radius in meters"
    )
    class_i_div_2_radius: float = Field(
        default=0.0, ge=0, description="Class I Division 2 zone radius in meters"
    )
    vertical_above: float = Field(
        default=0.46, ge=0, description="Vertical extent above grade in meters (default 18 inches)"
    )


class SetbackRules(BaseModel):
    """Property line and boundary setback rules."""

    property_line_default: float = Field(
        default=7.5, ge=0, description="Default setback from property line in meters"
    )
    property_line_front: Optional[float] = Field(
        default=None, ge=0, description="Front property line setback (overrides default)"
    )
    property_line_rear: Optional[float] = Field(
        default=None, ge=0, description="Rear property line setback (overrides default)"
    )
    property_line_side: Optional[float] = Field(
        default=None, ge=0, description="Side property line setback (overrides default)"
    )
    wetland_buffer: float = Field(
        default=15.0, ge=0, description="Buffer distance from wetlands in meters"
    )
    watercourse_buffer: float = Field(
        default=30.0, ge=0, description="Buffer distance from watercourses in meters"
    )


class RuleSet(BaseModel):
    """Complete rule set for site layout constraints.

    Rules can be overridden at request time via JSON merge patch.
    """

    # Property setbacks
    setbacks: SetbackRules = Field(default_factory=SetbackRules, description="Setback rules")

    # Equipment-to-boundary setbacks (by equipment type)
    equipment_to_boundary: Dict[str, float] = Field(
        default_factory=lambda: {
            "default": 5.0,
            "digester": 15.0,
            "flare": 30.0,
            "chemical_storage": 10.0,
        },
        description="Minimum distance from equipment to site boundary by type"
    )

    # Equipment-to-equipment clearances (pairwise matrix)
    # Keys are "type1_to_type2" or "type1_type2" for symmetric
    equipment_clearances: Dict[str, float] = Field(
        default_factory=lambda: {
            "default": 3.0,
            "digester_to_any": 10.0,
            "flare_to_any": 75.0,
            "digester_to_digester": 5.0,
            "aeration_to_clarifier": 5.0,
            "building_to_building": 6.0,
            "tank_to_tank": 3.0,
        },
        description="Minimum clearance between equipment pairs"
    )

    # Fire separation requirements (override clearances when larger)
    fire_separation: Dict[str, float] = Field(
        default_factory=lambda: {
            "default": 0.0,  # No fire separation by default
            "flare_to_any": 75.0,
            "chemical_storage_to_building": 15.0,
            "electrical_building_to_digester": 15.0,
        },
        description="Fire separation distances (takes precedence over clearances)"
    )

    # Access rules
    access: AccessRules = Field(default_factory=AccessRules, description="Road and access rules")

    # Topology/adjacency preferences (soft constraints)
    adjacency_preference_weight: float = Field(
        default=10.0, ge=0, description="Weight for topology adjacency soft constraints"
    )
    flow_direction_weight: float = Field(
        default=5.0, ge=0, description="Weight for honoring process flow direction"
    )

    # NFPA 820 hazardous area zone configurations
    nfpa820_zones: Dict[str, NFPA820ZoneConfig] = Field(
        default_factory=dict,
        description="NFPA 820 hazardous area zone radii by equipment type"
    )

    # Equipment types that must be outside hazard zones
    hazard_zone_exclusions: List[str] = Field(
        default_factory=lambda: [
            "electrical_building",
            "control_building",
            "motor_control_center",
            "office",
            "laboratory",
        ],
        description="Equipment types that must be placed outside Class I hazard zones"
    )

    def get_nfpa820_zone(self, equipment_type: str) -> Optional[NFPA820ZoneConfig]:
        """Get NFPA 820 zone configuration for an equipment type."""
        # Check exact match first
        if equipment_type in self.nfpa820_zones:
            zone_data = self.nfpa820_zones[equipment_type]
            if isinstance(zone_data, dict):
                return NFPA820ZoneConfig(**zone_data)
            return zone_data
        # Check partial matches
        for key, zone_data in self.nfpa820_zones.items():
            if key in equipment_type or equipment_type in key:
                if isinstance(zone_data, dict):
                    return NFPA820ZoneConfig(**zone_data)
                return zone_data
        return None

    def is_hazard_zone_exclusion(self, equipment_type: str) -> bool:
        """Check if equipment type must be placed outside hazard zones."""
        return any(excl in equipment_type or equipment_type in excl
                   for excl in self.hazard_zone_exclusions)

    def get_equipment_to_boundary(self, equipment_type: str) -> float:
        """Get equipment-to-boundary setback for a type."""
        # Check specific type first
        if equipment_type in self.equipment_to_boundary:
            return self.equipment_to_boundary[equipment_type]
        # Check partial matches (e.g., "digester" matches "digester_tank")
        for key, value in self.equipment_to_boundary.items():
            if key != "default" and key in equipment_type:
                return value
        return self.equipment_to_boundary.get("default", 5.0)

    def get_clearance(self, type1: str, type2: str) -> float:
        """Get minimum clearance between two equipment types.

        Checks in order:
        1. Specific pair "type1_to_type2"
        2. Specific pair "type2_to_type1"
        3. Wildcard "type1_to_any"
        4. Wildcard "type2_to_any"
        5. Fire separation requirements
        6. Default clearance
        """
        clearance = self.equipment_clearances.get("default", 3.0)

        # Check specific pairs
        key1 = f"{type1}_to_{type2}"
        key2 = f"{type2}_to_{type1}"
        if key1 in self.equipment_clearances:
            clearance = max(clearance, self.equipment_clearances[key1])
        if key2 in self.equipment_clearances:
            clearance = max(clearance, self.equipment_clearances[key2])

        # Check wildcards
        for t in [type1, type2]:
            wild_key = f"{t}_to_any"
            if wild_key in self.equipment_clearances:
                clearance = max(clearance, self.equipment_clearances[wild_key])

        # Check fire separation (takes precedence when larger)
        for key, fire_dist in self.fire_separation.items():
            if key == "default":
                continue
            parts = key.split("_to_")
            if len(parts) == 2:
                if (type1 in parts[0] or parts[0] in type1 or parts[0] == "any") and (
                    type2 in parts[1] or parts[1] in type2 or parts[1] == "any"
                ):
                    clearance = max(clearance, fire_dist)
                if (type2 in parts[0] or parts[0] in type2 or parts[0] == "any") and (
                    type1 in parts[1] or parts[1] in type1 or parts[1] == "any"
                ):
                    clearance = max(clearance, fire_dist)

        return clearance

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "RuleSet":
        """Load ruleset from YAML string."""
        import yaml
        data = yaml.safe_load(yaml_content)
        return cls(**data)

    def merge_override(self, override: Dict) -> "RuleSet":
        """Merge override dict into this ruleset (JSON merge patch semantics)."""
        import json
        base = json.loads(self.model_dump_json())
        _deep_merge(base, override)
        return RuleSet(**base)


def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base dict (modifies base in place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
