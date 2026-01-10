"""Contract validation and migration utilities.

This module provides validation of spatial contracts against the v1.0 schema
and migration from legacy unversioned formats.
"""

from .validator import (
    CURRENT_VERSION,
    SUPPORTED_VERSIONS,
    create_contract,
    validate_contract,
)

__all__ = [
    "CURRENT_VERSION",
    "SUPPORTED_VERSIONS",
    "create_contract",
    "validate_contract",
]
