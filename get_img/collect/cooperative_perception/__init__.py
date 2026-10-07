"""Shared UAV-Vehicle cooperative-perception collection infrastructure."""

from .cooperative_manager import CooperativeManager
from .ground_vehicle_manager import GroundVehicleManager
from .object_registry import ObjectRegistry
from .simulator_launcher import close_simulator, ensure_simulator

__all__ = [
    "CooperativeManager",
    "GroundVehicleManager",
    "ObjectRegistry",
    "ensure_simulator",
    "close_simulator",
]
