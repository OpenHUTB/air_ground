"""Shared visibility-state and identity enrichment."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def visibility_state(
    air_visible: bool,
    vehicle_visible: bool,
    previously_visible: bool = False,
) -> str:
    if air_visible and vehicle_visible:
        return "Joint Visible"
    if air_visible:
        return "UAV Dominant"
    if vehicle_visible:
        return "Vehicle Dominant"
    return "Occlusion" if previously_visible else "Short Missing"


def enrich_observations(
    observations: Iterable[Dict[str, Any]],
    object_registry: Any,
    sensor_id: str,
) -> List[Dict[str, Any]]:
    enriched = []
    for observation in observations:
        record = object_registry.enrich_annotation(dict(observation))
        record["sensor_id"] = str(sensor_id)
        enriched.append(record)
    return enriched


def target_event_record(
    world_frame_id: int,
    target_uuid: str,
    air_observation: Optional[Dict[str, Any]],
    vehicle_observation: Optional[Dict[str, Any]],
    previous_state: Optional[str],
) -> Dict[str, Any]:
    state = visibility_state(
        air_observation is not None,
        vehicle_observation is not None,
        previously_visible=previous_state not in (None, "Short Missing"),
    )
    events = []
    if previous_state and previous_state != state:
        events.append("view_switch_event")
    if previous_state in {"UAV Dominant", "Vehicle Dominant", "Occlusion"} and state == "Joint Visible":
        events.append("reacquisition_event")
    if previous_state == "UAV Dominant" and state == "Vehicle Dominant":
        events.append("target_transfer_event")
    if previous_state == "Vehicle Dominant" and state == "UAV Dominant":
        events.append("target_transfer_event")
    return {
        "world_frame_id": int(world_frame_id),
        "target_global_uuid": str(target_uuid),
        "visibility_state": state,
        "events": events,
        "air_visible": air_observation is not None,
        "vehicle_visible": vehicle_observation is not None,
    }

