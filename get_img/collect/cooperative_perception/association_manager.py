"""Truth-based cross-view association records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List


class AssociationManager:
    def build(
        self,
        observations_by_sensor: Dict[str, Iterable[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        views = defaultdict(list)
        for sensor_id, observations in observations_by_sensor.items():
            for observation in observations:
                object_uuid = observation.get("global_object_uuid")
                if object_uuid:
                    views[str(object_uuid)].append(str(sensor_id))
        records = []
        for object_uuid, sensor_ids in sorted(views.items()):
            unique = sorted(set(sensor_ids))
            for source_index, source in enumerate(unique):
                for target in unique[source_index + 1 :]:
                    records.append(
                        {
                            "global_object_uuid": object_uuid,
                            "source_sensor": source,
                            "target_sensor": target,
                            "relation": "same_object",
                            "truth_source": "carla_actor_registry",
                            "association_source": "ground_truth",
                            "usage": "annotation_and_evaluation_only",
                        }
                    )
        return records
