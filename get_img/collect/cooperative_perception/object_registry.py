"""Deterministic dataset UUID and scene-local ID allocation."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional

from .schema import DATASET_NAMESPACE


class ObjectRegistry:
    def __init__(
        self,
        scene_id: str,
        dataset_namespace: str = DATASET_NAMESPACE,
    ) -> None:
        self.scene_id = str(scene_id)
        self.dataset_namespace = str(dataset_namespace)
        self.namespace_uuid = uuid.uuid5(uuid.NAMESPACE_URL, self.dataset_namespace)
        self._by_actor_id: Dict[int, Dict[str, Any]] = {}
        self._next_scene_object_id = 1
        self._next_spawn_serial = 1

    def register(
        self,
        actor: Any,
        object_spawn_serial: Optional[int] = None,
    ) -> Dict[str, Any]:
        actor_id = int(actor.id)
        if actor_id in self._by_actor_id:
            return self._by_actor_id[actor_id]
        spawn_serial = (
            int(object_spawn_serial)
            if object_spawn_serial is not None
            else self._next_spawn_serial
        )
        self._next_spawn_serial = max(self._next_spawn_serial, spawn_serial + 1)
        stable_key = "%s:%s:%d" % (
            self.dataset_namespace,
            self.scene_id,
            spawn_serial,
        )
        record = {
            "global_object_uuid": str(uuid.uuid5(self.namespace_uuid, stable_key)),
            "scene_object_id": int(self._next_scene_object_id),
            "carla_actor_id": actor_id,
            "object_spawn_serial": spawn_serial,
            "blueprint": str(getattr(actor, "type_id", "unknown")),
        }
        self._next_scene_object_id += 1
        self._by_actor_id[actor_id] = record
        return record

    def register_many(self, actors: Iterable[Any]) -> List[Dict[str, Any]]:
        return [self.register(actor) for actor in actors if actor is not None]

    def lookup(self, actor_id: int) -> Optional[Dict[str, Any]]:
        return self._by_actor_id.get(int(actor_id))

    def enrich_annotation(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        actor_id = annotation.get("carla_actor_id")
        if actor_id is None:
            return dict(annotation)
        identity = self.lookup(int(actor_id))
        if identity is None:
            return dict(annotation)
        enriched = dict(annotation)
        enriched.update(identity)
        return enriched

    def records(self) -> List[Dict[str, Any]]:
        return sorted(
            self._by_actor_id.values(),
            key=lambda item: int(item["scene_object_id"]),
        )

