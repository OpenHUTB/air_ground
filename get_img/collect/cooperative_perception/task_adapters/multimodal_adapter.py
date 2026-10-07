"""Task 1 modality declaration."""


class MultimodalAdapter:
    task_id = "task1"
    route_profile = "coverage"
    air_modalities = ("rgb", "depth", "normal", "semantic", "lidar")
    ground_modalities = ("rgb", "depth", "normal", "semantic", "lidar")
    model_inputs = tuple(air_modalities)

