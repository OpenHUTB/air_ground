"""Task 2 modality and route declaration."""


class SOTAdapter:
    task_id = "task2"
    route_profile = "target_interaction"
    motion_mode = "rear_upper_follow"
    air_modalities = ("rgb", "depth", "semantic")
    ground_modalities = ("rgb", "depth", "semantic")
    model_inputs = ("rgb",)

