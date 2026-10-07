"""Task 3 fixed 3 virtual Air + 1 Ground Vehicle declaration."""


class MCMOTAdapter:
    task_id = "task3"
    route_profile = "multi_object_traversal"
    air_platform_count = 3
    ground_vehicle_count = 1
    air_platform_virtual = True
    air_modalities = ("rgb", "depth", "semantic")
    ground_modalities = ("rgb", "depth", "semantic")
    model_inputs = ("rgb",)

