# AirGroundCoopSuite coordinate convention

Schema version: `1.0.0`.

## Frame and time contract

Every committed sample stores `world_frame_id`, the canonical sensor
`timestamp`, world-snapshot `simulation_time`, consecutive `sample_index`, and
the originating `source_tick`. `world_frame_id` is the hard synchronization
key and `sample_index` never replaces it. Every required SensorData object must
report that same frame. Sensor timestamps must agree with the world snapshot
within `1e-4` seconds. Missing, late, newer-frame, or time-inconsistent sensor
data rejects the whole sample; copying a previous frame is forbidden.

Every platform state is recorded per committed frame and binds platform pose,
sensor poses, velocity, acceleration, heading, and control to the same frame
and time fields. A normal map derived from depth explicitly inherits the source
depth frame and timestamp.

All transforms are 4 x 4 homogeneous matrices serialized in row-major order.
Mathematical operations use column vectors:

`p_target = T_target_from_source @ p_source`

The transform name always states its direction. For example,
`T_world_from_sensor` maps a point expressed in the sensor coordinate system
into the CARLA world coordinate system.

## CARLA world and sensor coordinates

CARLA uses a left-handed coordinate system. World and actor axes are `x`
forward, `y` right, and `z` up. Locations are in meters. CARLA rotations may
also be saved as pitch, yaw, and roll in degrees, but degree-valued rotations
are metadata and are not a substitute for a transform matrix.

## OpenCV camera coordinates

OpenCV camera axes are `x` right, `y` down, and `z` forward. A point in the
CARLA sensor frame `[x_forward, y_right, z_up]` is converted to OpenCV camera
coordinates as `[y_right, -z_up, x_forward]`.

## Required transforms

- `T_world_from_platform`
- `T_platform_from_sensor`
- `T_world_from_sensor`
- `T_sensor_from_world`
- `T_target_sensor_from_source_sensor`

Pairwise transforms are calculated through the transform graph instead of
being hard-coded for each platform pair.

## Numerical validation

Every inverse pair must satisfy `T_A_from_B @ T_B_from_A ~= I`. The default
closure tolerance is `1e-5` maximum absolute element error. Camera
calibration must also pass 3D-box reprojection QA; its pixel threshold remains
`pilot_provisional` until the UAV-Vehicle pilot batch is inspected.
