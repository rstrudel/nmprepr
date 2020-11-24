from gym.envs.registration import register


robot_ids = {
    "Sphere": "sphere",
    "SShape": "s_shape",
}
num_samples = [16, 32, 64, 128, 180, 256, 512, 1024]

normals = [(False, ""), (True, "Normals")]

coord_frames = ["local", "global"]

"""
Boxes Environments
"""

for robot_register_name, robot_name in robot_ids.items():
    kwargs = {"robot_name": robot_name}
    env_str = f"{robot_register_name}-NoBoxes"
    register(
        id=f"{env_str}-v0",
        entry_point="mpenv.envs.boxes:boxes_noobst",
        kwargs=kwargs.copy(),
    )
    for ns in num_samples:
        kwargs["n_samples"] = ns
        kwargs_global = kwargs.copy()
        # Global Volume
        kwargs_global["on_surface"] = False
        kwargs_global["add_normals"] = False
        env_str = f"{robot_register_name}-Boxes-{ns}Pts-Volume"
        register(
            id=f"{env_str}-v0",
            entry_point="mpenv.envs.boxes:boxes_pointcloud",
            kwargs=kwargs_global.copy(),
        )
        # Global Surface
        kwargs_global["on_surface"] = True
        for add_normals, str_normals in normals:
            kwargs_global["add_normals"] = add_normals
            env_str = f"{robot_register_name}-Boxes-{ns}Pts-Surface{str_normals}"
            register(
                id=f"{env_str}-v0",
                entry_point="mpenv.envs.boxes:boxes_pointcloud",
                kwargs=kwargs_global.copy(),
            )
        # Ray Tracing
        kwargs_local = kwargs.copy()
        kwargs_local["n_rays"] = 256
        env_str = f"{robot_register_name}-Boxes-{ns}Pts-Rays"
        register(
            id=f"{env_str}-v0",
            entry_point="mpenv.envs.boxes:boxes_raytracing",
            kwargs=kwargs_local.copy(),
        )

"""
Boxes Qureshi
"""
for ns in num_samples:
    kwargs = {"n_samples": ns}
    kwargs_global = kwargs.copy()
    kwargs_global["on_surface"] = False
    env_str = f"Sphere-Qureshi-{ns}Pts-Volume"
    register(
        id=f"{env_str}-v0",
        entry_point="mpenv.envs.qureshi:qureshi_pointcloud",
        kwargs=kwargs_global.copy(),
    )
    kwargs_global["on_surface"] = True
    env_str = f"Sphere-Qureshi-{ns}Pts-SurfaceNormals"
    register(
        id=f"{env_str}-v0",
        entry_point="mpenv.envs.qureshi:qureshi_pointcloud",
        kwargs=kwargs_global.copy(),
    )
    kwargs_local = kwargs.copy()
    kwargs_local["n_rays"] = 256
    env_str = f"Sphere-Qureshi-{ns}Pts-Rays"
    register(
        id=f"{env_str}-v0",
        entry_point="mpenv.envs.qureshi:qureshi_raytracing",
        kwargs=kwargs_local.copy(),
    )

"""
Kavraki Slot
"""
kwargs = {"robot_name": "s_shape", "n_rays": 256}
env_str = "KavrakiShape-Slot"
register(
    id=f"{env_str}-v0", entry_point="mpenv.envs.slot:slot_noobst", kwargs=kwargs.copy(),
)
for ns in num_samples:
    kwargs["n_samples"] = ns
    env_str = f"SShape-Slot-{ns}Pts-Rays"
    register(
        id=f"{env_str}-v0",
        entry_point="mpenv.envs.slot:slot_raytracing",
        kwargs=kwargs.copy(),
    )

"""
Narrow 2D environments
"""

num_rooms = [None, 1, 2, 5, 10, 50, 100, 1000]
num_rooms_max_idx = [None, 1, 3, 12, 22, 155, 282, 2727]
visibility_dist = [0.2, 0.5, 1]

for (num_room, max_idx) in zip(num_rooms, num_rooms_max_idx):
    str_room = ""
    if num_room is not None:
        if num_room > 1:
            str_room = f"{num_room}Rooms"
        else:
            str_room = f"{num_room}Room"
    str_register_env = f"Narrow{str_room}-Corners-v0"
    kwargs = {"max_env_idx": max_idx}
    register(
        id=str_register_env,
        entry_point="mpenv.envs.narrow:narrow_corners",
        kwargs=kwargs.copy(),
    )
    for ns in num_samples:
        kwargs["n_samples"] = ns
        kwargs["on_surface"] = False
        kwargs["add_normals"] = False
        kwargs["coordinate_frame"] = "global"
        str_register_env = f"Narrow{str_room}-{ns}Pts-GlobalVolume-v0"
        register(
            id=str_register_env,
            entry_point="mpenv.envs.narrow:narrow_pointcloud",
            kwargs=kwargs.copy(),
        )
        kwargs["coordinate_frame"] = "local"
        str_register_env = f"Narrow{str_room}-{ns}Pts-LocalVolume-v0"
        register(
            id=str_register_env,
            entry_point="mpenv.envs.narrow:narrow_pointcloud",
            kwargs=kwargs.copy(),
        )
        for add_normals, str_normals in normals:

            kwargs["on_surface"] = True
            kwargs["add_normals"] = add_normals

            # Global Coordinate Frame
            str_register_env = f"Narrow{str_room}-{ns}Pts-GlobalSurface{str_normals}-v0"
            kwargs["coordinate_frame"] = "global"
            register(
                id=str_register_env,
                entry_point="mpenv.envs.narrow:narrow_pointcloud",
                kwargs=kwargs.copy(),
            )

            # Local Coordinate Frame
            str_register_env = f"Narrow{str_room}-{ns}Pts-LocalSurface{str_normals}-v0"
            kwargs["coordinate_frame"] = "local"
            register(
                id=str_register_env,
                entry_point="mpenv.envs.narrow:narrow_pointcloud",
                kwargs=kwargs.copy(),
            )

    # Global Image
    str_register_env = f"Narrow{str_room}-GlobalImage-v0"
    kwargs = {"max_env_idx": max_idx, "size": (64, 64), "pov": "global"}
    register(
        id=str_register_env,
        entry_point="mpenv.envs.narrow:narrow_image",
        kwargs=kwargs.copy(),
    )

    # Local Image
    str_register_env = f"Narrow{str_room}-LocalImage-v0"
    kwargs = {"max_env_idx": max_idx, "size": (64, 64), "pov": "local"}
    register(
        id=str_register_env,
        entry_point="mpenv.envs.narrow:narrow_image",
        kwargs=kwargs.copy(),
    )
