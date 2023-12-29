from gymnasium.envs.registration import register

register(
    id="onestep-v0",
    entry_point="voxelgym2D.envs:VoxelGymOneStep",
    nondeterministic=True,
)
