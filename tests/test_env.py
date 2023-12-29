"""Test the environment."""

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from voxelgym2D.envs import VoxelGymOneStep


def test_onsestep():
    """test onestep env"""
    env = gym.make("voxelgym2D:onestep-v0")
    check_env(env.unwrapped, skip_render_check=True)
    _, i = env.reset(seed=1234)
    # assert i is a dict
    assert isinstance(i, dict)

    del env
    # test continuous action space
    env = gym.make(
        "voxelgym2D:onestep-v0",
        discrete_actions=False,
    )
    check_env(env.unwrapped, skip_render_check=True)


def test_action_to_bins():
    """Test the action to bins function"""
    assert VoxelGymOneStep.action_to_bins(np.array([-1])) == 0
    assert VoxelGymOneStep.action_to_bins(np.array([-0.75])) == 1
    assert VoxelGymOneStep.action_to_bins(np.array([-0.5])) == 2
    assert VoxelGymOneStep.action_to_bins(np.array([-0.25])) == 3
    assert VoxelGymOneStep.action_to_bins(np.array([0])) == 4

    assert VoxelGymOneStep.action_to_bins(np.array([0.25])) == 5
    assert VoxelGymOneStep.action_to_bins(np.array([0.5])) == 6
    assert VoxelGymOneStep.action_to_bins(np.array([0.75])) == 7
    assert VoxelGymOneStep.action_to_bins(np.array([1])) == 7
