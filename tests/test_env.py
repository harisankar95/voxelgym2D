"""Test the environment."""

import gym
import numpy as np
import pytest
from gym.utils.env_checker import check_env

import voxelgym2D


def test_onsestep():
    """test onestep env"""
    env = gym.make("voxelgym2D:onestep-v0")
    check_env(env)
    _ = env.reset()
    _, i = env.reset(return_info=True)
    # assert i is a dict
    assert isinstance(i, dict)

    del env
    # test continuous action space
    env = gym.make(
        "voxelgym2D:onestep-v0",
        discrete_actions=False,
        inference_mode=True,
        multi_output=True,
    )
    check_env(env)
