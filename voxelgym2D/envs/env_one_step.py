"""Environment corresponding to Onestep action space"""

from typing import List, Tuple

import gym
import numpy as np

from .base_env import BaseEnv


class VoxelGymOneStep(BaseEnv):
    """Environment corresponding to Onestep action space"""

    def __init__(
        self,
        mapfile: str = "200x200x200_dense.npy",
        view_size: int = 21,
        max_collisions: int = 0,
        max_steps: int = 60,
        show_path: bool = True,
        multi_output: bool = False,
        partial_reward: bool = True,
        image_size: int = 42,
        test_mode: bool = False,
        inference_mode: bool = False,
        discrete_actions: bool = True,
    ):
        """
        Parameters
        ----------
        mapfile : str
            name of the map file in the maps folder
        view_size : int
            size of the view window for observation
        max_collisions : int
            maximum number of collisions allowed before episode ends
        max_steps : int
            maximum number of steps allowed before episode ends
        show_path : bool
            whether to show the last travesed action path in the observation
        multi_output : bool
            whether to add additional outputs in the observation
        partial_reward : bool
            whether to give rewards for each step
        image_size : int
            size of the image to be returned as observation
        test_mode : bool
            whether to run in test mode, for evaluation during training
        inference_mode : bool
            whether to run in inference mode
        discrete_actions : bool
            whether to use discrete actions
        """

        super().__init__(
            mapfile,
            view_size,
            max_collisions,
            max_steps,
            show_path,
            multi_output,
            partial_reward,
            image_size,
            test_mode,
            inference_mode,
        )

        self.discrete_actions = discrete_actions
        if self.discrete_actions:
            self.action_space = gym.spaces.Discrete(len(self.delta))
        else:
            self.action_space = gym.spaces.Box(-1, 1, shape=(len(self.delta),))

    def _compute_reward(self, completion_reward: bool = False) -> float:
        """
        Computes the reward for the current step

        Parameters
        ----------
        completion_reward : bool
            if True, returns the terminal reward for reaching the target,
            else returns the reward for the current step

        Returns
        -------
        reward : float
            reward for the current step
        """
        # eps_ = 1e-8
        reward_euc_astar = 0.0

        if completion_reward:
            reward_completion = 0.5

        else:
            reward_completion = -self.action_cost

            _new_astarPath, _cost_astar, self.astar_runs = self.run_astar(self._target_location)
            if len(_new_astarPath) > 0:
                self.astarPath = _new_astarPath
                self.cost_astar = _cost_astar

                reward_euc_astar = self.min_cost_astar - _cost_astar
            else:
                # fallback to euclidean distance if astar fails
                reward_euc_astar = self.min_distance_to_target - float(
                    np.linalg.norm(self._target_location - self._agent_location)
                )

        if completion_reward:
            return reward_completion
        if (self.cost_astar < self.min_cost_astar) and self.partial_reward:
            return round((2 * reward_euc_astar + reward_completion) / 10, 4)
        if (self.cost_astar > self.min_cost_astar) and self.partial_reward:
            return round((reward_euc_astar + reward_completion) / 10, 4)
        return round(reward_completion / 10, 4)

    def _take_action(self, action: np.ndarray) -> Tuple[List, bool]:
        """
        Takes the action and updates the agent location

        Parameters
        ----------
        action : np.ndarray
            action to be taken

        Returns
        -------
        new_agent_location : List
            new agent location along with the path taken
        collision : bool
            True if the agent collides with an obstacle, else False
        """
        if not self.discrete_actions:
            self.action = list(self.delta[np.argmax(action)])
        else:
            self.action = list(self.delta[int(action)])

        self.action_cost = float(np.linalg.norm(self.action))
        action_location = self._agent_location + np.array(self.action, dtype=np.int32)
        action_path = [action_location.tolist()]

        valid_action_path = []
        collision = False
        # check if the action is valid
        for pt in action_path:
            if pt == self._target_location.tolist():
                valid_action_path.append(pt)
                break
            if self._new_world[pt[1], pt[0]] == 1:
                collision = True
                break
            valid_action_path.append(pt)
        # complete path by including initial location
        valid_action_path.insert(0, self._agent_location.tolist())

        return valid_action_path, collision
