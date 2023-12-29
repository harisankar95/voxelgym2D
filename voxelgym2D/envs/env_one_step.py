"""Voxel environment corresponding to Onestep action space"""

from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .base_env import BaseEnv


class VoxelGymOneStep(BaseEnv):
    """Voxel environment corresponding to Onestep action space"""

    def __init__(
        self,
        render_mode: Optional[str] = None,
        mapfile: str = "600x600.npy",
        view_size: int = 21,
        image_size: int = 42,
        max_collisions: int = 0,
        max_steps: int = 60,
        show_path: bool = True,
        multi_output: bool = False,
        partial_reward: bool = True,
        inference_mode: bool = False,
        discrete_actions: bool = True,
        log_level: str = "ERROR",
    ):
        """
        Parameters
        ----------
        render_mode : Optional[str], optional
            render mode, by default None
        mapfile : str
            name of the map file in the maps folder
        view_size : int
            size of the view window for observation
        image_size : int
            size of the image to be returned as observation
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
        inference_mode : bool
            whether to run in inference mode
        discrete_actions : bool
            whether to use discrete actions
        log_level : str, optional
            log level, by default "ERROR". One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        """

        super().__init__(
            render_mode,
            mapfile,
            view_size,
            image_size,
            max_collisions,
            max_steps,
            show_path,
            multi_output,
            partial_reward,
            inference_mode,
            log_level,
        )

        self.discrete_actions = discrete_actions
        if self.discrete_actions:
            self.action_space = gym.spaces.Discrete(len(self.delta))
        else:
            # self.action_space = gym.spaces.Box(-1, 1, shape=(len(self.delta)))
            self.action_space = gym.spaces.Box(-1, 1, shape=[1])

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
            return reward_completion

        reward_completion = -self.action_cost

        _new_astarPath, _cost_astar, self.astar_runs = self._run_astar(self._target_location)
        if len(_new_astarPath) > 0:
            self.astarPath = _new_astarPath
            self.cost_astar = _cost_astar

            reward_euc_astar = self.min_cost_astar - _cost_astar
            improved = self.min_cost_astar > _cost_astar
        else:
            # fallback to euclidean distance if astar fails
            current_distance_to_target = float(np.linalg.norm(self._target_location - self._agent_location))
            reward_euc_astar = self.min_distance_to_target - current_distance_to_target
            improved = self.min_distance_to_target > current_distance_to_target

        if self.partial_reward:
            if improved:
                return round((2 * reward_euc_astar + reward_completion) / 10, 4)
            return round((reward_euc_astar + reward_completion) / 10, 4)

        return round(reward_completion / 10, 4)

    @staticmethod
    def action_to_bins(action: np.ndarray) -> int:
        """
        Converts the action to bins of size 1/4 and returns the bin number in the range [0, 7] for actions

        Parameters
        ----------
        action : np.ndarray
            action to be converted to bin number

        Returns
        -------
        bin : int
            bin number in the range [0, 7]
        """
        # Clip action to [-1, 1]
        clipped = np.clip(action, -1, 1)
        return min(int((clipped + 1) * 4), 7)

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
            # convert angle to bin number
            action_idx = self.action_to_bins(action)
            self.action = list(self.delta[action_idx])
        else:
            self.action = list(self.delta[int(action)])

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

        # compute action cost
        self.action_cost = 0.0
        for _, dx in enumerate(np.array(valid_action_path[1:]) - np.array(valid_action_path[:-1])):
            self.action_cost += np.sqrt(np.sum(dx**2))
        self.action_cost = round(self.action_cost, 4)

        return valid_action_path, collision
